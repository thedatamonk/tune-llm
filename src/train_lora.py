#!/usr/bin/env python
import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of SmolLM on MedQuad SFT data.")

    parser.add_argument(
        "--base-model-name-or-path",
        type=str,
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="Base model checkpoint to start from.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="Path to train SFT JSONL (with 'messages' column).",
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        required=True,
        help="Path to validation SFT JSONL (with 'messages' column).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/smollm_medquad_lora",
        help="Directory to save adapter checkpoints and logs.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length for tokenization.",
    )

    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (controls effective batch size).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (conservative = small).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=20,
        help="Log every N steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=400,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="When to run evaluation.",
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Run evaluation every N steps when evaluation_strategy='steps'.",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Use bfloat16 if supported by GPU.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use float16 if preferred. If both bf16 and fp16 are false, use full precision.",
    )


    parser.add_argument(
    "--sanity-check",
    action="store_true",
    help="Run a quick sanity check on a small subset of the data.",
    )

    parser.add_argument(
        "--sanity-train-size",
        type=int,
        default=200,
        help="Number of training samples for sanity mode."
    )

    parser.add_argument(
        "--sanity-val-size",
        type=int,
        default=50,
        help="Number of val samples for sanity mode."
    )

    parser.add_argument(
        "--sanity-max-steps",
        type=int,
        default=300,
        help="Max steps for sanity mode."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        use_fast=True,
    )

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        device_map="auto",
    )

    # Handle padding token (SmolLM may not have one -> use eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. Load datasets from JSONL
    train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    val_dataset = load_dataset("json", data_files=args.val_jsonl, split="train")

    if args.sanity_check:
        print("=== SANITY CHECK MODE ENABLED ===")
        
        train_subset_size = min(args.sanity_train_size, len(train_dataset))
        val_subset_size = min(args.sanity_val_size, len(val_dataset))

        train_dataset = train_dataset.select(range(train_subset_size))
        val_dataset = val_dataset.select(range(val_subset_size))

        print(f"Train subset: {train_subset_size}")
        print(f"Val subset:   {val_subset_size}")

    # 3. Define formatting function using the model's chat_template
    def formatting_prompts_func(examples):
        """
        TRL's SFTTrainer will call this with a batch of examples.
        We turn each 'messages' list into a single string using tokenizer.chat_template.
        """
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    # 4. LoRA config (conservative)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # This target_modules list is model-architecture-specific; adjust if needed.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Wrap model in PEFT adapter
    model = get_peft_model(model, lora_config)

    # 5. Training arguments
    # training_args = TrainingArguments(
    #     output_dir=str(output_dir),
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     per_device_eval_batch_size=args.per_device_eval_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     learning_rate=args.learning_rate,
    #     num_train_epochs=args.num_train_epochs,
    #     logging_steps=args.logging_steps,
    #     save_steps=args.save_steps,
    #     save_strategy=args.evaluation_strategy,
    #     eval_strategy=args.evaluation_strategy,
    #     eval_steps=args.eval_steps,
    #     save_total_limit=3,
    #     bf16=args.use_bf16,
    #     fp16=args.use_fp16,
    #     report_to=["none"],  # or ["wandb"] if you want
    #     load_best_model_at_end=((not args.sanity_check) and args.evaluation_strategy != "no"),
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    # )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        bf16=args.use_bf16,
        fp16=args.use_fp16,
        report_to=["none"],  # or ["wandb"] if you want
        max_length=args.max_seq_length,   # e.g. 512 / 1024 / 2048
        packing=False,                        # 1 example per sequence
        load_best_model_at_end=(
            (not args.sanity_check) and args.evaluation_strategy != "no"
        ),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if args.sanity_check:
        sft_config.max_steps = args.sanity_max_steps
        sft_config.num_train_epochs = 1
        sft_config.save_strategy = "no"
        sft_config.eval_strategy = "steps"
        print("Sanity mode: overriding training settings:")
        print(f"  max_steps = {sft_config.max_steps}")
        print(f"  save_strategy = {sft_config.save_strategy}")


    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.evaluation_strategy != "no" else None,
        args=sft_config,
        # formatting_func=formatting_prompts_func,
        # max_seq_length=args.max_seq_length,
        # packing=False,  # Keep one example per sequence for clarity
    )

    # 7. Train & save adapter
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))

    # Also save tokenizer (with pad_token set) and LoRA config
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    lora_config.save_pretrained(str(output_dir / "final_adapter"))

    print(f"Training complete. LoRA adapter saved to: {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()



