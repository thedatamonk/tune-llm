import argparse
import json
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_config import (BASE_MODEL_NAME_OR_PATH, BATCH_SIZE, DEVICE,
                          GEN_KWARGS, LORA_ADAPTER_PATH, MAX_INPUT_LENGTH,
                          PATHS, build_prompt)
from eval_datasets import load_dataset

def load_models_and_tokenizer():
    """
    Load tokenizer, base model, and LoRA model.
    Returns (tokenizer, base_model, lora_model).
    """
    print(f"Loading tokenizer from {BASE_MODEL_NAME_OR_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_OR_PATH)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        # standard trick: use eos_token as pad
        tokenizer.pad_token = tokenizer.eos_token

    # For decoder-only it is often convenient to left pad
    tokenizer.padding_side = "left"

    print(f"Loading base model from {BASE_MODEL_NAME_OR_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
    )
    base_model.to(DEVICE)
    base_model.eval()

    print(f"Loading LoRA adapters from {LORA_ADAPTER_PATH}...")
    # load a separate copy for LoRA
    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
    )
    lora_model = PeftModel.from_pretrained(
        lora_base,
        LORA_ADAPTER_PATH,
    )
    lora_model.to(DEVICE)
    lora_model.eval()

    return tokenizer, base_model, lora_model


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    questions: List[str],
) -> List[str]:
    """
    Generate answers for a batch of questions using the given model.
    Returns list of answers (same order as questions).
    """

    prompts = [build_prompt(q) for q in questions]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    padded_seq_len = input_ids.size(1)


    gen_kwargs = dict(GEN_KWARGS)
    if gen_kwargs.get("pad_token_id") is None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    outputs = model.generate(
        input_ids=input_ids,    
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # Decode answers by stripping the prompt tokens
    answers: List[str] = []
    for i in range(outputs.size(0)):
        generated_ids = outputs[i, padded_seq_len:]
        if generated_ids.numel() == 0:
            text = ""
        else:
            text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        answers.append(text.strip())
    return answers

def save_predictions(
    dataset_name: str,
    data: List[dict],
    base_answers: List[str],
    lora_answers: List[str],
):
    outputs_dir = Path(PATHS.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = outputs_dir / f"{dataset_name}_predictions.jsonl"
    print(f"Writing predictions to {out_path} ...")

    if not (len(data) == len(base_answers) == len(lora_answers)):
        raise ValueError(
            f"Length mismatch: data={len(data)}, base={len(base_answers)}, lora={len(lora_answers)}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for ex, base_ans, lora_ans in zip(data, base_answers, lora_answers):
            rec = {
                "id": ex["id"],
                "source": ex["source"],
                "question": ex["question"],
                "ground_truth": ex.get("answer"),
                "base_answer": base_ans,
                "lora_answer": lora_ans,
                "meta": ex.get("meta"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def run_eval_for_dataset(dataset_name: str):
    print(f"Loading dataset: {dataset_name}")
    data = load_dataset(dataset_name)
    print(f"Loaded {len(data)} examples.")

    tokenizer, base_model, lora_model = load_models_and_tokenizer()

    questions = [ex["question"] for ex in data]

    base_answers: List[str] = []
    lora_answers: List[str] = []

    for start in range(0, len(questions), BATCH_SIZE):
        batch_questions = questions[start : start + BATCH_SIZE]
        print(
            f"Generating for batch {start} - {start + len(batch_questions)} / {len(questions)}"
        )

        batch_base = generate_batch(base_model, tokenizer, batch_questions)
        base_answers.extend(batch_base)

        batch_lora = generate_batch(lora_model, tokenizer, batch_questions)
        lora_answers.extend(batch_lora)

    save_predictions(dataset_name, data, base_answers, lora_answers)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "medquad_val",
            "medquad_golden",
            "handwritten_eval_v1",
            "safety_probes_v1",
        ],
    )
    args = parser.parse_args()
    run_eval_for_dataset(args.dataset)


if __name__ == "__main__":
    main()