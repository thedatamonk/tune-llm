#!/usr/bin/env python
"""
Evaluate a (base or finetuned) LLM on the golden MedQuad set.

Usage (base model example):
    python evaluate_on_golden.py \
        --model-name-or-path HuggingFaceTB/SmolLM-135M-Instruct \
        --golden-path golden_medquad.jsonl \
        --output-predictions smollm_base_on_golden.jsonl \
        --output-summary smollm_base_on_golden_summary.json \
        --run-id smollm_base_2025-11-27_v1

Later, for finetuned model, just change --model-name-or-path and --run-id.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import (add_lengths_and_flags, compute_metrics, write_predictions,
                     write_summary)


def load_golden(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data

def build_prompt(question: str) -> str:
    """
    Single, consistent prompt template for both base and finetuned models.
    """
    prompt = (
        "You are a safe, helpful medical assistant. "
        "Answer the user's question in clear, concise language. "
        "You are not a substitute for a doctor. "
        "Encourage the user to consult a qualified healthcare professional "
        "for personal medical decisions.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

def generate_answers(
    model,
    tokenizer,
    golden_examples: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    
    model.to(device)
    model.eval()

    results = []

    # this function will yield a new batch everytime it's called
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    for batch in tqdm(list(chunked(golden_examples, batch_size)), desc="Generating answers", unit="batch"):
        # build prompts for the batch inputs
        prompts = [build_prompt(ex["question"]) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        input_ids = inputs["input_ids"]
        
        # real (non-pad) lengths per row
        pad_id = tokenizer.pad_token_id

        # shape: [batch_size]
        input_lengths = (input_ids != pad_id).sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        # decode per example in the batch
        for ex, out_ids, in_len in zip(
            batch,
            outputs,
            input_lengths,
        ):
            # keep only newly generated tokens
            gen_only_ids = out_ids[int(in_len):]

            text = tokenizer.decode(
                gen_only_ids,
                skip_special_tokens=True,
            ).strip()

            results.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold_answer": ex["answer"],
                    "model_answer": text,
                    "question_type": ex.get("question_type", None),
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a causal LLM on the golden set"
    )

    p.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HF model id or local path (base or finetuned).",
    )
    p.add_argument(
        "--golden-path",
        type=str,
        default="golden_medquad.jsonl",
        help="Path to golden JSONL file.",
    )
    p.add_argument(
        "--output-predictions",
        type=str,
        default="predictions_on_golden.jsonl",
        help="Where to write per-example predictions JSONL.",
    )
    p.add_argument(
        "--output-summary",
        type=str,
        default="summary_on_golden.json",
        help="Where to write aggregated metrics JSON.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="run_1",
        help="Tag for this evaluation run (used in outputs).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cuda", or "cpu" (default: auto).',
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per answer.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling.",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batched generation.",
    )

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    golden_path = Path(args.golden_path)
    preds_path = Path(args.output_predictions)
    summary_path = Path(args.output_summary)

    golden_examples = load_golden(golden_path)
    print(f"[INFO] Loaded {len(golden_examples)} golden examples from {golden_path}")

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[INFO] Using device: {device}")

    # Load model + tokenizer
    print(f"[INFO] Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"     # we need to set this for decoder-only models


    # ensure eos_token_id
    # if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
    #     tokenizer.eos_token_id = tokenizer.pad_token_id

    # if no pad token, use eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Record start time
    start_time = time.time()

    # Generate answers
    predictions = generate_answers(
        model=model,
        tokenizer=tokenizer,
        golden_examples=golden_examples,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Add length + safety flags
    add_lengths_and_flags(tokenizer, predictions)

    # Compute metrics (BERTScore, ROUGE-L, safety rates)
    metrics = compute_metrics(predictions)

    elapsed = time.time() - start_time
    print(f"[INFO] Generation + metrics time: {elapsed:.2f} seconds "
          f"({elapsed / len(predictions):.3f} s/example)")

    # Write predictions and metrics summary
    write_predictions(preds_path, predictions, args.model_name_or_path, args.run_id)
    write_summary(
        summary_path,
        model_name=args.model_name_or_path,
        run_id=args.run_id,
        num_examples=len(predictions),
        metrics=metrics,
        total_gen_time_in_secs=elapsed,
    )

    print(f"[INFO] Wrote predictions to {preds_path}")
    print(f"[INFO] Wrote summary to {summary_path}")
    print("[INFO] Done.")