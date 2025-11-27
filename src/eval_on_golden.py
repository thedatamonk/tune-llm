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
from pathlib import Path
from typing import Any, Dict, List
import torch

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
) -> List[Dict[str, Any]]:
    
    model.to(device)
    model.eval()

    results = []

    for ex in golden_examples:
        q = ex["question"]
        gold_a = ex["answer"]

        prompt = build_prompt(q)

        # tokenize the input prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        # generate output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Extract only the generated continuation
        generated_ids = output_ids[0]
        input_len = inputs["input_ids"].shape[1]
        gen_only_ids = generated_ids[input_len:]        # extract generated tokens only; not the input prompt tokens
        model_answer = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()         # deode the generated tokens

        result = {
            "id": ex["id"],
            "question": q,
            "gold_answer": gold_a,
            "model_answer": model_answer,
            "question_type": ex.get("question_type", None),
        }
        results.append(result)

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
    # ensure eos_token_id
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

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

    # Write outputs
    write_predictions(preds_path, predictions, args.model_name_or_path, args.run_id)
    write_summary(
        summary_path,
        model_name=args.model_name_or_path,
        run_id=args.run_id,
        num_examples=len(predictions),
        metrics=metrics,
    )

    print(f"[INFO] Wrote predictions to {preds_path}")
    print(f"[INFO] Wrote summary to {summary_path}")
    print("[INFO] Done.")