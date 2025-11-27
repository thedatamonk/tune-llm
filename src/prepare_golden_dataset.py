#!/usr/bin/env python
"""
Prepare a golden evaluation set from keivalya/MedQuad-MedicalQnADataset.

Usage (example):
    python prepare_golden_dataset.py \
        --dataset-name keivalya/MedQuad-MedicalQnADataset \
        --split train \
        --golden-size 300 \
        --seed 42 \
        --output-golden golden_medquad.jsonl \
        --output-indices golden_indices.json \
        --output-remaining remaining_medquad.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

def prepare_golden(
    dataset_name: str,
    split: str,
    golden_size: int,
    seed: int,
    output_golden: Path,
    output_indices: Path,
    output_remaining: Path | None = None,
) -> None:
    # Load dataset
    ds = load_dataset(dataset_name, split=split)

    # We expect columns: "qtype", "Question", "Answer"
    # Filter out empty rows if any
    valid_indices = []
    for i, row in enumerate(ds):
        q = row.get("Question", "")
        a = row.get("Answer", "")
        if isinstance(q, str) and isinstance(a, str):
            if q.strip() and a.strip():
                valid_indices.append(i)

    if not valid_indices:
        raise RuntimeError("No valid rows found in dataset after filtering.")
    
    # Sample golden indices
    random.seed(seed)
    if golden_size > len(valid_indices):
        print(
            f"[WARN] Requested golden_size={golden_size} "
            f"but only {len(valid_indices)} valid rows. Using all valid rows."
        )
        golden_size = len(valid_indices)

    golden_indices = sorted(random.sample(valid_indices, golden_size))

    # Write golden JSONL
    output_golden.parent.mkdir(parents=True, exist_ok=True)

    with output_golden.open("w", encoding="utf-8") as f_out:
        for idx_in_golden, original_idx in enumerate(golden_indices):
            row = ds[original_idx]
            qtype = row.get("qtype", None)
            question = row.get("Question", "")
            answer = row.get("Answer", "")

            record = {
                "id": f"golden_{idx_in_golden:06d}",
                "dataset_name": dataset_name,
                "original_index": original_idx,
                "question": question,
                "answer": answer,
                "question_type": qtype,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote golden set ({golden_size} examples) to {output_golden}")

    # Save golden indices metadata
    indices_payload = {
        "dataset_name": dataset_name,
        "split": split,
        "seed": seed,
        "golden_size": golden_size,
        "golden_indices": golden_indices,
    }

    output_indices.parent.mkdir(parents=True, exist_ok=True)
    output_indices.write_text(json.dumps(indices_payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote golden_indices metadata to {output_indices}")

    # Optionally write remaining examples as JSONL
    if output_remaining is not None:
        golden_set = set(golden_indices)
        output_remaining.parent.mkdir(parents=True, exist_ok=True)
        with output_remaining.open("w", encoding="utf-8") as f_rem:
            running_id = 0
            for i, row in enumerate(ds):
                if i in golden_set:
                    continue
                qtype = row.get("qtype", None)
                question = row.get("Question", "")
                answer = row.get("Answer", "")
                if not question.strip() or not answer.strip():
                    continue

                rec = {
                    "id": f"remaining_{running_id:06d}",
                    "dataset_name": dataset_name,
                    "original_index": i,
                    "question": question,
                    "answer": answer,
                    "question_type": qtype,
                }
                f_rem.write(json.dumps(rec, ensure_ascii=False) + "\n")
                running_id += 1

        print(
            f"[INFO] Wrote remaining set "
            f"(excluding golden) to {output_remaining}"
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare golden eval set"
    )

    p.add_argument(
        "--dataset-name",
        type=str,
        default="keivalya/MedQuad-MedicalQnADataset",
        help="HF dataset name (default: keivalya/MedQuad-MedicalQnADataset)",
    )

    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    p.add_argument(
        "--golden-size",
        type=int,
        default=300,
        help="Number of examples in golden set",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling golden set",
    )
    p.add_argument(
        "--output-golden",
        type=str,
        default="./data/golden_medquad.jsonl",
        help="Path to write golden JSONL",
    )
    p.add_argument(
        "--output-indices",
        type=str,
        default="./data/golden_indices.json",
        help="Path to write golden indices metadata JSON",
    )
    p.add_argument(
        "--output-remaining",
        type=str,
        default="./data/remaining_medquad.jsonl",
        help="Optional path to write remaining (non-golden) JSONL",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_golden = Path(args.output_golden)
    out_indices = Path(args.output_indices)
    out_remaining = Path(args.output_remaining) if args.output_remaining else None

    prepare_golden(
        dataset_name=args.dataset_name,
        split=args.split,
        golden_size=args.golden_size,
        seed=args.seed,
        output_golden=out_golden,
        output_indices=out_indices,
        output_remaining=out_remaining,
    )


    

