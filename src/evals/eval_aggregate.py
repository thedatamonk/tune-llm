import argparse
import json
from pathlib import Path
from typing import Dict, List

from eval_config import PATHS


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _aggregate_for_model(
    scores: List[Dict], model_key: str = "lora"
) -> Dict[str, float]:
    """
    Compute averages for A, B, C for given model_key ("base" or "lora").
    Ignores None values.
    """
    sum_A = 0.0
    sum_B = 0.0
    sum_C = 0.0
    count_A = 0
    count_B = 0
    count_C = 0

    for rec in scores:
        s = rec["scores"].get(model_key, {})
        A = s.get("A")
        B = s.get("B")
        C = s.get("C")

        if A is not None:
            sum_A += float(A)
            count_A += 1
        if B is not None:
            sum_B += int(B)
            count_B += 1
        if C is not None:
            sum_C += int(C)
            count_C += 1

    avg_A = sum_A / count_A if count_A > 0 else float("nan")
    avg_B = sum_B / count_B if count_B > 0 else float("nan")
    avg_C = sum_C / count_C if count_C > 0 else float("nan")

    return {
        "avg_A": avg_A,
        "avg_B": avg_B,
        "avg_C": avg_C,
        "n_A": count_A,
        "n_B": count_B,
        "n_C": count_C,
    }


def _unsafe_rate(scores: List[Dict], model_key: str = "lora") -> float:
    """
    For safety probes: fraction of examples where B == 0 (unsafe).
    Ignores None values.
    """
    total = 0
    unsafe = 0
    for rec in scores:
        s = rec["scores"].get(model_key, {})
        B = s.get("B")
        if B is None:
            continue
        total += 1
        if int(B) == 0:
            unsafe += 1
    if total == 0:
        return float("nan")
    return unsafe / total


def aggregate_scores(dataset_name: str):
    scores_path = Path(PATHS.outputs_dir) / f"{dataset_name}_scores.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    print(f"Loading scores from {scores_path} ...")
    scores = _read_jsonl(scores_path)
    print(f"Loaded {len(scores)} scored records.")

    base_stats = _aggregate_for_model(scores, "base")
    lora_stats = _aggregate_for_model(scores, "lora")

    print("\n=== Aggregated scores ===")
    print(f"Dataset: {dataset_name}")
    print("\nBase model:")
    print(
        f"  A (correctness/usefulness): {base_stats['avg_A']:.3f} (n={base_stats['n_A']})"
    )
    print(f"  B (safety):                 {base_stats['avg_B']:.3f} (n={base_stats['n_B']})")
    print(f"  C (clarity/tone):           {base_stats['avg_C']:.3f} (n={base_stats['n_C']})")

    print("\nLoRA model:")
    print(
        f"  A (correctness/usefulness): {lora_stats['avg_A']:.3f} (n={lora_stats['n_A']})"
    )
    print(f"  B (safety):                 {lora_stats['avg_B']:.3f} (n={lora_stats['n_B']})")
    print(f"  C (clarity/tone):           {lora_stats['avg_C']:.3f} (n={lora_stats['n_C']})")

    # Unsafe rate (mainly meaningful for safety_probes_v1, but harmless elsewhere)
    base_unsafe = _unsafe_rate(scores, "base")
    lora_unsafe = _unsafe_rate(scores, "lora")
    print("\nUnsafe rate (B == 0):")
    print(f"  Base: {base_unsafe * 100:.1f}%")
    print(f"  LoRA: {lora_unsafe * 100:.1f}%")

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
    aggregate_scores(args.dataset)


if __name__ == "__main__":
    main()