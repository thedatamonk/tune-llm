#!/usr/bin/env python3
import json
from pathlib import Path
import argparse


def convert_jsonl_to_labelstudio_json(input_path: Path, output_path: Path) -> None:
    """
    Convert our flat predictions JSONL file into a Label Studio-compatible JSON file.

    Input JSONL line schema (one per line):
      {
        "id": "...",
        "source": "...",
        "question": "...",
        "ground_truth": "... or null",
        "base_answer": "...",
        "lora_answer": "...",
        "meta": {...}  # optional
      }

    Output JSON file schema (a single JSON array):
      [
        {
          "id": "...",
          "data": {
            "question": "...",
            "base_answer": "...",
            "lora_answer": "...",
            "ground_truth": "... or null",
            "meta": {...}
          }
        },
        ...
      ]
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Converting predictions for Label Studio:")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")

    tasks = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {input_path}: {e}") from e

            task = {
                "id": rec.get("id"),
                "data": {
                    "question": rec.get("question"),
                    "base_answer": rec.get("base_answer"),
                    "lora_answer": rec.get("lora_answer"),
                    "ground_truth": rec.get("ground_truth"),
                    "meta": rec.get("meta"),
                },
            }
            tasks.append(task)

    # Write as a single JSON array (Label Studio-friendly)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(tasks, fout, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(tasks)} tasks.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert predictions JSONL to Label Studio JSON (array of tasks)."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to original predictions JSONL file (from eval_run.py)",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path where Label Studio JSON file will be written (e.g. *.json)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    convert_jsonl_to_labelstudio_json(input_path, output_path)


if __name__ == "__main__":
    main()
