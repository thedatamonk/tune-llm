#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are a safe, helpful medical assistant. "
    "You provide clear, concise, and cautious information based on established medical knowledge. "
    "You do not make definitive diagnoses or prescribe specific treatments or dosages. "
    "You encourage users to consult qualified healthcare professionals for personal medical decisions."
)


CANDIDATE_QUESTION_KEYS = [
    "question",
    "query",
    "input",
    "prompt",
    "user_question",
]

CANDIDATE_ANSWER_KEYS = [
    "answer",
    "response",
    "output",
    "target",
    "ground_truth",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_system_prompt(system_prompt_file: Path | None) -> str:
    if system_prompt_file is None:
        return DEFAULT_SYSTEM_PROMPT

    with system_prompt_file.open("r", encoding="utf-8") as f:
        return f.read().strip()

def build_messages(system_prompt: str, question: str, answer: str) -> List[Dict[str, str]]:
    """
    Chat-style messages compatible with HF chat_template-based models.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT data (messages + train/val split) from a JSONL file."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to JSONL containing 'everything else besides holdout'.",
    )
    parser.add_argument(
        "--output-train-jsonl",
        type=Path,
        required=True,
        help="Where to write train SFT JSONL.",
    )
    parser.add_argument(
        "--output-val-jsonl",
        type=Path,
        required=True,
        help="Where to write validation SFT JSONL.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of validation samples to carve out from the input.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional path to a text file containing the system prompt. "
             "If not provided, a default safe medical assistant prompt is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting.",
    )

    args = parser.parse_args()

    data = read_jsonl(args.input_jsonl)

    system_prompt = load_system_prompt(args.system_prompt_file)

    # Build messages field for each example
    processed = []
    for ex in data:
        question = str(ex["question"])
        answer = str(ex["answer"])

        messages = build_messages(system_prompt, question, answer)

        processed.append(
            {
                "messages": messages,
                "question": question,
                "answer": answer,
            }
        )

    # Shuffle & split
    random.seed(args.seed)
    random.shuffle(processed)

    val_size = min(args.val_size, len(processed) - 1) if len(processed) > 1 else len(processed)
    val_data = processed[:val_size]
    train_data = processed[val_size:]

    print(f"Total samples: {len(processed)} | train: {len(train_data)} | val: {len(val_data)}")

    write_jsonl(args.output_train_jsonl, train_data)
    write_jsonl(args.output_val_jsonl, val_data)

    print(f"Wrote train to {args.output_train_jsonl}")
    print(f"Wrote val   to {args.output_val_jsonl}")


if __name__ == "__main__":
    main()