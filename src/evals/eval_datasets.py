# src/eval_datasets.py

import json
from typing import Dict, List, Optional

from eval_config import PATHS

Example = Dict  # each example is a dict with id, source, question, answer, meta(optional)


def _read_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _canonicalize_examples(
    raw: List[Dict],
    source_name: str,
    id_prefix: str,
    has_answer: bool,
) -> List[Example]:
    examples: List[Example] = []
    for idx, r in enumerate(raw):
        ex_id = r.get("id") or f"{id_prefix}_{idx}"
        question = r.get("question")
        if question is None:
            raise ValueError(f"Example {idx} in {source_name} missing 'question' field")

        answer: Optional[str] = None
        if has_answer:
            answer = r.get("answer")
            if answer is None:
                raise ValueError(
                    f"Example {ex_id} in {source_name} expected to have 'answer' field"
                )

        examples.append(
            {
                "id": ex_id,
                "source": source_name,
                "question": question,
                "answer": answer,
                "meta": r.get("meta"),
            }
        )
    return examples


def load_medquad_val() -> List[Example]:
    raw = _read_jsonl(PATHS.medquad_val)
    return _canonicalize_examples(
        raw=raw,
        source_name="medquad_val",
        id_prefix="medquad_val",
        has_answer=True,
    )


def load_medquad_golden() -> List[Example]:
    raw = _read_jsonl(PATHS.medquad_golden)
    return _canonicalize_examples(
        raw=raw,
        source_name="medquad_golden",
        id_prefix="medquad_golden",
        has_answer=True,
    )


def load_handwritten_eval_v1() -> List[Example]:
    raw = _read_jsonl(PATHS.handwritten_eval_v1)
    return _canonicalize_examples(
        raw=raw,
        source_name="handwritten_eval_v1",
        id_prefix="handwritten_eval_v1",
        has_answer=False,
    )


def load_safety_probes_v1() -> List[Example]:
    raw = _read_jsonl(PATHS.safety_probes_v1)
    return _canonicalize_examples(
        raw=raw,
        source_name="safety_probes_v1",
        id_prefix="safety_probes_v1",
        has_answer=False,
    )


def load_dataset(name: str) -> List[Example]:
    if name == "medquad_val":
        return load_medquad_val()
    elif name == "medquad_golden":
        return load_medquad_golden()
    elif name == "handwritten_eval_v1":
        return load_handwritten_eval_v1()
    elif name == "safety_probes_v1":
        return load_safety_probes_v1()
    else:
        raise ValueError(f"Unknown dataset name: {name}")
