#!/bin/bash
set -e

python ./src/eval_on_golden.py \
  --model-name-or-path HuggingFaceTB/SmolLM-1.7B \
  --golden-path ./datasets/golden_medquad.jsonl \
  --output-predictions ./outputs/smollm_1_7B_base_on_golden.jsonl \
  --output-summary ./outputs/smollm_1_7B_base_on_golden_summary.json \
  --run-id smollm_1_7B_base_2025-11-27_v1 \
  --max-new-tokens 512 \
  --temperature 0.2 \
  --top-p 0.9 \
  --batch-size 8
  