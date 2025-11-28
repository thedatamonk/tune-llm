#!/bin/bash
set -e

python src/prepare_sft_data.py \
  --input-jsonl ./datasets/remaining_medquad.jsonl \
  --output-train-jsonl ./datasets/medquad_train_sft.jsonl \
  --output-val-jsonl ./datasets/medquad_val_sft.jsonl \
  --val-size 100
