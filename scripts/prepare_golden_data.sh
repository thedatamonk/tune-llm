#!/bin/bash
set -e

python ./src/prepare_golden_dataset.py \
  --dataset-name keivalya/MedQuad-MedicalQnADataset \
  --split train \
  --golden-size 100 \
  --seed 42 \
  --output-golden ./datasets/golden_medquad.jsonl \
  --output-indices ./datasets/golden_indices.json \
  --output-remaining ./datasets/remaining_medquad.jsonl