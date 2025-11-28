#!/bin/bash
set -e

python src/train_lora.py \
  --train-jsonl ./datasets/medquad_train_sft.jsonl \
  --val-jsonl ./datasets/medquad_val_sft.jsonl \
  --output-dir ./finetuned_outputs/smollm_135m_medquad_lora \
  --max-seq-length 512 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-train-epochs 2 \
  --lora-rank 8 \
  --use-bf16
