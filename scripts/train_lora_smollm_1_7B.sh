#!/bin/bash
set -e

python src/train_lora.py \
  --base-model-name-or-path HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --train-jsonl ./datasets/medquad_train_sft.jsonl \
  --val-jsonl ./datasets/medquad_val_sft.jsonl \
  --output-dir ./finetuned_outputs/smollm2_1_7b_medquad_lora_full_train \
  --max-seq-length 512 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --num-train-epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --use-bf16 \
  --evaluation-strategy "steps" \
  --eval-steps 200 \
  --save-steps 200 \
  --logging-steps 50
