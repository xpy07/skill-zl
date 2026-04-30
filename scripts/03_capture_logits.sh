#!/bin/bash
# 捕获 Teacher (Qwen3-235B) 的 logit 分布
# 需要 8×A100 80GB 或同等算力

set -e

MODEL_PATH="${TEACHER_MODEL:-Qwen/Qwen3-235B}"
DATASET_PATH="${DATASET_PATH:-./data/sft/train.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-./data/logits_capture}"
COMPRESS_CONFIG="${COMPRESS_CONFIG:-./configs/compression_config.yaml}"

echo "=== Patent Agent Distill: Logit Capture ==="
echo "Teacher Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "Compression Config: $COMPRESS_CONFIG"

python -m distillkit.sample_logits_vllm \
  --model "$MODEL_PATH" \
  --dataset "$DATASET_PATH" \
  --output "$OUTPUT_PATH" \
  --compression-config "$COMPRESS_CONFIG"

echo "=== Logit capture complete ==="
