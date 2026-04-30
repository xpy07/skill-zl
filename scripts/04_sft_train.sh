#!/bin/bash
# SFT 微调 Student 模型
# 需要 2×A100 80GB

set -e

CONFIG="${CONFIG:-./configs/sft_config.yaml}"

echo "=== Patent Agent Distill: SFT Training ==="

python -m trl sft \
  --model_name Qwen/Qwen3-8B \
  --dataset_name ./data/sft \
  --output_dir ./output/patent-agent-sft \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --max_seq_length 4096 \
  --gradient_checkpointing true \
  --optim paged_adamw_8bit \
  --logging_steps 10 \
  --save_steps 500

echo "=== SFT training complete ==="
echo "Model saved to: ./output/patent-agent-sft"
