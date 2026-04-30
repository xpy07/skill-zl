#!/bin/bash
# DistillKit Logit KD 蒸馏训练
# 需要 4×A100 80GB

set -e

CONFIG="${CONFIG:-./configs/distill_config.yaml}"

echo "=== Patent Agent Distill: Logit KD Training ==="
echo "Config: $CONFIG"

distillkit "$CONFIG"

echo "=== Distillation complete ==="
echo "Model saved to: ./output/patent-agent-model"
