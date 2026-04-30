#!/bin/bash
# 量化蒸馏后的模型用于部署

set -e

MODEL_PATH="${MODEL_PATH:-./output/patent-agent-model}"
QUANT_PATH="${QUANT_PATH:-./output/patent-agent-model-awq}"

echo "=== Patent Agent Distill: Quantization ==="
echo "Source: $MODEL_PATH"
echo "Output: $QUANT_PATH"

# AWQ 4-bit 量化
python -m awq.quantize \
  --model_path "$MODEL_PATH" \
  --quant_path "$QUANT_PATH" \
  --w_bit 4 \
  --q_group_size 128

echo "=== Quantization complete ==="
echo "Quantized model: $QUANT_PATH"
