#!/bin/bash
# 专利代理师蒸馏 - SFT 训练脚本
# 在有 GPU 的机器上运行此脚本
# 需要：2×A100 80GB 或同等算力

set -e

echo "=========================================="
echo "  专利代理师 SFT 训练"
echo "=========================================="

# 配置
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-./data/final}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/patent-agent-sft}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

echo "模型: $MODEL_NAME"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "训练轮数: $NUM_EPOCHS"
echo "学习率: $LR"
echo ""

# 检查依赖
echo "检查依赖..."
pip install -q torch transformers trl accelerate peft bitsandbytes wandb

# 检查数据
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "错误: 训练数据不存在: $DATA_DIR/train.jsonl"
    echo "请先运行: python scripts/merge_data.py"
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < "$DATA_DIR/train.jsonl")
VAL_SAMPLES=$(wc -l < "$DATA_DIR/val.jsonl" 2>/dev/null || echo "0")
echo "训练样本: $TRAIN_SAMPLES 条"
echo "验证样本: $VAL_SAMPLES 条"
echo ""

# 开始训练
echo "开始 SFT 训练..."
python -m trl sft \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --max_seq_length "$MAX_SEQ_LEN" \
    --gradient_checkpointing true \
    --optim paged_adamw_8bit \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --report_to wandb \
    --seed 42

echo ""
echo "=========================================="
echo "  SFT 训练完成！"
echo "  模型保存在: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 运行评估: python evaluation/eval_all.py"
echo "  2. 如需 Logit KD 蒸馏，运行: bash scripts/03_capture_logits.sh"
