#!/bin/bash
# 扩展训练数据 - 生成更多样本
# 用法: bash scripts/expand_data.sh [每类样本数]
# 默认每类生成 500 条

set -e

NUM_PER_TYPE="${1:-500}"

echo "=========================================="
echo "  扩展 SFT 训练数据"
echo "  每类生成: $NUM_PER_TYPE 条"
echo "=========================================="

cd "$(dirname "$0")/.."

# 1. 运行模板生成器
echo ""
echo "[1/2] 运行模板生成器..."
python3 scripts/generate_sft_data.py "$NUM_PER_TYPE"

# 2. 合并所有数据
echo ""
echo "[2/2] 合并数据..."
python3 scripts/merge_data.py

echo ""
echo "=========================================="
echo "  数据扩展完成！"
echo "  查看: data/final/"
echo "=========================================="
