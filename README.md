# 专利代理师蒸馏项目 (Patent Agent Distillation)

基于 [DistillKit](https://github.com/arcee-ai/DistillKit) 的专利代理师大模型蒸馏方案。

## 项目目标

将大模型的专利代理能力蒸馏到轻量模型中，覆盖专利代理师四大核心职责：

1. **创造性评估** — 三性分析（新颖性、创造性、实用性）
2. **文件撰写** — 权利要求书、说明书、摘要
3. **审查意见答复** — 区别技术特征争辩、权利要求修改
4. **客户咨询** — 流程指引、法条引用、风险提示

## 技术方案

两阶段蒸馏：
- **阶段一**：SFT 数据蒸馏（Teacher 生成高质量训练数据 → Student 微调）
- **阶段二**：DistillKit Logit KD（KL 散度 + CE + Hidden State 对齐）

推荐配置：
- Teacher: Qwen3-235B
- Student: Qwen3-8B

## 项目结构

```
patent-agent-distill/
├── SKILL.md              # 完整技术方案
├── data/
│   ├── prompts/          # Prompt 模板
│   ├── sft/              # SFT 训练数据
│   └── test/             # 评估测试集
├── configs/              # 训练配置文件
├── scripts/              # 训练/评估脚本
└── evaluation/           # 评估代码
```

## 快速开始

```bash
# 1. 安装 DistillKit
git clone https://github.com/arcee-ai/distillkit.git
cd distillkit && pip install -e .

# 2. 准备数据（见 data/ 目录）

# 3. SFT 微调
bash scripts/04_sft_train.sh

# 4. Logit KD 蒸馏
bash scripts/05_distill_train.sh

# 5. 评估
python evaluation/eval_all.py
```

## 许可证

MIT
