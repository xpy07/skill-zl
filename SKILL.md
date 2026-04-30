# 专利代理师蒸馏方案 (Patent Agent Distillation)

> 基于 arcee-ai/DistillKit，将大模型的专利代理能力蒸馏到小模型中

## 目标岗位能力矩阵

| 能力维度 | 任务描述 | 关键技能点 |
|---------|---------|-----------|
| **创造性评估** | 对客户专利案件进行新颖性、创造性、实用性三性评估 | 技术方案拆解、对比文件检索分析、创造性三步法判断 |
| **文件撰写** | 撰写权利要求书、说明书、摘要等专利代理文件 | 权利要求布局、说明书充分公开、技术术语精准 |
| **审查意见答复** | 针对审查员的审查意见进行答复和权利要求修改 | 区别技术特征分析、创造性争辩、权利要求修改策略 |
| **客户咨询** | 解答客户关于专利代理流程、策略、法律条款等问题 | 专利法/细则/指南条文引用、流程指引、风险提示 |

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Teacher Model                         │
│         (Qwen3-235B / DeepSeek-V3 / GPT-4)              │
│    具备深厚的专利法律知识 + 实务经验 + 中文法律写作能力      │
└──────────────┬──────────────────────────────────┬────────┘
               │ ① 知识蒸馏 (Logit KD)            │ ② 数据蒸馏 (SFT)
               ▼                                    ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│  DistillKit Logit KD     │    │  Teacher 生成高质量训练数据     │
│  - KL 散度损失            │    │  - 评估报告模板                │
│  - Hidden State 对齐     │    │  - 撰写样本                    │
│  - Cross-Entropy         │    │  - 审查意见答复样本             │
└──────────────┬───────────┘    │  - Q&A 对话                   │
               │                └──────────────┬───────────────┘
               ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│                   Student Model                          │
│              (Qwen3-8B / Llama-3.1-8B)                   │
│            轻量化专利代理师，可本地/边缘部署                  │
└─────────────────────────────────────────────────────────┘
```

## 蒸馏策略：两阶段

### 阶段一：领域数据蒸馏（SFT，先做这步）

用 Teacher 生成高质量专利代理训练数据，对 Student 做 SFT 微调。

**为什么先做 SFT？** 纯 Logit KD 需要大量 token，但专利领域数据稀缺。先用 SFT 让 Student 建立领域知识框架，再用 Logit KD 精调分布，效果远好于直接 KD。

### 阶段二：Logit 知识蒸馏（DistillKit）

用 DistillKit 做 teacher-student 的 logit 级别蒸馏，保留 teacher 的推理分布特征。

---

## 阶段一：数据构建（核心）

### 数据来源

| 来源 | 获取方式 | 数据量预估 |
|-----|---------|-----------|
| 国家知识产权局公开数据 | CNIPA 专利检索系统下载公开文本 | 100万+ 件专利全文 |
| 复审委决定/法院判决 | 裁判文书网、复审委决定书 | 10万+ 份 |
| 专利法/细则/指南 | 官方文本 | 基础法律条文 |
| 审查指南 | CNIPA 审查指南全文 | 实务操作规范 |
| 真实代理实务案例 | 需自行收集或购买 | 视情况 |

### 四类任务的数据模板

#### 任务 1：创造性评估

**Prompt 模板：**
```
你是一名资深专利代理师。请对以下技术方案进行专利创造性评估。

【技术方案】
{technical_solution}

【最近似对比文件】
{closest_prior_art}

【区别技术特征】
{distinguishing_features}

请从以下维度分析：
1. 新颖性判断（与对比文件逐项比对）
2. 创造性判断（三步法）：
   - 确定最接近的现有技术
   - 确定区别技术特征和实际解决的技术问题
   - 判断是否显而易见
3. 实用性评估
4. 结论与建议
```

**Teacher 输出示例（用于 SFT 训练）：**
```
## 创造性评估报告

### 一、新颖性分析
本申请权利要求1请求保护的技术方案与对比文件1（CN1234567A）公开的技术方案相比...

### 二、创造性分析（三步法）

**第一步：确定最接近的现有技术**
对比文件1公开了[...]，与本申请属于同一技术领域，解决了相近的技术问题，被确定为最接近的现有技术。

**第二步：确定区别技术特征和实际解决的技术问题**
本申请权利要求1与对比文件1的区别技术特征为：[...]
基于上述区别技术特征，本申请实际解决的技术问题是：[...]

**第三步：判断是否显而易见**
对比文件2（CN2345678A）公开了[...]，给出了将上述区别技术特征应用到对比文件1的技术启示...
[或] 对比文件2未给出将上述区别技术特征应用到对比文件1的技术启示，本领域技术人员需要付出创造性劳动才能...

### 三、结论
本申请权利要求1相对于对比文件1和对比文件2的结合具备创造性（专利法第22条第3款）。
建议：[具体建议]
```

#### 任务 2：专利文件撰写

**Prompt 模板：**
```
你是一名资深专利代理师。请根据以下技术交底书撰写专利申请文件。

【技术交底书】
{technical_disclosure}

【撰写要求】
- 权利要求书：独立权利要求 + 从属权利要求布局
- 说明书：技术领域、背景技术、发明内容、具体实施方式
- 摘要：不超过300字

请严格按照中国专利法实施细则的要求撰写。
```

**关键训练点：**
- 独立权利要求的上位概括 vs 具体限定
- 从属权利要求的梯度布局（核心特征 → 优选方案 → 具体实施）
- 说明书的充分公开（专利法第26条第3款）
- 功能性限定的使用规范
- 方法权利要求 vs 产品权利要求的撰写差异

#### 任务 3：审查意见答复

**Prompt 模板：**
```
你是一名资深专利代理师。请针对以下审查意见进行答复。

【申请号】{application_number}
【审查意见】
{office_action}

【当前权利要求】
{current_claims}

请提供：
1. 审查意见分析（逐条分析审查员的驳回理由）
2. 答复策略（争辩 vs 修改 vs 两者结合）
3. 修改后的权利要求书（如需修改）
4. 意见陈述书正文
```

**关键训练点：**
- 区别技术特征的认定与争辩
- 创造性答复的"三步法"反向论证
- 权利要求修改的合规性（不得超出原始记载范围，专利法第33条）
- 审查指南第二部分第四章的创造性判断标准
- 实用新型 vs 发明审查标准差异

#### 任务 4：客户咨询

**Prompt 模板：**
```
你是一名资深专利代理师。请回答以下客户咨询。

【客户问题】
{client_question}

【背景信息】
{context}

请提供专业、准确、易懂的解答，必要时引用相关法律条文。
```

**关键训练点：**
- 专利申请流程（受理 → 初审 → 公布 → 实审 → 授权）
- 专利保护期限（发明20年，实用新型10年，外观设计15年）
- PCT 国际申请流程
- 专利侵权判定原则（全面覆盖、等同原则）
- 专利无效宣告程序
- 费用减缴政策

### 数据生成流程

```bash
# Step 1: 准备原始素材（专利文本、法律条文、案例）
# Step 2: 构建 prompt 模板
# Step 3: 用 Teacher 批量生成训练数据
# Step 4: 人工审核 + 质量筛选
# Step 5: 格式化为 SFT 训练格式
```

**数据格式（JSONL）：**
```json
{
  "messages": [
    {"role": "system", "content": "你是一名资深专利代理师，精通中国专利法及相关实务..."},
    {"role": "user", "content": "请对以下技术方案进行创造性评估：..."},
    {"role": "assistant", "content": "## 创造性评估报告\n### 一、新颖性分析\n..."}
  ]
}
```

**数据量建议：**
- 创造性评估：5,000-10,000 条
- 文件撰写：3,000-5,000 条（长文本，注意 sequence_length）
- 审查意见答复：5,000-8,000 条
- 客户咨询：3,000-5,000 条
- **总计：16,000-28,000 条高质量样本**

---

## 阶段二：DistillKit Logit 蒸馏

### Teacher 模型选择

| 模型 | 优势 | 劣势 | 推荐度 |
|-----|------|------|-------|
| **Qwen3-235B** | 中文法律理解极强，开源可获取 logit | 需要大量 GPU 捕获 logit | ⭐⭐⭐⭐⭐ |
| **DeepSeek-V3/R1** | 推理能力强，适合创造性分析 | 需要 API 或本地部署 | ⭐⭐⭐⭐ |
| **GPT-4** | 综合能力强 | 无法获取 logit，只能做数据蒸馏 | ⭐⭐⭐ |

**推荐方案：** Qwen3-235B 作为 Teacher（开源，可捕获 logit），同时用 DeepSeek-R1 补充推理类数据。

### Student 模型选择

| 模型 | 参数量 | 优势 | 推荐场景 |
|-----|-------|------|---------|
| **Qwen3-8B** | 8B | 中文能力强，与 Qwen3-235B 同架构 | 首选，同架构蒸馏效果最好 |
| **Qwen3-4B** | 4B | 更轻量 | 边缘部署场景 |
| **Llama-3.1-8B** | 8B | 英文能力强 | 需要处理涉外专利 |

### Logit 捕获配置

```yaml
# capture_config.yaml
# 用 Teacher 模型对专利领域数据做推理，捕获 logit 分布

# 使用 vLLM 捕获
python -m distillkit.sample_logits_vllm \
  --model Qwen/Qwen3-235B \
  --dataset ./patent_sft_data/train.jsonl \
  --output ./patent_logits_capture/ \
  --compression-config ./compression_config.yaml
```

**压缩配置（推荐）：**
```yaml
# compression_config.yaml
logprob_compressor:
  d: 151936          # Qwen3 词表大小
  k: 128             # 每个 token 保留 128 个 logprob
  exact_k: 16        # 其中 16 个精确存储
  exact_dtype: bfloat16
  polynomial_terms: [0, 1, 2, 3, 4, "sqrt"]
  term_dtype: float32
  residual_bins: []
  delta_encoding: false
  error_diffusion: false
```

**存储预估：** ~300 bytes/token × 28,000 条 × 平均 1024 tokens/条 ≈ **~8.6 GB**（可接受）

### 蒸馏训练配置

```yaml
# distill_config.yaml
project_name: patent-agent-distill
model: Qwen/Qwen3-8B
output_path: ./patent-agent-model
sequence_length: 4096    # 专利文本较长，至少 4096
resize_embeddings_to_multiple_of: 64
use_flash_attention: true

dataset:
  train_dataset:
    repo_id: ./patent_logits_capture   # 本地捕获的 logit 数据
    split: train
  seed: 42
  prepacked: true

teacher:
  kind: dataset
  logprob_compressor:
    d: 151936
    k: 128
    exact_k: 16
    exact_dtype: bfloat16
    polynomial_terms: [0, 1, 2, 3, 4, "sqrt"]
    term_dtype: float32
    residual_bins: []
    delta_encoding: false
    error_diffusion: false

loss_functions:
  - function: cross_entropy
    weight: 0.3          # SFT 损失，保证生成质量
  - function: kl
    weight: 0.5          # KL 散度，学习 teacher 的分布
    temperature: 2.0     # 温度稍高，软化分布
    missing_probability_handling: zero
    sparse_chunk_length: 1024
  - function: hs_cosine
    weight: 0.2          # 隐藏状态对齐，保留推理路径
    layer_mapping: all

training_args:
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  save_steps: 256
  save_total_limit: 3
  logging_steps: 1
  learning_rate: 1.0e-6   # 领域微调用较小学习率
  weight_decay: 0.01
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  bf16: true
  remove_unused_columns: false
  optim: paged_adamw_8bit
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  report_to: wandb
```

### 启动训练

```bash
# 阶段一：SFT 微调（先让 Student 学会领域知识）
# 使用 LLaMA-Factory 或 TRL 的 SFTTrainer
python -m trl sft \
  --model_name Qwen/Qwen3-8B \
  --dataset_name ./patent_sft_data \
  --output_dir ./patent-agent-sft \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4

# 阶段二：Logit KD（精调分布）
distillkit distill_config.yaml
```

---

## 评估体系

### 自动评估

| 评估维度 | 评估方法 | 指标 |
|---------|---------|------|
| 法律条文引用准确性 | 人工标注 + 自动匹配 | 准确率 |
| 权利要求撰写规范性 | 结构化检查（前序部分+特征部分） | 合规率 |
| 创造性分析逻辑性 | 三步法结构检查 | 完整度 |
| 技术术语准确性 | 术语表比对 | 准确率 |

### 人工评估（必须做）

- 邀请 3-5 名持证专利代理师进行盲评
- 每个任务维度 100 个测试案例
- 评分标准：1-5 分（1=完全不可用，5=资深代理师水平）

### 基准测试集

```json
// test_creativity.json - 创造性评估测试集
[
  {
    "id": "CRE-001",
    "input": "一种锂电池正极材料，包含...",
    "closest_prior_art": "CN1234567A 公开了一种...",
    "expected_analysis": {
      "novelty": true,
      "inventive_step": true,
      "reasoning": "区别技术特征在于...对比文件2未给出..."
    }
  }
]

// test_drafting.json - 撰写测试集
// test_office_action.json - 审查意见答复测试集
// test_consultation.json - 客户咨询测试集
```

---

## 部署建议

### 硬件需求

| 阶段 | GPU 需求 | 时间预估 |
|-----|---------|---------|
| Teacher logit 捕获 | 8×A100 80GB（Qwen3-235B） | 2-4 小时 |
| SFT 微调 | 2×A100 80GB | 4-8 小时 |
| Logit KD 训练 | 4×A100 80GB | 8-16 小时 |
| Student 推理部署 | 1×A10 或 1×4090 | 实时响应 |

### 轻量化部署

```bash
# 量化 Student 模型用于部署
python -m awq.quantize \
  --model_path ./patent-agent-model \
  --quant_path ./patent-agent-model-awq \
  --w_bit 4 \
  --q_group_size 128

# 或使用 GGUF 格式部署到 CPU
python convert_hf_to_gguf.py ./patent-agent-model --outtype q4_k_m
```

---

## 项目结构

```
patent-agent-distill/
├── data/
│   ├── raw/                    # 原始专利数据
│   ├── prompts/                # Prompt 模板
│   ├── sft/                    # SFT 训练数据
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── test/                   # 测试集
│       ├── test_creativity.json
│       ├── test_drafting.json
│       ├── test_office_action.json
│       └── test_consultation.json
├── configs/
│   ├── capture_config.yaml     # Logit 捕获配置
│   ├── compression_config.yaml # 压缩配置
│   ├── sft_config.yaml         # SFT 训练配置
│   └── distill_config.yaml     # DistillKit 蒸馏配置
├── scripts/
│   ├── 01_data_prepare.py      # 数据预处理
│   ├── 02_teacher_generate.py  # Teacher 生成训练数据
│   ├── 03_capture_logits.sh    # 捕获 logit
│   ├── 04_sft_train.sh         # SFT 训练
│   ├── 05_distill_train.sh     # Logit KD 训练
│   ├── 06_eval.py              # 评估脚本
│   └── 07_quantize_deploy.sh   # 量化部署
├── evaluation/
│   ├── eval_creativity.py
│   ├── eval_drafting.py
│   ├── eval_office_action.py
│   └── eval_consultation.py
└── README.md
```

---

## 关键注意事项

1. **数据质量 > 数据数量**：专利领域错误代价极高，宁可少一点也要保证正确
2. **法律条文必须精确**：不能出现"编造法条"的问题，需要在 prompt 中强制引用原文
3. **中国专利法 vs 国际规则**：明确区分，不要混淆
4. **保守原则**：模型不确定时应建议"请咨询资深代理师"，而非给出可能错误的建议
5. **持续更新**：专利法修订后需要重新微调（如 2024 年专利法实施细则修改）
