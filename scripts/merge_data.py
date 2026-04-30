#!/usr/bin/env python3
"""
合并所有 SFT 训练数据源，生成最终训练集
"""

import json
import os
import glob
from pathlib import Path
from collections import Counter

def load_jsonl(path):
    """加载 JSONL 文件"""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'messages' in data and len(data['messages']) >= 2:
                    samples.append(data)
            except json.JSONDecodeError as e:
                print(f"  跳过无效行: {path}: {e}")
    return samples


def validate_sample(sample):
    """验证样本格式"""
    messages = sample.get('messages', [])
    if len(messages) < 2:
        return False
    
    # 检查必须有 system + user + assistant
    roles = [m['role'] for m in messages]
    if 'user' not in roles or 'assistant' not in roles:
        return False
    
    # 检查内容不为空
    for m in messages:
        if not m.get('content', '').strip():
            return False
    
    return True


def deduplicate(samples):
    """基于 user 消息去重"""
    seen = set()
    unique = []
    for s in samples:
        user_content = None
        for m in s['messages']:
            if m['role'] == 'user':
                user_content = m['content']
                break
        if user_content and user_content not in seen:
            seen.add(user_content)
            unique.append(s)
    return unique


def main():
    base_dir = Path(__file__).parent.parent / "data" / "sft"
    output_dir = base_dir.parent / "final"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("合并 SFT 训练数据")
    print("=" * 60)
    
    # 收集所有 JSONL 文件
    all_files = sorted(glob.glob(str(base_dir / "*.jsonl")))
    print(f"\n发现 {len(all_files)} 个数据文件：")
    
    all_samples = []
    category_counts = Counter()
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        samples = load_jsonl(fpath)
        valid = [s for s in samples if validate_sample(s)]
        
        # 分类统计
        if 'creativity' in fname:
            cat = '创造性评估'
        elif 'drafting' in fname:
            cat = '文件撰写'
        elif 'office_action' in fname:
            cat = '审查意见答复'
        elif 'consultation' in fname:
            cat = '客户咨询'
        elif 'claim' in fname:
            cat = '权利要求优化'
        else:
            cat = '其他'
        
        category_counts[cat] += len(valid)
        all_samples.extend(valid)
        
        print(f"  {fname}: {len(samples)} 条 (有效 {len(valid)} 条) [{cat}]")
    
    print(f"\n总计: {len(all_samples)} 条有效样本")
    
    # 去重
    unique_samples = deduplicate(all_samples)
    print(f"去重后: {len(unique_samples)} 条")
    
    # 打乱顺序
    import random
    random.seed(42)
    random.shuffle(unique_samples)
    
    # 按类别统计最终数据
    final_counts = Counter()
    for s in unique_samples:
        user_content = next((m['content'] for m in s['messages'] if m['role'] == 'user'), '')
        if '创造性' in user_content or '新颖性' in user_content:
            final_counts['创造性评估'] += 1
        elif '撰写' in user_content or '权利要求书' in user_content:
            final_counts['文件撰写'] += 1
        elif '审查意见' in user_content or '答复' in user_content:
            final_counts['审查意见答复'] += 1
        elif '咨询' in user_content or '请问' in user_content:
            final_counts['客户咨询'] += 1
        else:
            final_counts['其他'] += 1
    
    print("\n最终数据分布：")
    for cat, count in sorted(final_counts.items()):
        print(f"  {cat}: {count} 条")
    
    # 按 90:10 划分训练集和验证集
    split_idx = int(len(unique_samples) * 0.9)
    train_samples = unique_samples[:split_idx]
    val_samples = unique_samples[split_idx:]
    
    # 保存
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"\n训练集: {len(train_samples)} 条 → {train_path}")
    print(f"验证集: {len(val_samples)} 条 → {val_path}")
    
    # 数据质量报告
    print("\n" + "=" * 60)
    print("数据质量报告")
    print("=" * 60)
    
    # 检查平均长度
    user_lens = []
    assistant_lens = []
    for s in unique_samples:
        for m in s['messages']:
            if m['role'] == 'user':
                user_lens.append(len(m['content']))
            elif m['role'] == 'assistant':
                assistant_lens.append(len(m['content']))
    
    print(f"  用户消息平均长度: {sum(user_lens)//len(user_lens)} 字符")
    print(f"  助手回复平均长度: {sum(assistant_lens)//len(assistant_lens)} 字符")
    print(f"  最短用户消息: {min(user_lens)} 字符")
    print(f"  最长用户消息: {max(user_lens)} 字符")
    print(f"  最短助手回复: {min(assistant_lens)} 字符")
    print(f"  最长助手回复: {max(assistant_lens)} 字符")
    
    print("\n✅ 数据合并完成！")
    print(f"\n下一步：")
    print(f"  1. 检查 {output_dir} 中的数据质量")
    print(f"  2. 将数据传输到有 GPU 的机器")
    print(f"  3. 运行 SFT 训练: bash scripts/04_sft_train.sh")


if __name__ == "__main__":
    main()
