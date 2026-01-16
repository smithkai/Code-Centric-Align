import json
import os
from collections import Counter

# 1. 配置你的数据集路径
dataset_path = "data/align/v3_dpo_train_2600.json" # 请修改为你的实际路径

def analyze_dpo_dataset(path):
    if not os.path.exists(path):
        print(f"错误：找不到文件 {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        "total": len(data),
        "short_chosen": 0,    # chosen 太短
        "length_bias": 0,    # chosen 明显短于 rejected
        "overlap_issue": 0,  # chosen 包含了 prompt 内容
        "pass_count": 0      # 包含 pass 关键字
    }

    print(f"开始分析数据集: {path} (总计 {stats['total']} 条)")
    print("-" * 50)

    for entry in data:
        # 兼容不同格式，通常是 prompt, chosen, rejected
        p = entry.get("prompt", "")
        c = entry.get("chosen", "")
        r = entry.get("rejected", "")

        # 检查是否包含 pass
        if "pass" in c.split():
            stats["pass_count"] += 1

        # 检查长度偏差 (如果 chosen 长度不到 rejected 的 50%)
        if len(c) < len(r) * 0.5:
            stats["length_bias"] += 1

        # 检查 chosen 是否太短 (比如少于 10 个字符)
        if len(c.strip()) < 10:
            stats["short_chosen"] += 1

        # 检查 prompt 泄露 (chosen 是否复读了 prompt 的大幅内容)
        if len(p) > 20 and p[:20] in c:
            stats["overlap_issue"] += 1

    # 输出报告
    print(f"1. 懒惰倾向: {stats['pass_count']} 条样本包含 'pass'。")
    print(f"2. 长度偏差: {stats['length_bias']} 条样本的 Chosen 显著短于 Rejected (可能导致偷懒)。")
    print(f"3. 复读风险: {stats['overlap_issue']} 条样本的 Chosen 包含了 Prompt 开头。")
    print(f"4. 异常样本: {stats['short_chosen']} 条样本的 Chosen 几乎为空。")
    
    if stats['length_bias'] / stats['total'] > 0.3:
        print("\n⚠️ 预警：数据集存在严重的『负向长度偏差』，模型极大概率学会偷懒！")
    if stats['overlap_issue'] / stats['total'] > 0.2:
        print("\n⚠️ 预警：数据集存在严重的『复读问题』，这解释了为何 HumanEval 在复读题目！")

analyze_dpo_dataset(dataset_path)
