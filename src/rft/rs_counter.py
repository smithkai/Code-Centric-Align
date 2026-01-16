"""
统计rs_validator打分后的样本，有多少是有效可以用于DPO训练的
我要把rs_validator弄清楚
"""
import json

FILE = "data/rft/v3/v3_rs_verified_10k.jsonl"
total, perfect, dpo_pairs = 0, 0, 0

with open(FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        flags = item['passed_flags']
        total += 1
        if any(flags):
            perfect += 1 # Pass@10 成功
            if not all(flags):
                dpo_pairs += 1 # 理想偏好对

print(f"📊 最终统计报告:")
print(f" - 处理总量: {total}")
print(f" - 成功运行 (Pass@10): {perfect} ({perfect/total*100:.1f}%)")
print(f" - 理想 DPO 偏好对: {dpo_pairs} ({dpo_pairs/total*100:.1f}%)")
