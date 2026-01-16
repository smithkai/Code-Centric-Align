"""
统计gpo正负样本的长度，确定模式极短输出导致极低测评分数的原因
"""

import json
import numpy as np

def audit_dpo_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chosen_lens = []
    rejected_lens = []
    
    for item in data:
        # 假设数据格式为 [{"chosen": "...", "rejected": "...", ...}]
        # 如果是 LLaMA-Factory 的格式，请根据实际 key 名调整 (可能是 'output')
        c = item.get('chosen', '')
        r = item.get('rejected', '')
        
        # 如果 chosen/rejected 是 list (多轮对话格式)，取最后一条
        c_text = c[-1]['content'] if isinstance(c, list) else str(c)
        r_text = r[-1]['content'] if isinstance(r, list) else str(r)
        
        chosen_lens.append(len(c_text))
        rejected_lens.append(len(r_text))

    print(f"📊 数据集审计报告: {file_path}")
    print("-" * 40)
    print(f"样本总量: {len(data)}")
    print(f"Chosen 平均长度: {np.mean(chosen_lens):.2f}")
    print(f"Rejected 平均长度: {np.mean(rejected_lens):.2f}")
    print(f"长度比 (Chosen/Rejected): {np.mean(chosen_lens)/np.mean(rejected_lens):.2f}")
    
    short_chosen_count = sum(1 for c, r in zip(chosen_lens, rejected_lens) if c < r * 0.5)
    print(f"潜在风险: 有 {short_chosen_count} 个样本 Chosen 长度不足 Rejected 的一半 (约 {short_chosen_count/len(data)*100:.1f}%)")

if __name__ == "__main__":
    # 请将此处改为你 DPO 训练用的 json 路径
    audit_dpo_json("data/align/v3_dpo_train_2600.json")
