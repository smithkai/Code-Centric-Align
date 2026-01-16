"""
这个脚本会将 .jsonl 校验结果（评分后的）转换成 LLaMA-Factory 或 TRL 兼容的 .json 格式。
"""
import json
import os

INPUT_FILE = "data/rft/v3/v3_rs_verified_10k.jsonl"
OUTPUT_FILE = "data/align/v3_dpo_train_2600.json"

def build():
    dpo_data = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            instr = item['instruction']
            resps = item['responses']
            flags = item['passed_flags']
            
            # 必须包含至少一个成功和一个失败
            if any(flags) and not all(flags):
                # 1. 提取所有通过的回复及其长度
                chosen_candidates = [
                    (r, len(r)) for i, r in enumerate(resps) if flags[i] is True
                ]
                # 策略：选择最短的代码作为 Chosen
                # ! 其实这里就有些问题了，选择最短的代码可能导致dpo学习到代码越短越好的特性
                chosen_candidates.sort(key=lambda x: x[1])
                best_chosen = chosen_candidates[0][0]
                
                # 2. 提取所有失败的回复
                rejected_candidates = [
                    (r, len(r)) for i, r in enumerate(resps) if flags[i] is False
                ]
                # 策略：随机选一个或选第一个失败的
                # 这里的策略感觉非常不合理
                worst_rejected = rejected_candidates[0][0]
                
                dpo_data.append({
                    "instruction": instr,
                    "input": "",
                    "output": [], # 预留
                    "chosen": best_chosen,
                    "rejected": worst_rejected
                })

    # 保存为标准 JSON 格式
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ DPO 数据集构造完成！")
    print(f"📊 最终样本数: {len(dpo_data)}")
    print(f"📂 路径: {OUTPUT_FILE}")

if __name__ == "__main__":
    build()