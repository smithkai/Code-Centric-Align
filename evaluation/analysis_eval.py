import json
import random

# 指向你刚才测评生成的 json 文件
file_path = "eval_results/dpo_v3.1/generations.json"

with open(file_path, 'r') as f:
    data = json.load(f)

# HumanEval 有 164 题，我们随机看 3 题
samples = random.sample(data, 3)

for i, sample in enumerate(samples):
    print(f"\n{'='*30} 样本 {i+1} {'='*30}")
    # 注意：这里的 key 取决于 evaluation-harness 的版本，通常是 'generation' 或 0
    gen_code = sample[0] if isinstance(sample, list) else sample
    print("模型输出内容：")
    print("-" * 50)
    print(gen_code)
    print("-" * 50)
