"""
直接用deepseek从自动化选择出的样本对选择最具代表性的负样本和数据集正样本构建dpo数据对
"""
import json
import jsonlines
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# 配置
client = OpenAI(api_key="YOUR_DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")
MAX_WORKERS = 10  # 并发数，根据你的 API 限制调整
INPUT_FILE = "data/rft/v3/v3_rs_verified_10k.jsonl"
GT_FILE = "data/raw/magicoder_raw.jsonl"
OUTPUT_FILE = "v3_dpo_final.jsonl"

# 1. 加载 Ground Truth (Chosen 池)
ground_truth_map = {}
with jsonlines.open(GT_FILE) as reader:
    for obj in reader:
        ground_truth_map[obj['instruction']] = obj['output']

def process_item(item):
    instruction = item['instruction']
    if instruction not in ground_truth_map:
        return None
            
    chosen = ground_truth_map[instruction]
    
    # 获取失败的响应及其原始索引
    failed_entries = [
        {"idx": i, "content": resp} 
        for i, (resp, flag) in enumerate(zip(item['responses'], item['passed_flags'])) 
        if not flag
    ]
    
    if not failed_entries:
        return None

    # 构建极简 Prompt：让 DeepSeek 只选编号
    # 对 Candidate 进行长度截断（取前 1000 字符），节省输入 Token
    candidates_text = ""
    for entry in failed_entries:
        truncated_content = entry['content'][:1000].replace('\n', ' ')
        candidates_text += f"ID {entry['idx']}: {truncated_content}\n"

    prompt = f"""Instruction: {instruction[:500]}
Ground Truth: {chosen[:500]}

Candidates (Failed Responses):
{candidates_text}

Task: Which Candidate ID is the most representative 'failed' sample (contains logical errors or hallucinations like Thai/gibberish)? 
Return ONLY the ID number. If all are unreadable junk, return 'SKIP'."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # 降低随机性，选编号更准
            max_tokens=10    # 强制限制输出长度
        )
        
        result = response.choices[0].message.content.strip()
        
        # 提取数字
        match = re.search(r'\d+', result)
        if match:
            selected_idx = int(match.group())
            # 确保选出的 ID 在范围内
            if selected_idx < len(item['responses']):
                return {
                    "instruction": instruction,
                    "chosen": chosen,
                    "rejected": item['responses'][selected_idx]
                }
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# 2. 运行多线程处理
final_dpo_data = []
with jsonlines.open(INPUT_FILE) as reader:
    all_items = list(reader)

print(f"开始处理 {len(all_items)} 条数据...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(process_item, all_items))

# 过滤掉 None 结果
final_dpo_data = [r for r in results if r is not None]

# 3. 保存
with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    writer.write_all(final_dpo_data)

print(f"处理完成！成功生成 {len(final_dpo_data)} 对 DPO 数据。")