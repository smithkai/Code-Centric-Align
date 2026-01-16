import json
import os
import time
import concurrent.futures
from openai import OpenAI  # DeepSeek API 兼容 OpenAI 格式

# --- 核心配置 ---
API_KEY = "sk-debd9341cb2d434aafbc31028209ad33"
BASE_URL = "https://api.deepseek.com" # 或者对应的镜像地址
INPUT_FILE = "v3_final_safe_train.json"
OUTPUT_FILE = "v3_deepseek_verified_dpo.json"
BATCH_SIZE = 10  # 并发数，根据你的 API 额度调整

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

AUDIT_PROMPT = """你是一个高水平的代码评审专家。请判断以下【模型生成的代码】是否完整、正确地实现了【题目要求】。

【题目要求】:
{instruction}

【模型生成的代码】:
{output}

评审准则：
1. 代码是否逻辑正确且能完成题目任务？
2. 代码是否包含实质性内容（拒绝只有pass、复读题目或空壳函数）？
3. 如果代码是正确的，请只回答 "PASS"。
4. 如果代码逻辑错误、偷懒、或答非所问，请回答 "FAIL" 并简述原因（不超过20字）。

你的最终结论必须以 "RESULT: PASS" 或 "RESULT: FAIL" 结尾。"""

def audit_single_case(entry):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # 对应 DeepSeek-V3
            messages=[
                {"role": "system", "content": "You are a senior code reviewer."},
                {"role": "user", "content": AUDIT_PROMPT.format(
                    instruction=entry['instruction'],
                    output=entry['output']
                )}
            ],
            temperature=0.1, # 降低随机性
            max_tokens=50
        )
        result_text = response.choices[0].message.content
        if "RESULT: PASS" in result_text:
            return entry, True, "PASS"
        else:
            reason = result_text.split("RESULT: FAIL")[-1].strip()
            return entry, False, reason
    except Exception as e:
        return entry, None, str(e)

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🚀 开始调用 DeepSeek-V3 进行逻辑审计，总计 {len(data)} 条...")

    final_verified_data = []
    fail_count = 0

    # 使用线程池并发加速
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(audit_single_case, item) for item in data]

        from tqdm import tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            entry, is_passed, msg = future.result()

            if is_passed is True:
                final_verified_data.append(entry)
            elif is_passed is False:
                fail_count += 1
                # print(f"❌ 剔除样本: {msg}") # 调试时可开启
            else:
                print(f"⚠️ API 报错: {msg}")

    print(f"\n✅ 审计完成！")
    print(f"📊 过滤结果: {len(data)} -> {len(final_verified_data)}")
    print(f"📉 逻辑错误率: {(fail_count/len(data))*100:.1f}%")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_verified_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()