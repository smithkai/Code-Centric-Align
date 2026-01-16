"""
第二个构建DPO数据对的脚本计划重新从v3_rs_candidate.jsonl开始
重新设计自动化检验策略，以及dpo数据对挑选策略，构建dpo数据集
这个脚本也是后期出问题的脚本，把instruction丢了
"""
import json
import re
import multiprocessing
import subprocess
import os
from tqdm import tqdm

# --- 核心配置 ---
INPUT_FILE = "v3_rs_candidates.jsonl"
OUTPUT_FILE = "v3_dpo_cleaned_high_quality.json" # 最终直接用于训练的格式
NUM_WORKERS = 14
TIMEOUT = 5

def extract_python_code(text):
    if not isinstance(text, str): return ""
    # 匹配 Markdown 代码块
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip()

def is_lazy_code(code):
    """检测是否是偷懒代码"""
    code_clean = code.strip()
    # 1. 只有 pass
    if code_clean == "pass" or code_clean.endswith(": pass"): return True
    # 2. 长度过短（通常意味着没写核心逻辑）
    if len(code_clean) < 30: return True
    # 3. 只重复了 Docstring 而没有实现
    lines = [l for l in code_clean.split('\n') if l.strip() and not l.strip().startswith(('"', '#'))]
    if not lines or (len(lines) == 1 and "pass" in lines[0]): return True
    return False

def safe_execute(code):
    """安全执行校验"""
    if not code or is_lazy_code(code):
        return False, "Lazy or Empty"
    try:
        res = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        return (True, "Passed") if res.returncode == 0 else (False, "Syntax/Runtime Error")
    except Exception as e:
        return False, str(e)

def process_line(line):
    """核心逻辑：从多个响应中筛选最优对"""
    if not line.strip(): return None
    try:
        item = json.loads(line)
    except: return None

    prompt = item.get('prompt', '')
    responses = item.get('responses', [])
    
    passed_candidates = []
    failed_candidates = []

    for resp in responses:
        code = extract_python_code(resp)
        success, msg = safe_execute(code)
        
        # 计算特征：代码长度（作为质量的初步参考，避免极短代码）
        # 注意：不是越长越好，但要避免过短
        char_len = len(code)

        if success:
            passed_candidates.append({"text": resp, "len": char_len})
        else:
            # 只有语法错误的才作为 rejected，或者质量极差的
            failed_candidates.append({"text": resp, "len": char_len})

    # --- 构造偏好对策略 ---
    # 1. 如果没有成功的，或者没有失败的，无法构造对
    if not passed_candidates or not failed_candidates:
        return None

    # 2. 选择 Chosen：在 Passed 中选择长度最适中的（避免最短的那个，防止偷懒）
    # 按照长度排序，取中位或者偏长一点的
    passed_candidates.sort(key=lambda x: x['len'], reverse=True)
    chosen = passed_candidates[0]['text'] # 取逻辑最全的

    # 3. 选择 Rejected：选择 Failed 样本中长度也较长的（这种样本通常有尝试写逻辑但写错了）
    # 这样对比学习的效果最好（Hard Negative）
    failed_candidates.sort(key=lambda x: x['len'], reverse=True)
    rejected = failed_candidates[0]['text']

    return {
        "instruction": prompt, # 适配 LLaMA-Factory 格式
        "input": "",
        "output": chosen,
        "rejected": rejected
    }

def main():
    print(f"🚀 启动高质量代码 DPO 筛选流水线...")
    
    # 统计总数用于进度条
    with open(INPUT_FILE, 'r') as f:
        total = sum(1 for _ in f)

    final_data = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with open(INPUT_FILE, 'r') as f:
            # 使用 imap 处理并显示进度条
            for result in tqdm(pool.imap(process_line, f, chunksize=10), total=total):
                if result:
                    final_data.append(result)

    # 保存为标准 JSON 格式
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成！原始 {total} -> 优质对 {len(final_data)}")
    print(f"📊 损耗率: {(1 - len(final_data)/total)*100:.1f}% (剔除了不合规样本)")

    # 随后还有一个二次清洗的的脚本产生了一个新的v3_final_safe_train.json

    with open("v3_dpo_cleaned_high_quality.json", 'r') as f:
        data = json.load(f)
    
    safe_data = []
    for entry in data:
        c_len = len(entry['output'])
        r_len = len(entry['rejected'])
        
        # 策略：如果 Chosen 比 Rejected 短很多（超过 20%），且 Chosen 包含 pass/todo，直接剔除
        is_lazy = any(w in entry['output'].lower() for w in ['pass', 'todo', 'return none'])
        if c_len < r_len * 0.8 and is_lazy:
            continue
        
        safe_data.append(entry)

    print(f"二次清洗完成：{len(data)} -> {len(safe_data)}")
    with open("v3_final_safe_train.json", 'w') as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
