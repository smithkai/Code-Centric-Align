"""
第一次拒绝采样判定用的脚本，生成文件v3_rs_verified_10k.json
"""
import json
import re
import multiprocessing
import subprocess
import os
import time
from tqdm import tqdm

# --- 核心配置 ---
INPUT_FILE = "data/rft/v3/v3_rs_candidates.jsonl"
OUTPUT_FILE = "data/rft/v3/v3_rs_verified_10k.jsonl"
NUM_WORKERS = 30  # 32核服务器建议设为30
TIMEOUT = 4       # 稍微增加一点超时容忍度

# --- 工具函数：必须定义在顶层 ---

def extract_python_code(text):
    """提取 Markdown 中的 Python 代码块，处理多种可能的格式"""
    if not isinstance(text, str):
        return ""
    # 优先提取 ```python ... ```
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 次优提取 ``` ... ```
    pattern_alt = r"```\s*(.*?)\s*```"
    match_alt = re.search(pattern_alt, text, re.DOTALL)
    if match_alt:
        return match_alt.group(1).strip()
    
    # 如果都没有，返回原文本（处理模型没打标签的情况）
    return text.strip()

def safe_execute(code):
    """安全沙箱执行"""
    if not code:
        return False, "Empty code"
    
    try:
        # 使用 subprocess 隔离运行
        res = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env={"PATH": os.environ.get("PATH", "")}
        )
        if res.returncode == 0:
            return True, "Passed"
        else:
            # 获取最后一行报错
            err_msg = res.stderr.strip().split('\n')[-1] if res.stderr else "Unknown Error"
            return False, err_msg
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"Runtime Error: {str(e)}"

def process_line(line):
    """单行处理逻辑"""
    if not line or not line.strip():
        return None
    
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None

    responses = item.get('responses', [])
    passed_flags = []
    verify_logs = []
    
    for resp in responses:
        code = extract_python_code(resp)
        is_passed, msg = safe_execute(code)
        passed_flags.append(is_passed)
        verify_logs.append(msg)
    
    item['passed_flags'] = passed_flags
    item['verify_logs'] = verify_logs
    return item

# --- 主逻辑 ---

def main():
    print(f"🚀 启动 32 核断点续传流水线...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_FILE}")
        return

    # 1. 快速统计总行数
    print("📏 正在扫描原始文件行数...")
    with open(INPUT_FILE, 'rb') as f:
        total_lines = sum(1 for _ in f)
    print(f"📊 原始文件总计: {total_lines} 条指令")

    # 2. 检查断点续传进度
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'rb') as f:
            processed_count = sum(1 for _ in f)
        print(f"🔃 检测到已有进度：已完成 {processed_count} 条，将从第 {processed_count + 1} 条开始...")
    
    if processed_count >= total_lines:
        print("✅ 所有数据已处理完毕，无需重复运行。")
        return

    # 3. 准备统计变量
    dpo_pairs = 0
    pass_at_10 = 0

    # 4. 流式处理：使用进程池和文件句柄
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            
            # 跳过已处理部分
            for _ in range(processed_count):
                next(f_in)
            
            # 剩余部分通过 imap 流式读取
            # chunksize=10 可以平衡 32 核的负载，不至于频繁通信
            results = pool.imap(process_line, f_in, chunksize=10)
            
            pbar = tqdm(results, total=total_lines - processed_count, desc="校验进度")
            for res in pbar:
                if res:
                    # 更新实时统计指标（虽然是局部统计，但能给个预期）
                    if any(res['passed_flags']):
                        pass_at_10 += 1
                        if not all(res['passed_flags']):
                            dpo_pairs += 1
                    
                    # 写入并刷新
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    
                    # 每 100 条 flush 一次磁盘，平衡性能与安全
                    if pbar.n % 100 == 0:
                        f_out.flush()

    print(f"\n🎉 任务全量完成！")
    print(f"📂 最终结果存放在: {OUTPUT_FILE}")
    print(f"💡 建议：运行完毕后再次统计该文件的『理想 DPO 偏好对』数量。")

if __name__ == "__main__":
    main()