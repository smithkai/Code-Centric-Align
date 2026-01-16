import json
import re
import numpy as np
import os

def get_code_stats(code):
    """提取单段代码的统计特征"""
    if not code or not isinstance(code, str):
        return None
    
    # 移除 markdown 标签
    code_clean = re.sub(r"```python\s*(.*?)\s*```", r"\1", code, flags=re.DOTALL).strip()
    if not code_clean:
        return None

    length = len(code_clean)
    lines = code_clean.split('\n')
    total_lines = len(lines) if len(lines) > 0 else 1
    # 统计注释行 (以#开头的行)
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    
    return {
        "length": length,
        "lines": total_lines,
        "comment_ratio": comment_lines / total_lines
    }

def stream_analyze(file_path):
    """流式分析文件，支持 .json 和 .jsonl"""
    lengths = []
    lines_of_code = []
    comment_ratios = []
    
    print(f"📖 正在处理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 判断是 json 还是 jsonl
        if file_path.endswith('.jsonl'):
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                res = get_code_stats(item.get('output') or item.get('response') or "")
                if res:
                    lengths.append(res['length'])
                    lines_of_code.append(res['lines'])
                    comment_ratios.append(res['comment_ratio'])
        else:
            # 对于标准的 .json 列表，使用 ijson 或简单的流式加载
            # 这里针对你的 199 条小文件，直接加载即可
            data = json.load(f)
            for item in data:
                res = get_code_stats(item.get('output') or item.get('response') or "")
                if res:
                    lengths.append(res['length'])
                    lines_of_code.append(res['lines'])
                    comment_ratios.append(res['comment_ratio'])

    if not lengths:
        return None

    return {
        "Avg Length": np.mean(lengths),
        "Avg Lines": np.mean(lines_of_code),
        "Comment %": np.mean(comment_ratios) * 100,
        "Max Length": np.max(lengths),
        "Sample Count": len(lengths)
    }

def main():
    # 路径配置 (请根据实际情况微调)
    model_file = "data/align/v3_deepseek_verified_dpo.json"
    gt_file = "data/raw/magicoder_raw.jsonl"

    if not os.path.exists(model_file) or not os.path.exists(gt_file):
        print("❌ 错误：请检查文件路径是否正确")
        return

    # 1. 分析模型生成的 199 条数据
    model_results = stream_analyze(model_file)
    
    # 2. 分析原始巨量数据集
    gt_results = stream_analyze(gt_file)

    print("\n" + "="*60)
    print(f"{'指标 (Metric)':<15} | {'模型生成 (199条)':<15} | {'原始 GT 数据':<15}")
    print("-" * 60)
    
    for key in ["Avg Length", "Avg Lines", "Comment %", "Max Length", "Sample Count"]:
        m_val = model_results[key] if model_results else 0
        g_val = gt_results[key] if gt_results else 0
        print(f"{key:<15} | {m_val:<15.2f} | {g_val:<15.2f}")
    print("="*60)

if __name__ == "__main__":
    main()