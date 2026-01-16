import json
import re
import black
from tqdm import tqdm

def format_code(code):
    try:
        inner_code = re.sub(r"```python\s*(.*?)\s*```", r"\1", code, flags=re.DOTALL).strip()
        formatted = black.format_str(inner_code, mode=black.FileMode())
        return f"```python\n{formatted.strip()}\n```"
    except:
        return code

def get_fingerprint(text):
    """提取文本指纹：只保留字母和数字，忽略所有空格、换行和特殊符号"""
    if not text: return ""
    return re.sub(r'\W+', '', str(text)).lower()

def main():
    # 1. 加载 199 条 PASS 样本
    print("📖 加载 199 条 DeepSeek 审计通过样本...")
    with open("data/align/v3_deepseek_verified_dpo.json", 'r') as f:
        pure_passed = json.load(f)
    
    # 使用指纹识别已通过的 Prompt
    passed_fingerprints = {get_fingerprint(item['instruction']) for item in pure_passed if item['instruction']}
    print(f"DEBUG: passed_fingerprints 独立数量: {len(passed_fingerprints)}")

    # 2. 加载全量 GT 索引
    print("🎯 正在加载原始 GT 索引...")
    gt_map = {} # fingerprint -> raw_output
    with open("data/raw/magicoder_raw.jsonl", 'r') as f:
        for line in f:
            it = json.loads(line)
            # 兼容 magicoder 可能的字段名
            instr = it.get('instruction') or it.get('prompt')
            if instr:
                gt_map[get_fingerprint(instr)] = it.get('output') or it.get('response')

    # 3. 加载 2299 条待处理样本
    with open("data/align/v3_final_safe_train.json", 'r') as f:
        all_candidates = json.load(f)

    # 打印前 3 条指纹看看是不是全一样
    print(f"DEBUG: 前 3 条 candidate 指纹示例: {[get_fingerprint(it['instruction'])[:20] for it in all_candidates[:3]]}")

    final_dataset = []

    # 第一部分：添加 199 条黄金对
    for item in pure_passed:
        final_dataset.append({
            "instruction": item['instruction'],
            "output": format_code(item['output']),
            "rejected": format_code(item['rejected'])
        })

    # 第二部分：异源对
    print("🧪 正在构建 GT vs 模型错误对...")
    match_success = 0
    match_fail = 0
    skipped_pass = 0

    for item in tqdm(all_candidates):
        raw_instr = item['instruction']
        fp = get_fingerprint(raw_instr)
        
        if not fp: continue

        if fp in passed_fingerprints:
            skipped_pass += 1
            continue
            
        if fp in gt_map:
            final_dataset.append({
                "instruction": raw_instr,
                "output": format_code(gt_map[fp]),
                "rejected": format_code(item['output'])
            })
            match_success += 1
        else:
            match_fail += 1

    print(f"\n✅ 整合完成！")
    print(f"📊 同源对 (Model-Model): {len(pure_passed)}")
    print(f"📊 异源对 (GT-Model): {match_success}")
    print(f"⏭️ 跳过同源项: {skipped_pass}")
    print(f"❌ 匹配失败: {match_fail}")

    with open("data/align/v3_final_mixed_dpo.json", "w") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
