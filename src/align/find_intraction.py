"""
这是不知道哪个数据集开始出现指令丢失的问题，解决指令丢失问题做的脚本
"""


import json
import re
from tqdm import tqdm

def get_robust_key(code):
    """提取鲁棒性更强的匹配 Key：提取第一个出现的函数名和参数"""
    if not code: return ""
    # 匹配 def function_name(args):
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1) # 返回函数名
    return ""

def get_logic_content(code):
    """提取代码的主体逻辑，去掉注释和空白，只取前 200 个有效字符"""
    # 去掉 markdown 标签
    code = re.sub(r"```python\s*(.*?)\s*```", r"\1", code, flags=re.DOTALL)
    # 去掉所有 unittest 部分 (模型喜欢乱加这个)
    code = re.split(r'class Test|if __name__', code)[0]
    # 提取纯字母数字
    clean = re.sub(r'\W+', '', code).lower()
    return clean[:200]

def main():
    # 1. 建立二级索引
    print("🎯 正在建立双层回溯索引...")
    logic_to_instr = {}
    func_name_to_instr = {} # 备选方案

    with open("data/raw/magicoder_raw.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            it = json.loads(line)
            instr = it.get('instruction', '')
            out = it.get('output', '')
            
            logic_fp = get_logic_content(out)
            func_name = get_robust_key(out)
            
            if logic_fp: logic_to_instr[logic_fp] = (instr, out)
            if func_name: func_name_to_instr[func_name] = (instr, out)

    # 2. 加载待修复数据
    with open("data/align/v3_final_safe_train.json", 'r') as f:
        candidates = json.load(f)

    final_dataset = []
    recovered_count = 0

    print("🧪 正在尝试通过逻辑特征找回题目...")
    for item in tqdm(candidates):
        model_out = item.get('output', '')
        
        # 尝试逻辑匹配
        m_logic_fp = get_logic_content(model_out)
        match_res = logic_to_instr.get(m_logic_fp)
        
        # 如果逻辑没匹配上，尝试函数名匹配
        if not match_res:
            m_func_name = get_robust_key(model_out)
            match_res = func_name_to_instr.get(m_func_name)

        if match_res:
            instr, gt_out = match_res
            final_dataset.append({
                "instruction": instr,
                "output": gt_out,        # 既然模型没题目，这里的 output 建议用 GT
                "rejected": model_out   # 模型的输出作为负样本
            })
            recovered_count += 1

    print(f"\n✅ 修复报告:")
    print(f"📊 成功找回: {recovered_count} / {len(candidates)}")
    print(f"📦 导出数据集规模: {len(final_dataset)}")

    with open("data/align/v4_fixed_recovery.json", "w") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
