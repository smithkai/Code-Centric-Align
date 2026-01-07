# 数据体检与深度清洗
# 目标：剔除“垃圾内容”。代码模型最怕：太短的没逻辑、太长的超出显存、或者全是注释没有代码。
# 过滤这部分之后好好做一下，如何处理数据能提高数据质量


import jsonlines
from tqdm import tqdm

def is_quality_code(text):
    """简单的代码质量启发式检查"""
    if len(text) < 50: return False  # 太短的代码通常没有逻辑
    if "import" not in text and "def" not in text: 
        # 如果既没有导包也没有定义函数，可能只是段普通文本，根据需求调整
        pass 
    return True

def run_clean():
    input_file = "data/raw/magicoder_raw.jsonl"
    output_file = "data/processed/magicoder_cleaned.jsonl"
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 存在就不创建了

    cleaned_count = 0
    total_count = 0

    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for item in tqdm(reader, desc="Cleaning Data"):
            total_count += 1
            instruction = item['instruction']
            output = item['output']

            # 1. 长度过滤：限制在模型常见的 context 范围内 (比如字符长度在 100 到 10000 之间)
            if not (100 <= len(output) <= 10000):
                continue

            # 2. 启发式逻辑检查
            if not is_quality_code(output):
                continue

            # 3. 简单去污：比如包含 "HumanEval" 字样的可能是测试集泄露，直接干掉
            if "humaneval" in instruction.lower() or "mbpp" in instruction.lower():
                continue

            writer.write(item)
            cleaned_count += 1

    print(f"清洗完成！保留率: {cleaned_count/total_count:.2%} ({cleaned_count}/{total_count})")

if __name__ == "__main__":
    run_clean()