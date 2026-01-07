# 数据采集与流式下载
import jsonlines  # 用于处理JSONL格式文件
from datasets import load_dataset  # Hugging Face官方库，用于下载和处理数据集
from tqdm import tqdm  # 用于显示进度条，下载过程可视化
import os  # 处理文件路径和创建目录

def run_collect():
    # 配置数据集名称和保存路径
    dataset_name = "ise-uiuc/Magicoder-OSS-Instruct-75K"  # 这个路径是怎么回事，了解一下Dataset和Huggingface的关系
    save_path = "data/raw/magicoder_raw.jsonl"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"正在从 HuggingFace 远程流式加载数据集: {dataset_name}...")
    
    # streaming=True 关键：不会下载整个文件，而是像水龙头一样一行行读取
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # 打印第一条字段，确定一下数据名称
    first_item = next(iter(ds))
    print(f"数据示例及字段名: {first_item.keys()}")

    count = 0
    with jsonlines.open(save_path, mode='w') as writer:
        for item in tqdm(ds, desc="Downloading & Saving"):
            # 统一字段格式，方便后续处理
            # Magicoder 原本字段是 'instruction' 和 'response'
            formatted_item = {
                "instruction": item['problem'],
                "output": item['solution'],
                "metadata": {
                    "source": "magicoder_oss",
                    "lang": item.get('lang', 'unknown'), # 编程语言类型
                    "seed": item.get('seed', '')  # 生成数据时参考的原始代码片段或种子信息
                }
            }
            writer.write(formatted_item)
            count += 1
            
            # 初次尝试可以只取 10000 条，如果是完整项目可以一直跑完
            # if count >= 10000: break 

    print(f"采集完成！共计 {count} 条原始数据存入 {save_path}")

if __name__ == "__main__":# 打印第一条数据来看看字段名
    run_collect()