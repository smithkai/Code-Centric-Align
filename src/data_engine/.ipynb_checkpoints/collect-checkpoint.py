# 数据采集与流式下载
import jsonlines
from datasets import load_dataset
from tqdm import tqdm
import os

def run_collect():
    # 配置数据集名称和保存路径
    dataset_name = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    save_path = "data/raw/magicoder_raw.jsonl"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"正在从 HuggingFace 远程流式加载数据集: {dataset_name}...")
    
    # streaming=True 关键：不会下载整个文件，而是像水龙头一样一行行读取
    ds = load_dataset(dataset_name, split="train", streaming=True)

    count = 0
    with jsonlines.open(save_path, mode='w') as writer:
        for item in tqdm(ds, desc="Downloading & Saving"):
            # 统一字段格式，方便后续处理
            # Magicoder 原本字段是 'instruction' 和 'response'
            formatted_item = {
                "instruction": item['instruction'],
                "output": item['response'],
                "metadata": {"source": "magicoder_oss"}
            }
            writer.write(formatted_item)
            count += 1
            
            # 初次尝试可以只取 10000 条，如果是完整项目可以一直跑完
            # if count >= 10000: break 

    print(f"采集完成！共计 {count} 条原始数据存入 {save_path}")

if __name__ == "__main__":
    run_collect()