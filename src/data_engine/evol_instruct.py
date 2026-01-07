import requests
import jsonlines
import time
import json
import shutil # 用于快速复制文件
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置信息保持不变
API_KEY = "sk-2fb3ba5427a04f8c8d00517af99c1cb2"
API_URL = "https://api.deepseek.com/chat/completions"

def evolve_data(item):
    """
    演进函数：直接处理传入的字典副本
    """
    instruction = item.get('instruction', '')
    system_prompt = (
        "你是一个高级代码专家。你的任务是进行指令演进（Evol-Instruct）。\n"
        "1. 进化指令：将基础编程任务变得更难。增加算法复杂度、性能约束。\n"
        "2. 生成答案：编写高质量代码。\n"
        "请严格遵守 JSON 格式返回：{\"instruction\": \"...\", \"output\": \"...\"}"
    )
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请进化这个指令并给出答案: {instruction}"}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        evolved_result = json.loads(content)
        
        # 构造新对象，保留原有的其他元数据（如果有的话）
        new_item = item.copy() 
        new_item['instruction'] = evolved_result['instruction']
        new_item['output'] = evolved_result['output']
        new_item['data_tag'] = 'evolved_extra' # 标记这是额外增加的演进数据
        return new_item
    except Exception:
        return None

def run_evol_streaming():
    input_file = "data/processed/magicoder_final.jsonl"
    output_file = "data/processed/magicoder_evolved.jsonl"
    evolve_count = 20  # 只需要演进前20条作为额外补充
    max_workers = 5

    # --- 第一步：直接物理复制原文件到目标位置 ---
    # 这步不经过内存，速度极快，且完整保留了所有原始数据
    shutil.copyfile(input_file, output_file)
    print(f"原始数据已全量同步至 {output_file}")

    # --- 第二步：流式读取前 N 条进行演进 ---
    data_to_evolve = []
    with jsonlines.open(input_file) as reader:
        for i, item in enumerate(reader):
            if i >= evolve_count:
                break
            data_to_evolve.append(item) # 只读入需要演进的这几条，不占内存

    print(f"开始演进前 {evolve_count} 条数据并追加到文件末尾...")

    # --- 第三步：并发演进并实时追加写入 ---
    with jsonlines.open(output_file, mode='a') as writer: # 'a' 模式追加在原文件末尾
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(evolve_data, item): item for item in data_to_evolve}
            
            for future in tqdm(as_completed(future_to_item), total=len(data_to_evolve)):
                result = future.result()
                if result:
                    writer.write(result) # 发现一条，写入一条

if __name__ == "__main__":
    run_evol_streaming()