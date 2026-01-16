"""
这是rft的第一步
拒绝采样，用SFT_V3去跑数据集（出于时间和费用，只跑了前10k数据）
可以得到每个instruction对应模型的10个
"""

import json
import os
import math
import time
from vllm import LLM, SamplingParams

# --- 路径配置 ---
MODEL_PATH = "models/sft_v3_merged_model"
INPUT_PATH = "data/processed/magicoder_final_cleaned.jsonl"
# 加上 v2 标识，区分你之前失败的那次
OUTPUT_PATH = "data/rft/v3/v3_rs_candidates.jsonl" 

# --- 硬件与性能配置 (针对 4090D 24G 优化) ---
GPU_UTIL = 0.85          # 预留显存，防止调度峰值 OOM
MAX_MODEL_LEN = 2048     # 缩短上下文以容纳更多并发
MAX_NUM_SEQS = 128       # 适中的并发数，平衡吞吐量与内存 Swap 压力
SWAP_SPACE = 16          # 16GB CPU 交换空间
ENFORCE_EAGER = True     # 禁用 CUDA Graph，释放 ~2G 显存

# --- 采样策略配置 ---
DATA_LIMIT = 10000       # 本次实验先跑 10k
N_CANDIDATES = 10        # 增加采样深度，目标是把筛选率从 1% 提升到 5% 以上
CHUNK_SIZE = 1000        # 每 1000 条指令作为一个 Chunk 提交，降低内存压力

def run_sampling():
    start_time = time.time()
    
    # 1. 初始化 vLLM 引擎
    print(f"🚀 正在初始化引擎: {MODEL_PATH}")
    
    # 用vLLM引擎加载模型
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=ENFORCE_EAGER,  # 禁用cuda graph 直接省下约2G显存
        tensor_parallel_size=1,  # 单卡运行 不开启张量并行
        swap_space=SWAP_SPACE  # 显存紧张时利用内存保证程序不奔溃
    )
    
    # 2. 设置采样参数 (Best-of-N 模式)
    sampling_params = SamplingParams(
        n=N_CANDIDATES,      # 关键：对每个 prompt 生成 10 个回答  vllm能直接指定生成多少回答，但不理解之后多个回答的数是如何区分的
        temperature=0.8,     # 较高温度：为了让模型“发散”，生成不同的逻辑，方便挑好的   温度决定模型生成内容的发散性
        top_p=0.95,          # 截断概率：保证生成质量，不至于乱说话
        max_tokens=1024,     # 限制每个回答的最长 token 数
        stop=["<|im_end|>", "<|endoftext|>"] # 停止符，防止模型复读或胡言乱语   这个停止符和模型训练学到的自己生成的会不会有冲突? GPT给的解释是不会冲突是一个双保险但还是不太理解?
    )

    # 3. 加载原始数据并打上标记
    all_data = []
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误：找不到输入文件 {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= DATA_LIMIT: break
            item = json.loads(line)
            # 构造 ChatML 提示词
            prompt = f"<|im_start|>user\n{item['instruction']}\n<|im_end|>\n<|im_start|>assistant\n"
            # 记录原始索引，方便后续合并数据时去重  就是以后可能加大数据量训练 还有是方便回溯找指令
            # idx_maeker is mainly used for solving Deduplication, merging, and traceability issues
            item['idx_marker'] = f"part1_{i}"  
            item['formatted_prompt'] = prompt
            all_data.append(item)

    total_items = len(all_data)
    # ceiling是天花板 所以ceil是向上取整
    num_chunks = math.ceil(total_items / CHUNK_SIZE)
    print(f"📦 已加载 {total_items} 条指令。分 {num_chunks} 个批次执行。")

    # 4. 循环分段处理
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 使用追加模式 'a'，即使脚本中途崩溃，已生成的也不丢
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
        for chunk_idx in range(num_chunks):
            # 获取每个批次的数据范围
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total_items)
            
            chunk_data = all_data[start:end]
            chunk_prompts = [d['formatted_prompt'] for d in chunk_data]
            
            print(f"\n⚡ 批次 {chunk_idx + 1}/{num_chunks} | 进度: {start}-{end}")
            
            # 执行推理
            outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=True)

            # 保存结果
            for j, output in enumerate(outputs):
                result = {
                    "idx_marker": chunk_data[j]['idx_marker'],
                    "instruction": chunk_data[j]['instruction'],
                    "responses": [o.text.strip() for o in output.outputs]
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            f_out.flush() # 批次完成即刻落盘

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\n✨ 采样完成！耗时: {duration:.2f} 分钟")
    print(f"📂 数据保存在: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_sampling()