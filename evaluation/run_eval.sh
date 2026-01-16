#!/bin/bash

# 1. 配置模型路径
# 请确保这是你合并（Merged）后的最终模型路径
MODEL_PATH="/root/autodl-tmp/LLaMA-Factory/saves/qwen/dpo_v3_final_merged"
OUTPUT_DIR="eval_results/dpo_v3.1"

# 创建输出文件夹
mkdir -p $OUTPUT_DIR

echo "开始测评模型: $MODEL_PATH"
echo "评测任务: HumanEval (Python)"

# 2. 执行测评命令
# --n_samples 1 表示计算 Pass@1
# --allow_code_execution 允许运行模型生成的代码进行单元测试验证
python ../bigcode-evaluation-harness/main.py \
    --model $MODEL_PATH \
    --tasks humaneval \
    --max_length_generation 1024 \
    --precision bf16 \
    --allow_code_execution \
    --batch_size 1 \
    --do_sample False \
    --save_generations \
    --save_generations_path $OUTPUT_DIR

# 3. 提示结果位置
echo "----------------------------------------"
echo "测评完成！"
echo "生成的代码保存在: $OUTPUT_DIR/generations.json"
echo "你可以通过查看终端上方的 Pass@1 分数来评估效果。"
