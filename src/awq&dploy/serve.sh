#!/bin/bash

# --- 关键修复：让子进程认识 conda ---
# 这里的路径通常是 /root/miniconda3/... 或 /opt/conda/...
# 如果你不确定，在终端输入: echo $CONDA_EXE 然后把 bin/conda 换成 etc/profile.d/conda.sh
source /root/miniconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh

# --- 激活项目目录下的环境 ---
# 使用绝对路径最稳妥，假设你的项目在 /root/autodl-tmp/Code-Centric-Align
conda activate /root/autodl-tmp/align_env

# --- 启动服务 ---
python3 -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Code-Centric-Align/models/qwen2.5-coder-7b-awq \
    --quantization awq \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
