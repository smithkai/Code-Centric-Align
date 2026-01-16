import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "models/Qwen2.5-Coder-7B"
adapter_path = "models/sft_output_v3"
save_path = "models/sft_v3_merged_model"

# 1. 优先从 adapter_path 加载 Tokenizer，因为它包含你训练时的所有配置（如 chat_template）
print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 2. 加载基座模型
# 使用 device_map="auto"：会自动利用 GPU，比纯 CPU 速度快几十倍
# low_cpu_mem_usage=True：能显著降低加载时的峰值内存占用，防止 "Killed"
print("正在加载基座模型 (使用 GPU/CPU 自动分配)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto", 
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 3. 加载适配器
print("正在加载适配器 (Adapter)...")
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path,
    device_map="auto" # 保持和基座模型一致的分配逻辑
)

# 4. 合并权重
# 这一步在 GPU 上进行矩阵相加非常快
print("正在合并权重...")
merged_model = model.merge_and_unload()

# 5. 保存合并后的完整模型
# safe_serialization=True 会保存为 .safetensors 格式，这是目前最推荐的格式
print(f"正在保存完整模型至 {save_path}...")
merged_model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

print(f"\n🎉 成功！模型已成功合并并保存至: {save_path}")