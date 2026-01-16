from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_path = "models/stf_v3_merged_model" # 你合并后的模型路径
quant_path = "models/qwen2.5-coder-7b-awq"    # 量化模型保存路径


# 1. 权重路径（指向你 DPO 合并后的完整模型）
model_path = "/root/autodl-tmp/Code-Centric-Align/models/sft_v3_merged_model" 
# 2. Tokenizer 路径（关键：指向你最初的基座模型，或者 Qwen 官方模型路径）
# 如果你本地有原版 Qwen 文件夹，填那个路径；如果没有，填 "Qwen/Qwen2.5-Coder-7B-Instruct"
base_tokenizer_path = "/root/autodl-tmp/Code-Centric-Align/models/Qwen2.5-Coder-7B" 

def quantize_model():
    print("--- 步骤 1: 使用兼容模式加载 Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path, trust_remote_code=True)
    
    print("--- 步骤 2: 加载 DPO 合并后的权重 ---")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        safetensors=True,
        low_cpu_mem_usage=True
    )

    print("--- 步骤 3: 准备本地代码校准数据 (纯文本列表) ---")
    # AutoAWQ 要求列表元素是纯字符串
    # 建议准备 32-64 条样本以获得更好的量化效果
    calib_data = [
        "def quick_sort(arr): return arr if len(arr) <= 1 else quick_sort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quick_sort([x for x in arr[1:] if x >= arr[0]])",
        "import torch\nclass Net(torch.nn.Module):\n    def __init__(self):\n        super(Net, self).__init__()\n        self.fc = torch.nn.Linear(10, 1)",
        "using namespace std;\nint main() { cout << 'Hello World' << endl; return 0; }",
        "SELECT * FROM users WHERE age > 18 ORDER BY create_time DESC;",
        "// 实现一个单例模式\npublic class Singleton { private static Singleton instance; private Singleton() {} }"
    ] * 8  # 简单通过重复增加样本量到 40 条，满足量化器的最小统计需求

    print("--- 步骤 4: 启动 AWQ 量化 ---")
    # 这里的 version 建议从 "GEMM" 改为 "AUTO"，让它自动适配你的环境
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    
    # 直接传入字符串列表 calib_data
    # AutoAWQ 内部会判断：如果是 list of str，则不会去下载 pile 数据集
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    print("--- 步骤 5: 保存结果 ---")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    print(f"量化圆满完成！离线校准成功。保存位置: {quant_path}")

if __name__ == "__main__":
    quantize_model()