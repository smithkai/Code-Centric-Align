import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

# 1. 设置环境变量，减少显存碎片，防止 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. 核心 Collator：手动处理 Tokenization 和 Masking
# Instruction Masking（指令屏蔽） 这其实是模型训练的一个关键
class QwenDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, response_template, max_length=2048):
        super().__init__(tokenizer=tokenizer, mlm=False)  # 明确告诉模型,我们是Causal LM不是Masked LM
        self.response_template = response_template
        self.max_length = max_length

    def __call__(self, examples):
        texts = [example["text"] for example in examples]  # 这不会内存爆炸吗? call方法处理的是一个batch所以不会炸
        # 转id
        # batch 是一个类似于 dict（字典）的对象
        batch = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        # 默认的labels就是和input_ids一样的
        labels = batch["input_ids"].clone()

        # 下面这行代码没有特别理解
        # 好像是不加encode会16 numeral system
        response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)  # 我们是要滑窗找,加special_tokens会完蛋 不知道special tokens什么时候用
        
        for i in range(len(texts)):
            input_ids_list = batch["input_ids"][i].tolist()
            found_idx = None
            for idx in range(len(input_ids_list) - len(response_token_ids) + 1):
                if input_ids_list[idx : idx + len(response_token_ids)] == response_token_ids:
                    found_idx = idx + len(response_token_ids)
                    break
            
            if found_idx is not None:
                labels[i, :found_idx] = -100
            
            labels[i][batch["attention_mask"][i] == 0] = -100
            
        batch["labels"] = labels
        return batch

def train():
    model_id = "models/Qwen2.5-Coder-7B"
    dataset_path = "data/processed/magicoder_final_cleaned.jsonl" 
    output_dir = "models/sft_output_v3"

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = "<|endoftext|>"  # pad_token要在编码阶段和屏蔽阶段配合逻辑发挥作用,有些疑惑。
    
    # 2. 模型加载 (4-bit QLoRA)量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 它是目前 4-bit 量化中精度损失最低的一种，比传统的线性量化（FP4）强很多。
        bnb_4bit_compute_dtype=torch.bfloat16,  # 定义计算精度，告诉模型计算时要转会bfloat16精度
        bnb_4bit_use_double_quant=True,  # 双重量化
    )

    # LoRA理论上可以用于很多架构
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # 参数默认是None,会把模型放到cpu上
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    """
    为什么需要 trust_remote_code？ 
    如果 Qwen 的作者发明了一种全新的计算公式（比如优化了计算顺序），现有的 transformers 库里没有这个公式的 Python 代码，
    它就必须从云端下载作者写的 modeling_qwen.py 来告诉电脑“怎么算”。
    """
    # 低精度的模型为了能微调需要做一些调整，低精度可以推理,但为了应对微调需要计算的梯度,需要
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA 设置
    peft_config = LoraConfig(
        r=64, lora_alpha=64,
        # 瞄准Transfomrer的替代方案
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # 训练中随机丢弃5%的路径,防止过拟合
        bias="none", 
        task_type="CAUSAL_LM",  # LoRA也需要知道任务的类型
    )
    """
    o_proj等等这些是 Qwen（以及 Llama）架构中 MLP（多层感知机） 层的组成部分。Qwen 的 MLP 并不是传统的 FNN，而是采用了 SwiGLU 激活函数的结构
    """
    model = get_peft_model(model, peft_config)  # 外挂LoRA + 冻结基座 => new model

    # 4. 数据预处理 加载数据进内存 如果是特别大的数据会用内存映射
    # load_dataset会帮你把json类型转换成Python字典的形式
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train") #指定 train 会返回一个 Dataset 对象而不是 DatasetDict 
    
    def formatting_func(example):
        text = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>")
        return {"text": text}
    
    dataset = raw_dataset.map(formatting_func, remove_columns=raw_dataset.column_names).shuffle(seed=42)
    
    # 5. 训练参数（已优化显存占用）
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,      # 降到 2 以应对长样本
        gradient_accumulation_steps=8,     # 相应增加到 8，保持总 batch 16 不变 用小显存模拟大batch的训练效果
        learning_rate=2e-5,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,                    # 每 200 步保存一次
        bf16=True,  # Brain Float 16 精度，相比传统的 float16，它在处理大数值时更稳，不容易出现 Loss 变成 NaN 的情况
        tf32=True,  # 在 A100/30 系列显卡上加速矩阵乘法
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        save_total_limit=2,                # 只保留最近两个 checkpoint，节省磁盘
        remove_unused_columns=False,
    )

    # 6. 使用原生 Trainer
    collator = QwenDataCollator(tokenizer=tokenizer, response_template="<|im_start|>assistant\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # 7. 开始训练（支持自动从断点恢复）
    # 如果 output_dir 下有 checkpoint，它会自动加载最新进度
    # 智能恢复：如果文件夹里有 checkpoint 就恢复，没有就从头开始
    checkpoint = None
    if os.path.exists(output_dir) and os.listdir(output_dir):
        checkpoint = True

    trainer.train(resume_from_checkpoint=checkpoint)
    
    # 8. 保存模型
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()