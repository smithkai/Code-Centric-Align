"""
2025/01/09
这个脚本主要是针对第一次SFT40%的准确率重新进行设计
1. v1用了packing,就是不用填充，多条指令塞满一个序列，会出问题。所以v2用了batching，一个序列就是一个样本。
2. 刚刚改过来存在梯度消失的问题是，消除packing后，formatting_prompts_func函数存在[]被str的问题，导致模型出现一堆奇怪东西，但为什么会梯度消失我还是不太理解
3. v1实验使用的学习率过大，2x10^-4,于是我在v2把它降到2x10^-5
4. LoRA Alpha的问题，也是太大，然后学习率和噪声都大，就会造成训练过程的不稳定，效果也不好
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM  # 这个工具是只对回答部分计算损失DataCollatorForCompletionOnlyLM
from datasets import load_dataset

def train():
    # 1. 路径配置
    model_id = "models/Qwen2.5-Coder-7B"
    dataset_path = "data/processed/magicoder_evolved.jsonl"
    output_dir = "models/sft_output_v2" # 换个目录，防止覆盖

    # 2. Tokenizer 优化
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # 使用特殊 token 作为 pad，避免占用 eos_token 导致无法正常结束
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" 

    # 3. QLoRA 配置 (保持 NF4 压缩)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA 配置 (降低干扰强度)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64, # 修改：alpha=r 是最稳健的配比，128 太激进了
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. 训练参数设定 (大幅下调学习率)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,          # 修改：从 2e-4 降到 2e-5，防止权重被“砸烂”
        num_train_epochs=1,          # 修改：10k 数据 1 个 epoch 足够对齐，多跑易过拟合
        max_steps=625,               # 显式指定总步数为 625 (10000/16)
        lr_scheduler_type="cosine",  # 余弦退火有助于模型后期收敛
        warmup_ratio=0.1,            # 增加预热比例，让模型平滑进入微调状态
        logging_steps=10,
        save_steps=100,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        save_total_limit=3           # 只保留最后的几个点，节省磁盘
    )

    # 7. 数据格式处理
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select(range(10000))

    # def formatting_prompts_func(example):
    #     # 包装成列表返回
    #     text = (
    #         f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
    #         f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    #     )
    #     return [text] # 必须加中括号变成 List
    
    # 针对后面print部分数据，AI发现数据和加载的，破案结果
    def formatting_prompts_func(example):
        output_texts = []
        # 判断输入是单条还是多条（SFTTrainer 在 packing=False 时可能传入 batch）
        if isinstance(example['instruction'], list):
            for i in range(len(example['instruction'])):
                text = (
                    f"<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n"
                    f"<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
                )
                output_texts.append(text)
        else:
            # 单条情况
            text = (
                f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>"
            )
            output_texts.append(text)
        return output_texts
    
    # 8. 启动 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
        packing=False, # 修改：关闭 packing，让模型学会独立处理每一条指令的边界
    )
    print(f"Sample Data: {tokenizer.decode(trainer.train_dataset[0]['input_ids'])}")
    
    trainer.train()
    trainer.save_model(output_dir)
    print(f"🎉 训练完成！新模型已保存至: {output_dir}")

if __name__ == "__main__":
    train()