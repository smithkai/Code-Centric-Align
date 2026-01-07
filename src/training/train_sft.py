import os
import torch
from transformers import (
    AutoModelForCausalLM,  # 自动根据名称加载不同的模型
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # 轻量化微调相关包
from trl import SFTTrainer  # SFTTrainer专门为SFT设计的，简化了数据处理和模型训练的代码量
from datasets import load_dataset

def train():
    # 1. 配置路径与模型
    model_id = "Qwen/Qwen2.5-Coder-7B" # 建议选择专门的代码底座
    dataset_path = "data/processed/final_training_data.jsonl"
    output_dir = "models/sft_output"

    # 2.  tokenizer 加载
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 代码模型常用右填充

    # 3. 量化配置 (QLoRA) - 极大降低显存需求
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
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))}, # 分布式适配
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # 开启 FlashAttention
    )
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA 配置 (针对代码逻辑，我们覆盖所有 Linear 层)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. 训练参数设定
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True, # 确保显卡支持
        logging_first_step=True,
        deepspeed="configs/ds_z3_config.json", # 启用 DeepSpeed
        report_to="wandb" # 建议开启监控
    )

    # 7. 数据加载与格式转换
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_prompts_func(example):
        output_text = []
        for i in range(len(example['instruction'])):
            # 严格对齐 ChatML 格式，包含系统提示词
            text = f"<|im_start|>system\n{example['system'][i]}<|im_end|>\n" \
                   f"<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n" \
                   f"<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
            output_text.append(text)
        return output_text

    # 8. 启动 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=4096, # 根据显存调整
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()