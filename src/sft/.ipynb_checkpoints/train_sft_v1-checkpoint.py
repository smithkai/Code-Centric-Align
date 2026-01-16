import os
import torch
# Transformer这个库有必要好好看一下
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,  # 训练参数这个还没看
    BitsAndBytesConfig  # 负责压缩模型的配置
)
# prepare_model_for_kbit_training这个方法只有peft时才会用，所以时这个库提供的
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # 轻量化微调相关包
# trl库是将微调流程推向全自动化的关键一步
from trl import SFTTrainer  # SFTTrainer专门为SFT设计的，简化了数据处理和模型训练的代码量
from datasets import load_dataset

def train():
    # 1. 配置路径与模型
    model_id = "models/Qwen2.5-Coder-7B" # 建议选择专门的代码底座
    dataset_path = "data/processed/magicoder_evolved.jsonl"  # 使用指令演进后的数据集
    output_dir = "models/sft_output"

    # 2.  tokenizer 加载
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # 允许执行模型仓库中的自定义代码，例如可能有自定义的分词逻辑
    tokenizer.pad_token = tokenizer.eos_token  # 要指定padding token
    tokenizer.padding_side = "right" # 告诉模型，句子短了从右边补齐

    # 3. 量化配置 (QLoRA) - 极大降低显存需求
    """模型压缩的四大金刚
    量化，剪枝，蒸馏，结构设计/矩阵分解
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 16bit权重压缩到4bit加载
        bnb_4bit_quant_type="nf4",  # 使用正态浮点数(NF4)量化，比常规的4bit更精准
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算升维到BF16 保证精度
        bnb_4bit_use_double_quant=True,  # 对量化后的参数再量化一次，进一步省显存
    )

    # 4. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        # device_map={"": int(os.environ.get("LOCAL_RANK", 0))}, # 分布式适配,这个参数决定权重如何往显卡分配
        device_map = "auto",
        trust_remote_code=True,  # 部分模型会在架构中加入一些自定义代码，需要加这个参数让它自己执行
        attn_implementation="flash_attention_2" # 开启 FlashAttention2,架构层面改变注意力的计算效率,改用adpa了，FlashAttention涉及底层cuda和上层cuda要保持一致
    )
    # 下面这行代码，推理的时候可以不用，全参数微调也不存在，因为不能用量化，只有LoRA这种冻结微调的时候才需要，恢复部分精度，加检查点
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA 配置 (针对代码逻辑，我们覆盖所有 Linear 层)
    # 这里量化配置和LoRA配置不是很理解
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. 训练参数设定
    # 训练参数配置也不理解，为什么就训练3个epoch
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 原来是4,1,4
        gradient_accumulation_steps=4,  # 原来是4,16,4
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=3,
        save_steps=100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True, # 确保显卡支持
        tf32=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,       # 必须开启，以节省显存
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # torch_compile=True,                # 开启编译优化
        logging_first_step=True,
        # deepspeed="configs/ds_z3_config.json", # 启用 DeepSpeed,deepseed一般是用来来多卡训练的，这里单卡，改成deepspeed2
        report_to="wandb" # 建议开启监控
    )

    # 7. 数据加载与格式转换
    # datasets部分也有很多问题
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select(range(10000))  # 先取10000条数据试一下
    if "data_tag" in dataset.column_names:
        dataset = dataset.remove_columns(["data_tag"])
    # def formatting_prompts_func(example):
    #     output_text = []
    #     for i in range(len(example['instruction'])):
    #         # 严格对齐 ChatML 格式，包含系统提示词
    #         text = f"<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n" \
    #                f"<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
    #         output_text.append(text)
    #     return output_text
    def formatting_prompts_func(example):
        # 注意：packing=True 时，example 是单条数据字典
        text = (
            f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
        return text

    
    # 8. 启动 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,  # 有了SFTTrainer就不用load_peft_model了
        max_seq_length=1024, #根据显存调整
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
        packing=True,
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()
