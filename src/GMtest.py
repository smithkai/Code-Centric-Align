from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. 准备数据（模拟一些古诗/对联）
data = {
    "text": [
        "白日依山尽，黄河入海流。",
        "举头望明月，低头思故乡。",
        "欲穷千里目，更上一层楼。"
    ]
}
dataset = Dataset.from_dict(data)

# 2. 加载分词器
model_id = "gpt2" # 也可以换成 "uer/gpt2-chinese-cluecorpussmall" 更好的支持中文
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GPT2 没有 padding token，需要手动指定，否则训练会报错
tokenizer.pad_token = tokenizer.eos_token

# 3. 预处理函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=32)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Data Collator：负责把数据整理成批次，并处理 Padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. 加载模型
model = AutoModelForCausalLM.from_pretrained(model_id)

# 6. 定义训练参数 (这就是你的配置文件)
training_args = TrainingArguments(
    output_dir="./gpt2-poet",       # 训练结果保存路径
    overwrite_output_dir=True,      # 覆盖旧文件
    num_train_epochs=5,             # 训练轮数
    per_device_train_batch_size=4,  # 每个设备的批次大小
    save_steps=10,                  # 每多少步保存一次
    logging_steps=5,                # 每多少步打印一次日志
    learning_rate=5e-5,             # 学习率
    weight_decay=0.01,              # 权重衰减
)

# 7. 启动训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

# 8. 保存模型和分词器
trainer.save_model("./my_final_poet_model")
tokenizer.save_pretrained("./my_final_poet_model")

# --- 华丽的分界线：假设这是在另一个脚本中 ---

# 9. 加载并推理
from transformers import pipeline

poet_bot = pipeline("text-generation", model="./my_final_poet_model", tokenizer="./my_final_poet_model")

# 生成一首诗
result = poet_bot("白日依山尽", max_length=20, num_return_sequences=1)
print(result[0]['generated_text'])