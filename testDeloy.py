import openai

# 关键：现在可以直接访问本地的 8000 端口
client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key="empty"
)

response = client.chat.completions.create(
    model="/root/autodl-tmp/Code-Centric-Align/models/qwen2.5-coder-7b-awq",
    messages=[{"role": "user", "content": "用 Python 写一个简单的爬虫"}]
)

print(response.choices[0].message.content)
