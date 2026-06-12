import os
from openai import OpenAI
import time
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key=os.getenv("DASHSCOPE_API_KEY"), # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
start = time.time()
client = OpenAI(

    api_key="na",
    base_url="https://u508661-jlx8-31132355.bjb1.seetacloud.com:8443/v1",
)

completion = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user',
         'content': 'Tell me something about large language models.'}
    ]
)

print(completion.choices[0].message.content)
print("\n总时间：", time.time() - start, "秒")