# 调用本地/服务器大模型
#封装与 vLLM/Ollama 等大模型推理后端的网络通信。

from openai import OpenAI
from config import LLM_BASE_URL, LLM_MODEL_NAME, LLM_API_KEY


class LLMClient:
    def __init__(self):
        # 借助 openai 包的通用性，只需替换 base_url 即可调用本地模型
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

    def generate(self, prompt: str) -> str:
        # 发起流式或阻塞式的 ChatCompletion 请求
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个基于知识库资料回答问题的助手。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,            # 降低 temperature 可减少大模型随机性，更适合严谨的 RAG 任务
            max_tokens=1024             # 限制最大生成长度
        )

        return response.choices[0].message.content
