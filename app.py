#可选：FastAPI接口
#将整个 RAG 系统封装为标准 RESTful API，方便前端页面（如 Vue/React 或小程序）调用


from fastapi import FastAPI
from pydantic import BaseModel

from embedding_model import EmbeddingModel
from vector_store import VectorStore
from prompt_builder import build_prompt
from llm_client import LLMClient

# 初始化 FastAPI 实例
app = FastAPI(title="Local RAG Demo")

# 在服务启动时将模型和知识库挂载到内存中，避免每次请求重复加载
embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
vector_store.load()
llm = LLMClient()

# 定义请求体的入参结构，具有数据校验功能
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    # API 的核心逻辑与 ask.py 极其类似：检索 -> 组装 -> 生成
    docs = vector_store.search(req.question)

    if not docs:
        return {
            "answer": "知识库中未找到相关依据。",
            "references": []
        }

    prompt = build_prompt(req.question, docs)
    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "references": [
            {
                "source": doc["source"],
                "score": doc["score"],
                "chunk_id": doc["chunk_id"]
            }
            for doc in docs
        ]
    }
