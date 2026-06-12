下面给你一套适合个人电脑运行的**小型 Python RAG 项目布置方案**。你照着建文件、装依赖、复制代码，就可以跑通：

```text
文档 → 切分 → 向量化 → 保存知识库 → 检索 → 拼 Prompt → 调用你的 Qwen/vLLM/Ollama → 输出答案
```

这里默认你已经有一个可访问的大模型接口，例如：

```text
vLLM:   http://服务器IP:6006/v1
Ollama: http://localhost:11434
```

---

## 1. 项目目录结构

建议新建目录：

```text
rag-demo/
│
├─ docs/                         # 放你的知识库原始文档
│  ├─ demo.txt
│  └─ project.md
│
├─ storage/                      # 自动生成，保存向量库和文本块
│  ├─ faiss.index
│  └─ chunks.json
│
├─ config.py                     # 配置项
├─ document_loader.py            # 文档加载
├─ text_splitter.py              # 文本切分
├─ embedding_model.py            # 向量模型
├─ vector_store.py               # FAISS 向量库
├─ prompt_builder.py             # Prompt 构造
├─ llm_client.py                 # 调用本地/服务器大模型
├─ build_kb.py                   # 构建知识库脚本
├─ ask.py                        # 命令行问答脚本
├─ app.py                        # 可选：FastAPI接口
└─ requirements.txt              # 依赖列表
```

---

## 2. 创建虚拟环境

进入项目目录：

```bash
cd rag-demo
```

创建虚拟环境：

```bash
python -m venv .venv
```

Windows 激活：

```bash
.venv\Scripts\activate
```

Linux / macOS 激活：

```bash
source .venv/bin/activate
```

---

## 3. requirements.txt

创建：

```text
requirements.txt
```

内容如下：

```txt
openai>=1.40.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
numpy>=1.26.0
pypdf>=4.0.0
python-dotenv>=1.0.1
fastapi>=0.110.0
uvicorn>=0.29.0
```

安装：

```bash
pip install -r requirements.txt
```

如果安装 `faiss-cpu` 失败，可以先换源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. config.py

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
STORAGE_DIR = BASE_DIR / "storage"

FAISS_INDEX_PATH = STORAGE_DIR / "faiss.index"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5
MIN_SCORE = 0.35

# 如果你用 vLLM/OpenAI兼容接口
LLM_BASE_URL = "http://localhost:6006/v1"
LLM_MODEL_NAME = "Qwen3-8B"
LLM_API_KEY = "EMPTY"
```

如果你是在个人电脑调用服务器模型，把：

```python
LLM_BASE_URL = "http://localhost:6006/v1"
```

改成：

```python
LLM_BASE_URL = "http://服务器IP:6006/v1"
```

---

## 5. document_loader.py

```python
from pathlib import Path
from pypdf import PdfReader


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n[第{i + 1}页]\n{text}")

    return "\n".join(pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return load_txt(path)

    if suffix == ".pdf":
        return load_pdf(path)

    raise ValueError(f"暂不支持的文件类型: {path}")


def load_all_documents(docs_dir: Path):
    files = []
    for path in docs_dir.rglob("*"):
        if path.suffix.lower() in [".txt", ".md", ".pdf"]:
            files.append(path)

    documents = []
    for path in files:
        text = load_document(path)
        documents.append({
            "source": str(path),
            "text": text
        })

    return documents
```

---

## 6. text_splitter.py

```python
def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
```

---

## 7. embedding_model.py

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vectors.astype("float32")
```

第一次运行会下载 embedding 模型，需要联网。下载完成后会缓存到本地。

---

## 8. vector_store.py

```python
import json
import faiss
from pathlib import Path
from config import FAISS_INDEX_PATH, CHUNKS_PATH, STORAGE_DIR, TOP_K, MIN_SCORE


class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]):
        self.chunks = chunks

        texts = [item["content"] for item in chunks]
        vectors = self.embedding_model.encode(texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def save(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self):
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE):
        query_vector = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if float(score) < min_score:
                continue

            item = self.chunks[idx]

            results.append({
                "score": float(score),
                "content": item["content"],
                "source": item["source"],
                "chunk_id": item["chunk_id"]
            })

        return results
```

这里使用的是 FAISS：

```text
IndexFlatIP
```

因为向量已经 normalize，所以内积相似度等价于余弦相似度。

---

## 9. prompt_builder.py

```python
def build_prompt(question: str, docs: list[dict]) -> str:
    context = []

    for i, doc in enumerate(docs, start=1):
        context.append(
            f"""[资料{i}]
来源：{doc["source"]}
相关度：{doc["score"]:.3f}
内容：
{doc["content"]}
"""
        )

    context_text = "\n".join(context)

    prompt = f"""
你是一个严谨的企业知识库问答助手。

请严格根据【知识库资料】回答【用户问题】。
如果资料中没有明确答案，请回答：“知识库中未找到相关依据。”
不要编造资料中不存在的内容。
回答最后请列出参考来源。

【知识库资料】
{context_text}

【用户问题】
{question}

请给出回答：
""".strip()

    return prompt
```

---

## 10. llm_client.py

```python
from openai import OpenAI
from config import LLM_BASE_URL, LLM_MODEL_NAME, LLM_API_KEY


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

    def generate(self, prompt: str) -> str:
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
            temperature=0.2,
            max_tokens=1024
        )

        return response.choices[0].message.content
```

---

## 11. build_kb.py

```python
from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from document_loader import load_all_documents
from text_splitter import split_text
from embedding_model import EmbeddingModel
from vector_store import VectorStore


def main():
    print("开始加载文档...")
    documents = load_all_documents(DOCS_DIR)

    if not documents:
        print("docs 目录下没有可用文档，请先放入 txt/md/pdf 文件。")
        return

    all_chunks = []

    for doc in documents:
        chunks = split_text(
            doc["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "content": chunk
            })

    print(f"文档数量：{len(documents)}")
    print(f"文本块数量：{len(all_chunks)}")

    print("开始向量化并构建 FAISS 索引...")
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    vector_store.build(all_chunks)
    vector_store.save()

    print("知识库构建完成，已保存到 storage/ 目录。")


if __name__ == "__main__":
    main()
```

---

## 12. ask.py

```python
from embedding_model import EmbeddingModel
from vector_store import VectorStore
from prompt_builder import build_prompt
from llm_client import LLMClient


def main():
    embedding_model = EmbeddingModel()

    vector_store = VectorStore(embedding_model)
    vector_store.load()

    llm = LLMClient()

    print("RAG 问答系统已启动，输入 exit 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        docs = vector_store.search(question)

        if not docs:
            print("知识库中未找到相关依据。")
            continue

        prompt = build_prompt(question, docs)
        answer = llm.generate(prompt)

        print("\n回答：")
        print(answer)

        print("\n参考片段：")
        for i, doc in enumerate(docs, start=1):
            print(f"{i}. {doc['source']} | score={doc['score']:.3f}")


if __name__ == "__main__":
    main()
```

---

## 13. app.py，可选 API 服务

如果你想把 RAG 做成接口：

```python
from fastapi import FastAPI
from pydantic import BaseModel

from embedding_model import EmbeddingModel
from vector_store import VectorStore
from prompt_builder import build_prompt
from llm_client import LLMClient


app = FastAPI(title="Local RAG Demo")

embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
vector_store.load()
llm = LLMClient()


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
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
```

启动：

```bash
uvicorn app:app --host 127.0.0.1 --port 9000
```

测试：

```bash
curl -X POST http://127.0.0.1:9000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"请介绍一下项目内容\"}"
```

Windows PowerShell 可以用：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:9000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"请介绍一下项目内容"}'
```

---

## 14. 放入测试文档

创建：

```text
docs/demo.txt
```

示例内容：

```txt
本项目是一个小型RAG检索增强生成系统。
系统使用Python开发，使用FAISS作为本地向量数据库。
系统流程包括文档加载、文本切分、向量化、相似度检索、Prompt构造和大模型生成。
大模型通过vLLM提供的OpenAI兼容接口进行调用。
```

---

## 15. 运行流程

### 第一步：构建知识库

```bash
python build_kb.py
```

成功后会生成：

```text
storage/faiss.index
storage/chunks.json
```

### 第二步：启动问答

```bash
python ask.py
```

输入：

```text
这个项目使用什么向量数据库？
```

应该能回答：

```text
使用 FAISS 作为本地向量数据库
```

### 第三步：可选，启动 API

```bash
uvicorn app:app --host 127.0.0.1 --port 9000
```

---

## 16. 使用时注意

如果你调用的是服务器上的 vLLM 模型，先确认服务通：

```bash
curl http://服务器IP:6006/v1/models
```

如果能返回模型列表，再运行 RAG。

`config.py` 里的模型名必须和 `/v1/models` 返回的 `id` 一致：

```python
LLM_MODEL_NAME = "Qwen3-8B"
```

---

## 17. 后续维护建议

你后面可以逐步升级：

```text
1. 支持 Word 文档
2. 支持 Excel 文档
3. 支持更好的中文切分
4. 加入 reranker
5. 加入 BM25 关键词检索
6. 加入日志
7. 加入 Web 页面
8. 把 FAISS 换成 Milvus 或 Qdrant
```

但第一版就按上面的目录和代码做，最容易跑通。