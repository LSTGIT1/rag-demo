#配置
import os
from pathlib import Path


# 获取当前文件所在目录的绝对路径，作为项目的基准目录
BASE_DIR = Path(__file__).resolve().parent

# 定义各种资源的存储路径
DOCS_DIR = BASE_DIR / "docs"          # 原始文档存放目录
STORAGE_DIR = BASE_DIR / "storage"    # 向量索引和切片数据的持久化目录
FAISS_INDEX_PATH = STORAGE_DIR / "faiss.index" # FAISS 向量库文件路径
CHUNKS_PATH = STORAGE_DIR / "chunks.json"      # 文本块原始内容备份文件

# Embedding 模型配置
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5" # 使用的中文向量化模型

# 文本切分策略参数
CHUNK_SIZE = 500      # 每个文本块的最大长度（字符数）
CHUNK_OVERLAP = 100   # 相邻文本块之间的重叠长度，防止关键信息在边界处被截断

# 检索策略参数
TOP_K = 5             # 每次检索召回的最相关文本块数量
MIN_SCORE = 0.35      # 相似度阈值，低于此分数的片段将被过滤，保证检索质量

# 大模型 API 配置（适配兼容 OpenAI 格式的 vLLM/Ollama 接口）
#LLM_BASE_URL = "http://127.0.0.1:6006/v1" # 大模型服务的地址
LLM_BASE_URL = "https://u508661-jlx8-31132355.bjb1.seetacloud.com:8443/v1"

LLM_MODEL_NAME = "Qwen3-8B"               # 模型名称，需与启动的模型一致
LLM_API_KEY = "EMPTY"                     # 本地或私有部署通常无需严格鉴权，设为占位符




# 是否启用 Reranker
USE_RERANKER = True

# Reranker 模型名称
#RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"


# FAISS 第一阶段召回数量。加入 Reranker 后不要只召回 5 个，建议先召回 20 个
RETRIEVE_TOP_K = 10

# Reranker 重排后最终给大模型的片段数量
RERANK_TOP_N = 5

# Reranker 分数阈值。第一版先不强过滤，设为 None
RERANK_MIN_SCORE = None


# =========================
# Qdrant 配置
# =========================

# 推荐：连接 Docker 中运行的 Qdrant 服务
QDRANT_MODE = "server"

# Qdrant HTTP 服务地址
QDRANT_URL = "http://127.0.0.1:6333"

# 如果不用 Docker，可以改成 local，并使用这个本地路径
QDRANT_LOCAL_PATH = "F:/Project/rag-demo/qdrant_storage"

# Collection 可以理解为一个知识库表
QDRANT_COLLECTION_NAME = "rag_demo"

# Qdrant 第一阶段召回数量
QDRANT_TOP_K = 20

# 实验阶段建议 True：每次构建都删除旧 collection 并重建
QDRANT_RECREATE_COLLECTION = True