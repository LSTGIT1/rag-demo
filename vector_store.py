# FAISS 向量库
# 负责高维向量的存储和近邻检索（ANN）


import json
import faiss
from pathlib import Path
from config import FAISS_INDEX_PATH, CHUNKS_PATH, STORAGE_DIR, TOP_K, MIN_SCORE


class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []    # 用于保存向量对应的原始文本及其元数据

    def build(self, chunks: list[dict]):
        self.chunks = chunks

        # 调用 Embedding 模型将所有文本块转化为向量矩阵
        texts = [item["content"] for item in chunks]

        # 调用 Embedding 模型将所有文本块转化为向量矩阵
        vectors = self.embedding_model.encode(texts)

        # 构建基于内积（Inner Product）的暴力检索索引
        # 因为向量已经过归一化，此时的内积数学上等价于余弦相似度（Cosine Similarity）
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def save(self):
        # 将 FAISS 索引和原始文本块的 JSON 数据持久化到硬盘
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self):
        # 启动时将本地的索引和文本块加载到内存中
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE):
        # 1. 将用户查询转化为向量
        query_vector = self.embedding_model.encode([query])

        # 2. 在 FAISS 库中检索最相似的 Top K 个向量
        # scores 存储相似度得分，indices 存储命中向量的内部 ID
        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS 没找到足够的数据时会返回 -1
                continue

            if float(score) < min_score:    # 滤除低于阈值的无关干扰片段
                continue


            # 3. 根据 ID 找回原始文本块
            item = self.chunks[idx]
            results.append({
                "score": float(score),
                "content": item["content"],
                "source": item["source"],
                "chunk_id": item["chunk_id"]
            })

        return results
