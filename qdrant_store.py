"""
Qdrant 向量库封装。

作用：
1. 连接 Qdrant。
2. 创建 / 重建 collection。
3. 写入 chunk、向量和元数据。
4. 根据用户问题召回相关片段。
"""

from typing import List, Dict
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    QDRANT_MODE,
    QDRANT_URL,
    QDRANT_LOCAL_PATH,
    QDRANT_COLLECTION_NAME,
)


class QdrantStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.collection_name = QDRANT_COLLECTION_NAME

        # server：连接 Docker Qdrant 服务
        # local：不启动服务，直接写入本地目录
        if QDRANT_MODE == "local":
            self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
        else:
            self.client = QdrantClient(url=QDRANT_URL)

    def get_vector_size(self) -> int:
        """自动检测 embedding 向量维度。"""
        vector = self.embedding_model.encode(["测试向量维度"])[0]
        return len(vector)

    def recreate_collection(self):
        """删除旧 collection 并重新创建。"""
        vector_size = self.get_vector_size()

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        print(f"Collection 已创建：{self.collection_name}")
        print(f"向量维度：{vector_size}")

    def create_collection_if_not_exists(self):
        """如果 collection 不存在，则创建。"""
        if not self.client.collection_exists(self.collection_name):
            vector_size = self.get_vector_size()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert_chunks(self, chunks: List[Dict], batch_size: int = 64):
        """将 chunks 批量写入 Qdrant。"""
        if not chunks:
            print("没有可写入的 chunk")
            return

        total = len(chunks)

        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            texts = [str(item.get("content", "")) for item in batch]

            # 生成向量
            vectors = self.embedding_model.encode(texts)

            points = []
            for idx, (chunk, vector) in enumerate(zip(batch, vectors)):
                source = str(chunk.get("source", ""))
                content = str(chunk.get("content", ""))

                payload = {
                    "content": content,
                    "source": source,
                    "file_name": source.split("/")[-1].split("\\")[-1],
                    "chunk_index": start + idx,
                }

                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            print(f"已写入：{min(start + batch_size, total)} / {total}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """根据用户问题从 Qdrant 召回相关片段。"""
        query_vector = self.embedding_model.encode([str(query)])[0]

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        docs = []
        for point in result.points:
            payload = point.payload or {}
            docs.append({
                "content": payload.get("content", ""),
                "source": payload.get("source", ""),
                "score": point.score,
                "payload": payload,
            })

        return docs

    def count(self) -> int:
        """统计 collection 中的 point 数量。"""
        return self.client.count(
            collection_name=self.collection_name,
            exact=True
        ).count
