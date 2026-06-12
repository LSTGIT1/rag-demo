"""
Reranker 重排序模块。

作用：
1. 接收用户问题 query 和 FAISS 初步召回的 docs。
2. 使用 BAAI/bge-reranker-v2-m3 对每个 query-document 对重新打分。
3. 按 Reranker 分数从高到低排序。
4. 返回最相关的前 N 个片段。

注意：
- FAISS 是第一阶段粗召回，速度快。
- Reranker 是第二阶段精排，准确率更高，但会增加耗时。
"""

import os
import platform
from typing import List, Dict

# 和 embedding_model.py 一样，设置模型缓存目录与镜像源
# 这几行必须放在 sentence_transformers import 之前
if platform.system() == "Windows":
    os.environ["HF_HOME"] = "F:/AI_Models_Cache"
else:
    os.environ["HF_HOME"] = "/root/autodl-tmp/AI_Models_Cache"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#os.environ["HF_ENDPOINT"] = "https://huggingface.co"


from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME, RERANK_TOP_N, RERANK_MIN_SCORE


class RerankerModel:
    def __init__(self):
        """
        初始化 Reranker 模型。

        CrossEncoder：
        - 输入是一对文本：[query, document]
        - 输出是这对文本的相关性分数
        - 适合用于 RAG 中的精排阶段

        注意：
        - 这里不再使用 FlagReranker
        - 因为你当前环境中 FlagEmbedding 和 transformers 可能存在兼容问题
        """
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            trust_remote_code=True
        )

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        top_n: int = RERANK_TOP_N
    ) -> List[Dict]:
        """
        对 FAISS 召回结果进行重排。

        Args:
            query: 用户问题。
            docs: FAISS 初步召回的文档片段列表。
            top_n: 重排后保留前几个。

        Returns:
            重排后的 docs，每个 doc 会新增 rerank_score 字段。
        """
        if not docs:
            return []

        query = str(query)

        pairs = []
        valid_docs = []

        for doc in docs:
            content = str(doc.get("content", "")).strip()

            if not content:
                continue

            # CrossEncoder 接收的是 query-document 文本对
            pairs.append((query, content))
            valid_docs.append(doc)

        if not pairs:
            return []

        # CrossEncoder 使用 predict 进行打分
        scores = self.reranker.predict(pairs)

        reranked_docs = []

        for doc, score in zip(valid_docs, scores):
            new_doc = dict(doc)
            new_doc["rerank_score"] = float(score)

            if RERANK_MIN_SCORE is not None and new_doc["rerank_score"] < RERANK_MIN_SCORE:
                continue

            reranked_docs.append(new_doc)

        # rerank_score 越高，说明 query 和文档片段越相关
        reranked_docs.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_docs[:top_n]
