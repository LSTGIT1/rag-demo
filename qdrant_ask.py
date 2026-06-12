
"""
Qdrant + Reranker + LLM 问答入口。
"""

from embedding_model import EmbeddingModel
from qdrant_store import QdrantStore
from prompt_builder import build_prompt
from llm_client import LLMClient
from reranker_model import RerankerModel

from config import USE_RERANKER, QDRANT_TOP_K, RERANK_TOP_N


def print_docs(docs):
    """打印参考片段。"""
    print("\n参考片段：")

    for i, doc in enumerate(docs, start=1):
        if "rerank_score" in doc:
            print(
                f"{i}. {doc.get('source', '')} | "
                f"qdrant_score={doc.get('score', 0):.3f} | "
                f"rerank_score={doc.get('rerank_score', 0):.3f}"
            )
        else:
            print(
                f"{i}. {doc.get('source', '')} | "
                f"qdrant_score={doc.get('score', 0):.3f}"
            )


def main():
    embedding_model = EmbeddingModel()
    qdrant_store = QdrantStore(embedding_model)
    llm = LLMClient()

    reranker = RerankerModel() if USE_RERANKER else None

    print("Qdrant RAG 问答系统已启动，输入 exit 退出。")
    print(f"Qdrant 召回数量：{QDRANT_TOP_K}")
    print(f"Reranker 状态：{'已启用' if USE_RERANKER else '未启用'}")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        try:
            # 1. Qdrant 召回
            docs = qdrant_store.search(question, top_k=QDRANT_TOP_K)

            if not docs:
                print("Qdrant 中未找到相关片段")
                continue

            # 2. Reranker 精排
            if USE_RERANKER and reranker is not None:
                docs = reranker.rerank(question, docs, top_n=RERANK_TOP_N)

            # 3. 构造 prompt
            prompt = build_prompt(question, docs)

            # 4. 调用大模型
            answer = llm.generate(prompt)

            print("\n回答：")
            print(answer)

            print_docs(docs)

        except Exception as e:
            print(f"处理问题时发生错误：{e}")


if __name__ == "__main__":
    main()