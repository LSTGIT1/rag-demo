"""
带 Reranker 的命令行 RAG 问答入口。

运行方式：
    python reranker_ask.py

这个文件的作用：
1. 不修改原来的 ask.py，单独提供一个“增强检索版”的问答入口。
2. 原始 ask.py 流程是：FAISS TopK -> Prompt -> LLM。
3. 本文件流程是：FAISS 先召回更多候选 -> Reranker 精排 -> Prompt -> LLM。
4. 这样方便你对比“未加 Reranker”和“加入 Reranker”后的效果差异。

运行前提：
1. 已经执行 python build_kb.py 构建知识库。
2. storage/faiss.index 和 storage/chunks.json 已存在。
3. 服务器上的 vLLM / Qwen3-8B 服务正在运行。
4. 已安装 FlagEmbedding：pip install FlagEmbedding
5. reranker_model.py 文件存在，并能加载 BAAI/bge-reranker-v2-m3。
"""

import config

# ==============================
# Reranker 配置读取与默认值兜底
# ==============================
# 说明：
# 你后续可以把这些变量正式写入 config.py。
# 如果 config.py 里已经有这些配置，本文件会优先使用 config.py 的值。
# 如果 config.py 暂时没有，本文件会使用下面的默认值，避免直接启动失败。

# 是否启用 Reranker
USE_RERANKER = getattr(config, "USE_RERANKER", True)

# 不启用 Reranker 时，直接从 FAISS 取多少条
TOP_K = getattr(config, "TOP_K", 5)

# 启用 Reranker 时，FAISS 第一阶段先召回多少候选片段
RETRIEVE_TOP_K = getattr(config, "RETRIEVE_TOP_K", 20)

# Reranker 重排后，最终保留多少片段进入 Prompt
RERANK_TOP_N = getattr(config, "RERANK_TOP_N", 5)

# Reranker 模型名称
'''
因为 getattr 的逻辑是：
如果 config 里有 RERANKER_MODEL_NAME，就用 config 里的值；
如果没有，才用默认值。

'''
RERANKER_MODEL_NAME = getattr(
    config,
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3"
)

# Reranker 分数过滤阈值。
# 第一版建议先设为 None，不过滤，只按分数排序。
RERANK_MIN_SCORE = getattr(config, "RERANK_MIN_SCORE", None)

# reranker_model.py 内部可能会 from config import RERANKER_MODEL_NAME 等变量。
# 如果你的 config.py 暂时没写这些变量，这里先动态补到 config 模块对象上，
# 这样 reranker_model.py 导入时也能正常找到它们。
config.USE_RERANKER = USE_RERANKER
config.TOP_K = TOP_K
config.RETRIEVE_TOP_K = RETRIEVE_TOP_K
config.RERANK_TOP_N = RERANK_TOP_N
config.RERANKER_MODEL_NAME = RERANKER_MODEL_NAME
config.RERANK_MIN_SCORE = RERANK_MIN_SCORE

from embedding_model import EmbeddingModel
from vector_store import VectorStore
from prompt_builder import build_prompt
from llm_client import LLMClient
from reranker_model import RerankerModel


def print_docs(docs):
    """
    打印最终进入 Prompt 的参考片段信息。

    加入 Reranker 后，每条结果会同时包含：
    - faiss_score：第一阶段 FAISS 粗召回相似度分数。
    - rerank_score：第二阶段 Reranker 精排相关性分数。

    注意：
    不同 reranker 模型的分数范围不一定是 0-1。
    你只需要理解为：rerank_score 越大，说明相关性越高。
    """
    print("\n参考片段：")

    for i, doc in enumerate(docs, start=1):
        if "rerank_score" in doc:
            print(
                f"{i}. {doc['source']} | "
                f"faiss_score={doc['score']:.3f} | "
                f"rerank_score={doc['rerank_score']:.3f}"
            )
        else:
            print(
                f"{i}. {doc['source']} | "
                f"score={doc['score']:.3f}"
            )


def main():
    """
    命令行问答主流程。

    完整流程：
    1. 加载 Embedding 模型。
    2. 加载 FAISS 知识库。
    3. 加载大模型客户端。
    4. 可选加载 Reranker。
    5. 用户输入问题。
    6. FAISS 粗召回。
    7. Reranker 重排。
    8. 构造 Prompt。
    9. 调用 Qwen/vLLM 生成答案。
    10. 输出答案和参考片段。
    """

    # 1. 加载 Embedding 模型。
    # 用户问题需要先变成向量，才能去 FAISS 中做相似度检索。
    embedding_model = EmbeddingModel()

    # 2. 加载已经构建好的本地 FAISS 知识库。
    # 这里不会重新读取 docs，也不会重新向量化。
    # 它会直接读取 storage/faiss.index 和 storage/chunks.json。
    vector_store = VectorStore(embedding_model)
    vector_store.load()

    # 3. 创建大模型客户端。
    # 这个客户端会调用 config.py 中配置的 vLLM/OpenAI 兼容接口。
    llm = LLMClient()

    # 4. 按配置决定是否启用 Reranker。
    # 启用时会加载 BAAI/bge-reranker-v2-m3。
    # 第一次运行可能需要下载模型，启动会慢一些。
    reranker = RerankerModel() if USE_RERANKER else None

    print("RAG 问答系统已启动，输入 exit 退出。")
    print(f"Reranker 状态：{'已启用' if USE_RERANKER else '未启用'}")

    if USE_RERANKER:
        print(f"FAISS 第一阶段召回数量：{RETRIEVE_TOP_K}")
        print(f"Reranker 模型：{RERANKER_MODEL_NAME}")
        print(f"Reranker 最终保留数量：{RERANK_TOP_N}")
    else:
        print(f"FAISS 直接召回数量：{TOP_K}")

    while True:
        question = input("\n请输入问题：").strip()

        # 支持输入 exit / quit / q 退出命令行程序。
        if question.lower() in ["exit", "quit", "q"]:
            break

        # 空问题不处理，直接继续等待下一次输入。
        if not question:
            continue

        try:
            if USE_RERANKER:
                # ==============================
                # 第一阶段：FAISS 粗召回
                # ==============================
                # 加入 Reranker 后，FAISS 不应该只取 5 条，
                # 而是先取更多候选，比如 20 条。
                candidate_docs = vector_store.search(
                    question,
                    top_k=RETRIEVE_TOP_K
                )

                if not candidate_docs:
                    print("知识库中未找到相关依据。")
                    continue

                # ==============================
                # 第二阶段：Reranker 精排
                # ==============================
                # Reranker 会重新判断“问题”和“每个片段”的相关性，
                # 排序后只保留前 RERANK_TOP_N 个片段。
                docs = reranker.rerank(
                    question,
                    candidate_docs,
                    top_n=RERANK_TOP_N
                )

            else:
                # 不启用 Reranker 时，保持和 ask.py 类似的原始流程。
                docs = vector_store.search(
                    question,
                    top_k=TOP_K
                )

            if not docs:
                print("知识库中未找到相关依据。")
                continue

            # ==============================
            # 第三阶段：构造 RAG Prompt
            # ==============================
            # Prompt 中会包含用户问题和重排后的知识库片段。
            prompt = build_prompt(question, docs)

            # ==============================
            # 第四阶段：调用大模型生成答案
            # ==============================
            # llm.generate 会请求服务器上的 Qwen/vLLM 服务。
            answer = llm.generate(prompt)

            print("\n回答：")
            print(answer)

            # 打印参考片段，方便观察 Reranker 是否改变了排序。
            print_docs(docs)

        except Exception as exc:
            # 避免某一次提问报错后整个程序退出，方便连续调试。
            print(f"处理问题时发生错误：{exc}")


if __name__ == "__main__":
    main()
