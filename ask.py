# 命令行问答脚本
#命令行形式的交互控制台，将在线链路串联起来。


from embedding_model import EmbeddingModel
from vector_store import VectorStore
from prompt_builder import build_prompt
from llm_client import LLMClient


def main():
    # 1. 唤醒基础设施：加载模型与本地磁盘上的知识库
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

        # 2. 检索阶段：去向量库中寻找高相关性片段
        docs = vector_store.search(question)

        if not docs:
            print("知识库中未找到相关依据。")
            continue

        # 3. 增强与生成阶段：构造 Prompt 并请求大模型
        prompt = build_prompt(question, docs)
        answer = llm.generate(prompt)

        # 4. 终端输出结果展示
        print("\n回答：")
        print(answer)

        print("\n参考片段：")
        for i, doc in enumerate(docs, start=1):
            print(f"{i}. {doc['source']} | score={doc['score']:.3f}")


if __name__ == "__main__":
    main()
