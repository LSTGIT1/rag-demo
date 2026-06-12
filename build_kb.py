# 构建知识库脚本
# document_loader.py text_splitter.py embedding_model.py vector_store.py 编排上述所有的离线模块，将 docs/ 目录下的文档加工并固化到 storage/ 中。

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from document_loader import load_all_documents
from text_splitter import split_text
from embedding_model import EmbeddingModel
from vector_store import VectorStore


def main():
    print("开始加载文档...")                      # 1. 提取所有文档
    documents = load_all_documents(DOCS_DIR)

    if not documents:
        print("docs 目录下没有可用文档，请先放入 txt/md/pdf 文件。")
        return

    all_chunks = []
    # 2. 对每个文档进行切分，并打上来源、ID 等元数据标签
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

    # 3. 初始化模型和向量库
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    # 4. 执行向量化并保存
    vector_store.build(all_chunks)
    vector_store.save()

    print("知识库构建完成，已保存到 storage/ 目录。")


if __name__ == "__main__":
    main()
