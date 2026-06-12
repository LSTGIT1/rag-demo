"""
构建 Qdrant 知识库。
"""

from document_loader import load_documents
from text_splitter import split_documents
from embedding_model import EmbeddingModel
from qdrant_store import QdrantStore
from config import QDRANT_RECREATE_COLLECTION


def main():
    print("开始构建 Qdrant 知识库...")

    documents = load_documents()
    print(f"加载文档数量：{len(documents)}")

    if not documents:
        print("docs 目录没有可用文档")
        return

    chunks = split_documents(documents)
    print(f"切分 chunk 数量：{len(chunks)}")

    embedding_model = EmbeddingModel()
    qdrant_store = QdrantStore(embedding_model)

    if QDRANT_RECREATE_COLLECTION:
        qdrant_store.recreate_collection()
    else:
        qdrant_store.create_collection_if_not_exists()

    qdrant_store.upsert_chunks(chunks)

    print(f"Qdrant 当前 point 数量：{qdrant_store.count()}")
    print("Qdrant 知识库构建完成")


if __name__ == "__main__":
    main()
