# 文本切分
#长文本处理引擎。由于大模型的上下文窗口有限，且段落越长向量化效果越模糊，需要将长文档切分成带有重叠（Overlap）的短文本块（Chunks）。


def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0

    # 统一换行符格式，清洗文本
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 经典的滑动窗口切分算法
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # 下一次切分的起点向后移动 chunk_size - overlap，从而实现重叠
        start += chunk_size - overlap

    return chunks

def split_documents(documents, chunk_size: int = 500, overlap: int = 100):
    """
    将多个文档切分成 chunks。

    Args:
        documents: document_loader 返回的文档列表，格式通常是：
            [
                {
                    "source": "文档路径",
                    "text": "文档全文"
                }
            ]

        chunk_size: 每个 chunk 的最大字符数。
        overlap: 相邻 chunk 之间的重叠字符数。

    Returns:
        all_chunks: 统一后的 chunk 列表，格式为：
            [
                {
                    "source": "文档路径",
                    "content": "切分后的文本片段",
                    "chunk_index": 0
                }
            ]
    """
    all_chunks = []

    for doc in documents:
        source = doc.get("source", "")
        text = doc.get("text", "")

        # 对单个文档全文进行切分
        chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # 给每个 chunk 补充来源和序号，方便后续溯源
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "content": chunk,
                "chunk_index": i
            })

    return all_chunks