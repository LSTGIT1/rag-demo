# 文档加载
# 充当数据适配器。根据不同的后缀，使用不同的解析策略来提取提纯文本

from pathlib import Path
from pypdf import PdfReader
from config import DOCS_DIR

def load_txt(path: Path) -> str:
    # 读取纯文本文件，忽略无法识别的编码错误，防止中断
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    # 使用 pypdf 读取 PDF 文件
    reader = PdfReader(str(path))

    pages = []
    # 逐页提取文本，并加上页码标识，方便后续溯源

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n[第{i + 1}页]\n{text}")

    return "\n".join(pages)


def load_document(path: Path) -> str:
    # 路由函数：根据文件后缀名分发给具体的加载函数
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return load_txt(path)

    if suffix == ".pdf":
        return load_pdf(path)

    raise ValueError(f"暂不支持的文件类型: {path}")


def load_all_documents(docs_dir: Path):
    # 遍历目标目录及其子目录下的所有支持的文件
    files = []
    for path in docs_dir.rglob("*"):
        if path.suffix.lower() in [".txt", ".md", ".pdf"]:
            files.append(path)

    documents = []
    for path in files:
        text = load_document(path)
        documents.append({
            "source": str(path),
            "text": text
        })

    return documents
def load_documents():
    """
    默认加载 config.py 中 DOCS_DIR 指定目录下的所有文档。

    这样其他模块可以直接：
        documents = load_documents()

    不需要每次都传 docs_dir。
    """
    return load_all_documents(Path(DOCS_DIR))