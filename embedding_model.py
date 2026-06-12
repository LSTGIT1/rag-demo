# 向量模型
#将人类语言转化为机器可理解的高维稠密向量
import os

import platform

if platform.system() == "Windows":
    os.environ["HF_HOME"] = "F:/AI_Models_Cache"
else:
    os.environ["HF_HOME"] = "/root/autodl-tmp/AI_Models_Cache"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(self):
        # 初始化并加载预训练的嵌入模型（首次运行会从 HuggingFace/ModelScope 下载）
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        # 将传入的文本列表批量转换为向量
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,      # 关键操作：L2归一化。归一化后，向量长度为1。
            convert_to_numpy=True           # 转换为 numpy 数组以便 FAISS 处理
        )
        return vectors.astype("float32")    # 强制转换为 float32，这是 FAISS C++ 底层要求的标准数据类型
