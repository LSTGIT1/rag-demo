# Prompt 构造
#将召回的上下文（Context）和用户的问题（Query）拼接成结构化的 Prompt，强制大模型基于给定的资料作答，降低幻觉（Hallucination）

def build_prompt(question: str, docs: list[dict]) -> str:
    context = []

    # 格式化组装检索到的知识库片段
    for i, doc in enumerate(docs, start=1):
        context.append(
            f"""[资料{i}]
                来源：{doc["source"]}
                相关度：{doc["score"]:.3f}
                内容：
                {doc["content"]}
            """
        )

    context_text = "\n".join(context)

    prompt = f"""
你是一个严谨的企业知识库问答助手。

请严格根据【知识库资料】回答【用户问题】。
如果资料中没有明确答案，请回答：“知识库中未找到相关依据。”
不要编造资料中不存在的内容。
回答最后请列出参考来源。

【知识库资料】
{context_text}

【用户问题】
{question}

请给出回答：
""".strip()

    return prompt
