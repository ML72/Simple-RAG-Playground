from langchain_core.prompts import ChatPromptTemplate

def simple_prompt():
    """
    The simple RAG prompt.
    """
    return ChatPromptTemplate.from_template(
        """You are a precise question answering system. Your task is to use the context below to provide a short answer to the specified question.
        Please be concise and answer in only a single short paragraph.

        Context: {context}

        Question: {input}

        Answer:"""
    )

def custom_prompt():
    """
    The custom RAG prompt.
    """
    return ChatPromptTemplate.from_template(
        """You are a helpful chatbot. Use the context below to answer the question at the end. Answer in a single paragraph.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {input}

        Answer:"""
    )

PROMPT_REGISTRY = {
    "simple": simple_prompt,
    "custom": custom_prompt,
}

def get_prompt(name):
    """
    Retrieve a prompt template by name.
    """
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt '{name}' not found. Available prompts: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]()
