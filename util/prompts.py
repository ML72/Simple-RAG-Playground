from langchain_core.prompts import ChatPromptTemplate

def default_prompt():
    """
    The default RAG prompt.
    """
    return ChatPromptTemplate.from_template(
        """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {input}
        Answer:"""
    )

PROMPT_REGISTRY = {
    "default": default_prompt,
}

def get_prompt(name):
    """
    Retrieve a prompt template by name.
    """
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt '{name}' not found. Available prompts: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]()
