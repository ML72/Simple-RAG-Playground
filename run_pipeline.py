"""
RAG Pipeline Evaluation Script
Evaluates a RAG system using Giskard's RAG Evaluation Toolkit (RAGET)
with explicit LangChain Expression Language (LCEL) pipelines.
"""
import os
import argparse
import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path
from operator import itemgetter

# --- Modern LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Giskard Imports ---
from giskard.rag import KnowledgeBase

# --- Local Imports ---
from util.prompts import get_prompt
from util.loaders import (
    load_documents_from_json,
    load_test_data_from_json,
    create_giskard_testset_from_data
)
from util.evaluation import (
    evaluate_with_raget,
    calculate_quality_score,
    save_results_as_markdown
)

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def format_docs(docs):
    """Helper for LCEL: Join retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def setup_rag_pipeline(documents, prompt_template, chunk_size=500, chunk_overlap=100):
    """
    Setup the RAG pipeline using modern LCEL (LangChain Expression Language).
    """
    print("\n" + "="*60)
    print("Setting up RAG Pipeline (LCEL Mode)...")
    print("="*60 + "\n")
    
    # 1. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")
    
    # 2. Vector Store
    print("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. LLM & Prompt
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    
    # 4. Define LCEL Chain
    # This replaces the "Black Box" create_retrieval_chain
    # We explicitly define how data flows from input -> retriever -> prompt -> llm
    
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    print("✅ Modern RAG pipeline setup complete\n")
    return rag_chain


def main():
    parser = argparse.ArgumentParser(description='Evaluate Modern LCEL RAG pipeline')
    parser.add_argument('--documents', type=str, default='data/document_texts.json')
    parser.add_argument('--test-data', type=str, default='data/test_data.json')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--chunk-overlap', type=int, default=100)
    parser.add_argument('--prompt', type=str, default='default', help='Name of the prompt to use (default: default)')
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Error: OPENAI_API_KEY not set.")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    documents = load_documents_from_json(args.documents)
    test_data = load_test_data_from_json(args.test_data)
    
    # Get Prompt
    try:
        prompt_template = get_prompt(args.prompt)
        print(f"Using prompt: {args.prompt}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Setup Pipeline
    rag_chain = setup_rag_pipeline(
        documents,
        prompt_template,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Setup Giskard Knowledge Base
    print("Creating knowledge base for evaluation...")
    df_docs = pd.DataFrame({"content": [doc.page_content for doc in documents]})
    knowledge_base = KnowledgeBase.from_pandas(df_docs, columns=["content"])
    
    # Setup Testset
    testset = create_giskard_testset_from_data(test_data)
    
    # Evaluate
    report = evaluate_with_raget(rag_chain, testset, knowledge_base)
    quality_score = calculate_quality_score(report)
    
    # Save Results
    print("\n" + "="*60)
    print(f"🔒 Overall Quality Score: {quality_score}/100")
    print("="*60 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = output_dir / f"evaluation_report_{timestamp}.md"
    
    save_results_as_markdown(
        report, quality_score, md_path, 
        len(test_data), len(documents)
    )
    
    print(f"✅ Report saved to: {md_path}")

if __name__ == "__main__":
    main()