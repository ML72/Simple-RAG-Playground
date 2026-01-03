"""
Generate Test Set from Knowledge Base Documents
Uses Giskard's RAGET to generate test questions for RAG evaluation
"""
import argparse
import json
import os
import warnings
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import giskard
from giskard.rag import KnowledgeBase, generate_testset
from giskard.rag.question_generators import (
    simple_questions,
    complex_questions,
    distracting_questions,
    situational_questions,
    double_questions
)

QUESTION_TYPES = {
    "simple": simple_questions,
    "complex": complex_questions,
    "distracting": distracting_questions,
    "situational": situational_questions,
    "double": double_questions
}

# Suppress Pydantic serializer warnings from Giskard
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def load_documents_from_json(json_path):
    """
    Load documents from JSON file
    
    Args:
        json_path: Path to JSON file containing documents
        
    Returns:
        List of document dictionaries
    """
    print(f"Loading documents from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    return documents


def create_knowledge_base(documents):
    """
    Create Giskard KnowledgeBase from documents
    
    Args:
        documents: List of document dictionaries with 'page_content' key
        
    Returns:
        KnowledgeBase object
    """
    print("Creating knowledge base...")
    
    # Extract page content from documents
    df = pd.DataFrame({
        "content": [doc["page_content"] for doc in documents]
    })
    
    knowledge_base = KnowledgeBase.from_pandas(df, columns=["content"])
    print(f"Knowledge base created with {len(documents)} documents")
    
    return knowledge_base


def generate_test_questions(knowledge_base, num_questions, agent_description=None, question_types=None):
    """
    Generate test set using Giskard RAGET
    
    Args:
        knowledge_base: Giskard KnowledgeBase object
        num_questions: Number of test questions to generate
        agent_description: Optional description of the RAG agent
        question_types: Optional list of question types to generate
        
    Returns:
        Generated testset
    """
    print(f"\nGenerating {num_questions} test questions...")
    
    if agent_description is None:
        agent_description = "A RAG-based question answering system"
    
    # Configure Giskard LLM and embedding models
    giskard.llm.set_llm_model("openai/gpt-4o")
    giskard.llm.set_embedding_model("openai/text-embedding-3-small")
    
    # Resolve question generators
    generators = None
    if question_types:
        generators = []
        for q_type in question_types:
            if q_type in QUESTION_TYPES:
                generators.append(QUESTION_TYPES[q_type])
            else:
                print(f"Warning: Unsupported question type '{q_type}'. Skipping.")
        
        if not generators:
            print("Warning: No valid question generators found. Using default.")
            generators = None

    # Generate testset
    testset = generate_testset(
        knowledge_base,
        num_questions=num_questions,
        language='en',
        agent_description=agent_description,
        question_generators=generators
    )
    
    print(f"Generated {len(testset)} test questions")
    return testset


def save_testset(testset, output_path):
    """
    Save testset to JSON file
    
    Args:
        testset: Giskard testset object
        output_path: Path to save JSON file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert testset to DataFrame and then to JSON
    df = testset.to_pandas()
    
    # Save to JSON
    df.to_json(output_path, orient='records', indent=2, force_ascii=False)
    
    print(f"\nSaved test set to {output_path}")
    print(f"Test set contains {len(df)} questions")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate test set from knowledge base documents using Giskard RAGET"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/document_texts.json",
        help="Path to input JSON file with documents (default: data/document_texts.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/test_data.json",
        help="Path to output JSON file for test set (default: data/test_data.json)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of test questions to generate (default: 50)"
    )
    parser.add_argument(
        "--agent-description",
        type=str,
        default=None,
        help="Description of the RAG agent (helps guide question generation)"
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=list(QUESTION_TYPES.keys()),
        default=list(QUESTION_TYPES.keys()),
        help=f"List of question types to generate. Available: {', '.join(QUESTION_TYPES.keys())}"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key (required by Giskard)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("Giskard's test generation requires an OpenAI API key.")
        print("Please set it before running: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load documents
    documents = load_documents_from_json(args.input)
    
    if not documents:
        print("❌ No documents found in input file. Please generate documents first.")
        return
    
    # Create knowledge base
    knowledge_base = create_knowledge_base(documents)
    
    # Generate test set
    print(f"Selected question types: {args.question_types}")
    testset = generate_test_questions(
        knowledge_base,
        num_questions=args.num_questions,
        agent_description=args.agent_description,
        question_types=args.question_types
    )
    
    # Save test set
    save_testset(testset, args.output)
    
    print(f"\n✅ Successfully generated test set with {args.num_questions} questions")


if __name__ == "__main__":
    main()
