"""
Generate Knowledge Base from Wikipedia
Loads documents about a specified topic and saves them to JSON
"""
import argparse
import json
import warnings
from pathlib import Path
from langchain_community.document_loaders import WikipediaLoader

# Suppress BeautifulSoup parser warning from wikipedia library
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")


def load_wikipedia_documents(topic, max_docs=10, max_chars=5000):
    """
    Load documents from Wikipedia about a specific topic
    
    Args:
        topic: Topic to search for on Wikipedia
        max_docs: Maximum number of documents to load
        max_chars: Maximum characters per document
        
    Returns:
        List of loaded documents
    """
    print(f"Loading documents about '{topic}' from Wikipedia...")
    
    loader = WikipediaLoader(
        query=topic,
        load_max_docs=max_docs,
        doc_content_chars_max=max_chars
    )
    
    documents = loader.load()
    
    return documents


def save_documents_to_json(documents, output_path):
    """
    Save loaded documents to JSON file
    
    Args:
        documents: List of LangChain documents
        output_path: Path to save JSON file
    """
    # Convert documents to JSON-serializable format
    documents_data = []
    for doc in documents:
        doc_dict = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        documents_data.append(doc_dict)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(documents_data)} documents to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Load Wikipedia documents about a topic and save to JSON"
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="ATT&CK, cyber kill chain, STIX (structured information expression), zero trust security model, SQL injection",
        help="Topics to search for on Wikipedia, delimited by commas"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help="Maximum number of documents to load (default: 10)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=5000,
        help="Maximum characters per document (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/document_texts.json",
        help="Output JSON file path (default: data/document_texts.json)"
    )
    
    args = parser.parse_args()
    
    # Load documents
    queries = [q.strip() for q in args.topics.split(",")]
    documents = []
    for query in queries:
        next_documents = load_wikipedia_documents(
            topic=query,
            max_docs=args.max_docs,
            max_chars=args.max_chars
        )
        documents.extend(next_documents)

    # Simple check for low quality documents
    documents = [doc for doc in documents if len(doc.page_content.strip()) > 1000]
    print(f"Filtered to {len(documents)} documents after simple quality check")
    
    # Save to JSON
    save_documents_to_json(documents, args.output)
    
    print(f"\n✅ Successfully generated knowledge base for topics: '{args.topics}'")


if __name__ == "__main__":
    main()
