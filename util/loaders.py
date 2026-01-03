import json
import pandas as pd
import uuid
from langchain_core.documents import Document
from giskard.rag import QATestset

def load_documents_from_json(json_path):
    """Load documents from JSON file."""
    print(f"Loading documents from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = Document(
            page_content=item.get('page_content', ''),
            metadata=item.get('metadata', {})
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} documents")
    return documents


def load_test_data_from_json(json_path):
    """Load test data from JSON file."""
    print(f"Loading test data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test questions")
    return data


def create_giskard_testset_from_data(test_data):
    """
    Convert test data JSON to Giskard QATestset.
    
    Args:
        test_data: List of test cases with question, reference_answer, etc.
    
    Returns:
        QATestset object for Giskard evaluation
    """
    print("\n" + "="*60)
    print("Creating Giskard testset from test data...")
    print("="*60 + "\n")
    
    # Prepare data for DataFrame
    questions = []
    reference_answers = []
    reference_contexts = []
    ids = []
    conversation_histories = []
    metadatas = []
    
    for item in test_data:
        questions.append(item.get('question', ''))
        reference_answers.append(item.get('reference_answer', ''))
        reference_contexts.append(item.get('reference_context', ''))
        ids.append(str(uuid.uuid4()))
        conversation_histories.append([])
        # Ensure metadata has 'question_type' as Giskard expects it for reporting
        metadata = item.get('metadata', {})
        if 'question_type' not in metadata:
            metadata['question_type'] = 'unknown'
        metadatas.append(metadata)
    
    # Create DataFrame with all required columns for QuestionSample
    df = pd.DataFrame({
        'id': ids,
        'question': questions,
        'reference_answer': reference_answers,
        'reference_context': reference_contexts,
        'conversation_history': conversation_histories,
        'metadata': metadatas
    })
    
    # Create QATestset using from_pandas
    testset = QATestset.from_pandas(df)
    
    print(f"Created testset with {len(testset)} questions\n")
    return testset
