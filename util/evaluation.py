from datetime import datetime
from giskard.rag import evaluate

def evaluate_with_raget(rag_chain, testset, knowledge_base):
    """Evaluate using Giskard's RAGET."""
    print("\n" + "="*60)
    print("Running RAGET Evaluation...")
    print("="*60 + "\n")
    
    # Modern LCEL chains usually return the string directly via StrOutputParser
    def predict_fn(question: str, history=None) -> str:
        return rag_chain.invoke({"input": question})
    
    report = evaluate(
        predict_fn,
        testset=testset,
        knowledge_base=knowledge_base
    )
    
    return report


def calculate_quality_score(report):
    """Calculate aggregate quality score (0-100)."""
    try:
        # Try to get score from dataframe first (most accurate)
        df = None
        if hasattr(report, 'to_pandas'):
            df = report.to_pandas()
        elif hasattr(report, '_dataframe'):
            df = report._dataframe
            
        if df is not None and 'correctness' in df.columns:
            # Calculate percentage of correct answers
            correct_count = df['correctness'].sum()
            total_count = len(df)
            if total_count > 0:
                return round((correct_count / total_count) * 100, 1)

        # Fallback to report attributes
        metrics = {}
        if hasattr(report, 'correctness_rate'):
            metrics['correctness'] = report.correctness_rate
        if hasattr(report, 'faithfulness_rate'):
            metrics['faithfulness'] = report.faithfulness_rate
        
        component_scores = []
        if hasattr(report, 'get_components'):
            for component in report.get_components():
                if hasattr(component, 'pass_rate'):
                    component_scores.append(component.pass_rate * 100)
        
        if component_scores:
            quality_score = sum(component_scores) / len(component_scores)
        elif metrics:
            quality_score = sum(metrics.values()) / len(metrics) * 100
        else:
            quality_score = 0.0  # Default to 0 if no metrics found
        
        return round(quality_score, 1)
    
    except Exception as e:
        print(f"Warning: Could not calculate quality score: {e}")
        return 0.0


def save_results_as_markdown(report, quality_score, output_path, test_count, doc_count):
    """Save results as markdown."""
    # Determine quality level
    if quality_score >= 90:
        level, emoji = "EXCELLENT", "✅"
    elif quality_score >= 75:
        level, emoji = "GOOD", "👍"
    elif quality_score >= 60:
        level, emoji = "FAIR", "⚠️"
    elif quality_score >= 40:
        level, emoji = "POOR", "❌"
    else:
        level, emoji = "CRITICAL", "🚨"
    
    # Extract DataFrame for detailed metrics
    df = None
    try:
        if hasattr(report, 'to_pandas'):
            df = report.to_pandas()
        elif hasattr(report, '_dataframe'):
            df = report._dataframe
    except Exception:
        pass

    md_content = f"""# RAG Evaluation Report

## {emoji} Overall Quality Score: {quality_score}/100 ({level})

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Method:** RAGET (Modern LCEL Pipeline)  
**Test Questions:** {test_count}  
**Knowledge Base Documents:** {doc_count}

## 📊 Detailed Metrics

"""
    
    if df is not None and 'correctness' in df.columns:
        # 1. Overall Correctness
        correct_count = df['correctness'].sum()
        total_count = len(df)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        md_content += f"### 🎯 Overall Performance\n\n"
        md_content += f"- **Correct Answers:** {correct_count} / {total_count}\n"
        md_content += f"- **Accuracy:** {accuracy:.1f}%\n\n"
        
        # 2. Category Breakdown
        md_content += f"### 📂 Performance by Category\n\n"
        
        # Extract question_type from metadata
        if 'metadata' in df.columns:
            # Helper to safely get question_type
            def get_type(meta):
                if isinstance(meta, dict):
                    return meta.get('question_type', 'unknown')
                return 'unknown'
            
            df['question_type'] = df['metadata'].apply(get_type)
            
            # Group by category
            category_stats = df.groupby('question_type')['correctness'].agg(['sum', 'count'])
            category_stats['accuracy'] = (category_stats['sum'] / category_stats['count'] * 100)
            
            md_content += "| Category | Correct | Total | Accuracy |\n"
            md_content += "|----------|---------|-------|----------|\n"
            
            for category, row in category_stats.iterrows():
                cat_name = str(category).title()
                correct = int(row['sum'])
                total = int(row['count'])
                acc = row['accuracy']
                
                # Add status emoji
                if acc >= 90: status = "✅"
                elif acc >= 75: status = "👍"
                elif acc >= 60: status = "⚠️"
                else: status = "❌"
                
                md_content += f"| {status} {cat_name} | {correct} | {total} | {acc:.1f}% |\n"
            
            md_content += "\n"

    # Detailed Results Section
    md_content += "## 📋 Detailed Results\n\n"
    
    try:
        # Try to convert to pandas DataFrame to get detailed results
        df = None
        if hasattr(report, 'to_pandas'):
            df = report.to_pandas()
        elif hasattr(report, '_dataframe'):
            df = report._dataframe
            
        if df is not None:
            # Iterate through results and format them
            # Use enumerate to avoid index type issues (index might be UUID string)
            for i, (index, row) in enumerate(df.iterrows()):
                question = row.get('question', 'N/A')
                ref_answer = row.get('reference_answer', 'N/A')
                agent_answer = row.get('agent_answer', 'N/A')
                
                # Determine correctness status
                correctness = row.get('correctness', None)
                if correctness is True:
                    status = "✅ Correct"
                elif correctness is False:
                    status = "❌ Incorrect"
                else:
                    status = "❓ Unknown"
                
                # Format the entry
                md_content += f"### Q{i + 1}: {question}\n\n"
                md_content += f"**Status:** {status}\n\n"
                
                md_content += "**Reference Answer:**\n"
                md_content += f"> {ref_answer}\n\n"
                
                md_content += "**Agent Answer:**\n"
                md_content += f"> {agent_answer}\n\n"
                
                # Add metadata if useful (e.g. question type)
                metadata = row.get('metadata', {})
                if isinstance(metadata, dict) and 'question_type' in metadata:
                    md_content += f"*Type: {metadata['question_type']}*\n\n"
                
                md_content += "---\n\n"
        else:
            md_content += "*Detailed results could not be extracted (to_pandas not available).*\n"
            
    except Exception as e:
        md_content += f"*Error extracting detailed results: {str(e)}*\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return output_path
