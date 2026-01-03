# Simple RAG Playground

A basic Retrieval-Augmented Generation (RAG) demo using LangChain with Giskard RAGET (RAG Evaluation Toolkit) for quality assessment.

## Overview

This project demonstrates:
- **RAG Pipeline**: Uses LangChain to build a question-answering system with retrieval
- **Pre-loaded Knowledge**: Uses a small set of Wikipedia articles related to cybersecurity
- **RAGET Evaluation**: Uses Giskard's RAG Evaluation Toolkit to test for hallucinations, correctness, and faithfulness
- **Vector Store**: Uses FAISS for efficient similarity search
- **LLM**: Uses OpenAI's GPT-4o for generation
- **Automated Scoring**: Provides a single quality score (0-100) for easy tracking

## Prerequisites

- Python 3.12 or higher
- OpenAI API key

## Setup

1. **Create Conda environment (recommended)**:

If you have Conda installed, it is recommended to create a separate Python environment:
```bash
conda create -n simple_rag_playground python=3.12
```

Then activate the environment:
```bash
conda activate simple_rag_playground
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set your OpenAI API key**:

Create a `.env` file in the root directory (you can copy `.env.example`):
```bash
cp .env.example .env
```

Then edit `.env` and add your API key:
```
OPENAI_API_KEY=sk-...
```

## Running the Application

### 1. Generate Knowledge Base (Optional)

If you want to create a new knowledge base from Wikipedia topics:
```bash
python scripts/generate_knowledge_base.py "Climate Change, Renewable Energy" --output data/document_texts.json
```

### 2. Generate Test Set (Optional)

If you want to generate new test questions from your knowledge base:
```bash
python scripts/generate_test_set.py --input data/document_texts.json --output data/test_data.json --num-questions 20
```

### 3. Run RAG Pipeline & Evaluation

Run the main pipeline to evaluate the RAG system:
```bash
python run_pipeline.py
```

**Options:**
- `--documents`: Path to document JSON (default: `data/document_texts.json`)
- `--test-data`: Path to test data JSON (default: `data/test_data.json`)
- `--prompt`: Prompt template to use (default: `default`)
- `--output-dir`: Directory for results (default: `results`)

Example:
```bash
python run_pipeline.py --prompt default --chunk-size 1000
```

## What It Does

1. **Loads Data**: Reads pre-processed documents and test questions from JSON files.
2. **Builds RAG Pipeline**: 
   - Chunks the document text.
   - Creates embeddings and stores them in FAISS.
   - Sets up a modern LCEL (LangChain Expression Language) chain.
3. **Evaluates Quality**: 
   - Runs the test questions through the pipeline.
   - Uses Giskard's RAGET to evaluate answers against the knowledge base.
   - Checks for correctness, faithfulness, and context relevance.
4. **Generates Report**: Creates a detailed markdown report with:
   - Overall Quality Score (0-100).
   - Performance breakdown by question category (Simple, Complex, Distracting, etc.).
   - Detailed logs of every question, answer, and evaluation result.

## Output

The script will:
- Print progress and the final quality score to the console.
- Generate a report in the `results/` directory, e.g., `results/evaluation_report_20240101_120000.md`.
- The report includes:
  - **Executive Summary**: Score and key metrics.
  - **Category Analysis**: How the model performed on different types of questions.
  - **Detailed Logs**: Full trace of inputs and outputs.

## Key Components

### RAG Pipeline (`run_pipeline.py`)
- Orchestrates the loading, setup, and evaluation process.
- Uses `util/` modules for modular functionality.

### Scripts (`scripts/`)
- `generate_knowledge_base.py`: Fetches content (e.g., from Wikipedia) and saves it to JSON.
- `generate_test_set.py`: Uses Giskard to generate synthetic test questions from the knowledge base.

### Evaluation (`util/evaluation.py`)
- Calculates accuracy-based scores.
- Generates the "pretty" Markdown report with tables and emojis.
- Returns comprehensive evaluation report

### Quality Score (`calculate_quality_score`)
- Aggregates RAGET metrics into a single score (0-100)
- 90-100: Excellent quality
- 75-89: Good quality
- 60-74: Fair quality (needs improvement)
- Below 60: Poor quality (significant issues)

## Customization

- **Change topics**: Use `scripts/generate_knowledge_base.py` with different topics.
- **Change chunk size**: Use the `--chunk-size` argument when running `run_pipeline.py`.
- **Add more tests**: Use `scripts/generate_test_set.py` with a higher `--num-questions`.
- **Modify Prompts**: Edit `util/prompts.py` to add or modify prompt templates.

## Notes

- Uses RAGET (RAG Evaluation Toolkit) specifically designed for RAG systems
- Automatically generates test questions from your knowledge base
- Provides actionable insights for improving RAG quality
- Quality score makes it easy to track improvements over time
- For production use, consider increasing `num_questions` for more thorough evaluation
