# Document Retrieval System

A modular document retrieval system using Google's Gemini embeddings and SQLite for efficient semantic search. This project now includes a FastAPI interface for easy integration and deployment with intelligent similarity search to avoid redundant processing.

## Project Structure

```
.
├── .gitignore
├── api.py                    # FastAPI application
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Dockerfile for building the application image
├── env.example               # Example environment file
├── main.py                   # Main script to run the application
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── questions_answers.jsonl   # Permanent storage for questions and answers
├── request_logs.jsonl        # Request logs (existing functionality)
├── test_qa_storage.py        # Test script for QA storage functionality
├── test_similarity_search.py # Test script for similarity search functionality
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration and device setup
│   ├── document_manager.py   # Main orchestrator class
│   ├── document_processing/  # Document loading and chunking
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── embeddings/          # Embedding generation using Gemini
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── database/            # SQLite operations for embeddings
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── search/              # Semantic search engine
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── fuzzy.py         # Fuzzy string matching utilities
│   │   └── similarity_search.py # Similarity search for Q&A
└── sample_documents/
```

## Setup

### Local Development

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up environment**:
    Create a `.env` file from the `env.example` and add your API keys:
    ```
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```
    Get your Google API key from: https://aistudio.google.com/app/apikey

3.  **Run the application**:
    ```bash
    python main.py
    ```
    This will start the FastAPI server on `http://localhost:8000`.

### Docker

1.  **Set up environment**:
    Create a `.env` file as described above.

2.  **Build and run the container**:
    ```bash
    docker-compose build
    docker-compose up
    ```
    The application will be available at `http://localhost:8000`.

## API Usage

You can interact with the API using any HTTP client, such as `curl`.

### Process a document and ask questions

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your_token_here' \
  -d '{
  "documents": "https://arxiv.org/pdf/1706.03762.pdf",
  "questions": [
    "What is the title of the paper?",
    "What are the main components of the Transformer architecture?"
  ]
}'
```

### Retrieve stored questions and answers

```bash
curl http://localhost:8000/qa
```

This endpoint returns all questions and answers that have been processed and stored permanently.

### Health Check

```bash
curl http://localhost:8000/health
```

## Intelligent Similarity Search

The system now includes intelligent similarity search functionality that optimizes performance by avoiding redundant processing:

### How It Works

1. **Pre-processing Check**: Before processing any new questions, the system checks the stored `questions_answers.jsonl` file for similar questions using fuzzy string matching.

2. **Similarity Threshold**: Questions with 90% or higher similarity are considered matches and their cached answers are returned immediately.

3. **Hybrid Processing**: If some questions have matches and others don't, only the unmatched questions are processed through the full pipeline.

4. **Performance Benefits**: 
   - **Faster Response Times**: Matched questions return instantly
   - **Reduced Processing**: Avoids redundant document processing and embedding generation
   - **Cost Savings**: Reduces API calls to external services

### Similarity Search Algorithm

The system uses the `thefuzz` library with the `fuzz.ratio()` function to calculate similarity scores between questions. This algorithm:

- Compares questions character by character
- Handles typos, word order changes, and minor variations
- Returns scores from 0-100 (100 being exact match)
- Is case-insensitive for better matching

### Example Scenarios

**Scenario 1: All Questions Match**
```
Input: ["What is Newton's definition of quantity of motion?"]
Result: Returns cached answer immediately (no document processing)
```

**Scenario 2: Partial Matches**
```
Input: ["What is Newton's definition?", "What is the main topic?"]
Result: Returns cached answer for first question, processes second question
```

**Scenario 3: No Matches**
```
Input: ["What is the main topic of this document?"]
Result: Processes all questions normally
```

## Data Storage

### Questions and Answers Storage

The system includes permanent storage for questions and answers in `questions_answers.jsonl`. This file stores:

- **Timestamp**: When the Q&A was processed
- **Document Name**: Identifier for the processed document
- **Document URL**: Source URL of the document
- **Questions**: List of questions asked
- **Answers**: List of corresponding answers

Each entry is stored as a JSON object on a separate line (JSONL format).

### Request Logs

The existing `request_logs.jsonl` file continues to store comprehensive request information including:
- Timing data and processing details
- Similarity search usage statistics
- Cached vs. new answer counts

### Testing the Storage

You can test the questions and answers storage functionality using the provided test scripts:

```bash
# Test basic Q&A storage
python test_qa_storage.py

# Test similarity search functionality
python test_similarity_search.py
```

## Configuration

Edit `src/config.py` to modify:
- **Model settings**: Change embedding model or dimensions
- **Search parameters**: Adjust top-k results
- **Default paths**: Set default document locations
- **Device preferences**: Force specific device usage

## File Descriptions

- **`main.py`**: Entry point to run the FastAPI application.
- **`api.py`**: Defines the FastAPI application and its endpoints.
- **`Dockerfile`**: Instructions for building the Docker image.
- **`docker-compose.yml`**: Defines the services for Docker Compose.
- **`src/config.py`**: Centralized configuration.
- **`src/document_manager.py`**: High-level orchestrator for document processing.
- **`src/document_processing/processor.py`**: Handles document loading and chunking.
- **`src/embeddings/generator.py`**: Generates embeddings using Gemini.
- **`src/database/manager.py`**: Manages the SQLite database for embeddings.
- **`src/search/engine.py`**: Implements the semantic search functionality.
