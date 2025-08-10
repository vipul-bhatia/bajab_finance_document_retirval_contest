# Bajaj Finance Document Retrieval System

A sleek FastAPI service for answering questions from documents via semantic search. It uses OpenAI embeddings with FAISS for vector search, dynamic chunking for large docs, neighbor-chunk expansion for better context, and a similarity cache to reuse past answers.

## Project Structure

```
.
├── api.py                          # FastAPI application
├── main.py                         # Local dev entrypoint (uvicorn)
├── Dockerfile                      # Container image
├── docker-compose.yml              # Compose setup
├── requirements.txt                # Python deps
├── env.example                     # Example environment variables
├── README.md
├── document_cache/                 # Cached text chunks per document
├── document_embeddings/            # FAISS indexes (auto-created)
├── questions_answers.jsonl         # Persistent Q&A store
├── request_logs.jsonl              # Request logs
├── sample_documents/               # Example inputs
└── src/
    ├── __init__.py
    ├── config.py                   # Global config (device, models, paths)
    ├── document_manager.py         # Orchestrates processing/search
    ├── document_processing/
    │   ├── __init__.py
    │   └── processor.py            # Load, OCR, dynamic chunking
    ├── embeddings/
    │   ├── __init__.py
    │   └── generator.py            # OpenAI embedding calls
    ├── database/
    │   ├── __init__.py
    │   └── manager.py              # FAISS read/write, migration
    ├── query_processing/
    │   ├── __init__.py
    │   ├── analyzer.py             # Query decomposition
    │   └── enhancer.py             # Answer synthesis w/ prompt rules
    └── search/
        ├── __init__.py
        ├── engine.py               # Retrieval + neighbor expansion
        └── similarity_search.py    # Q&A similarity cache
```

## Environment

Copy `.env` from `env.example` and fill values:

```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_api_key
EXPECTED_TOKEN=your_api_bearer_token

# Optional (defaults shown)
EMBEDDINGS_DIR=document_embeddings
```

Notes:
- Mac with Apple Silicon uses MPS automatically when available.
- `EXPECTED_TOKEN` is required for the API bearer auth.

## Quick Start

### Local Development

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Set up environment:
    - Create `.env` from `env.example` and populate the keys above.
    - Get Google API key from: https://aistudio.google.com/app/apikey

3.  Run the application:
    ```bash
    python main.py
    ```
    This will start the FastAPI server on `http://localhost:8000`.

### Docker

1.  **Set up environment**:
    Create a `.env` file as described above.

2.  Build and run the container:
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

## How it works (high level)

- Query analysis: decomposes complex questions to multiple focused queries.
- Retrieval: FAISS similarity search over dynamic chunks; each hit expands with 1 neighbor up/down.
- Answering: enhancer synthesizes a concise, source-grounded answer; responds in the source language if not English.
- Similarity cache: skips work for previously answered questions per document.

## Data & Storage

- `document_cache/`: cached raw text chunks per document (speeds re-processing)
- `document_embeddings/`: FAISS indexes per document (auto-migrated here)
- `questions_answers.jsonl`: persistent Q&A history
- `request_logs.jsonl`: request/timing logs

## Configuration

Edit `src/config.py` to modify:
- Model settings (embedding model, dimensions)
- Search parameters (top-k, device)
- Paths (default document, embeddings dir)

## Tips

- Increase concurrency knobs only if your API limits and machine can support it.
- Large PDFs benefit from the dynamic chunking and neighbor expansion already enabled.
