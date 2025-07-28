# Document Retrieval System

A modular document retrieval system using Google's Gemini embeddings and SQLite for efficient semantic search. This project now includes a FastAPI interface for easy integration and deployment.

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
│   └── search/              # Semantic search engine
│       ├── __init__.py
│       └── engine.py
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
  'http://localhost:8000/process-document' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "documents": "https://arxiv.org/pdf/1706.03762.pdf",
  "questions": [
    "What is the title of the paper?",
    "What are the main components of the Transformer architecture?"
  ]
}'
```

### Health Check

```bash
curl http://localhost:8000/health
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
