# Document Retrieval System

A modular document retrieval system using Google's Gemini embeddings and SQLite for efficient semantic search.

## Project Structure

```
src/
├── __init__.py
├── config.py                 # Configuration and device setup
├── document_manager.py       # Main orchestrator class
├── document_processing/      # Document loading and chunking
│   ├── __init__.py
│   └── processor.py
├── embeddings/              # Embedding generation using Gemini
│   ├── __init__.py
│   └── generator.py
├── database/                # SQLite operations for embeddings
│   ├── __init__.py
│   └── manager.py
└── search/                  # Semantic search engine
    ├── __init__.py
    └── engine.py
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   Get your API key from: https://aistudio.google.com/app/apikey

3. **Prepare documents**:
   Place your text documents in the project directory or specify the full path.

## Usage

### Using the Interactive Interface

Run the main script:
```bash
python main.py
```

This provides an interactive menu to:
- Load and process documents
- Search through document content
- Manage document databases
- Delete stored embeddings

## Configuration

Edit `src/config.py` to modify:
- **Model settings**: Change embedding model or dimensions
- **Search parameters**: Adjust top-k results
- **Default paths**: Set default document locations
- **Device preferences**: Force specific device usage

## File Descriptions

- **`main.py`**: New modular main interface
- **`main.py`**: Original monolithic version (kept for reference)
- **`src/config.py`**: Centralized configuration
- **`src/document_manager.py`**: High-level orchestrator
- **`src/document_processing/processor.py`**: Document loading and chunking
- **`src/embeddings/generator.py`**: Gemini embedding generation
- **`src/database/manager.py`**: SQLite operations
- **`src/search/engine.py`**: Semantic search functionality