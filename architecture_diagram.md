# Document Retrieval System Architecture

## Complete System Architecture Diagram

```mermaid
graph TB
    %% External Components
    subgraph "External Services"
        GOOGLE[Google Gemini API]
        OPENAI[OpenAI Embeddings API]
        URL[Document URL]
    end

    %% API Layer
    subgraph "API Layer"
        FASTAPI[FastAPI Server]
        AUTH[Bearer Token Auth]
        HEALTH[Health Check]
    end

    %% Main Application Flow
    subgraph "Application Core"
        MAIN[main.py]
        API[api.py]
        DM[DocumentManager]
    end

    %% Document Processing Pipeline
    subgraph "Document Processing"
        DP[DocumentProcessor]
        DOWNLOAD[Download from URL]
        PARSE[Parse Document]
        CHUNK[Text Chunking]
        LANGCHAIN[LangChain Splitter]
    end

    %% Embedding Generation
    subgraph "Embedding System"
        EG[EmbeddingGenerator]
        BATCH[Batch Processing]
        PARALLEL[Parallel Workers]
        OPENAI_EMB[OpenAI text-embedding-3-large]
    end

    %% Database Layer
    subgraph "Vector Database"
        DB[DatabaseManager]
        FAISS[FAISS Index]
        STORE[Store Embeddings]
        LOAD[Load Index]
    end

    %% Search Engine
    subgraph "Search Engine"
        SE[SearchEngine]
        SEMANTIC[Semantic Search]
        FAISS_SEARCH[FAISS Search]
    end

    %% Query Processing
    subgraph "Query Processing"
        QA[QueryAnalyzer]
        QE[QueryEnhancer]
        DECOMPOSE[Query Decomposition]
        ENHANCE[Result Enhancement]
        GEMINI[Gemini 2.5 Flash Lite]
    end

    %% Configuration
    subgraph "Configuration"
        CONFIG[config.py]
        ENV[Environment Variables]
        DEVICE[Device Selection]
    end

    %% Data Flow Connections
    URL --> DOWNLOAD
    DOWNLOAD --> PARSE
    PARSE --> CHUNK
    CHUNK --> LANGCHAIN
    LANGCHAIN --> EG
    
    EG --> BATCH
    BATCH --> PARALLEL
    PARALLEL --> OPENAI_EMB
    OPENAI_EMB --> DB
    
    DB --> STORE
    STORE --> FAISS
    FAISS --> LOAD
    LOAD --> SE
    
    SE --> SEMANTIC
    SEMANTIC --> FAISS_SEARCH
    FAISS_SEARCH --> QA
    
    QA --> DECOMPOSE
    DECOMPOSE --> GEMINI
    GEMINI --> QE
    QE --> ENHANCE
    
    %% API Flow
    FASTAPI --> API
    API --> AUTH
    API --> DM
    DM --> DP
    DM --> SE
    DM --> DB
    
    %% Configuration Flow
    CONFIG --> ENV
    CONFIG --> DEVICE
    CONFIG --> EG
    CONFIG --> SE
    CONFIG --> DP
    
    %% External API Connections
    GOOGLE --> GEMINI
    OPENAI --> OPENAI_EMB
    
    %% Health and Monitoring
    FASTAPI --> HEALTH
    
    
    class FASTAPI,API,AUTH,HEALTH apiLayer
    class DP,DOWNLOAD,PARSE,CHUNK,LANGCHAIN processing
    class EG,BATCH,PARALLEL,OPENAI_EMB embedding
    class DB,FAISS,STORE,LOAD database
    class SE,SEMANTIC,FAISS_SEARCH search
    class QA,QE,DECOMPOSE,ENHANCE,GEMINI query
    class CONFIG,ENV,DEVICE config
    class GOOGLE,OPENAI,URL external
```

## Detailed Component Architecture

```mermaid
graph LR
    %% User Request Flow
    subgraph "1. User Request"
        USER[User]
        REQUEST[POST /api/v1/hackrx/run]
        DOC_URL[Document URL]
        QUESTIONS[Questions List]
    end

    %% Authentication
    subgraph "2. Authentication"
        BEARER[Bearer Token]
        VERIFY[Token Verification]
    end

    %% Document Processing Pipeline
    subgraph "3. Document Processing"
        DOWNLOAD[Download Document]
        PROCESS[Process Document]
        CHUNK[Split into Chunks]
        CHUNK_SIZE[1400 chars/chunk]
        OVERLAP[100 chars overlap]
    end

    %% Embedding Generation
    subgraph "4. Embedding Generation"
        EMBED[Generate Embeddings]
        BATCH_SIZE[Batch Size: 50]
        WORKERS[Max Workers: 10]
        MODEL[text-embedding-3-large]
        DIM[3072 dimensions]
    end

    %% Vector Storage
    subgraph "5. Vector Storage"
        FAISS_CREATE[Create FAISS Index]
        FAISS_STORE[Store on Disk]
        FAISS_LOAD[Load for Search]
        INDEX_PATH[document_embeddings_*.faiss]
    end

    %% Query Processing
    subgraph "6. Query Processing"
        ANALYZE[Query Analysis]
        DECOMPOSE[Complex Query Decomposition]
        SIMPLE[Simple Query Handling]
        PARALLEL_SEARCH[Parallel Search]
    end

    %% Search Execution
    subgraph "7. Search Execution"
        SEMANTIC_SEARCH[Semantic Search]
        TOP_K[Top 5 Results]
        SCORE[Similarity Scoring]
        RANK[Result Ranking]
    end

    %% Answer Generation
    subgraph "8. Answer Generation"
        ENHANCE[Result Enhancement]
        SYNTHESIZE[Information Synthesis]
        DIRECT_ANSWER[Direct Answer Generation]
        CONCISE[1-2 Sentence Response]
    end

    %% Response
    subgraph "9. Response"
        ANSWERS[Answers List]
        TIMING[Timing Data]
        METADATA[Processing Metadata]
    end

    %% Flow Connections
    USER --> REQUEST
    REQUEST --> DOC_URL
    REQUEST --> QUESTIONS
    REQUEST --> BEARER
    
    BEARER --> VERIFY
    VERIFY --> DOWNLOAD
    
    DOWNLOAD --> PROCESS
    PROCESS --> CHUNK
    CHUNK --> CHUNK_SIZE
    CHUNK --> OVERLAP
    
    CHUNK --> EMBED
    EMBED --> BATCH_SIZE
    EMBED --> WORKERS
    EMBED --> MODEL
    EMBED --> DIM
    
    EMBED --> FAISS_CREATE
    FAISS_CREATE --> FAISS_STORE
    FAISS_STORE --> INDEX_PATH
    FAISS_STORE --> FAISS_LOAD
    
    QUESTIONS --> ANALYZE
    ANALYZE --> DECOMPOSE
    ANALYZE --> SIMPLE
    DECOMPOSE --> PARALLEL_SEARCH
    SIMPLE --> PARALLEL_SEARCH
    
    FAISS_LOAD --> SEMANTIC_SEARCH
    PARALLEL_SEARCH --> SEMANTIC_SEARCH
    SEMANTIC_SEARCH --> TOP_K
    SEMANTIC_SEARCH --> SCORE
    SCORE --> RANK
    
    RANK --> ENHANCE
    ENHANCE --> SYNTHESIZE
    SYNTHESIZE --> DIRECT_ANSWER
    DIRECT_ANSWER --> CONCISE
    
    CONCISE --> ANSWERS
    ANSWERS --> TIMING
    ANSWERS --> METADATA
    
    
    class USER,REQUEST,DOC_URL,QUESTIONS request
    class BEARER,VERIFY auth
    class DOWNLOAD,PROCESS,CHUNK,CHUNK_SIZE,OVERLAP processing
    class EMBED,BATCH_SIZE,WORKERS,MODEL,DIM embedding
    class FAISS_CREATE,FAISS_STORE,FAISS_LOAD,INDEX_PATH storage
    class ANALYZE,DECOMPOSE,SIMPLE,PARALLEL_SEARCH query
    class SEMANTIC_SEARCH,TOP_K,SCORE,RANK search
    class ENHANCE,SYNTHESIZE,DIRECT_ANSWER,CONCISE answer
    class ANSWERS,TIMING,METADATA response
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant DocumentManager
    participant DocumentProcessor
    participant EmbeddingGenerator
    participant DatabaseManager
    participant SearchEngine
    participant QueryAnalyzer
    participant QueryEnhancer
    participant OpenAI
    participant Gemini
    participant FAISS

    Client->>FastAPI: POST /api/v1/hackrx/run
    Note over Client,FastAPI: {documents: URL, questions: [list]}
    
    FastAPI->>FastAPI: Verify Bearer Token
    FastAPI->>DocumentManager: initialize_document_from_url()
    
    DocumentManager->>DocumentProcessor: download_and_process_document_with_size()
    DocumentProcessor->>DocumentProcessor: Download from URL
    DocumentProcessor->>DocumentProcessor: Parse document (PDF/TXT/DOCX)
    DocumentProcessor->>DocumentProcessor: Split into chunks (1400 chars)
    DocumentProcessor-->>DocumentManager: Return chunks
    
    DocumentManager->>DatabaseManager: embeddings_exist()
    DatabaseManager->>FAISS: Check if index exists
    
    alt Index exists
        DatabaseManager->>FAISS: load_faiss_index()
        FAISS-->>DatabaseManager: Return existing index
    else Index doesn't exist
        DatabaseManager->>EmbeddingGenerator: get_embeddings_in_parallel()
        EmbeddingGenerator->>OpenAI: Batch embedding requests
        OpenAI-->>EmbeddingGenerator: Return embeddings
        EmbeddingGenerator-->>DatabaseManager: Return embeddings tensor
        DatabaseManager->>FAISS: Create and store index
    end
    
    DatabaseManager-->>DocumentManager: Return FAISS index
    DocumentManager->>SearchEngine: load_embeddings()
    SearchEngine-->>DocumentManager: Index loaded
    
    DocumentManager->>SearchEngine: process_multiple_queries()
    
    SearchEngine->>QueryAnalyzer: process_multiple_queries()
    QueryAnalyzer->>Gemini: Batch query analysis
    Gemini-->>QueryAnalyzer: Return decomposed queries
    QueryAnalyzer-->>SearchEngine: Return query lists
    
    loop For each query
        SearchEngine->>SearchEngine: find_relevant_chunks()
        SearchEngine->>FAISS: Semantic search
        FAISS-->>SearchEngine: Return top-k results
        SearchEngine->>QueryEnhancer: get_most_relevant_chunk()
        QueryEnhancer->>Gemini: Generate answer
        Gemini-->>QueryEnhancer: Return enhanced answer
        QueryEnhancer-->>SearchEngine: Return answer
    end
    
    SearchEngine-->>DocumentManager: Return all answers
    DocumentManager-->>FastAPI: Return answers list
    FastAPI-->>Client: Return {answers: [list]}

    Note over Client,FAISS: Complete pipeline with timing data
```

## System Components Overview

```mermaid
mindmap
  root((Document Retrieval System))
    API Layer
      FastAPI Server
      Bearer Token Auth
      Health Check Endpoint
      Request/Response Models
    Document Processing
      URL Download
      Multi-format Support
        PDF Processing
        TXT Processing
        DOCX Processing
        EML/MSG Support
      Text Chunking
        LangChain Splitter
        Configurable Chunk Size
        Overlap Management
    Embedding System
      OpenAI Embeddings
        text-embedding-3-large
        3072 Dimensions
      Parallel Processing
        Batch Processing
        Thread Pool Executor
        Error Handling
    Vector Database
      FAISS Index
        IndexFlatL2
        Disk Storage
        Index Management
      Database Manager
        Store Operations
        Load Operations
        Existence Checks
    Search Engine
      Semantic Search
        FAISS Search
        Similarity Scoring
        Result Ranking
      Intelligent Search
        Query Decomposition
        Parallel Search
        Result Deduplication
    Query Processing
      Query Analyzer
        Gemini 2.5 Flash Lite
        Complex Query Decomposition
        Simple Query Handling
        Batch Processing
      Query Enhancer
        Result Enhancement
        Direct Answer Generation
        Information Synthesis
    Configuration
      Environment Variables
      Device Selection
        CPU/GPU/MPS
      Model Configuration
      Processing Parameters
```

## Performance Optimization Features

```mermaid
graph TD
    subgraph "Performance Optimizations"
        subgraph "Parallel Processing"
            BATCH[Batch Processing]
            THREADS[Thread Pool Executor]
            CONCURRENT[Concurrent Futures]
        end
        
        subgraph "Caching & Storage"
            FAISS_CACHE[FAISS Index Caching]
            DISK_STORE[Disk Storage]
            INDEX_REUSE[Index Reuse]
        end
        
        subgraph "Memory Management"
            IN_MEMORY[In-Memory Processing]
            STREAM[Stream Processing]
            CLEANUP[Memory Cleanup]
        end
        
        subgraph "API Optimization"
            RETRY[Retry Logic]
            BACKOFF[Exponential Backoff]
            TIMEOUT[Request Timeouts]
        end
        
        subgraph "Search Optimization"
            TOP_K[Top-K Results]
            DEDUP[Result Deduplication]
            RANKING[Smart Ranking]
        end
    end
    
    BATCH --> THREADS
    THREADS --> CONCURRENT
    FAISS_CACHE --> DISK_STORE
    DISK_STORE --> INDEX_REUSE
    IN_MEMORY --> STREAM
    STREAM --> CLEANUP
    RETRY --> BACKOFF
    BACKOFF --> TIMEOUT
    TOP_K --> DEDUP
    DEDUP --> RANKING
    
    class BATCH,THREADS,CONCURRENT,FAISS_CACHE,DISK_STORE,INDEX_REUSE,IN_MEMORY,STREAM,CLEANUP,RETRY,BACKOFF,TIMEOUT,TOP_K,DEDUP,RANKING optimization
```

This comprehensive architecture diagram shows:

1. **Complete System Flow**: From user request to response
2. **Component Interactions**: How each module communicates
3. **Data Processing Pipeline**: Document processing, embedding generation, and search
4. **External Dependencies**: Google Gemini and OpenAI APIs
5. **Performance Optimizations**: Parallel processing, caching, and memory management
6. **Configuration Management**: Environment variables and device selection

The system is designed for high-performance document retrieval with intelligent query processing and semantic search capabilities. 