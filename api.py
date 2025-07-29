#!/usr/bin/env python3
"""
FastAPI Document Retrieval API
Processes documents from URLs and answers questions using intelligent search
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import time
import traceback

from src.document_manager import DocumentManager

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Document Retrieval API",
    description="Process documents from URLs and answer questions using intelligent semantic search",
    version="1.0.0"
)

# Security scheme for Bearer token
security = HTTPBearer()

# Expected Bearer token
EXPECTED_TOKEN = os.getenv("EXPECTED_TOKEN")

# Pydantic models for request/response
class DocumentQuery(BaseModel):
    documents: HttpUrl  # Single document URL
    questions: List[str]

class AnswersResponse(BaseModel):
    answers: List[str]

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.post("/api/v1/hackrx/run", response_model=AnswersResponse)
async def process_document_and_questions(
    request: DocumentQuery,
    token: str = Depends(verify_token)
):
    """
    Process a document from URL and answer multiple questions
    
    Args:
        request: DocumentQuery containing document URL and list of questions
        
    Returns:
        AnswersResponse with list of answers
    """
    total_start_time = time.time()

    print(f"Received document URL: {request.documents}")
    
    try:
        # Initialize document manager
        print(f"🚀 Starting document processing pipeline...")
        document_manager = DocumentManager()
        
        # Step 1: Download and process document
        download_start = time.time()
        print(f"📥 Step 1: Downloading document from URL...")
        print(f"   URL: {request.documents}")
        print(f"   Chunk size: 2500 characters")
        
        success, document_name = document_manager.initialize_document_from_url(str(request.documents), 1400)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Failed to download or process the document. Please check the URL and try again."
            )
        
        download_time = time.time() - download_start
        
        # Get detailed timing from document manager
        if hasattr(document_manager, 'timing_data'):
            timing_data = document_manager.timing_data
        
        print(f"✅ Document download & processing completed in {download_time:.2f}s")
        
        # Step 2: Get document info and embedding stats
        doc_info = document_manager.get_document_info()
        print(f"📄 Document ready: {doc_info['chunk_count']} chunks loaded")
        
        # Step 3: Process all questions in parallel
        query_start = time.time()
        print(f"🎯 Step 2: Processing {len(request.questions)} questions...")
        answers = document_manager.process_multiple_queries(request.questions)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start_time
        
        print(f"🏁 All questions processed in {query_time:.2f}s")
        print(f"🏁 Total pipeline time: {total_time:.2f}s")
        
        # Print comprehensive timing summary
        print(f"\n📊 COMPREHENSIVE TIMING SUMMARY:")
        print(f"   • Document Download & Processing: {timing_data.get('download_and_processing', 0):.2f}s")
        if 'embedding_generation' in timing_data:
            print(f"   • Embedding Generation: {timing_data['embedding_generation']:.2f}s")
        print(f"   • Database Operations: {timing_data.get('database_load', 0):.2f}s")
        print(f"   • Search Engine Load: {timing_data.get('search_engine_load', 0):.2f}s")
        print(f"   • Query Processing: {query_time:.2f}s")
        print(f"   • Total Pipeline: {total_time:.2f}s")
        
        return AnswersResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Document Retrieval API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Retrieval API",
        "version": "1.0.0",
        "description": "Process documents from URLs and answer questions using intelligent semantic search",
        "endpoints": {
            "POST /api/v1/hackrx/run": "Main endpoint for document processing and question answering",
            "GET /health": "Health check endpoint",
            "GET /docs": "Interactive API documentation"
        }
    }

# Note: Use main.py to run the server with proper reload functionality 