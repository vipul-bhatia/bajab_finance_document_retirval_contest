#!/usr/bin/env python3
"""
FastAPI Document Retrieval API
Processes documents from URLs and answers questions using intelligent search
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import time
import traceback

from src.document_manager import DocumentManager

app = FastAPI(
    title="Document Retrieval API",
    description="Process documents from URLs and answer questions using intelligent semantic search",
    version="1.0.0"
)

# Pydantic models for request/response
class DocumentQuery(BaseModel):
    documents: HttpUrl  # Single document URL
    questions: List[str]
    chunk_size: int = 2500  # Optional: adjust chunk size for performance vs accuracy

class AnswersResponse(BaseModel):
    answers: List[str]
    processing_time: float
    document_chunks: int
    total_questions: int

@app.post("/process-document", response_model=AnswersResponse)
async def process_document_and_questions(request: DocumentQuery):
    """
    Process a document from URL and answer multiple questions
    
    Args:
        request: DocumentQuery containing document URL and list of questions
        
    Returns:
        AnswersResponse with list of answers and metadata
    """
    start_time = time.time()
    
    try:
        # Initialize document manager
        document_manager = DocumentManager()
        
        # Download and process document
        print(f"🚀 Processing document: {request.documents}")
        print(f"📏 Using chunk size: {request.chunk_size} characters")
        success, document_name = document_manager.initialize_document_from_url(str(request.documents), request.chunk_size)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Failed to download or process the document. Please check the URL and try again."
            )
        
        # Get document info
        doc_info = document_manager.get_document_info()
        print(f"✅ Document ready: {doc_info['chunk_count']} chunks loaded")
        
        # Process all questions in parallel
        print(f"🎯 Processing {len(request.questions)} questions...")
        answers = document_manager.process_multiple_queries(request.questions)
        
        processing_time = time.time() - start_time
        
        print(f"🏁 All questions processed in {processing_time:.2f}s")
        
        return AnswersResponse(
            answers=answers,
            processing_time=round(processing_time, 2),
            document_chunks=doc_info['chunk_count'],
            total_questions=len(request.questions)
        )
        
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
            "POST /process-document": "Main endpoint for document processing and question answering",
            "GET /health": "Health check endpoint",
            "GET /docs": "Interactive API documentation"
        }
    }

# Note: Use main.py to run the server with proper reload functionality 