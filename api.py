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
import json
import aiofiles
import asyncio
from datetime import datetime

from src.document_manager import DocumentManager
from src.search.similarity_search import find_similar_questions, get_answers_for_matched_questions

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

async def save_questions_answers(questions: List[str], answers: List[str], document_name: str, document_url: str):
    """
    Save questions and answers to a separate JSON file for permanent storage
    """
    try:
        qa_entry = {
            "timestamp": datetime.now().isoformat(),
            "document_name": document_name,
            "document_url": document_url,
            "questions": questions,
            "answers": answers
        }
        
        qa_file = "questions_answers.jsonl"
        
        async with aiofiles.open(qa_file, mode='a') as f:
            await f.write(json.dumps(qa_entry) + '\n')
            
        print(f"✅ Questions and answers saved to {qa_file}")
        
    except Exception as qa_error:
        # Log error but don't fail the request
        print(f"Warning: Failed to save questions and answers: {qa_error}")

async def log_request(log_entry: dict):
    """
    Asynchronously write a log entry to the request log file.
    """
    try:
        log_file = "request_logs.jsonl"
        async with aiofiles.open(log_file, mode='a') as f:
            await f.write(json.dumps(log_entry) + '\n')
    except Exception as log_error:
        print(f"Warning: Failed to log request details: {log_error}")

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

    print("Request documents", request.documents)
    print("Request questions", request.questions)
    
    try:
        print(f"🚀 Starting document processing pipeline...")
        
        # Step 0: Check for similar questions in stored data
        print(f"🔍 Step 0: Checking for similar questions in stored data...")
        similarity_start = time.time()
        
        similarity_result = await find_similar_questions(request.questions, str(request.documents), threshold=100)
        similarity_time = time.time() - similarity_start
        
        # Always log the request, regardless of cache hit or miss
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "document_url": str(request.documents),
            "questions": request.questions,
            "answers": [],  # Placeholder, will be filled later
            "similarity_search_used": similarity_result['found_similar'],
            "cached_answers_count": 0,
            "new_answers_count": 0,
            "document_name": "", # Placeholder
            "total_pipeline_time": 0.0
        }

        if similarity_result['found_similar']:
            print(f"✅ Found {len(similarity_result['matched_questions'])} similar questions!")
            
            # If all questions have matches, return cached answers
            if len(similarity_result['matched_questions']) == len(request.questions):
                print(f"🎯 All questions matched! Returning cached answers.")
                print(f"⏳ Adding 3-second delay for matched queries as per requirement...")
                
                
                
                # Extract answers in the correct order
                cached_answers = []
                for question in request.questions:
                    for match in similarity_result['matched_questions']:
                        if match['new_question'] == question:
                            cached_answers.append(match['answer'])
                            break
                
                total_time = time.time() - total_start_time
                print(f"🏁 Similarity search completed in {similarity_time:.2f}s")
                print(f"🏁 Total pipeline time: {total_time:.2f}s")
                
                # Log the similarity search results
                print(f"\n📊 SIMILARITY SEARCH RESULTS:")
                for match in similarity_result['matched_questions']:
                    print(f"   • '{match['new_question']}' → '{match['matched_question']}' ({match['similarity_score']}%) ")
                
                # Update log entry for cached response
                log_entry["answers"] = cached_answers
                log_entry["cached_answers_count"] = len(cached_answers)
                if similarity_result['matched_questions']:
                    log_entry["document_name"] = similarity_result['matched_questions'][0].get('document_name', 'Unknown')
                
                await log_request(log_entry)
                
                return AnswersResponse(answers=cached_answers)
            
            # If some questions have matches, process only unmatched questions
            else:
                print(f"🔄 Processing {len(similarity_result['unmatched_questions'])} new questions...")
                questions_to_process = similarity_result['unmatched_questions']
                cached_answers = get_answers_for_matched_questions(similarity_result['matched_questions'])
        else:
            if similarity_result['document_matched']:
                print(f"❌ Document found but no similar questions. Processing all {len(request.questions)} questions...")
            else:
                print(f"❌ Document not found in stored data. Processing all {len(request.questions)} questions...")
            questions_to_process = request.questions
            cached_answers = []
        
        # Initialize document manager
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
        log_entry["document_name"] = document_name
        
        # Get detailed timing from document manager
        if hasattr(document_manager, 'timing_data'):
            timing_data = document_manager.timing_data
        
        print(f"✅ Document download & processing completed in {download_time:.2f}s")
        
        # Step 2: Get document info and embedding stats
        doc_info = document_manager.get_document_info()
        print(f"📄 Document ready: {doc_info['chunk_count']} chunks loaded")
        print(f"🔍 Search engines loaded: FAISS={doc_info.get('faiss_loaded', False)}, BM25={doc_info.get('bm25_loaded', False)}")
        if doc_info.get('bm25_stats'):
            bm25_stats = doc_info['bm25_stats']
            print(f"📊 BM25 Index: {bm25_stats.get('vocabulary_size', 'N/A')} terms, avg doc length: {bm25_stats.get('average_document_length', 'N/A'):.1f}")
        print(f"🚀 Using Hybrid Search: RRF fusion of FAISS (semantic) + BM25 (keyword) with intelligent query processing")
        
        # Step 3: Process unmatched questions with Hybrid Search (RRF)
        query_start = time.time()
        print(f"🎯 Step 2: Processing {len(questions_to_process)} new questions with Hybrid Search (FAISS + BM25 + RRF)...")
        new_answers = document_manager.process_multiple_queries_hybrid(
            questions_to_process,
            use_intelligent_search=True,  # Enable intelligent query decomposition
            final_top_k=10               # Consider top 10 chunks per query
        )
        query_time = time.time() - query_start
        
        # Combine cached and new answers
        all_answers = cached_answers + new_answers
        
        total_time = time.time() - total_start_time
        
        print(f"🏁 All questions processed with Hybrid Search in {query_time:.2f}s")
        print(f"🏁 Total pipeline time: {total_time:.2f}s")
        
        # Print comprehensive timing summary
        print(f"\n📊 COMPREHENSIVE TIMING SUMMARY:")
        print(f"   • Similarity Search: {similarity_time:.2f}s")
        print(f"   • Document Download & Processing: {timing_data.get('download_and_processing', 0):.2f}s")
        if 'embedding_generation' in timing_data:
            print(f"   • FAISS Embedding Generation: {timing_data['embedding_generation']:.2f}s")
        if 'bm25_load' in timing_data:
            print(f"   • BM25 Index Generation: {timing_data['bm25_load']:.2f}s")
        print(f"   • Database Operations: {timing_data.get('database_load', 0):.2f}s")
        print(f"   • Search Engine Load: {timing_data.get('search_engine_load', 0):.2f}s")
        print(f"   • Hybrid Query Processing (FAISS+BM25+RRF): {query_time:.2f}s")
        print(f"   • Total Pipeline: {total_time:.2f}s")
        print(f"🎯 Search Method: Hybrid RRF Fusion (FAISS semantic + BM25 keyword + intelligent decomposition)")
        
        # Print similarity search results if any matches were found
        if similarity_result['found_similar']:
            print(f"\n📊 SIMILARITY SEARCH RESULTS:")
            for match in similarity_result['matched_questions']:
                print(f"   • '{match['new_question']}' → '{match['matched_question']}' ({match['similarity_score']}%) ")
        
        # Save questions and answers to separate JSON file (only for new questions)
        if questions_to_process:
            await save_questions_answers(questions_to_process, new_answers, document_name, str(request.documents))
        
        # Update and write the final log entry
        log_entry["answers"] = all_answers
        log_entry["cached_answers_count"] = len(cached_answers)
        log_entry["new_answers_count"] = len(new_answers)
        log_entry["total_pipeline_time"] = total_time
        await log_request(log_entry)
        
        return AnswersResponse(answers=all_answers)
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
            "GET /qa": "Retrieve all stored questions and answers",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.get("/qa")
async def get_questions_answers():
    """Retrieve all stored questions and answers"""
    try:
        qa_file = "questions_answers.jsonl"
        qa_data = []
        
        if os.path.exists(qa_file):
            async with aiofiles.open(qa_file, mode='r') as f:
                content = await f.read()
                lines = content.strip().split('\n')
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        try:
                            qa_entry = json.loads(line)
                            qa_data.append(qa_entry)
                        except json.JSONDecodeError:
                            continue
        
        return {
            "total_entries": len(qa_data),
            "questions_answers": qa_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving questions and answers: {str(e)}"
        )

# Note: Use main.py to run the server with proper reload functionality 