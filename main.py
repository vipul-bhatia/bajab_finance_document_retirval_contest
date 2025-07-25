#!/usr/bin/env python3
"""
Document Retrieval API Server
Launch the FastAPI server for document processing and question answering
"""

import uvicorn

def main():
    """Launch the FastAPI server"""
    print("🚀 Starting Document Retrieval API Server")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 API Root: http://localhost:8000")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    # Start the server using import string for reload functionality
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 