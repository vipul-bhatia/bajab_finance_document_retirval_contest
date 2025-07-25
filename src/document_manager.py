import os
import hashlib
from .config import DOCUMENT_PATH
from .document_processing import DocumentProcessor
from .database import DatabaseManager
from .search import SearchEngine


class DocumentManager:
    """Main class that orchestrates document processing, embedding storage, and search"""
    
    def __init__(self):
        self.search_engine = SearchEngine()
        self.current_document = None
    
    def initialize_document_from_url(self, document_url: str, chunk_size: int = None):
        """Initialize embeddings from document URL - download, generate embeddings, then load for search"""
        
        # Generate document name from URL hash for consistency
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:12]
        document_name = f"doc_{url_hash}"
        
        # Download and load document chunks
        if chunk_size:
            chunks = DocumentProcessor.download_and_process_document_with_size(document_url, chunk_size)
        else:
            chunks = DocumentProcessor.download_and_process_document(document_url)
        if not chunks:
            print("Failed to download or process document from URL.")
            return False, document_name
        
        chunk_count = len(chunks)
        print(f"📄 Document processed: {chunk_count} chunks extracted")
        
        # Check if embeddings exist and match the number of chunks
        if DatabaseManager.embeddings_exist(document_name, chunk_count):
            print(f"✅ Embeddings database already exists, loading from database...")
            embeddings, document_chunks = DatabaseManager.load_embeddings(chunk_count, document_name)
        else:
            print(f"🔄 Generating new embeddings for document...")
            DatabaseManager.store_embeddings(chunks, document_name)
            embeddings, document_chunks = DatabaseManager.load_embeddings(chunk_count, document_name)
        
        # Load into search engine
        self.search_engine.load_embeddings(embeddings, document_chunks, document_name)
        self.current_document = document_name
        
        return True, document_name
    
    def initialize_document(self, document_path: str = DOCUMENT_PATH, document_name: str = None):
        """Initialize embeddings - generate if needed, then load for search"""
        
        # Generate document name from file path if not provided
        if not document_name:
            document_name = os.path.splitext(os.path.basename(document_path))[0]
            # Clean document name for use in file name (alphanumeric and underscore only)
            document_name = ''.join(c for c in document_name if c.isalnum() or c == '_')
        
        # Load document chunks
        chunks = DocumentProcessor.load_document(document_path)
        if not chunks:
            print("No document chunks found. Please provide a valid document.")
            return False, document_name
        
        chunk_count = len(chunks)
        
        # Check if embeddings exist and match the number of chunks
        if DatabaseManager.embeddings_exist(document_name, chunk_count):
            print(f"✅ Embeddings database already exists, loading from database...")
            embeddings, document_chunks = DatabaseManager.load_embeddings(chunk_count, document_name)
        else:
            print(f"🔄 Generating new embeddings for {document_name}...")
            DatabaseManager.store_embeddings(chunks, document_name)
            embeddings, document_chunks = DatabaseManager.load_embeddings(chunk_count, document_name)
        
        # Load into search engine
        self.search_engine.load_embeddings(embeddings, document_chunks, document_name)
        self.current_document = document_name
        
        return True, document_name
    
    def get_best_answer(self, query: str):
        """Get the single best answer for a query"""
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        return self.search_engine.get_best_answer(query)
    
    def process_multiple_queries(self, questions: list) -> list:
        """
        Process multiple queries in parallel and return list of answers
        
        Args:
            questions: List of user queries
            
        Returns:
            List of answers corresponding to each question
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        return self.search_engine.process_multiple_queries(questions)
    
    def get_document_info(self):
        """Get information about the currently loaded document"""
        if not self.current_document:
            return None
        
        return {
            "name": self.current_document,
            "chunk_count": len(self.search_engine.document_chunks)
        } 