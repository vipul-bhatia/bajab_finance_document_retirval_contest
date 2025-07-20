import os
from .config import DOCUMENT_PATH
from .document_processing import DocumentProcessor
from .database import DatabaseManager
from .search import SearchEngine


class DocumentManager:
    """Main class that orchestrates document processing, embedding storage, and search"""
    
    def __init__(self):
        self.search_engine = SearchEngine()
        self.current_document = None
    
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
    
    def search(self, query: str, top_k: int = None):
        """Search for relevant chunks"""
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if top_k is None:
            return self.search_engine.search_and_display(query)
        else:
            return self.search_engine.search_and_display(query, top_k)
    
    def get_document_info(self):
        """Get information about the currently loaded document"""
        if not self.current_document:
            return None
        
        return {
            "name": self.current_document,
            "chunk_count": len(self.search_engine.document_chunks)
        } 