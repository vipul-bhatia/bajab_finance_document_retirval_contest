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
        """Initialize FAISS index from document URL - download, generate embeddings, then load for search"""
        import time
        
        total_start = time.time()
        timing_data = {}
        
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:12]
        document_name = f"doc_{url_hash}"
        
        download_start = time.time()
        print(f"   📥 Downloading document...")
        if chunk_size:
            chunks = DocumentProcessor.download_and_process_document_with_size(document_url, chunk_size)
        else:
            chunks = DocumentProcessor.download_and_process_document(document_url)
        
        if not chunks:
            print("Failed to download or process document from URL.")
            return False, document_name
        
        download_time = time.time() - download_start
        timing_data['download_and_processing'] = round(download_time, 2)
        print(f"   ✅ Download & processing: {download_time:.2f}s")
        
        chunk_count = len(chunks)
        print(f"   📄 Document processed: {chunk_count} chunks extracted")
        
        embedding_start = time.time()
        if DatabaseManager.embeddings_exist(document_name, chunk_count):
            print(f"   💾 Loading existing FAISS index from disk...")
            faiss_index = DatabaseManager.load_faiss_index(document_name)
            db_time = time.time() - embedding_start
            timing_data['faiss_load'] = round(db_time, 2)
            print(f"   ✅ FAISS index load: {db_time:.2f}s")
        else:
            print(f"   🔄 Generating new FAISS index...")
            embedding_gen_start = time.time()
            DatabaseManager.store_embeddings(chunks, document_name)
            embedding_gen_time = time.time() - embedding_gen_start
            timing_data['embedding_generation'] = round(embedding_gen_time, 2)
            print(f"   ✅ Embedding generation: {embedding_gen_time:.2f}s")
            
            db_load_start = time.time()
            faiss_index = DatabaseManager.load_faiss_index(document_name)
            db_load_time = time.time() - db_load_start
            timing_data['faiss_load'] = round(db_load_time, 2)
            print(f"   ✅ FAISS index load: {db_load_time:.2f}s")
        
        search_load_start = time.time()
        self.search_engine.load_search_indices(faiss_index, chunks, document_name)
        self.current_document = document_name
        search_load_time = time.time() - search_load_start
        timing_data['search_engine_load'] = round(search_load_time, 2)
        
        total_time = time.time() - total_start
        timing_data['total_initialization'] = round(total_time, 2)
        
        print(f"   ✅ Search engine load: {search_load_time:.2f}s")
        print(f"   🏁 Total initialization: {total_time:.2f}s")
        
        self.timing_data = timing_data
        
        return True, document_name
    
    def initialize_document(self, document_path: str = DOCUMENT_PATH, document_name: str = None):
        """Initialize FAISS index - generate if needed, then load for search"""
        
        if not document_name:
            document_name = os.path.splitext(os.path.basename(document_path))[0]
            document_name = ''.join(c for c in document_name if c.isalnum() or c == '_')
        
        chunks = DocumentProcessor.load_document(document_path)
        if not chunks:
            print("No document chunks found. Please provide a valid document.")
            return False, document_name
        
        chunk_count = len(chunks)
        
        if DatabaseManager.embeddings_exist(document_name, chunk_count):
            print(f"✅ FAISS index already exists, loading from disk...")
            faiss_index = DatabaseManager.load_faiss_index(document_name)
        else:
            print(f"🔄 Generating new FAISS index for {document_name}...")
            DatabaseManager.store_embeddings(chunks, document_name)
            faiss_index = DatabaseManager.load_faiss_index(document_name)
        
        self.search_engine.load_search_indices(faiss_index, chunks, document_name)
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