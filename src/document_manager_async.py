import os
import hashlib
import json
import asyncio
import time
from .config import DOCUMENT_PATH
from .document_processing import DocumentProcessor
from .database.manager_async import AsyncDatabaseManager
from .search.engine_async import AsyncSearchEngine
from typing import List


class AsyncDocumentManager:
    """Async main class that orchestrates document processing, embedding storage, and search"""

    def __init__(self):
        self.search_engine = AsyncSearchEngine()
        self.current_document = None
        self.timing_data = {}

    async def initialize_document_from_url_async(self, document_url: str, chunk_size: int = None):
        """Async initialize FAISS index from document URL - download, generate embeddings, then load for search"""
        total_start = time.time()
        timing_data = {}

        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:12]
        document_name = f"doc_{url_hash}"
        cache_path = f"document_cache/{document_name}_chunks.json"

        # Check for cached chunks first
        if os.path.exists(cache_path):
            print(f"   📄 Loading chunks from cache...")
            with open(cache_path, 'r') as f:
                chunks = json.load(f)
            timing_data['download_and_processing'] = 0.0
            print(f"   ✅ Loaded {len(chunks)} chunks from cache.")
        else:
            download_start = time.time()
            print(f"   📥 Downloading document...")
            
            # Run document processing in executor (I/O bound but potentially CPU intensive)
            loop = asyncio.get_event_loop()
            if chunk_size:
                chunks = await loop.run_in_executor(
                    None,
                    DocumentProcessor.download_and_process_document_with_size,
                    document_url,
                    chunk_size
                )
            else:
                chunks = await loop.run_in_executor(
                    None,
                    DocumentProcessor.download_and_process_document,
                    document_url
                )

            if not chunks:
                print("Failed to download or process document from URL.")
                return False, document_name

            download_time = time.time() - download_start
            timing_data['download_and_processing'] = round(download_time, 2)
            print(f"   ✅ Download & processing: {download_time:.2f}s")

            # Cache the chunks
            os.makedirs("document_cache", exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(chunks, f)
            print(f"   💾 Saved chunks to cache: {cache_path}")

        chunk_count = len(chunks)
        print(f"   📄 Document processed: {chunk_count} chunks extracted")

        embedding_start = time.time()
        if AsyncDatabaseManager.embeddings_exist(document_name, chunk_count):
            print(f"   💾 Loading existing FAISS index from disk...")
            # Run in executor since FAISS loading can be CPU intensive
            loop = asyncio.get_event_loop()
            faiss_index = await loop.run_in_executor(
                None,
                AsyncDatabaseManager.load_faiss_index,
                document_name
            )
            db_time = time.time() - embedding_start
            timing_data['faiss_load'] = round(db_time, 2)
            print(f"   ✅ FAISS index load: {db_time:.2f}s")
        else:
            print(f"   🔄 Generating new FAISS index (async)...")
            embedding_gen_start = time.time()
            await AsyncDatabaseManager.store_embeddings_async(chunks, document_name)
            embedding_gen_time = time.time() - embedding_gen_start
            timing_data['embedding_generation'] = round(embedding_gen_time, 2)
            print(f"   ✅ Async embedding generation: {embedding_gen_time:.2f}s")

            db_load_start = time.time()
            loop = asyncio.get_event_loop()
            faiss_index = await loop.run_in_executor(
                None,
                AsyncDatabaseManager.load_faiss_index,
                document_name
            )
            db_load_time = time.time() - db_load_start
            timing_data['faiss_load'] = round(db_load_time, 2)
            print(f"   ✅ FAISS index load: {db_load_time:.2f}s")

        search_load_start = time.time()
        self.search_engine.load_embeddings(faiss_index, chunks, document_name)
        self.current_document = document_name
        search_load_time = time.time() - search_load_start
        timing_data['search_engine_load'] = round(search_load_time, 2)

        total_time = time.time() - total_start
        timing_data['total_initialization'] = round(total_time, 2)

        print(f"   ✅ Search engine load: {search_load_time:.2f}s")
        print(f"   🏁 Total async initialization: {total_time:.2f}s")

        self.timing_data = timing_data

        return True, document_name
    
    async def initialize_document_async(self, document_path: str = DOCUMENT_PATH, document_name: str = None):
        """Async initialize FAISS index - generate if needed, then load for search"""
        
        if not document_name:
            document_name = os.path.splitext(os.path.basename(document_path))[0]
            document_name = ''.join(c for c in document_name if c.isalnum() or c == '_')
        
        # Run document loading in executor
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            DocumentProcessor.load_document,
            document_path
        )
        
        if not chunks:
            print("No document chunks found. Please provide a valid document.")
            return False, document_name
        
        chunk_count = len(chunks)
        
        if AsyncDatabaseManager.embeddings_exist(document_name, chunk_count):
            print(f"✅ FAISS index already exists, loading from disk...")
            faiss_index = await loop.run_in_executor(
                None,
                AsyncDatabaseManager.load_faiss_index,
                document_name
            )
        else:
            print(f"🔄 Generating new FAISS index for {document_name} (async)...")
            await AsyncDatabaseManager.store_embeddings_async(chunks, document_name)
            faiss_index = await loop.run_in_executor(
                None,
                AsyncDatabaseManager.load_faiss_index,
                document_name
            )
        
        self.search_engine.load_embeddings(faiss_index, chunks, document_name)
        self.current_document = document_name
        
        return True, document_name
    
    async def get_best_answer_async(self, query: str):
        """Async get the single best answer for a query"""
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document_async() first.")
        
        return await self.search_engine.get_best_answer_async(query)
    
    async def process_multiple_queries_async(self, questions: list) -> list:
        """
        Async process multiple queries in parallel and return list of answers
        
        Args:
            questions: List of user queries
            
        Returns:
            List of answers corresponding to each question
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document_async() first.")
        
        return await self.search_engine.process_multiple_queries_async(questions)
    
    def get_document_info(self):
        """Get information about the currently loaded document"""
        if not self.current_document:
            return None
        
        return {
            "name": self.current_document,
            "chunk_count": len(self.search_engine.document_chunks),
            "timing_data": self.timing_data
        }


# Async wrapper functions for backward compatibility
async def initialize_and_query_async(document_url: str, query: str):
    """Initialize document and get answer for a single query"""
    async_manager = AsyncDocumentManager()
    success, doc_name = await async_manager.initialize_document_from_url_async(document_url)
    
    if not success:
        return None
    
    return await async_manager.get_best_answer_async(query)

async def initialize_and_query_multiple_async(document_url: str, queries: List[str]):
    """Initialize document and get answers for multiple queries"""
    async_manager = AsyncDocumentManager()
    success, doc_name = await async_manager.initialize_document_from_url_async(document_url)
    
    if not success:
        return []
    
    return await async_manager.process_multiple_queries_async(queries)

# Sync wrappers for compatibility with existing code
def initialize_and_query_sync(document_url: str, query: str):
    """Sync wrapper around async document processing and query"""
    return asyncio.run(initialize_and_query_async(document_url, query))

def initialize_and_query_multiple_sync(document_url: str, queries: List[str]):
    """Sync wrapper around async document processing and multiple queries"""
    return asyncio.run(initialize_and_query_multiple_async(document_url, queries))


# Performance testing and comparison
async def performance_comparison_test():
    """Compare performance between sync and async implementations"""
    print("🧪 Performance Comparison: Async vs Sync Document Processing")
    
    # Test document URL (replace with actual test document)
    test_url = "https://example.com/test_document.pdf"
    test_queries = [
        "What is the main topic of this document?",
        "What are the key features mentioned?",
        "How does the process work?",
        "What are the requirements?",
        "Are there any limitations?",
    ]
    
    print("Note: Replace test_url with actual document URL for full testing")
    
    # Example of how to use the async implementation:
    """
    # Test async approach
    print("🚀 Testing async approach...")
    async_start = time.time()
    
    async_manager = AsyncDocumentManager()
    success, doc_name = await async_manager.initialize_document_from_url_async(test_url)
    
    if success:
        # Test single query
        single_result = await async_manager.get_best_answer_async(test_queries[0])
        
        # Test multiple queries
        multiple_results = await async_manager.process_multiple_queries_async(test_queries)
        
        async_time = time.time() - async_start
        
        print(f"📊 Async Results:")
        print(f"   • Total Time: {async_time:.2f}s")
        print(f"   • Document: {doc_name}")
        print(f"   • Queries Processed: {len(multiple_results)}")
        
        # Show timing breakdown
        timing_info = async_manager.get_document_info()
        if timing_info and 'timing_data' in timing_info:
            print(f"   • Timing Breakdown: {timing_info['timing_data']}")
    """

if __name__ == "__main__":
    asyncio.run(performance_comparison_test())
