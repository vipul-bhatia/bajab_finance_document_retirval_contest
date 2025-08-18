import torch
import time
import faiss
import asyncio
from ..config import TOP_K, device
from ..embeddings.generator_async import AsyncEmbeddingGenerator, get_embedding_async
from ..query_processing.analyzer_async import AsyncQueryAnalyzer
from ..query_processing.enhancer_async import AsyncQueryEnhancer
from typing import List, Dict, Any
import concurrent.futures


class AsyncSearchEngine:
    """Async search engine with intelligent query processing using asyncio for I/O operations"""
    
    def __init__(self):
        self.faiss_index = None
        self.document_chunks = []
        self.document_name = None
        
        # Thread pool for CPU-bound operations (FAISS searches)
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,  # CPU-bound operations
            thread_name_prefix="search_cpu"
        )
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'cpu_executor'):
            self.cpu_executor.shutdown(wait=False)
    
    def load_embeddings(self, faiss_index, chunks: list, document_name: str):
        """Load FAISS index and chunks for search"""
        self.faiss_index = faiss_index
        self.document_chunks = chunks
        self.document_name = document_name

    def _build_expanded_text(self, center_index: int, window: int = 1) -> str:
        """Return text that includes the chunk at center_index and its neighbors within the window."""
        if not self.document_chunks:
            return ""
        start_index = max(0, center_index - window)
        end_index = min(len(self.document_chunks) - 1, center_index + window)
        # Join with blank lines to preserve readability between chunks
        return "\n\n".join(self.document_chunks[start_index:end_index + 1])
    
    def _find_relevant_chunks_sync(self, query: str, top_k: int = TOP_K):
        """Sync version of find_relevant_chunks for use with run_in_executor"""
        if self.faiss_index is None or not self.document_chunks:
            raise RuntimeError("FAISS index not initialized. Load index first.")
        
        # Get embedding synchronously (this will be called from executor)
        import asyncio
        try:
            # Try to get current loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context but need sync execution
                # This is a bit tricky - we'll use the sync embedding fallback
                from ..embeddings.generator import EmbeddingGenerator
                q_vec = EmbeddingGenerator.get_embedding(query).cpu().numpy().reshape(1, -1)
            else:
                # No running loop, we can use async
                q_vec = asyncio.run(get_embedding_async(query)).cpu().numpy().reshape(1, -1)
        except RuntimeError:
            # No event loop, use sync version
            from ..embeddings.generator import EmbeddingGenerator
            q_vec = EmbeddingGenerator.get_embedding(query).cpu().numpy().reshape(1, -1)
        
        distances, indices = self.faiss_index.search(q_vec, top_k)
        
        results = []
        for dist, i in zip(distances[0], indices[0]):
            # Expand each hit by including one chunk above and below for fuller context
            expanded_text = self._build_expanded_text(i, window=1)
            results.append({
                "chunk_index": int(i),
                "text": expanded_text,
                "score": float(1 - dist)  # Convert distance to similarity score
            })
        
        return results
    
    async def find_relevant_chunks_async(self, query: str, top_k: int = TOP_K):
        """Async version of find_relevant_chunks using executor for CPU-bound FAISS operations"""
        if self.faiss_index is None or not self.document_chunks:
            raise RuntimeError("FAISS index not initialized. Load index first.")
        
        # Run FAISS search in thread executor (CPU-bound)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.cpu_executor,
            self._find_relevant_chunks_sync,
            query,
            top_k
        )
        
        return results
    
    async def intelligent_search_async(self, query: str, return_all_results: bool = True):
        """
        Async enhanced search using OpenAI for query analysis and parallel search
        
        Args:
            query: User query (can be complex like "46M, knee surgery, Pune, 3-month policy")
            return_all_results: If True, returns all unique results from parallel search
            
        Returns:
            All unique results from parallel search (no additional LLM filtering)
        """
        print(f"🔍 Starting async intelligent search for: '{query}'")
        
        # Step 1: Decompose query using async analyzer
        decompose_start = time.time()
        async with AsyncQueryAnalyzer() as analyzer:
            search_queries = await analyzer.analyze_and_decompose_query_async(query)
        decompose_time = time.time() - decompose_start
        
        # Step 2: Execute parallel searches asynchronously
        parallel_start = time.time()
        async with AsyncQueryAnalyzer() as analyzer:
            all_results = await analyzer.parallel_search_async(self, search_queries, top_k_per_query=TOP_K)
        parallel_time = time.time() - parallel_start
        
        print(f"📊 Returning all {len(all_results)} unique results from async parallel search")
        print(f"⏱️  Search Timing: Decomposition {decompose_time:.2f}s + Parallel Search {parallel_time:.2f}s = {decompose_time + parallel_time:.2f}s total")
        
        return all_results
    
    async def get_best_answer_async(self, query: str):
        """
        Async version of get_best_answer using async query processing
        
        Args:
            query: User query
            
        Returns:
            Response with direct answer
        """
        total_start = time.time()
        
        print(f"🎯 Analyzing query async: '{query}'")
        
        # Get ALL results from intelligent search
        search_start = time.time()
        all_results = await self.intelligent_search_async(query)
        search_time = time.time() - search_start
        
        if not all_results:
            print("❌ No relevant results found")
            return None
        
        print(f"🧠 Generating answer from {len(all_results)} chunks...")
        
        # Use async enhancer to provide direct answer
        selection_start = time.time()
        async with AsyncQueryEnhancer() as enhancer:
            response = await enhancer.get_most_relevant_chunk_async(query, all_results)
        selection_time = time.time() - selection_start
        
        total_time = time.time() - total_start
        
        if response:
            print(f"\n💬 ANSWER:")
            print(f"   {response.get('answer', 'No answer generated')}")
            
            print(f"\n📊 METADATA:")
            print(f"   • Chunks Analyzed: {response.get('total_chunks_analyzed', 0)}")
            print(f"   • Source Chunks: {response.get('source_chunks', [])}")
            
            print(f"\n⏱️  TIMING:")
            print(f"   • Search: {search_time:.2f}s")
            print(f"   • Answer Generation: {selection_time:.2f}s")
            print(f"   • Total: {total_time:.2f}s")
            
        return response 
    
    async def process_multiple_queries_async(self, questions: List[str]) -> List[str]:
        """
        Async process multiple queries in parallel and return list of answers
        
        Args:
            questions: List of user queries
            
        Returns:
            List of answers corresponding to each question
        """
        total_start = time.time()
        
        print(f"🎯 Processing {len(questions)} queries in parallel (async)...")
        
        # Step 1: Analyze which queries need decomposition
        analysis_start = time.time()
        async with AsyncQueryAnalyzer() as analyzer:
            query_search_lists = await analyzer.process_multiple_queries_async(questions)
        analysis_time = time.time() - analysis_start
        
        # Step 2: Execute all searches in parallel
        search_start = time.time()
        all_search_results = []
        
        # Create semaphore to limit concurrent searches
        search_semaphore = asyncio.Semaphore(8)
        
        async def search_with_semaphore(search_queries, query_index):
            async with search_semaphore:
                if len(search_queries) == 1:
                    # Single query search
                    return await self.find_relevant_chunks_async(search_queries[0], 5), query_index
                else:
                    # Multi-query parallel search
                    async with AsyncQueryAnalyzer() as analyzer:
                        results = await analyzer.parallel_search_async(self, search_queries, 5)
                    return results, query_index
        
        # Create tasks for all searches
        search_tasks = [
            search_with_semaphore(search_queries, i)
            for i, search_queries in enumerate(query_search_lists)
        ]
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process search results
        results_by_index = {}
        for result in search_results:
            if isinstance(result, Exception):
                print(f"   ❌ Search failed: {result}")
                continue
            
            search_result, query_index = result
            results_by_index[query_index] = search_result
            print(f"   ✅ Query {query_index + 1}: {len(search_result)} chunks found")
        
        # Ensure results are in order
        for i in range(len(questions)):
            all_search_results.append(results_by_index.get(i, []))
        
        search_time = time.time() - search_start
        
        # Step 3: Generate answers for all queries in parallel
        answer_start = time.time()
        
        # Prepare data for batch enhancement
        queries_and_results = [
            (question, search_results) 
            for question, search_results in zip(questions, all_search_results)
            if search_results  # Only include queries with search results
        ]
        
        # Process enhancements in parallel
        if queries_and_results:
            async with AsyncQueryEnhancer() as enhancer:
                enhancement_results = await enhancer.process_multiple_enhancements_async(queries_and_results)
        else:
            enhancement_results = []
        
        # Map results back to original question order
        final_answers = []
        enhancement_index = 0
        
        for i, (question, search_results) in enumerate(zip(questions, all_search_results)):
            if search_results and enhancement_index < len(enhancement_results):
                response = enhancement_results[enhancement_index]
                answer = response.get('answer', 'Unable to generate answer.') if response else 'Unable to generate answer.'
                final_answers.append(answer)
                enhancement_index += 1
                print(f"   ✅ Answer {i + 1}: Generated")
            else:
                final_answers.append("No relevant information found for this query.")
                print(f"   ⚠️ Answer {i + 1}: No search results")
        
        answer_time = time.time() - answer_start
        total_time = time.time() - total_start
        
        print(f"\n📊 ASYNC BATCH PROCESSING COMPLETE:")
        print(f"   • Total Queries: {len(questions)}")
        print(f"   • Successful Answers: {len([a for a in final_answers if not a.startswith('Error') and not a.startswith('No relevant') and not a.startswith('Unable')])}")
        print(f"\n⏱️  TIMING:")
        print(f"   • Query Analysis: {analysis_time:.2f}s")
        print(f"   • Parallel Search: {search_time:.2f}s")
        print(f"   • Answer Generation: {answer_time:.2f}s")
        print(f"   • Total Time: {total_time:.2f}s")
        
        return final_answers


# Async wrapper functions for backward compatibility
async def search_async(search_engine, query: str):
    """Async wrapper for single query search."""
    return await search_engine.get_best_answer_async(query)

async def search_multiple_async(search_engine, queries: List[str]) -> List[str]:
    """Async wrapper for multiple query processing."""
    return await search_engine.process_multiple_queries_async(queries)

# Sync wrappers for compatibility with existing code
def search_sync(search_engine, query: str):
    """Sync wrapper around async search."""
    async_engine = AsyncSearchEngine()
    async_engine.load_embeddings(
        search_engine.faiss_index, 
        search_engine.document_chunks, 
        search_engine.document_name
    )
    return asyncio.run(async_engine.get_best_answer_async(query))

def search_multiple_sync(search_engine, queries: List[str]) -> List[str]:
    """Sync wrapper around async multiple search."""
    async_engine = AsyncSearchEngine()
    async_engine.load_embeddings(
        search_engine.faiss_index, 
        search_engine.document_chunks, 
        search_engine.document_name
    )
    return asyncio.run(async_engine.process_multiple_queries_async(queries))


# Performance testing
async def performance_test_search_engine():
    """Test performance of async search engine."""
    print("🧪 Testing async search engine performance...")
    
    # This would require a loaded search engine for full testing
    print("Note: Full performance test requires loaded document and FAISS index")
    print("Use this class with your existing DocumentManager for complete testing")
    
    # Example usage pattern:
    """
    async def example_usage():
        # Initialize document manager
        doc_manager = DocumentManager()
        success, doc_name = doc_manager.initialize_document_from_url("your_document_url")
        
        if success:
            # Create async search engine
            async_engine = AsyncSearchEngine()
            async_engine.load_embeddings(
                doc_manager.search_engine.faiss_index,
                doc_manager.search_engine.document_chunks,
                doc_manager.search_engine.document_name
            )
            
            # Test single query
            result = await async_engine.get_best_answer_async("Your query here")
            
            # Test multiple queries
            queries = ["Query 1", "Query 2", "Query 3"]
            answers = await async_engine.process_multiple_queries_async(queries)
    """

if __name__ == "__main__":
    asyncio.run(performance_test_search_engine())
