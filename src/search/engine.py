import torch
import time
from ..config import TOP_K, device
from ..embeddings import EmbeddingGenerator
from ..query_processing import QueryAnalyzer, QueryEnhancer


class SearchEngine:
    """Handles semantic search operations with intelligent query processing"""
    
    def __init__(self):
        self.embeddings = None
        self.document_chunks = []
        self.document_name = None
        self.query_analyzer = QueryAnalyzer()
        self.query_enhancer = QueryEnhancer()
    
    def load_embeddings(self, embeddings: torch.Tensor, chunks: list, document_name: str):
        """Load embeddings and chunks for search"""
        self.embeddings = embeddings
        self.document_chunks = chunks
        self.document_name = document_name
    
    def find_relevant_chunks(self, query: str, top_k: int = TOP_K):
        """Find relevant document chunks using semantic similarity"""
        if self.embeddings is None or not self.document_chunks:
            raise RuntimeError("Embeddings not initialized. Load embeddings first.")
        
        # Get query embedding
        q_vec = EmbeddingGenerator.get_embedding(query)
        
        # Calculate similarities
        similarities = self.embeddings @ q_vec  # (num_chunks,) on device
        
        # Get top results
        scores, indices = torch.topk(similarities, k=min(top_k, len(self.document_chunks)))
        
        results = []
        for score, i in zip(scores, indices):
            results.append({
                "chunk_index": i.item(),
                "text": self.document_chunks[i.item()],
                "score": score.item()
            })
        
        return results
    
    def intelligent_search(self, query: str, return_all_results: bool = True):
        """
        Enhanced search using Gemini for query analysis and parallel search
        
        Args:
            query: User query (can be complex like "46M, knee surgery, Pune, 3-month policy")
            return_all_results: If True, returns all unique results from parallel search
            
        Returns:
            All unique results from parallel search (no additional LLM filtering)
        """
        import time
        
        print(f"🔍 Starting intelligent search for: '{query}'")
        
        # Step 1: Decompose query using Gemini
        decompose_start = time.time()
        search_queries = self.query_analyzer.analyze_and_decompose_query(query)
        decompose_time = time.time() - decompose_start
        
        # Step 2: Execute parallel searches and return ALL unique results
        parallel_start = time.time()
        all_results = self.query_analyzer.parallel_search(self, search_queries, top_k_per_query=3)
        parallel_time = time.time() - parallel_start
        
        print(f"📊 Returning all {len(all_results)} unique results from parallel search")
        print(f"⏱️  Search Timing: Decomposition {decompose_time:.2f}s + Parallel Search {parallel_time:.2f}s = {decompose_time + parallel_time:.2f}s total")
        
        return all_results
    

    
    def get_best_answer(self, query: str):
        """
        Get the single best answer for a query using Gemini enhancement
        This is the ONLY method that uses LLM to filter results.
        
        Args:
            query: User query
            
        Returns:
            Single best result selected by Gemini
        """
        import time
        total_start = time.time()
        
        print(f"🎯 Finding best answer for: '{query}'")
        
        # Get ALL results from intelligent search (no LLM filtering)
        search_start = time.time()
        all_results = self.intelligent_search(query)
        search_time = time.time() - search_start
        
        if not all_results:
            print("❌ No relevant results found")
            return None
        
        print(f"🧠 Using Gemini to select best result from {len(all_results)} candidates...")
        
        # Use Gemini to select the most relevant from ALL results
        selection_start = time.time()
        best_result = self.query_enhancer.get_most_relevant_chunk(query, all_results)
        selection_time = time.time() - selection_start
        
        total_time = time.time() - total_start
        
        if best_result:
            print(f"\n🏆 Best Answer (Chunk {best_result['chunk_index']}):")
            print(f"Score: {best_result['score']:.4f}")
            if 'source_query' in best_result:
                print(f"Found via: '{best_result['source_query']}'")
            print(f"\n⏱️  Timing:")
            print(f"   • Intelligent Search: {search_time:.2f}s")
            print(f"   • Gemini Processing: {selection_time:.2f}s")
            print(f"   • Total Time: {total_time:.2f}s")
            print(f"\nContent:\n{best_result['text']}")
            
        return best_result 