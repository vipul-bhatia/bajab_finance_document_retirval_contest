import torch
import time
import faiss
from ..config import TOP_K, device
from ..embeddings import EmbeddingGenerator
from ..query_processing import QueryAnalyzer, QueryEnhancer
from .rrf_fusion import HybridSearchEngine
from typing import List, Dict, Any
import concurrent.futures


class SearchEngine:
    """Handles semantic search operations with intelligent query processing"""
    
    def __init__(self):
        self.faiss_index = None
        self.document_chunks = []
        self.document_name = None
        self.query_analyzer = QueryAnalyzer()
        self.query_enhancer = QueryEnhancer()
        self.hybrid_engine = HybridSearchEngine()  # RRF fusion engine
    
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
    
    def find_relevant_chunks(self, query: str, top_k: int = TOP_K):
        """Find relevant document chunks using FAISS"""
        if self.faiss_index is None or not self.document_chunks:
            raise RuntimeError("FAISS index not initialized. Load index first.")
        
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
        all_results = self.query_analyzer.parallel_search(self, search_queries, top_k_per_query=TOP_K)
        parallel_time = time.time() - parallel_start
        
        print(f"📊 Returning all {len(all_results)} unique results from parallel search")
        print(f"⏱️  Search Timing: Decomposition {decompose_time:.2f}s + Parallel Search {parallel_time:.2f}s = {decompose_time + parallel_time:.2f}s total")
        
        return all_results
    

    
    def get_best_answer(self, query: str):
        """
        Get a direct, concise answer for a query using Gemini
        
        Args:
            query: User query
            
        Returns:
            Simple response with direct answer
        """
        import time
        total_start = time.time()
        
        print(f"🎯 Analyzing query: '{query}'")
        
        # Get ALL results from intelligent search (no LLM filtering)
        search_start = time.time()
        all_results = self.intelligent_search(query)
        search_time = time.time() - search_start
        
        if not all_results:
            print("❌ No relevant results found")
            return None
        
        print(f"🧠 Generating answer from {len(all_results)} chunks...")
        
        # Use Gemini to provide direct answer
        selection_start = time.time()
        response = self.query_enhancer.get_most_relevant_chunk(query, all_results)
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
    
    def process_multiple_queries(self, questions: List[str]) -> List[str]:
        """
        Process multiple queries in parallel and return list of answers
        
        Args:
            questions: List of user queries
            
        Returns:
            List of answers corresponding to each question
        """
        import time
        total_start = time.time()
        
        print(f"🎯 Processing {len(questions)} queries in parallel...")
        
        # Step 1: Analyze which queries need decomposition
        analysis_start = time.time()
        query_search_lists = self.query_analyzer.process_multiple_queries(questions)
        analysis_time = time.time() - analysis_start
        
        # Step 2: Execute all searches in parallel
        search_start = time.time()
        all_search_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            search_futures = []
            
            for i, search_queries in enumerate(query_search_lists):
                if len(search_queries) == 1:
                    # Single query search
                    future = executor.submit(self.find_relevant_chunks, search_queries[0], 5)
                    search_futures.append((future, i, 'single'))
                else:
                    # Multi-query parallel search
                    future = executor.submit(self.query_analyzer.parallel_search, self, search_queries, 5)
                    search_futures.append((future, i, 'parallel'))
            
            # Collect search results
            results_by_index = {}
            for future, query_index, search_type in search_futures:
                try:
                    results = future.result()
                    results_by_index[query_index] = results
                    print(f"   ✅ Query {query_index + 1}: {len(results)} chunks found")
                except Exception as e:
                    print(f"   ❌ Query {query_index + 1} search failed: {e}")
                    results_by_index[query_index] = []
            
            # Ensure results are in order
            for i in range(len(questions)):
                all_search_results.append(results_by_index.get(i, []))
        
        search_time = time.time() - search_start

        # print('all_search_resultssss', all_search_results)
        
        # Step 3: Generate answers for all queries in parallel
        answer_start = time.time()
        answers = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            answer_futures = []
            
            for i, (question, search_results) in enumerate(zip(questions, all_search_results)):
                if search_results:
                    future = executor.submit(self.query_enhancer.get_most_relevant_chunk, question, search_results)
                    answer_futures.append((future, i))
                else:
                    answers.append(f"No relevant information found for this query.")
            
            # Collect answers in order
            answers_by_index = {}
            for future, query_index in answer_futures:
                try:
                    response = future.result()
                    answer = response.get('answer', 'Unable to generate answer.') if response else 'Unable to generate answer.'
                    answers_by_index[query_index] = answer
                    print(f"   ✅ Answer {query_index + 1}: Generated")
                except Exception as e:
                    print(f"   ❌ Answer {query_index + 1} generation failed: {e}")
                    answers_by_index[query_index] = f"Error generating answer: {str(e)}"
            
            # Fill in answers that were processed
            final_answers = []
            for i in range(len(questions)):
                if i in answers_by_index:
                    final_answers.append(answers_by_index[i])
                elif i < len(answers):
                    final_answers.append(answers[i])  # No search results case
                else:
                    final_answers.append("Unable to process this query.")
        
        answer_time = time.time() - answer_start
        total_time = time.time() - total_start
        
        print(f"\n📊 BATCH PROCESSING COMPLETE:")
        print(f"   • Total Queries: {len(questions)}")
        print(f"   • Successful Answers: {len([a for a in final_answers if not a.startswith('Error') and not a.startswith('No relevant') and not a.startswith('Unable')])}")
        print(f"\n⏱️  TIMING:")
        print(f"   • Query Analysis: {analysis_time:.2f}s")
        print(f"   • Parallel Search: {search_time:.2f}s")
        print(f"   • Answer Generation: {answer_time:.2f}s")
        print(f"   • Total Time: {total_time:.2f}s")
        
        return final_answers
    
    def hybrid_search_with_rrf(
        self,
        bm25_engine,
        query: str,
        top_k_per_engine: int = 20,
        final_top_k: int = TOP_K,
        use_mmr: bool = True,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform hybrid search using both FAISS and BM25 with RRF fusion
        
        Args:
            bm25_engine: BM25 search engine instance
            query: Search query
            top_k_per_engine: Results to get from each engine
            final_top_k: Final number of results after fusion
            use_mmr: Whether to apply MMR for diversity
            bm25_weight: Weight for BM25 results (0.0 to 1.0)
            faiss_weight: Weight for FAISS results (0.0 to 1.0)
            
        Returns:
            Dictionary with hybrid search results and metadata
        """
        if self.faiss_index is None or not self.document_chunks:
            raise RuntimeError("FAISS index not initialized. Load index first.")
        
        if not bm25_engine:
            raise RuntimeError("BM25 engine not provided.")
        
        print(f"🔍 Hybrid Search (RRF): '{query}' | FAISS weight: {faiss_weight:.1f}, BM25 weight: {bm25_weight:.1f}")
        
        return self.hybrid_engine.hybrid_search(
            faiss_engine=self,
            bm25_engine=bm25_engine,
            query=query,
            top_k_per_engine=top_k_per_engine,
            final_top_k=final_top_k,
            use_mmr=use_mmr,
            bm25_weight=bm25_weight,
            faiss_weight=faiss_weight
        )
    
    def intelligent_hybrid_search(
        self,
        bm25_engine,
        query: str,
        final_top_k: int = TOP_K,
        use_mmr: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced hybrid search using Gemini for query analysis and RRF fusion
        
        Args:
            bm25_engine: BM25 search engine instance
            query: User query (can be complex)
            final_top_k: Final number of results after fusion
            use_mmr: Whether to apply MMR for diversity
            
        Returns:
            Dictionary with hybrid search results and metadata
        """
        import time
        
        print(f"🧠 Intelligent Hybrid Search: '{query}'")
        
        # Step 1: Decompose query using Gemini
        decompose_start = time.time()
        search_queries = self.query_analyzer.analyze_and_decompose_query(query)
        decompose_time = time.time() - decompose_start
        
        # Step 2: Run parallel searches for all decomposed queries
        parallel_start = time.time()
        
        # Collect results from all search queries
        all_faiss_results = []
        all_bm25_results = []
        
        for search_query in search_queries:
            # FAISS search
            faiss_results = self.find_relevant_chunks(search_query, TOP_K)
            all_faiss_results.extend(faiss_results)
            
            # BM25 search
            bm25_results = bm25_engine.search(search_query, TOP_K)
            all_bm25_results.extend(bm25_results)
        
        # Remove duplicates based on chunk_index
        seen_faiss = set()
        unique_faiss = []
        for result in all_faiss_results:
            if result['chunk_index'] not in seen_faiss:
                unique_faiss.append(result)
                seen_faiss.add(result['chunk_index'])
        
        seen_bm25 = set()
        unique_bm25 = []
        for result in all_bm25_results:
            if result['chunk_index'] not in seen_bm25:
                unique_bm25.append(result)
                seen_bm25.add(result['chunk_index'])
        
        parallel_time = time.time() - parallel_start
        
        # Step 3: Fuse results using RRF
        fusion_start = time.time()
        fused_results = self.hybrid_engine.rrf_fusion.fuse_rankings(
            unique_bm25, unique_faiss, final_top_k * 2
        )
        
        # Apply MMR if requested
        final_results = fused_results
        if use_mmr and len(fused_results) > final_top_k:
            final_results = self.hybrid_engine.mmr_processor.apply_mmr(fused_results, final_top_k)
        else:
            final_results = fused_results[:final_top_k]
        
        fusion_time = time.time() - fusion_start
        
        total_time = decompose_time + parallel_time + fusion_time
        
        print(f"📊 Intelligent Hybrid Search: {len(final_results)} results")
        print(f"⏱️  Timing: Query Analysis {decompose_time:.2f}s + Search {parallel_time:.2f}s + Fusion {fusion_time:.2f}s = {total_time:.2f}s")
        
        return {
            'results': final_results,
            'metadata': {
                'query': query,
                'search_queries': search_queries,
                'faiss_results_count': len(unique_faiss),
                'bm25_results_count': len(unique_bm25),
                'final_results_count': len(final_results),
                'timing': {
                    'query_analysis': decompose_time,
                    'parallel_search': parallel_time,
                    'fusion': fusion_time,
                    'total': total_time
                },
                'used_mmr': use_mmr
            }
        }
    
    def get_best_hybrid_answer(
        self,
        bm25_engine,
        query: str,
        use_intelligent_search: bool = True,
        final_top_k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Get the best answer using hybrid search with RRF fusion
        
        Args:
            bm25_engine: BM25 search engine instance
            query: User query
            use_intelligent_search: Whether to use intelligent query decomposition
            final_top_k: Number of chunks to consider for answer generation
            
        Returns:
            Response with answer and hybrid search metadata
        """
        import time
        total_start = time.time()
        
        print(f"🎯 Hybrid Answer Generation: '{query}'")
        
        # Get hybrid search results
        if use_intelligent_search:
            search_results = self.intelligent_hybrid_search(bm25_engine, query, final_top_k)
        else:
            search_results = self.hybrid_search_with_rrf(bm25_engine, query, final_top_k=final_top_k)
        
        hybrid_results = search_results['results']
        
        if not hybrid_results:
            print("❌ No relevant results found in hybrid search")
            return None
        
        print(f"🧠 Generating answer from {len(hybrid_results)} hybrid chunks...")
        
        # Convert to format expected by query enhancer
        chunks_for_answer = []
        for result in hybrid_results:
            chunk_data = {
                'chunk_index': result['chunk_index'],
                'text': result['text'],
                'score': result.get('rrf_score', result.get('weighted_rrf_score', 0))
            }
            chunks_for_answer.append(chunk_data)
        
        # Use Gemini to provide direct answer
        selection_start = time.time()
        response = self.query_enhancer.get_most_relevant_chunk(query, chunks_for_answer)
        selection_time = time.time() - selection_start
        
        total_time = time.time() - total_start
        
        if response:
            # Enhance response with hybrid search metadata
            response['hybrid_search_metadata'] = search_results['metadata']
            response['answer_generation_time'] = selection_time
            response['total_time'] = total_time
            
            print(f"\n💬 HYBRID ANSWER:")
            print(f"   {response.get('answer', 'No answer generated')}")
            
            print(f"\n📊 HYBRID METADATA:")
            print(f"   • FAISS Chunks: {search_results['metadata']['faiss_results_count']}")
            print(f"   • BM25 Chunks: {search_results['metadata']['bm25_results_count']}")
            print(f"   • Fused Chunks: {search_results['metadata']['final_results_count']}")
            print(f"   • Source Chunks: {response.get('source_chunks', [])}")
            
            print(f"\n⏱️  TIMING:")
            print(f"   • Hybrid Search: {search_results['metadata']['timing']['total']:.2f}s")
            print(f"   • Answer Generation: {selection_time:.2f}s")
            print(f"   • Total: {total_time:.2f}s")
        
        return response 