import os
import hashlib
import json
import time
import concurrent.futures
from typing import List
from .config import DOCUMENT_PATH
from .document_processing import DocumentProcessor
from .database import DatabaseManager
from .search import SearchEngine
from .search.bm25_search import BM25SearchEngine
from .logging import get_hybrid_logger


class DocumentManager:
    """Main class that orchestrates document processing, embedding storage, and search"""

    def __init__(self):
        self.search_engine = SearchEngine()  # FAISS-based search
        self.bm25_engine = None  # BM25-based search
        self.current_document = None

    def initialize_document_from_url(self, document_url: str, chunk_size: int = None):
        """Initialize FAISS index from document URL - download, generate embeddings, then load for search"""
        import time

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

            # Cache the chunks
            os.makedirs("document_cache", exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(chunks, f)
            print(f"   💾 Saved chunks to cache: {cache_path}")


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
        self.search_engine.load_embeddings(faiss_index, chunks, document_name)
        search_load_time = time.time() - search_load_start
        timing_data['search_engine_load'] = round(search_load_time, 2)
        
        # Load or build BM25 index
        bm25_start = time.time()
        if DatabaseManager.bm25_index_exists(document_name, chunk_count):
            print(f"   💾 Loading existing BM25 index from disk...")
            self.bm25_engine = DatabaseManager.load_bm25_index(document_name)
        else:
            print(f"   🔄 Generating new BM25 index...")
            DatabaseManager.store_bm25_index(chunks, document_name)
            self.bm25_engine = DatabaseManager.load_bm25_index(document_name)
        
        bm25_time = time.time() - bm25_start
        timing_data['bm25_load'] = round(bm25_time, 2)
        print(f"   ✅ BM25 index load: {bm25_time:.2f}s")
        
        self.current_document = document_name

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
        
        self.search_engine.load_embeddings(faiss_index, chunks, document_name)
        
        # Load or build BM25 index
        if DatabaseManager.bm25_index_exists(document_name, chunk_count):
            print(f"✅ BM25 index already exists, loading from disk...")
            self.bm25_engine = DatabaseManager.load_bm25_index(document_name)
        else:
            print(f"🔄 Generating new BM25 index for {document_name}...")
            DatabaseManager.store_bm25_index(chunks, document_name)
            self.bm25_engine = DatabaseManager.load_bm25_index(document_name)
        
        self.current_document = document_name
        
        return True, document_name
    
    ###
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
    
    ###
    def get_bm25_results(self, query: str, top_k: int = 10):
        """
        Get BM25 search results for a query
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of BM25 search results with scores and text
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if not self.bm25_engine:
            raise RuntimeError("BM25 engine not initialized.")
        
        return self.bm25_engine.search(query, top_k)
    
    ####
    def get_parallel_search_results(self, query: str, top_k: int = 10):
        """
        Get search results from both FAISS and BM25 engines in parallel
        
        Args:
            query: Search query string
            top_k: Number of top results to return from each engine
            
        Returns:
            Dictionary with 'faiss_results' and 'bm25_results' keys
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if not self.bm25_engine:
            raise RuntimeError("BM25 engine not initialized.")
        
        import concurrent.futures
        import time
        
        print(f"🔍 Running parallel search: FAISS + BM25 for query: '{query}'")
        search_start = time.time()
        
        def get_faiss_results():
            return self.search_engine.find_relevant_chunks(query, top_k)
        
        def get_bm25_results():
            return self.bm25_engine.search(query, top_k)
        
        # Execute both searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            faiss_future = executor.submit(get_faiss_results)
            bm25_future = executor.submit(get_bm25_results)
            
            faiss_results = faiss_future.result()
            bm25_results = bm25_future.result()
        
        search_time = time.time() - search_start
        
        print(f"📊 Parallel search completed in {search_time:.3f}s")
        print(f"   • FAISS results: {len(faiss_results)}")
        print(f"   • BM25 results: {len(bm25_results)}")
        
        return {
            'faiss_results': faiss_results,
            'bm25_results': bm25_results,
            'search_time': search_time
        }
    
    def get_document_info(self):
        """Get information about the currently loaded document"""
        if not self.current_document:
            return None
        
        info = {
            "name": self.current_document,
            "chunk_count": len(self.search_engine.document_chunks),
            "faiss_loaded": self.search_engine.faiss_index is not None,
            "bm25_loaded": self.bm25_engine is not None
        }
        
        if self.bm25_engine:
            bm25_stats = self.bm25_engine.get_index_stats()
            info["bm25_stats"] = bm25_stats
        
        return info
    
    ####
    def get_hybrid_search_results(
        self,
        query: str,
        top_k_per_engine: int = 20,
        final_top_k: int = 10,
        use_mmr: bool = True,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5
    ) -> dict:
        """
        Get hybrid search results using RRF fusion of FAISS and BM25
        
        Args:
            query: Search query string
            top_k_per_engine: Results to get from each engine before fusion
            final_top_k: Final number of fused results to return
            use_mmr: Whether to apply MMR for diversity
            bm25_weight: Weight for BM25 results (0.0 to 1.0)
            faiss_weight: Weight for FAISS results (0.0 to 1.0)
            
        Returns:
            Dictionary with hybrid search results and metadata
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if not self.bm25_engine:
            raise RuntimeError("BM25 engine not initialized.")
        
        return self.search_engine.hybrid_search_with_rrf(
            bm25_engine=self.bm25_engine,
            query=query,
            top_k_per_engine=top_k_per_engine,
            final_top_k=final_top_k,
            use_mmr=use_mmr,
            bm25_weight=bm25_weight,
            faiss_weight=faiss_weight
        )
    ####
    def get_intelligent_hybrid_results(
        self,
        query: str,
        final_top_k: int = 10,
        use_mmr: bool = True
    ) -> dict:
        """
        Get intelligent hybrid search results with query decomposition and RRF fusion
        
        Args:
            query: Search query string (can be complex)
            final_top_k: Final number of fused results to return
            use_mmr: Whether to apply MMR for diversity
            
        Returns:
            Dictionary with hybrid search results and metadata
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if not self.bm25_engine:
            raise RuntimeError("BM25 engine not initialized.")
        
        return self.search_engine.intelligent_hybrid_search(
            bm25_engine=self.bm25_engine,
            query=query,
            final_top_k=final_top_k,
            use_mmr=use_mmr
        )
    ####
    def get_best_hybrid_answer(
        self,
        query: str,
        use_intelligent_search: bool = True,
        final_top_k: int = 10
    ) -> dict:
        """
        Get the best answer using hybrid search with RRF fusion
        
        Args:
            query: User query
            use_intelligent_search: Whether to use intelligent query decomposition
            final_top_k: Number of chunks to consider for answer generation
            
        Returns:
            Response with answer and hybrid search metadata
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if not self.bm25_engine:
            raise RuntimeError("BM25 engine not initialized.")
        
        return self.search_engine.get_best_hybrid_answer(
            bm25_engine=self.bm25_engine,
            query=query,
            use_intelligent_search=use_intelligent_search,
            final_top_k=final_top_k
        )
    
    #imp
    def process_multiple_queries_hybrid(
        self,
        questions: List[str],
        use_intelligent_search: bool = True,
        final_top_k: int = 10
    ) -> List[str]:
        """
        Process multiple queries using hybrid search with RRF fusion
        
        Args:
            questions: List of user queries
            use_intelligent_search: Whether to use intelligent query decomposition
            final_top_k: Number of chunks per query for answer generation
            
        Returns:
            List of answers corresponding to each question
        """
        if not self.current_document:
            raise RuntimeError("No document loaded. Call initialize_document() first.")
        
        if not self.bm25_engine:
            raise RuntimeError("BM25 engine not initialized.")
        
        import concurrent.futures
        
        total_start = time.time()
        
        # Initialize hybrid logger
        logger = get_hybrid_logger()
        logger.start_session(self.current_document, len(questions))
        logger.log_document_info(self.get_document_info())
        
        print(f"🎯 Processing {len(questions)} queries with Hybrid Search (RRF)...")
        print(f"📝 Detailed logging enabled - check hybrid_search_logs/ directory")
        
        # Process all queries in TRUE PARALLEL with detailed logging
        answers = [None] * len(questions)  # Pre-allocate to maintain order
        
        try:
            # Start logging for all questions
            for i, question in enumerate(questions):
                logger.log_question_start(i, question)
            
            # OPTIMIZATION: Use batch query analysis for multiple questions when intelligent search is enabled
            all_decomposed_queries = None
            if use_intelligent_search and len(questions) > 1:
                print(f"🧠 Batch analyzing {len(questions)} questions for intelligent decomposition...")
                batch_start = time.time()
                
                # Use batch processing for query analysis
                all_decomposed_queries = self.search_engine.query_analyzer.process_multiple_queries(questions)
                batch_analysis_time = time.time() - batch_start
                
                print(f"✅ Batch analysis completed in {batch_analysis_time:.3f}s")
                print(f"📊 Query decomposition summary:")
                for i, decomposed in enumerate(all_decomposed_queries):
                    print(f"   Question {i+1}: {len(decomposed)} sub-queries")
            
            # Execute all questions in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                for i, question in enumerate(questions):
                    # Pass pre-decomposed queries if available
                    pre_decomposed = all_decomposed_queries[i] if all_decomposed_queries else None
                    future = executor.submit(
                        self._get_hybrid_answer_with_logging, 
                        question, i, use_intelligent_search, final_top_k, logger, pre_decomposed
                    )
                    futures.append((future, i))
                
                print(f"🚀 Processing {len(questions)} questions in parallel...")
                
                # Collect results as they complete (maintaining order)
                for future in concurrent.futures.as_completed([f[0] for f in futures]):
                    # Find the question index for this future
                    question_index = None
                    for f, i in futures:
                        if f == future:
                            question_index = i
                            break
                    
                    if question_index is None:
                        continue
                    
                    question_start_time = time.time()
                    
                    try:
                        # Get hybrid search results with detailed component tracking
                        response = future.result()
                        
                        if response:
                            answer = response.get('answer', 'Unable to generate answer.')
                            answers[question_index] = answer
                            print(f"   ✅ Hybrid Answer {question_index + 1}: Generated ({len(answer)} chars)")
                        else:
                            answer = 'No relevant information found.'
                            answers[question_index] = answer
                            print(f"   ⚠️  Hybrid Answer {question_index + 1}: No results")
                        
                        # Log final results
                        if response and 'hybrid_search_metadata' in response:
                            metadata = response['hybrid_search_metadata']
                            final_results = metadata.get('results', [])
                            logger.log_final_results(question_index, final_results, answer)
                        
                        # Log performance metrics
                        question_time = time.time() - question_start_time
                        logger.log_performance_metrics(question_index, question_time)
                        
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        answers[question_index] = error_msg
                        print(f"   ❌ Hybrid Answer {question_index + 1} failed: {e}")
                        
                        # Log the error
                        logger.log_final_results(question_index, [], error_msg)
                        question_time = time.time() - question_start_time
                        logger.log_performance_metrics(question_index, question_time)
                
                # Verify all questions were processed
                if None in answers:
                    print("⚠️  Warning: Some questions may not have been processed")
                    for i, answer in enumerate(answers):
                        if answer is None:
                            answers[i] = "Question processing failed or timed out"
        
        finally:
            # End logging session
            logger.end_session()
        
        total_time = time.time() - total_start
        
        successful_answers = len([a for a in answers if not a.startswith('Error') and not a.startswith('Unable') and not a.startswith('No relevant')])
        
        print(f"\n📊 HYBRID BATCH PROCESSING COMPLETE:")
        print(f"   • Total Queries: {len(questions)}")
        print(f"   • Successful Answers: {successful_answers}")
        print(f"   • Total Time: {total_time:.2f}s")
        print(f"   • Average Time per Query: {total_time/len(questions):.2f}s")
        print(f"📋 Detailed logs available:")
        print(logger.get_log_files_info())
        
        return answers
    

    # most important function of the code calling most of the functions
    def _get_hybrid_answer_with_logging(
        self,
        question: str,
        question_index: int,
        use_intelligent_search: bool,
        final_top_k: int,
        logger,
        pre_decomposed_queries=None
    ) -> dict:
        """Get hybrid answer with detailed component logging and TRUE parallel execution"""
        
        import concurrent.futures
        
        # Step 1: Query Analysis and Decomposition (INTELLIGENT SEARCH)
        analysis_start = time.time()
        
        if use_intelligent_search:
            # Use pre-decomposed queries if available, otherwise decompose now
            if pre_decomposed_queries:
                search_queries = pre_decomposed_queries
                print(f"🧠 Question {question_index + 1}: Using pre-decomposed queries ({len(search_queries)} components)")
            else:
                # Fall back to individual decomposition
                search_queries = self.search_engine.query_analyzer.process_multiple_queries([question])[0]
                print(f"🧠 Question {question_index + 1}: Query decomposed into {len(search_queries)} components")
            
            # Execute parallel searches for decomposed queries
            all_faiss_results = []
            all_bm25_results = []
            
            # Search each decomposed query in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(search_queries))) as executor:
                faiss_futures = []
                bm25_futures = []
                
                for sub_query in search_queries:
                    faiss_future = executor.submit(self.search_engine.find_relevant_chunks, sub_query, top_k=20)
                    bm25_future = executor.submit(self.bm25_engine.search, sub_query, top_k=20)
                    faiss_futures.append((faiss_future, sub_query))
                    bm25_futures.append((bm25_future, sub_query))
                
                # Collect FAISS results
                for future, sub_query in faiss_futures:
                    try:
                        results = future.result()
                        for result in results:
                            result['source_query'] = sub_query
                        all_faiss_results.extend(results)
                    except Exception as e:
                        print(f"   ⚠️ FAISS search failed for sub-query '{sub_query}': {e}")
                
                # Collect BM25 results
                for future, sub_query in bm25_futures:
                    try:
                        results = future.result()
                        for result in results:
                            result['source_query'] = sub_query
                        all_bm25_results.extend(results)
                    except Exception as e:
                        print(f"   ⚠️ BM25 search failed for sub-query '{sub_query}': {e}")
            
            faiss_results = all_faiss_results
            bm25_results = all_bm25_results
            
            # For intelligent search, we need to calculate timing
            faiss_time = 0.0  # Will be calculated from individual searches
            bm25_time = 0.0   # Will be calculated from individual searches
            # Calculate total time for intelligent search (analysis + search)
            search_end = time.time()
            parallel_time = search_end - analysis_start
            
        else:
            # Simple search - single query
            search_queries = [question]
            print(f"🧠 Question {question_index + 1}: Using simple search (no decomposition)")
            
            # Run FAISS and BM25 in parallel for single query
            parallel_start = time.time()
            
            def run_faiss_with_timing():
                faiss_start = time.time()
                results = self.search_engine.find_relevant_chunks(question, top_k=20)
                faiss_time = time.time() - faiss_start
                return results, faiss_time
            
            def run_bm25_with_timing():
                bm25_start = time.time()
                results = self.bm25_engine.search(question, top_k=20)
                bm25_time = time.time() - bm25_start
                return results, bm25_time
            
            # Execute FAISS and BM25 in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                faiss_future = executor.submit(run_faiss_with_timing)
                bm25_future = executor.submit(run_bm25_with_timing)
                
                faiss_results, faiss_time = faiss_future.result()
                bm25_results, bm25_time = bm25_future.result()
            
            parallel_time = time.time() - parallel_start
        
        analysis_time = time.time() - analysis_start
        
        # Log individual search results with their timings
        logger.log_faiss_results(question_index, faiss_results, faiss_time)
        logger.log_bm25_results(question_index, bm25_results, bm25_time)
        
        # Log search strategy and timing
        if use_intelligent_search:
            logger.hybrid_logger.info(
                f"INTELLIGENT SEARCH SUMMARY:\n"
                f"  • Query Analysis Time: {analysis_time:.3f}s\n"
                f"  • Sub-queries Generated: {len(search_queries)}\n"
                f"  • FAISS Results: {len(faiss_results)} chunks\n"
                f"  • BM25 Results: {len(bm25_results)} chunks\n"
                f"  • Total Search Time: {parallel_time:.3f}s\n"
                f"  • Search Strategy: Intelligent decomposition"
            )
        else:
            # Log parallel execution summary for simple search
            max_search_time = max(faiss_time, bm25_time)
            parallel_efficiency = (faiss_time + bm25_time) / parallel_time if parallel_time > 0 else 1
            
            logger.hybrid_logger.info(
                f"PARALLEL SEARCH SUMMARY:\n"
                f"  • FAISS Time: {faiss_time:.3f}s\n"
                f"  • BM25 Time: {bm25_time:.3f}s\n"
                f"  • Parallel Wall Time: {parallel_time:.3f}s\n"
                f"  • Sequential Would Take: {faiss_time + bm25_time:.3f}s\n"
                f"  • Time Saved: {(faiss_time + bm25_time) - parallel_time:.3f}s\n"
                f"  • Parallel Efficiency: {parallel_efficiency:.1f}x speedup\n"
                f"  • Bottleneck: {'FAISS' if faiss_time > bm25_time else 'BM25'}\n"
                f"  • Search Strategy: Simple parallel search"
            )
        
        # Step 2: Apply RRF fusion using the parallel results
        fusion_start = time.time()
        
        # Use RRF fusion directly instead of calling intelligent_hybrid_search again
        from .search.rrf_fusion import RRFFusion, HybridSearchEngine
        
        if not hasattr(self, '_hybrid_engine'):
            self._hybrid_engine = HybridSearchEngine()
        
        # Apply RRF fusion to the parallel results
        fused_results = self._hybrid_engine.rrf_fusion.fuse_rankings(
            bm25_results, faiss_results, final_top_k * 2
        )
        
        # Apply MMR if needed
        final_results = fused_results
        if len(fused_results) > final_top_k:
            final_results = self._hybrid_engine.mmr_processor.apply_mmr(fused_results, final_top_k)
        else:
            final_results = fused_results[:final_top_k]
        
        fusion_time = time.time() - fusion_start
        
        # Create search results metadata
        search_results = {
            'results': final_results,
            'metadata': {
                'query': question,
                'search_strategy': 'intelligent_decomposition' if use_intelligent_search else 'simple_parallel',
                'sub_queries': search_queries,
                'faiss_results_count': len(faiss_results),
                'bm25_results_count': len(bm25_results),
                'final_results_count': len(final_results),
                'timing': {
                    'query_analysis': analysis_time,
                    'faiss_search': faiss_time,
                    'bm25_search': bm25_time,
                    'parallel_search': parallel_time,
                    'fusion': fusion_time,
                    'total': parallel_time + fusion_time
                },
                'intelligent_search': {
                    'enabled': use_intelligent_search,
                    'sub_queries_count': len(search_queries),
                    'analysis_time': analysis_time,
                    'decomposition_ratio': len(search_queries) / 1 if len(search_queries) > 0 else 1
                },
                'parallel_execution': {
                    'sequential_time': faiss_time + bm25_time,
                    'parallel_time': parallel_time,
                    'time_saved': (faiss_time + bm25_time) - parallel_time,
                    'speedup': (faiss_time + bm25_time) / parallel_time if parallel_time > 0 else 1,
                    'bottleneck': 'FAISS' if faiss_time > bm25_time else 'BM25'
                }
            }
        }
        
        # Log RRF fusion details
        logger.log_rrf_fusion(question_index, search_results['metadata'], fusion_time)
        
        # Step 3: Generate answer
        hybrid_results = final_results
        if not hybrid_results:
            return None
        
        # Convert to format expected by query enhancer
        chunks_for_answer = []
        for result in hybrid_results:
            chunk_data = {
                'chunk_index': result['chunk_index'],
                'text': result['text'],
                'score': result.get('rrf_score', result.get('weighted_rrf_score', 0))
            }
            chunks_for_answer.append(chunk_data)
        
        # Generate answer
        response = self.search_engine.query_enhancer.get_most_relevant_chunk(question, chunks_for_answer)
        
        if response:
            # Enhance response with hybrid search metadata
            response['hybrid_search_metadata'] = search_results['metadata']
            response['hybrid_search_metadata']['results'] = hybrid_results
        
        return response 