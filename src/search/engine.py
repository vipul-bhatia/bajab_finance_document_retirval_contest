import torch
import time
from ..config import TOP_K, device
from ..embeddings import EmbeddingGenerator
from ..query_processing import QueryAnalyzer, QueryEnhancer
from typing import List
import concurrent.futures


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
                    future = executor.submit(self.query_analyzer.parallel_search, self, search_queries, 3)
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