import torch
import time
import faiss
from ..config import TOP_K, device
from ..embeddings import EmbeddingGenerator
from ..query_processing import QueryAnalyzer, QueryEnhancer
from typing import List
import concurrent.futures
from rank_bm25 import BM25Okapi


class SearchEngine:
    """Handles semantic search operations with intelligent query processing"""
    
    def __init__(self):
        self.faiss_index = None
        self.bm25_index = None
        self.document_chunks = []
        self.document_name = None
        self.query_analyzer = QueryAnalyzer()
        self.query_enhancer = QueryEnhancer()
    
    def load_search_indices(self, faiss_index, chunks: list, document_name: str):
        """Load FAISS index and create BM25 index."""
        self.faiss_index = faiss_index
        self.document_chunks = chunks
        self.document_name = document_name
        
        # Create BM25 index
        tokenized_corpus = [doc.split(" ") for doc in self.document_chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
    
    def find_relevant_chunks(self, query: str, top_k: int = TOP_K):
        """Find relevant document chunks using FAISS"""
        if self.faiss_index is None or not self.document_chunks:
            raise RuntimeError("FAISS index not initialized. Load index first.")
        
        q_vec = EmbeddingGenerator.get_embedding(query).cpu().numpy().reshape(1, -1)
        
        distances, indices = self.faiss_index.search(q_vec, top_k)
        
        results = []
        for dist, i in zip(distances[0], indices[0]):
            results.append({
                "chunk_index": i,
                "text": self.document_chunks[i],
                "score": 1 - dist  # Convert distance to similarity score
            })
        
        return results
        
    def _bm25_search(self, query: str, top_k: int = TOP_K):
        """Find relevant document chunks using BM25"""
        if self.bm25_index is None or not self.document_chunks:
            raise RuntimeError("BM25 index not initialized. Load index first.")
            
        tokenized_query = query.split(" ")
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        
        top_n = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n:
            results.append({
                "chunk_index": i,
                "text": self.document_chunks[i],
                "score": doc_scores[i]
            })
            
        return results

    def intelligent_search(self, query: str, return_all_results: bool = True):
        """
        Enhanced search using hybrid vector and keyword search.
        
        Args:
            query: User query
            return_all_results: If True, returns all unique results from parallel search
            
        Returns:
            All unique results from the hybrid search.
        """
        import time
        
        print(f"🔍 Starting intelligent search for: '{query}'")
        
        # Step 1: Execute parallel searches (Vector + BM25)
        parallel_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            vector_future = executor.submit(self.find_relevant_chunks, query, top_k=TOP_K)
            bm25_future = executor.submit(self._bm25_search, query, top_k=TOP_K)
            
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()

        # Step 2: Combine and re-rank results using Reciprocal Rank Fusion (RRF)
        combined_results = {}
        # RRF scoring constant
        k = 60
        for result in vector_results:
            if result["chunk_index"] not in combined_results:
                combined_results[result["chunk_index"]] = {"text": result["text"], "score": 0}
            combined_results[result["chunk_index"]]["score"] += 1 / (k + result["score"])

        for result in bm25_results:
            if result["chunk_index"] not in combined_results:
                combined_results[result["chunk_index"]] = {"text": result["text"], "score": 0}
            combined_results[result["chunk_index"]]["score"] += 1 / (k + result["score"])
            
        # Sort by RRF score
        sorted_results = sorted(combined_results.items(), key=lambda item: item[1]["score"], reverse=True)
        
        all_results = [{"chunk_index": r[0], "text": r[1]["text"], "score": r[1]["score"]} for r in sorted_results]
        
        parallel_time = time.time() - parallel_start
        
        print(f"📊 Returning all {len(all_results)} unique results from hybrid search")
        print(f"⏱️  Search Timing: {parallel_time:.2f}s total")
        
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
        
        # Step 1: Execute all searches in parallel
        search_start = time.time()
        all_search_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            search_futures = {executor.submit(self.intelligent_search, q): q for q in questions}
            
            results_by_question = {}
            for future in concurrent.futures.as_completed(search_futures):
                question = search_futures[future]
                try:
                    results = future.result()
                    results_by_question[question] = results
                    print(f"   ✅ Query '{question}': {len(results)} chunks found")
                except Exception as e:
                    print(f"   ❌ Query '{question}' search failed: {e}")
                    results_by_question[question] = []
            
            # Ensure results are in order
            for q in questions:
                all_search_results.append(results_by_question.get(q, []))
        
        search_time = time.time() - search_start
        
        # Step 2: Generate answers for all queries in parallel
        answer_start = time.time()
        answers = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            answer_futures = {}
            for i, (question, search_results) in enumerate(zip(questions, all_search_results)):
                if search_results:
                    future = executor.submit(self.query_enhancer.get_most_relevant_chunk, question, search_results)
                    answer_futures[future] = i
                else:
                    # Pre-fill answer for questions with no results
                    answers.append((i, f"No relevant information found for this query."))

            # Collect answers in order
            answers_by_index = {}
            for future in concurrent.futures.as_completed(answer_futures):
                query_index = answer_futures[future]
                try:
                    response = future.result()
                    answer = response.get('answer', 'Unable to generate answer.') if response else 'Unable to generate answer.'
                    answers_by_index[query_index] = answer
                    print(f"   ✅ Answer {query_index + 1}: Generated")
                except Exception as e:
                    print(f"   ❌ Answer {query_index + 1} generation failed: {e}")
                    answers_by_index[query_index] = f"Error generating answer: {str(e)}"
            
            # Add pre-filled answers
            for i, answer in answers:
                answers_by_index[i] = answer

            # Combine all answers in the correct order
            final_answers = [answers_by_index[i] for i in sorted(answers_by_index.keys())]

        answer_time = time.time() - answer_start
        total_time = time.time() - total_start
        
        print(f"\n📊 BATCH PROCESSING COMPLETE:")
        print(f"   • Total Queries: {len(questions)}")
        print(f"   • Successful Answers: {len([a for a in final_answers if not a.startswith('Error') and not a.startswith('No relevant') and not a.startswith('Unable')])}")
        print(f"\n⏱️  TIMING:")
        print(f"   • Parallel Search: {search_time:.2f}s")
        print(f"   • Answer Generation: {answer_time:.2f}s")
        print(f"   • Total Time: {total_time:.2f}s")
        
        return final_answers 