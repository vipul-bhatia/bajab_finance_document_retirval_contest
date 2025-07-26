import google.generativeai as genai
import concurrent.futures
from typing import List, Dict, Any
from ..embeddings import EmbeddingGenerator


class QueryAnalyzer:
    """Intelligent query analysis using a single, unified model call to decompose complex queries."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def analyze_and_decompose_query(self, query: str) -> List[str]:
        """
        Uses a single Gemini call to analyze a query.
        - If the query is simple, it's returned as is.
        - If the query is complex, it's broken into searchable components.
        """
        prompt = f"""You are an expert at analyzing document search queries. Your task is to determine if a query is simple or complex and act accordingly.

User Query: "{query}"

**RULES:**
1.  **Analyze the query's structure.**
2.  **If the query is a single, direct question, it is SIMPLE.** Do NOT decompose it. Return the original query exactly as it is.
    - Simple questions often start with: "What is", "How does", "Is there", "Does the policy cover", "What is the waiting period for".
3.  **If the query contains multiple distinct concepts or conditions, it is COMPLEX.** Break it down into 3-5 focused, searchable components.
    - Complex queries often contain lists of items, or combine demographic info with medical procedures (e.g., "46M, knee surgery, Pune").

**Your output must be a single line containing either the original query (for simple cases) or a comma-separated list of new queries (for complex cases).**

--- EXAMPLES ---

**User Query:** "How does the policy define a 'Hospital'?"
**Your Response:** How does the policy define a 'Hospital'?

**User Query:** "What is the waiting period for cataract surgery?"
**Your Response:** What is the waiting period for cataract surgery?

**User Query:** "46M, knee surgery, Pune, 3-month policy"
**Your Response:** 46 year old male patient eligibility, knee surgery coverage insurance, Pune medical providers network, 3 month waiting period policy, orthopedic surgery claim requirements

**User Query:** "Are there sub-limits on room rent and ICU charges for Plan A?"
**Your Response:** sub-limits on room rent for Plan A, sub-limits on ICU charges for Plan A, Plan A room rent and ICU coverage details

--- YOUR TASK ---

**User Query:** "{query}"
**Your Response:**"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            search_queries = [q.strip() for q in response_text.split(',') if q.strip()]
            
            if not search_queries:
                search_queries = [query]
            
            if len(search_queries) > 1:
                print(f"🧠 Query decomposed into {len(search_queries)} components:")
                for i, q in enumerate(search_queries, 1):
                    print(f"   {i}. {q}")
            else:
                print(f"🧠 Query is simple, using as is: '{query}'")
             
            return search_queries
            
        except Exception as e:
            print(f"Error in query analysis: {e}. Falling back to original query.")
            return [query]
    
    def parallel_search(self, search_engine, search_queries: List[str], top_k_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Execute multiple searches in parallel and combine results.
        (This method remains largely the same but is called by the search engine)
        """
        all_results = []
        
        print(f"🔍 Executing {len(search_queries)} parallel searches...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(search_engine.find_relevant_chunks, query, top_k_per_query): query 
                for query in search_queries
            }
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    for result in results:
                        result['source_query'] = query
                    all_results.extend(results)
                    print(f"   ✅ Query '{query}': {len(results)} results")
                except Exception as e:
                    print(f"   ❌ Query '{query}' failed: {e}")
        
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            chunk_id = result['chunk_index']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 Combined results: {len(unique_results)} unique chunks from {len(all_results)} total results")
        
        return unique_results
    
    def process_multiple_queries(self, queries: List[str]) -> List[List[str]]:
        """
        Process multiple queries in parallel, using a single model call for each to analyze and decompose.
        """
        print(f"🧠 Analyzing and decomposing {len(queries)} queries in parallel...")
        
        all_search_queries = [[]] * len(queries)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {
                executor.submit(self.analyze_and_decompose_query, query): i 
                for i, query in enumerate(queries)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    decomposed_queries = future.result()
                    all_search_queries[index] = decomposed_queries
                except Exception as e:
                    original_query = queries[index]
                    print(f"   ❌ Error processing query '{original_query[:30]}...': {e}. Using original query.")
                    all_search_queries[index] = [original_query]
        
        return all_search_queries
    
 