import google.generativeai as genai
import json
import concurrent.futures
from typing import List, Dict, Any
from ..embeddings import EmbeddingGenerator


class QueryAnalyzer:
    """Intelligent query analysis using Gemini to break down complex queries"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def analyze_and_decompose_query(self, query: str) -> List[str]:
        """
        Use Gemini to analyze a complex query and break it into searchable components
        
        Args:
            query: Complex user query (e.g., "46M, knee surgery, Pune, 3-month policy")
            
        Returns:
            List of focused search queries
        """
        prompt = f"""You are an expert at analyzing document search queries. Your task is to break down a complex query into focused, searchable components that will help find relevant information in insurance/medical documents.

User Query: "{query}"

Please analyze this query and break it down into 3-5 focused search components. Each component should target a specific aspect that would be found in documents.

For insurance/medical queries, consider these aspects:
- Demographics (age, gender, location)
- Medical conditions/procedures 
- Insurance policy details (duration, coverage, claims)
- Locations/providers
- Eligibility criteria
- Coverage limitations
- Waiting periods

Return your response as a JSON array of strings, where each string is a focused search query. Make the queries natural and specific.

Example for "46M, knee surgery, Pune, 3-month policy":
[
    "46 year old male patient eligibility",
    "knee surgery coverage insurance",
    "Pune medical providers network",
    "3 month waiting period policy",
    "orthopedic surgery claim requirements"
]

Your response (JSON array only):"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            if response_text.startswith('[') and response_text.endswith(']'):
                search_queries = json.loads(response_text)
            else:
                # Fallback: try to find JSON in the response
                import re
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    search_queries = json.loads(json_match.group())
                else:
                    # Ultimate fallback: split the original query
                    search_queries = self._fallback_decompose(query)
            
            # Validate and clean queries
            search_queries = [q.strip() for q in search_queries if q.strip()]
            
            # Ensure we have at least the original query
            if not search_queries:
                search_queries = [query]
            
            print(f"🧠 Query decomposed into {len(search_queries)} components:")
            for i, q in enumerate(search_queries, 1):
                print(f"   {i}. {q}")
            
            return search_queries
            
        except Exception as e:
            print(f"Error in query analysis: {e}")
            return self._fallback_decompose(query)
    
    def _fallback_decompose(self, query: str) -> List[str]:
        """Fallback method to decompose query if Gemini fails"""
        # Simple keyword-based decomposition
        components = []
        
        # Split by common separators
        parts = query.replace(',', ' ').replace('/', ' ').replace('-', ' ').split()
        
        # Group related terms
        if len(parts) >= 2:
            # Take first half and second half
            mid = len(parts) // 2
            components.append(' '.join(parts[:mid]))
            components.append(' '.join(parts[mid:]))
        
        # Always include the original query
        components.append(query)
        
        return list(set(components))  # Remove duplicates
    
    def parallel_search(self, search_engine, search_queries: List[str], top_k_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Execute multiple searches in parallel and combine results
        
        Args:
            search_engine: The SearchEngine instance
            search_queries: List of search queries to execute
            top_k_per_query: Number of results per query
            
        Returns:
            Combined and deduplicated search results
        """
        all_results = []
        
        print(f"🔍 Executing {len(search_queries)} parallel searches...")
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(search_engine.find_relevant_chunks, query, top_k_per_query): query 
                for query in search_queries
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    # Add query context to results
                    for result in results:
                        result['source_query'] = query
                    all_results.extend(results)
                    print(f"   ✅ Query '{query}': {len(results)} results")
                except Exception as e:
                    print(f"   ❌ Query '{query}' failed: {e}")
        
        # Deduplicate results based on chunk_index
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            chunk_id = result['chunk_index']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Sort by score (highest first)
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 Combined results: {len(unique_results)} unique chunks from {len(all_results)} total results")
        
        return unique_results
    
 