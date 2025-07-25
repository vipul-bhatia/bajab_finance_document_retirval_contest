import google.generativeai as genai
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
- Insurance policy details (policy name, duration, coverage, claims)
- Locations/providers
- Eligibility criteria
- Coverage limitations
- Waiting periods

Return your response as comma-separated search queries. Make the queries natural and specific.

Example 1: "46M, knee surgery, Pune, 3-month policy":
46 year old male patient eligibility, knee surgery coverage insurance, Pune medical providers network, 3 month waiting period policy, orthopedic surgery claim requirements

Example 2: "46 year male has knee surgery in Pune in 2024 and the male has a 3-month policy":
46 year old male patient eligibility, knee surgery coverage insurance, Pune medical providers network, 3 month waiting period policy, orthopedic surgery claim requirements

Example 3: "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?":
grace period for premium payment, National Parivar Mediclaim Plus Policy document, premium payment rules in National Parivar Mediclaim Plus Policy, grace period clause in insurance policy

Your response (comma-separated queries only):"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Split by commas and clean up
            search_queries = [q.strip() for q in response_text.split(',') if q.strip()]
            
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
    
    def should_decompose_query(self, query: str) -> bool:
        """
        Use Gemini to decide whether a query should be decomposed or not
        
        Args:
            query: Single query to analyze
            
        Returns:
            True if query should be decomposed, False otherwise
        """
        prompt = f"""Analyze this query and decide if it should be broken down into multiple search components or kept as a single search.

Query: "{query}"

Guidelines:
- Simple, direct questions about specific policy terms should NOT be decomposed.
- Complex queries with multiple aspects (e.g., age + procedure + location + policy name) SHOULD be decomposed.
- Questions asking "what is," "how much," or "does policy cover" for a single item are usually simple.
- Queries containing multiple demographic, medical, or location factors are usually complex.
- A query about a term within a specifically named policy (e.g., "grace period in National Parivar Mediclaim Plus Policy") is complex because it requires finding the specific policy document and then the specific term within it.

Examples:
- "What is the grace period?" -> NO (Simple, direct question)
- "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?" -> YES (Complex, as it involves a specific policy context)
- "Does policy cover maternity?" -> NO (Simple coverage question)
- "46M, knee surgery, Pune, 3-month policy" -> YES (Multiple aspects: age, procedure, location, policy detail)
- "How does policy define Hospital?" -> NO (Simple definition question)

Based on these guidelines, should the query be decomposed? Answer only: YES or NO"""

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip().upper()
            return "YES" in answer
        except Exception as e:
            print(f"Error in decomposition decision: {e}")
            # Fallback: decompose if query has commas or multiple aspects
            return ',' in query or len(query.split()) > 8

    def process_multiple_queries(self, queries: List[str]) -> List[List[str]]:
        """
        Process multiple queries in parallel, deciding decomposition for each
        
        Args:
            queries: List of user queries
            
        Returns:
            List of search query lists (some single, some decomposed)
        """
        print(f"🧠 Analyzing {len(queries)} queries for decomposition...")
        
        # First, decide decomposition for all queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            decompose_futures = {
                executor.submit(self.should_decompose_query, query): i 
                for i, query in enumerate(queries)
            }
            
            decomposition_decisions = {}
            for future in concurrent.futures.as_completed(decompose_futures):
                query_index = decompose_futures[future]
                query = queries[query_index]
                try:
                    should_decompose = future.result()
                    decomposition_decisions[query_index] = should_decompose
                    status = "DECOMPOSE" if should_decompose else "KEEP SINGLE"
                    print(f"   • '{query[:50]}...' -> {status}")
                except Exception as e:
                    print(f"   ❌ Error analyzing '{query[:30]}...': {e}")
                    decomposition_decisions[query_index] = False
        
        # Initialize result list with correct size
        all_search_queries = [None] * len(queries)
        
        # Process queries that need decomposition
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            query_futures = {}
            
            for i, query in enumerate(queries):
                if decomposition_decisions.get(i, False):
                    # Decompose this query
                    future = executor.submit(self.analyze_and_decompose_query, query)
                    query_futures[future] = i
                else:
                    # Keep as single query
                    all_search_queries[i] = [query]
            
            # Collect decomposed queries
            for future in concurrent.futures.as_completed(query_futures):
                query_index = query_futures[future]
                try:
                    decomposed = future.result()
                    all_search_queries[query_index] = decomposed
                except Exception as e:
                    query = queries[query_index]
                    print(f"   ❌ Error decomposing '{query[:30]}...': {e}")
                    all_search_queries[query_index] = [query]  # Fallback to single
        
        return all_search_queries
    
 