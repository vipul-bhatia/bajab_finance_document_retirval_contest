import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import json
import os

class AsyncQueryAnalyzer:
    """Async query analysis using OpenAI API with aiohttp for better I/O performance."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
        self.model = "gpt-4.1-mini-2025-04-14"
        
        # Connection pool settings
        self.connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=20,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        self.timeout = aiohttp.ClientTimeout(
            total=90,
            connect=10,
            sock_read=60
        )
        
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_api_call_async(self, messages: List[Dict], max_retries: int = 2) -> str:
        """
        Make async API call to OpenAI with retry logic and exponential backoff.
        
        Args:
            messages: Messages to send to the API
            max_retries: Maximum number of retry attempts
        
        Returns:
            Response text from the API
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.1
                }
                
                async with self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content'].strip()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API Error {response.status}: {error_text}")
                        
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = 1 * (2 ** attempt)  # Exponential backoff
                    print(f"   ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"   ❌ All {max_retries} attempts failed. Final error: {e}")
        
        raise last_exception
    
    async def analyze_and_decompose_query_async(self, query: str) -> List[str]:
        """
        Async version of query analysis and decomposition.
        
        Args:
            query: User query to analyze
            
        Returns:
            List of search queries (original or decomposed)
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
            messages = [
                {"role": "system", "content": "You are an expert query analyzer for document search systems."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self._make_api_call_async(messages)
            
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
            print(f"Error in query analysis after all retries: {e}. Falling back to original query.")
            return [query]
    
    async def _batch_analyze_and_decompose_async(self, queries: List[str]) -> List[List[str]]:
        """
        Async batch processing of multiple queries.
        
        Args:
            queries: List of queries to analyze
            
        Returns:
            List of decomposed query lists
        """
        prompt = f"""You are a sophisticated research analyst AI with enhanced query understanding capabilities. Your function is to analyze user intent and deconstruct queries into optimized search phrases for comprehensive information retrieval.

**STEP 1: INTENT CLASSIFICATION**
For each query, first classify its intent:
- **FACTUAL**: Direct fact lookup (What is X?, When does Y occur?)
- **PROCEDURAL**: Process or step-by-step explanation (How to do X?)
- **CONDITIONAL**: Scenario-based reasoning (If X happens, then what?)
- **COMPARATIVE**: Comparison between options (X vs Y, differences between)
- **ANALYTICAL**: Understanding relationships or implications (Why X?, What causes Y?)

**STEP 2: DECOMPOSITION STRATEGY**
Based on intent, apply appropriate decomposition:

**1. FACTUAL Queries:** Keep simple if single fact, decompose if multiple facts needed.
   * *Simple:* "What is the waiting period for cataract surgery?" -> `["waiting period for cataract surgery"]`
   * *Complex:* "What are the limits and exclusions for dental treatment?" -> `["dental treatment coverage limits", "dental treatment exclusions", "dental procedure waiting periods"]`

**2. PROCEDURAL Queries:** Break into process steps + requirements + exceptions.
   * *Example:* "How do I file a claim?" -> `["claim filing process steps", "required documents for claims", "claim submission timeline", "claim processing exceptions"]`

**3. CONDITIONAL Queries:** Decompose into condition + outcomes + exceptions + definitions.
   * *Example:* "If I'm hospitalized for 2 days, what's covered?" -> `["minimum hospitalization duration requirements", "coverage for short-term hospitalization", "exclusions for brief hospital stays", "definition of eligible hospitalization"]`

**4. COMPARATIVE Queries:** Break into individual components for comparison.
   * *Example:* "What's the difference between Plan A and Plan B benefits?" -> `["Plan A coverage benefits", "Plan B coverage benefits", "Plan A vs Plan B comparison", "differences in plan benefits"]`

**5. ANALYTICAL Queries:** Decompose into cause + effect + context + exceptions.
   * *Example:* "Why was my claim rejected?" -> `["common claim rejection reasons", "claim assessment criteria", "policy exclusions", "claim documentation requirements"]`

**ENHANCED CONTEXT GENERATION:**
For ANY query, always include:
- **Definitions** of key terms mentioned
- **Prerequisites** or conditions that apply
- **Exceptions** or limitations
- **Related processes** that might affect the answer
- **Cross-references** to connected topics

**DISAMBIGUATION FOCUS:**
- Resolve ambiguous pronouns (it, this, that, they)
- Clarify context-dependent terms
- Include related concepts that provide complete understanding
- Address potential misinterpretations

**RELATIONSHIP AWARENESS:**
- Include queries that find connected information
- Search for cause-effect relationships
- Look for procedural dependencies
- Find conditional variations

**OPTIMIZATION PRINCIPLES:**
* **Intent-Driven**: Tailor decomposition to query intent
* **Context-Complete**: Ensure all necessary context is captured
* **Disambiguation-Ready**: Include terms that resolve ambiguities
* **Relationship-Aware**: Connect related concepts and processes


**Input Format:** A JSON list of strings, where each string is a user query.
**Output Format:** You MUST return a JSON list of lists. Each inner list corresponds to a query from the input and contains the decomposed parts.

**Input:**
{json.dumps(queries, ensure_ascii=False)}

**Output (JSON):**
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert query analyzer for document search systems. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self._make_api_call_async(messages, max_retries=2)

            print("📥 DEBUG: Raw response from OpenAI:")
            print(f"{'='*50}")
            print(response_text)
            print(f"{'='*50}")
            
            # The model might return the JSON wrapped in markdown, so we extract it.
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            decomposed = json.loads(response_text.strip())
            
            if not isinstance(decomposed, list) or len(decomposed) != len(queries):
                raise ValueError("LLM output is not a list of the same length as the input queries.")

            # Sanitize the output to ensure it's a list of lists of strings
            result = []
            for item in decomposed:
                if isinstance(item, list):
                    result.append([str(q).strip() for q in item if str(q).strip()])
                elif isinstance(item, str):
                    result.append([item.strip()])
                else: # Fallback for unexpected types
                    result.append([queries[len(result)]])

            return result

        except Exception as e:
            print(f"Error in batch query analysis after all retries: {e}. Falling back to original queries.")
            return [[q] for q in queries]
    
    async def process_multiple_queries_async(self, queries: List[str]) -> List[List[str]]:
        """
        Async processing of multiple queries using batching.
        
        Args:
            queries: List of queries to process
            
        Returns:
            List of decomposed query lists
        """
        print(f"🧠 Analyzing and decomposing {len(queries)} queries (async batching mode)...")
        BATCH_SIZE = 30
        
        if len(queries) <= BATCH_SIZE:
            return await self._batch_analyze_and_decompose_async(queries)
        
        # Split into batches and process concurrently
        batches = [queries[i:i+BATCH_SIZE] for i in range(0, len(queries), BATCH_SIZE)]
        all_results = [[] for _ in range(len(queries))]
        
        # Process batches concurrently with semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent batch requests
        
        async def process_batch_with_semaphore(batch, batch_idx):
            async with semaphore:
                return await self._batch_analyze_and_decompose_async(batch), batch_idx
        
        # Create tasks for all batches
        tasks = [
            process_batch_with_semaphore(batch, batch_idx)
            for batch_idx, batch in enumerate(batches)
        ]
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"   ❌ Error processing batch: {result}")
                continue
            
            batch_decomposed, batch_idx = result
            batch_start = batch_idx * BATCH_SIZE
            
            for i, decomposed in enumerate(batch_decomposed):
                if batch_start + i < len(all_results):
                    all_results[batch_start + i] = decomposed
        
        # Fill any missing results with original queries
        for i, result in enumerate(all_results):
            if not result and i < len(queries):
                all_results[i] = [queries[i]]
        
        return all_results
    
    async def parallel_search_async(self, search_engine, search_queries: List[str], top_k_per_query: int = 5) -> List[Dict[str, Any]]:
        """
        Execute multiple searches in parallel asynchronously.
        
        Args:
            search_engine: Search engine instance
            search_queries: List of queries to search
            top_k_per_query: Number of results per query
            
        Returns:
            Combined and deduplicated search results
        """
        all_results = []
        
        print(f"🔍 Executing {len(search_queries)} parallel searches (async)...")
        
        # Create semaphore to limit concurrent searches
        semaphore = asyncio.Semaphore(8)  # Limit concurrent searches
        
        async def search_with_semaphore(query):
            async with semaphore:
                # Convert sync search to async using run_in_executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    search_engine.find_relevant_chunks, 
                    query, 
                    top_k_per_query
                )
        
        # Create tasks for all searches
        tasks = [search_with_semaphore(query) for query in search_queries]
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for query, results in zip(search_queries, search_results):
            if isinstance(results, Exception):
                print(f"   ❌ Query '{query}' failed: {results}")
                continue
            
            for result in results:
                result['source_query'] = query
            all_results.extend(results)
            print(f"   ✅ Query '{query}': {len(results)} results")
        
        # Remove duplicates
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


# Async wrapper functions for backward compatibility
async def analyze_query_async(query: str) -> List[str]:
    """Async wrapper for single query analysis."""
    async with AsyncQueryAnalyzer() as analyzer:
        return await analyzer.analyze_and_decompose_query_async(query)

async def process_queries_async(queries: List[str]) -> List[List[str]]:
    """Async wrapper for multiple query processing."""
    async with AsyncQueryAnalyzer() as analyzer:
        return await analyzer.process_multiple_queries_async(queries)

# Sync wrapper for compatibility with existing code
def process_queries_sync(queries: List[str]) -> List[List[str]]:
    """Sync wrapper around async query processing."""
    return asyncio.run(process_queries_async(queries))


# Performance testing
async def performance_test_analyzer():
    """Test performance of async vs sync query analysis."""
    test_queries = [
        "What is the waiting period for dental treatment?",
        "How do I file a claim for hospitalization?",
        "46M, diabetes, Mumbai, premium plan coverage",
        "Are maternity expenses covered under Plan B?",
        "What are the exclusions for pre-existing diseases?",
    ]
    
    print("🧪 Testing async query analysis performance...")
    
    start_time = time.perf_counter()
    async with AsyncQueryAnalyzer() as analyzer:
        results = await analyzer.process_multiple_queries_async(test_queries)
    async_time = time.perf_counter() - start_time
    
    print(f"📊 Async Results:")
    print(f"   • Time: {async_time:.2f}s")
    print(f"   • Queries processed: {len(results)}")
    print(f"   • Rate: {len(results) / async_time:.1f} queries/sec")
    
    for i, (original, decomposed) in enumerate(zip(test_queries, results)):
        print(f"   Query {i+1}: {original}")
        print(f"   Decomposed: {decomposed}")
        print()

if __name__ == "__main__":
    asyncio.run(performance_test_analyzer())
