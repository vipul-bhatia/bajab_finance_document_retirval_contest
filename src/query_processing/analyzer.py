import concurrent.futures
import time
from typing import List, Dict, Any
import json
import openai
import google.generativeai as genai

class QueryAnalyzer:
    """Intelligent query analysis using a single, unified model call to decompose complex queries."""
    
    def __init__(self):
        # Initialize OpenAI client - API key should be set in environment variables
        self.client = openai.OpenAI()
        self.model = "gpt-4.1-mini-2025-04-14"
        # self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _retry_with_backoff(self, func, max_retries=2, backoff_factor=1):
        """
        Execute a function with retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of attempts (default: 2)
            backoff_factor: Base delay between retries in seconds
        
        Returns:
            Result of the function or raises the last exception
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    delay = backoff_factor * (2 ** attempt)
                    print(f"   ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ All {max_retries} attempts failed. Final error: {e}")
        
        raise last_exception
    
    def analyze_and_decompose_query(self, query: str) -> List[str]:
        """
        Uses a single OpenAI call to analyze a query.
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

        def _make_api_call():
            # response = self.model.generate_content(prompt)
            # return response.text.strip()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert query analyzer for document search systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()

        try:
            response_text = self._retry_with_backoff(_make_api_call)
            
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
    
    def parallel_search(self, search_engine, search_queries: List[str], top_k_per_query: int = 5) -> List[Dict[str, Any]]:
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
    
    def _batch_analyze_and_decompose(self, queries: List[str]) -> List[List[str]]:
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
        
        def _make_batch_api_call():
            # response = self.model.generate_content(prompt)
            # response_text = response.text.strip()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert query analyzer for document search systems. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            response_text = response.choices[0].message.content.strip()

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

        try:
            result = self._retry_with_backoff(_make_batch_api_call)
            print(f"🧠 Batch decomposed {len(queries)} queries.")
            return result
            
        except Exception as e:
            print(f"Error in batch query analysis after all retries: {e}. Falling back to original queries.")
            return [[q] for q in queries]
    
    def process_multiple_queries(self, queries: List[str]) -> List[List[str]]:
        """
        Process multiple queries using batching: if <=30, send as one batch; if >30, split into batches and process concurrently.
        """
        print(f"🧠 Analyzing and decomposing {len(queries)} queries (batching mode)...")
        BATCH_SIZE = 30
        if len(queries) <= BATCH_SIZE:
            return self._batch_analyze_and_decompose(queries)
        # Split into batches and process concurrently
        batches = [queries[i:i+BATCH_SIZE] for i in range(0, len(queries), BATCH_SIZE)]
        all_results = [[] for _ in range(len(queries))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(25, len(batches))) as executor:
            future_to_batch_idx = {
                executor.submit(self._batch_analyze_and_decompose, batch): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                batch_start = batch_idx * BATCH_SIZE
                try:
                    batch_results = future.result()
                    for i, decomposed in enumerate(batch_results):
                        all_results[batch_start + i] = decomposed
                except Exception as e:
                    print(f"   ❌ Error processing batch {batch_idx+1}: {e}. Using original queries.")
                    for i in range(len(batches[batch_idx])):
                        all_results[batch_start + i] = [batches[batch_idx][i]]
        return all_results
    
 