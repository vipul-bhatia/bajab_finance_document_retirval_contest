import aiohttp
import asyncio
import json
from typing import List, Dict, Any, Optional
import os

async def call_tool_get_data_async(session: aiohttp.ClientSession, url: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """Async version of the tool to make GET requests."""
    print(f"🚀 Making async GET request to: {url}")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                print(f"✅ Request Successful!")
                return await response.json()
            print(f"❌ Request failed [{response.status}]: {await response.text()}")
    except Exception as e:
        print(f"❌ Request exception: {e}")
    return None


class AsyncQueryEnhancer:
    """Async query enhancement using OpenAI API with aiohttp for better I/O performance."""
    
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
            total=120,  # Longer timeout for enhancement
            connect=10,
            sock_read=90
        )
        
        self.session = None
        
        # Function tools for API calls
        self.functions = [
            {
                "name": "call_tool_get_data",
                "description": "Make a GET HTTP request to a given URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "API endpoint URL"},
                        "timeout": {"type": "integer", "description": "Request timeout in seconds"}
                    },
                    "required": ["url"]
                }
            }
        ]
    
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
    
    async def _make_chat_completion_async(self, messages: List[Dict], functions: Optional[List[Dict]] = None, function_call: str = "auto") -> Dict:
        """
        Make async chat completion request to OpenAI API.
        
        Args:
            messages: Messages for the conversation
            functions: Available functions for the model
            function_call: Function call mode
            
        Returns:
            Response from OpenAI API
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 1500,
        }
        
        if functions:
            payload["functions"] = functions
            payload["function_call"] = function_call
        
        async with self.session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"OpenAI API Error {response.status}: {error_text}")
    
    async def get_most_relevant_chunk_async(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Async version of get_most_relevant_chunk with function calling support.
        
        Args:
            query: User query
            search_results: List of search results from the search engine
            
        Returns:
            Dictionary with answer, source chunks, and metadata
        """
        try:
            if not search_results:
                return {"answer": None, "source_chunks": [], "total_chunks_analyzed": 0}

            # 1) Build the combined context
            results_text = ""
            for i, r in enumerate(search_results, 1):
                results_text += f"Context {i}: {r['text']}\n"
                if r.get("source_query") and r["source_query"] != query:
                    results_text += f"[via sub-query: {r['source_query']}]\n"
                results_text += "\n"

            system_msg = {
                "role": "system",
                "content": "You are an expert AI assistant specializing in information extraction."
            }
            user_msg = {
                "role": "user",
                "content": f"""User Query: "{query}"

Retrieved Information:
{results_text}

**Instructions:**
1. **Analyze Thoroughly & Synthesize:** Carefully review all the provided information. Identify every detail relevant to the query and combine these pieces into a single, coherent answer.

2. **Strictly Source-Based:** Base your answer **exclusively** on the given text. **Do not** use outside knowledge or make assumptions. **Your response must reflect only what the author has written, not what is known from modern knowledge on the topic.** If the query cannot be fully answered with the provided material, clearly state that the information is not available in the text.

3. **Describe the Author's Method:** **When explaining a proof, derivation, or argument, you must describe the specific method and reasoning used *by the author in the provided text*. Do not substitute, supplement, or append modern or alternative explanations, even if they lead to the same conclusion.**

4. **Direct Grounding & Precision:** Use exact wording or terminology verbatim for definitions, laws, or named principles (put those in quotation marks). When paraphrasing, preserve quantitative relationships precisely.

5. **Complete & Specific Answer:** Aim for completeness. Address every part of the query. Include all relevant details, conditions, and exceptions mentioned in the text.

6. **Concise but Complete Presentation:** Your primary goal is to provide a full and complete answer. Be as concise as possible ***without sacrificing completeness or accuracy***. If a complete answer requires more than a few sentences, that is acceptable.

7. **Clear Tone and Professional Phrasing:** Write the answer in a clear, straightforward manner.

* **Strictly Forbidden Phrases:** **Do not** use introductory phrases that refer to the sources or the chunks, such as "The provided text states..." or "According to the sources..." or "According to chunk 43". Mentioning about the source or chunk number in the answer is strictly forbidden.

8. **Language of Response:** If the user query or the retrieved information is primarily in a language other than English (e.g., Malayalam), respond in that language. Provide concise, direct answers in that language.

9. **Final Self-Check:** Before providing the answer, verify that it:

* Fully **answers the query**.

* Is **100% accurate** and directly supported by the provided text.

* **Accurately represents the author's own methodology and reasoning, without adding outside information or modern interpretations.**

* Reads smoothly on its own.

* When answering arithmetic questions, follow the exact calculation method shown in the chunks and context, even if it differs from standard mathematics , if no imformation is provided in chunks do normal calculation according to standard maths(Exmaple - 1 + 1 = 2). Use the same pattern or rule demonstrated, without explaining(critical no need for explanation) or correcting the approach.

***Use case specific(only if calling an endpoint is required to get the answer):***
If retrieving the answer requires calling an endpoint, use the tool call_tool_get_data.
You may call it multiple times. Understand all steps, then call to get the answer.
while giving the final answer, do not mention any thing about the source or chunk number in the answer. Just reply with what is asked with a short description.
***End of Use case specific part***


Example Query: "What is the grace period for premium payment?"

✅ Good Answer: "You have 30 days after your premium due date to make the payment. During this grace period, your policy stays active, and paying within this time keeps your continuity benefits intact."

❌ Poor Answer: "According to chunk 1, there is a grace period of 30 days mentioned for premium payments."

Example 2: "Is abortion covered?"

✅ Good Answer: "The policy covers lawful medical termination of pregnancy only on medical grounds or due to an accident. Voluntary termination within the first 12 weeks is not covered."

Example 3: "If I change my religion, can the government stop me?"

✅ Good Answer: "Under Article 25, every person has the freedom of conscience and the right to freely profess, practice, and propagate religion, subject to public order, morality, and health."

❌ Poor Answer(uses special characters, mentions the source): "Under /Article 25/, every person has the "freedom" of conscience and the right to freely profess, practice, and propagate religion,/n/n subject to public order, morality, and health.(Chunk 43)"

Your response:
"""
            }

            # 2) Initialize the message history
            messages = [system_msg, user_msg]

            # 3) Loop until we get a text response (no function_call)
            while True:
                resp_data = await self._make_chat_completion_async(
                    messages, 
                    functions=self.functions, 
                    function_call="auto"
                )
                
                msg = resp_data['choices'][0]['message']

                # 4) Did the model request a function call?
                if 'function_call' in msg and msg['function_call'] is not None:
                    fn_name = msg['function_call']['name']
                    fn_args = json.loads(msg['function_call']['arguments'])

                    # 5) Execute the function asynchronously
                    if fn_name == "call_tool_get_data":
                        result = await call_tool_get_data_async(
                            self.session,
                            url=fn_args["url"],
                            timeout=fn_args.get("timeout", 60)
                        ) or {}
                    else:
                        result = {}

                    # 6) Append the assistant's function_call and our function result
                    messages.append({
                        "role": "assistant", 
                        "content": None, 
                        "function_call": {
                            "name": fn_name,
                            "arguments": msg['function_call']['arguments']
                        }
                    })
                    messages.append({
                        "role": "function", 
                        "name": fn_name, 
                        "content": json.dumps(result)
                    })

                    # Loop back and let the model decide the next step
                    continue

                # 7) No function_call → final answer
                final_answer = msg.get('content', '').strip()
                break

            return {
                "answer": final_answer,
                "source_chunks": [r.get("chunk_index") for r in search_results],
                "total_chunks_analyzed": len(search_results)
            }
            
        except Exception as e:
            print(f"❌ Error in async get_most_relevant_chunk: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "source_chunks": [r.get("chunk_index") for r in search_results] if search_results else [],
                "total_chunks_analyzed": len(search_results) if search_results else 0
            }
    
    async def process_multiple_enhancements_async(
        self,
        queries_and_results: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple query enhancements in parallel.
        
        Args:
            queries_and_results: List of (query, search_results) tuples
            
        Returns:
            List of enhancement results
        """
        print(f"🧠 Processing {len(queries_and_results)} enhancements in parallel (async)...")
        
        # Create semaphore to limit concurrent enhancements
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent enhancements
        
        async def enhance_with_semaphore(query, search_results):
            async with semaphore:
                return await self.get_most_relevant_chunk_async(query, search_results)
        
        # Create tasks for all enhancements
        tasks = [
            enhance_with_semaphore(query, search_results)
            for query, search_results in queries_and_results
        ]
        
        # Execute enhancements concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"   ❌ Enhancement {i+1} failed: {result}")
                final_results.append({
                    "answer": f"Error processing query: {str(result)}",
                    "source_chunks": [],
                    "total_chunks_analyzed": 0
                })
            else:
                final_results.append(result)
                print(f"   ✅ Enhancement {i+1} completed")
        
        return final_results


# Async wrapper functions for backward compatibility
async def enhance_query_async(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Async wrapper for single query enhancement."""
    async with AsyncQueryEnhancer() as enhancer:
        return await enhancer.get_most_relevant_chunk_async(query, search_results)

async def enhance_multiple_queries_async(queries_and_results: List[tuple]) -> List[Dict[str, Any]]:
    """Async wrapper for multiple query enhancements."""
    async with AsyncQueryEnhancer() as enhancer:
        return await enhancer.process_multiple_enhancements_async(queries_and_results)

# Sync wrapper for compatibility with existing code
def enhance_query_sync(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sync wrapper around async query enhancement."""
    return asyncio.run(enhance_query_async(query, search_results))


# Performance testing
async def performance_test_enhancer():
    """Test performance of async query enhancement."""
    # Mock search results for testing
    mock_search_results = [
        {
            "chunk_index": 1,
            "text": "The insurance policy covers medical expenses up to the sum insured amount. Waiting periods apply for certain conditions.",
            "score": 0.95,
            "source_query": "insurance coverage"
        },
        {
            "chunk_index": 2,
            "text": "Pre-existing diseases have a waiting period of 2 years before coverage begins. Emergency treatments are covered immediately.",
            "score": 0.88,
            "source_query": "waiting period"
        }
    ]
    
    test_queries = [
        "What is the waiting period for pre-existing diseases?",
        "How much coverage does the policy provide?",
        "Are emergency treatments covered immediately?",
    ]
    
    queries_and_results = [(query, mock_search_results) for query in test_queries]
    
    print("🧪 Testing async query enhancement performance...")
    
    import time
    start_time = time.perf_counter()
    async with AsyncQueryEnhancer() as enhancer:
        results = await enhancer.process_multiple_enhancements_async(queries_and_results)
    async_time = time.perf_counter() - start_time
    
    print(f"📊 Async Enhancement Results:")
    print(f"   • Time: {async_time:.2f}s")
    print(f"   • Enhancements: {len(results)}")
    print(f"   • Rate: {len(results) / async_time:.1f} enhancements/sec")
    
    for i, (query, result) in enumerate(zip(test_queries, results)):
        print(f"\n   Query {i+1}: {query}")
        print(f"   Answer: {result.get('answer', 'No answer')[:100]}...")


if __name__ == "__main__":
    asyncio.run(performance_test_enhancer())
