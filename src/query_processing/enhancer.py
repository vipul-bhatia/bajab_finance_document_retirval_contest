import openai
import requests
import json
from typing import List, Dict, Any, Optional

def call_tool_get_data(url: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """Make a GET request and return JSON (or None on failure)."""
    print(f"🚀 Making GET request to: {url}")
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ Request Successful!")
            return response.json()
        print(f"❌ Request failed [{response.status_code}]: {response.text}")
    except Exception:
        pass
    return None


class QueryEnhancer:
    def __init__(self):
        self.client = openai.OpenAI()
        self.model = "gpt-4.1-mini-2025-04-14"

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

    def get_most_relevant_chunk(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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

8. **Final Self-Check:** Before providing the answer, verify that it:

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
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    functions=self.functions,
                    function_call="auto",
                    max_completion_tokens=1500,
                )
                msg = resp.choices[0].message

                # 4) Did the model request a function call?
                if hasattr(msg, "function_call") and msg.function_call is not None:
                    fn_name = msg.function_call.name
                    fn_args = json.loads(msg.function_call.arguments)

                    # 5) Execute the function
                    result = call_tool_get_data(
                        url=fn_args["url"],
                        timeout=fn_args.get("timeout", 60)
                    ) or {}

                    # 6) Append the assistant's function_call and our function result
                    messages.append(
                        {"role": "assistant", "content": None, "function_call": {
                            "name": fn_name,
                            "arguments": msg.function_call.arguments
                        }}
                    )
                    messages.append(
                        {"role": "function", "name": fn_name, "content": json.dumps(result)}
                    )

                    # Loop back and let the model decide the next step
                    continue

                # 7) No function_call → final answer
                final_answer = msg.content.strip()
                break

            return {
                "answer": final_answer,
                "source_chunks": [r.get("chunk_index") for r in search_results],
                "total_chunks_analyzed": len(search_results)
            }
        except Exception as e:
            print(f"❌ Error in get_most_relevant_chunk: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "source_chunks": [r.get("chunk_index") for r in search_results] if search_results else [],
                "total_chunks_analyzed": len(search_results) if search_results else 0
            }
