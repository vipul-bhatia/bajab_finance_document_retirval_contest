import openai
from typing import List, Dict, Any, Optional


class QueryEnhancer:
    """Uses GPT-4o to enhance search results by selecting most relevant items"""
    
    def __init__(self):
        # Initialize OpenAI client - API key should be set in environment variables
        self.client = openai.OpenAI()
        self.model = "gpt-4.1-mini-2025-04-14"
        # self.model = "gpt-5-mini-2025-08-07"
    
    def get_most_relevant_chunk(self, query: str, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use GPT-4o to provide a direct, concise, and user-friendly answer based on search results.
        """
        if not search_results:
            return None
        
        results_text = ""
        for i, result in enumerate(search_results, 1):
            results_text += f"Context: {result['text']}\n"
            if 'source_query' in result and result['source_query'] != query:
                results_text += f"[Found via sub-query: '{result['source_query']}']\n"
            results_text += "\n"

#         prompt = f"""You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer a user's query with complete accuracy, based strictly on the provided text chunks.

# **User Query:** "{query}"

# **Retrieved Information:**
# {results_text}

# **Instructions:**
# 1. **Analyze & Synthesize:** Carefully analyze all provided information to craft a *COMPLETE*, *ACCURATE*, and natural-sounding response that directly answers the query.

# 2. **Natural, Direct Tone:** Provide answers in a clear, conversational tone as if explaining to a colleague. Avoid mechanical phrases like "the text states" or "according to the document." Simply present the information naturally.

# 3. **Accuracy & Completeness:** 
#    - Base your answer strictly on the provided information
#    - Synthesize multiple relevant pieces into a coherent response
#    - Do not add information or make assumptions beyond what's given
#    - If you cannot find a complete answer, acknowledge this clearly

# 4. **Verification:** Before responding, verify that your answer:
#    - Addresses all aspects of the query
#    - Maintains complete accuracy to the source material
#    - Sounds natural and conversational
#    - Includes all relevant details while remaining clear and focused

# 5. **Length of the answer:** The answer must be concise. It should be of approx 2 sentences ensuring *completeness* and factual accuracy without unnecessary elaboration.

# Example Query: "What is the grace period for premium payment?"

# ✅ Good Answer:
# "You have 30 days after your premium due date to make the payment. During this grace period, your policy stays active and paying within this time ensures you keep your continuity benefits."

# ❌ Poor Answer:
# "According to chunk 1, there is a grace period of 30 days mentioned for premium payments."

# Your response:"""
        
# This is the prompt for your get_most_relevant_chunk method
        prompt = f"""

You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer the user's query with complete accuracy, using only the provided text chunks as your source of information.

User Query: "{query}"

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

* **Forbidden Phrases:** **Do not** use introductory phrases that refer to the sources, such as "The provided text states..." or "According to the sources...".

8. **Final Self-Check:** Before providing the answer, verify that it:

* Fully **answers the query**.

* Is **100% accurate** and directly supported by the provided text.

* **Accurately represents the author's own methodology and reasoning, without adding outside information or modern interpretations.**

* Reads smoothly on its own.

* When answering arithmetic questions, follow the exact calculation method shown in the chunks and context, even if it differs from standard mathematics , if no imformation is provided in chunks do normal calculation according to standard maths(Exmaple - 1 + 1 = 2). Use the same pattern or rule demonstrated, without explaining(critical no need for explanation) or correcting the approach.

Example Query: "What is the grace period for premium payment?"

✅ Good Answer: "You have 30 days after your premium due date to make the payment. During this grace period, your policy stays active, and paying within this time keeps your continuity benefits intact."

❌ Poor Answer: "According to chunk 1, there is a grace period of 30 days mentioned for premium payments."

Example 2: “Is abortion covered?”

✅ Good Answer: "The policy covers lawful medical termination of pregnancy only on medical grounds or due to an accident. Voluntary termination within the first 12 weeks is not covered."

Example 3: “If I change my religion, can the government stop me?”

✅ Good Answer: "Under Article 25, every person has the freedom of conscience and the right to freely profess, practice, and propagate religion, subject to public order, morality, and health."

❌ Poor Answer(uses special characters): "Under /Article 25/, every person has the "freedom" of conscience and the right to freely profess, practice, and propagate religion,/n/n subject to public order, morality, and health."

Your response:

"""




        # print("--------------------------------")
        # print("--------------------------------")
        # print(f"query: {query}")
        # print("--------------------------------")
        # print("--------------------------------")
        # print(len(results_text))
        # print(f"chuncks input: {results_text}")
        # print("--------------------------------")
        # print("--------------------------------")




        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI assistant. Your task is to accurately answer user queries based *only* on the provided text. Be direct and concise."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
                max_completion_tokens=1500,
                # temperature=0.0
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            result = {
                'answer': answer_text,
                'source_chunks': [result['chunk_index'] for result in search_results],
                'total_chunks_analyzed': len(search_results)
            }
            
            print(f"🧠 Generated enhanced answer based on {len(search_results)} chunks.")
            
            return result
            
        except Exception as e:
            print(f"Error using GPT-4o for analysis: {e}")
            return {
                'answer': f"Unable to analyze the query due to an error: {str(e)}",
                'source_chunks': [result['chunk_index'] for result in search_results],
                'total_chunks_analyzed': len(search_results)
            }
    
 