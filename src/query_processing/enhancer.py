import openai
from typing import List, Dict, Any, Optional


class QueryEnhancer:
    """Uses GPT-4o to enhance search results by selecting most relevant items"""
    
    def __init__(self):
        # Initialize OpenAI client - API key should be set in environment variables
        self.client = openai.OpenAI()
        self.model = "gpt-4.1-mini-2025-04-14"
    
    def get_most_relevant_chunk(self, query: str, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use GPT-4o to provide a direct, concise, and user-friendly answer based on search results.
        """
        if not search_results:
            return None
        
        results_text = ""
        for i, result in enumerate(search_results, 1):
            results_text += f"\n--- Chunk {result['chunk_index']} ---\n"
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
        prompt = f"""You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer the user's query with **complete accuracy**, using **only** the provided text chunks as your source of information.

**User Query:** "{query}"

**Retrieved Information:**
{results_text}

**Instructions:**

1.  **Strictly Source-Based:** Base your answer **exclusively** on the given text. Do not use outside knowledge. If the answer is not in the text, you must state that the information is not available.

2.  **Answer ONLY What is Asked:**
    * **This is the most important rule.** Directly address the user's specific question and nothing more.
    * Do not provide extra background information, related procedures, or general context unless it is absolutely necessary to answer the specific question.
    * **Synthesize complex information into a direct conclusion.** Avoid showing your step-by-step reasoning unless the query explicitly asks for an explanation.

3.  **Prioritize Completeness, then Conciseness:** Your primary goal is a **complete and accurate** answer that addresses all parts of the query. Once that is achieved, the answer must be as **concise** as possible. Do not add unnecessary words or repeat information.

4.  **Clear Tone and Professional Phrasing:** Write the answer in a clear, straightforward manner.
    * **Forbidden Phrases:** Do not use introductory phrases that refer to the sources, such as "The provided text states...", "According to the sources...", or "Based on the information given...". The answer should be a direct statement of fact.

5.  **Final Self-Check:** Before providing the answer, verify that it:
    * Fully and completely answers the query.
    * Is 100% accurate and directly supported by the provided text.
    * **Contains no information that was not explicitly asked for.**
    * Is as concise and direct as possible.

---
### Examples

**Example 1: A question asking for a specific value (Demonstrates Conciseness).**
* **Query:** "What is the ideal spark plug gap recommended?"
* **Source Text:** "The ideal spark plug gap recommended is 0.8-0.9 mm. To check the gap, use a feeler gauge. The washer must be in good condition before tightening."
* ✅ **Good Answer:** "The ideal spark plug gap is 0.8-0.9 mm."
* ❌ **Poor Answer (Not Concise):** "The ideal spark plug gap recommended is 0.8-0.9 mm. You should check this using a wire-type feeler gauge, and if adjustment is necessary, the side electrode should be carefully bent. The spark plug washer must also be in good condition."

**Example 2: A yes/no question requiring synthesis (Demonstrates Synthesizing to a Conclusion).**
* **Query:** "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?"
* **Source Text:** "Hydrocele is listed under the 'Specific Disease/Procedure Waiting Period' with a two-year (24 months) waiting period. The waiting period applies from the date of inception of the first policy and requires continuous coverage."
* ✅ **Good Answer:** "Yes, you can raise a claim. Your 6 years of continuous coverage exceeds the 24-month waiting period for Hydrocele."
* ❌ **Poor Answer (Shows "Too Much Work"):** "To raise a claim for Hydrocele, you must complete a waiting period of 24 months. The waiting period requires continuous coverage. Since you have been a customer for 6 years, which is longer than 24 months, you are eligible to raise a claim."

---

**Your response:**
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
    
 