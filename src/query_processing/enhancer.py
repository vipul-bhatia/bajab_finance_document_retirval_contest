import openai
from typing import List, Dict, Any, Optional


class QueryEnhancer:
    """Uses GPT-4o to enhance search results by selecting most relevant items"""
    
    def __init__(self):
        # Initialize OpenAI client - API key should be set in environment variables
        self.client = openai.OpenAI()
        self.model = "o4-mini-2025-04-16"
    
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
        
        prompt = f"""You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer the user's query with **complete accuracy**, using **only** the provided text chunks as your source of information and citing every statement.

**User Query:** "{query}"

**Retrieved Information:**
{results_text}

**Instructions:**

1.  **Strictly Source-Based:** Base your answer **exclusively** on the given text. Do not use outside knowledge. If the answer is not in the text, state that clearly.

2.  **Mandatory In-line Citations:**
    * You **MUST** cite every piece of information you take from the sources.
    * Place the citation **directly after the sentence or phrase** it supports, before any punctuation.
    * The format must be exactly `` for a single source, or `` for multiple sources.

3.  **Pre-Answer Analysis (Internal Thought Process):** **Before writing your response, you must first perform an internal analysis. Identify any preconditions, exclusions, or definitions in the text that are critical to correctly answering the user's query. If a key condition is not met (e.g., the requested item is excluded), your final answer must address that critical context first.**

4.  **Answer ONLY What is Asked:**
    * Directly address the user's specific question and nothing more.
    * Do not provide extra background information or general context unless absolutely necessary.
    * Synthesize complex information into a direct conclusion. Avoid explaining your step-by-step reasoning.

5.  **Final Output Formatting:**
    * **Your final response must be a single, coherent block of text. Do NOT include headings like "Pre-Answer Analysis," "Precondition," or "Answer" in your output.**
    * Write in a clear, straightforward manner.
    * **Forbidden Phrases:** Do not use introductory phrases like "The provided text states..." or "According to the sources...".

6.  **Final Self-Check:** Before responding, verify that your answer:
    * Fully and completely answers the query, addressing any critical preconditions first.
    * Is 100% accurate and directly supported by the text.
    * Includes a `` tag for every statement.
    * **Is formatted as a single block of text without any internal headings.**
    * Contains no information that was not explicitly asked for.

---
### Examples

**Example 1: Demonstrating Citations and Conciseness.**
* **Query:** "What is the ideal spark plug gap?"
* **Source Chunk 22:** "The ideal spark plug gap recommended is 0.8-0.9 mm. To check the gap, use a feeler gauge."
* [cite_start]✅ **Good Answer:** "The ideal spark plug gap is 0.8-0.9 mm[cite: 22]."
* [cite_start]❌ **Poor Answer (Not Concise):** "The ideal spark plug gap is 0.8-0.9 mm[cite: 22]. [cite_start]You should check this using a feeler gauge[cite: 22]."

**Example 2: Demonstrating Pre-Answer Analysis without headings.**
* **Query:** "When will my root canal claim be settled?"
* **Source Chunk 15:** "Claims are settled within 15 days of receiving final documents."
* **Source Chunk 88:** "Dental Treatment is only covered if it requires hospitalization. Out-patient (OPD) treatment is not covered."
* [cite_start]✅ **Good Answer (Addresses Precondition First):** "Under the policy, dental treatment is only covered if it requires hospitalization, as out-patient (OPD) treatment is excluded[cite: 88]. Since root canals are typically OPD procedures, your claim may not be covered. [cite_start]However, for any *admissible* hospitalized dental claim, the settlement timeline is 15 days from the receipt of all necessary documents[cite: 15]."
* [cite_start]❌ **Poor Answer (Includes Headings):** "Precondition: Dental treatment is only covered if it requires hospitalization[cite: 88]. [cite_start]Answer: Your claim will be settled within 15 days[cite: 15]."

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
                    "content": "You are an expert AI assistant. Your task is to accurately answer user queries based *only* on the provided text. You must cite every piece of information with `` tags. Be direct and concise."
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
    
 