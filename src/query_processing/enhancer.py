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
        
        prompt = f"""You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer the user's query with **complete accuracy**, using **only** the provided text chunks as your source of information.

**User Query:** "{query}"

**Retrieved Information:**
{results_text}

**Instructions:**

1. **Analyze Thoroughly & Synthesize:** Carefully review all the provided information. Identify every detail relevant to the query and combine these pieces into a single, coherent answer.

2. **Strictly Source-Based:** Base your answer **exclusively** on the given text. **Do not** use outside knowledge or make assumptions. If the query cannot be fully answered with the provided material, clearly state that the information is not available in the text.

3. **Direct Grounding & Precision:** Use exact wording or terminology verbatim for definitions, laws, or named principles (put those in quotation marks). When paraphrasing, preserve quantitative relationships precisely. Avoid vague rephrasings when a specific phrase is provided in the source.

4. **Complete & Specific Answer:** Aim for completeness. Address every part of the query. Include all relevant details, conditions, and exceptions mentioned in the text. Be specific — for example, reference exact article numbers, sections, or terms as given.

5. **Clear and Formal Tone:** Write the answer in a clear, straightforward manner, as if explaining to a colleague. Maintain the tone of the source material. **Do not** mention any "chunks," file names, or the retrieval process.

6. **Concise Presentation:** Keep your answer concise and focused. Aim for about **three - four sentences**, but if you find that you need more than that to fully answer the question, then you can add more sentences. These sentences should fully answer the question while omitting any unnecessary fluff or repetition.

7. **Internal Consistency Check:** For any answer involving mathematical or proportional relationships, instantiate the relation with a simple example in the answer to verify you are not misapplying it.

8. **Formatting:** Use plain text formatting. You may use numeric symbols (e.g., “2%”) and quotation marks when directly quoting source material.

9. **Final Self-Check:** Before providing the answer, verify that it:
   - Fully **answers the query** (all parts are addressed).
   - Is **100% accurate** and directly supported by the provided text.
   - Uses **exact phrasing** or close paraphrasing from the Retrieved Information text.
   - Reads smoothly on its own, without needing the source chunks.
   

*Example Query:* "What is the grace period for premium payment?"

✅ *Good Answer:* "You have 30 days after your premium due date to make the payment. During this grace period, your policy stays active, and paying within this time keeps your continuity benefits intact."

❌ *Poor Answer:* "According to chunk 1, there is a **grace period of 30 days** mentioned for premium payments."

Example 2: *“Is abortion covered?”*  
✅ Good Answer: "The policy covers lawful medical termination of pregnancy only on medical grounds or due to an accident. Voluntary termination within the first 12 weeks is not covered."

Example 3: *“If I change my religion, can the government stop me?”*  
✅ Good Answer: "Under Article 25, every person has the freedom of conscience and the right to freely profess, practice, and propagate religion, subject to public order, morality, and health."

❌ *Poor Answer(uses special characters):* "Under /Article 25/, *every* person has the "freedom" of conscience and the right to freely profess, practice, and propagate religion,/n/n subject to public order, morality, and health."

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
                        "content": "You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer the user's query with **complete accuracy**, using **only** the provided text chunks as your source of information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1
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
    
 