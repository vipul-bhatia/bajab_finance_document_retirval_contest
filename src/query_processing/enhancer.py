import google.generativeai as genai
from typing import List, Dict, Any, Optional


class QueryEnhancer:
    """Uses Gemini to enhance search results by selecting most relevant items"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def get_most_relevant_chunk(self, query: str, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use Gemini to provide a direct, concise, and user-friendly answer based on search results.
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

        prompt = f"""You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer a user's query with complete accuracy, based *strictly* on the provided text chunks.

**User Query:** "{query}"

**Retrieved Information:**
{results_text}

**Instructions:**
1.  **Analyze Thoroughly:** Carefully read all the provided text chunks to find the most relevant information(that completely answers the query) to answer the query.
2.  **Synthesize a Clear, User-Friendly Answer:** Aim for the conciseness of 1-2 sentences, but always prioritize providing a complete and accurate response over meeting a specific sentence count.
3.  **Direct, User-Friendly Tone:** Adopt a helpful, expert, human-like tone. **Crucially, do NOT use phrases like 'The provided text states,' 'According to the document,' or 'Based on the information.'** Just state the answer directly as a fact from the policy.
4.  **Be Factual, Precise and Absolute Fidelity to the Source:** Base your answer strictly on the details found in the `Retrieved Information`. Do not infer or add information not present in the text. If the context is insufficient, state that the answer cannot be found in the provided information.
5.  **Synthesize for Completeness:** Carefully analyze all provided text chunks. If multiple chunks contribute to the answer, you must synthesize them into a single, coherent, and comprehensive response. Ensure every part of the user's query is addressed. You *MUST NOT* omit any relevant details to make it shorter. Accuracy is more important than conciseness.

--- EXAMPLES ---

Chunk 1: "Grace Period means a period of thirty days after the Premium due date, for payment of the Premium."
Chunk 2: "During the grace period the policy shall be in force. Renewing within this period ensures you retain your continuity benefits, such as the waiting periods you have already served."

**Good Answer (Accurate, Complete, Direct):**
"You have a grace period of thirty days after the premium due date to make your payment. Your policy remains active during this time, and paying within this period ensures you do not lose any continuity benefits."

**Bad Answer (References the document):**
"According to the text, a grace period of thirty days is mentioned for premium payment."

**Bad Answer (Incomplete):**
"The grace period is thirty days." (This misses the crucial context about the policy remaining active and the continuity benefits.)

--- YOUR TASK ---

**Your Answer:**"""

        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text.strip()
            
            result = {
                'answer': answer_text,
                'source_chunks': [result['chunk_index'] for result in search_results],
                'total_chunks_analyzed': len(search_results)
            }
            
            print(f"🧠 Generated enhanced answer based on {len(search_results)} chunks.")
            
            return result
            
        except Exception as e:
            print(f"Error using Gemini for analysis: {e}")
            return {
                'answer': f"Unable to analyze the query due to an error: {str(e)}",
                'source_chunks': [result['chunk_index'] for result in search_results],
                'total_chunks_analyzed': len(search_results)
            }
    
 