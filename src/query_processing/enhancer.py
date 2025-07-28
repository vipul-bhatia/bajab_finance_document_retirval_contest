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

        prompt = f"""You are an expert insurance policy analyst. Your task is to answer the user's query in a clear, user-friendly way, based *only* on the provided text chunks from a policy document.

**User Query:** "{query}"

**Retrieved Information:**
{results_text}

**Instructions:**
1.  **Analyze Thoroughly:** Carefully read all the provided text chunks to find the most relevant information to answer the query.
2.  **Synthesize a Clear, User-Friendly Answer:** Formulate a concise and helpful answer in 1-2 sentences.
3.  **Speak Directly to the User:** Adopt a helpful, expert tone. **Crucially, do NOT use phrases like 'The provided text states,' 'According to the document,' or 'Based on the information.'** Just state the answer directly as a fact from the policy.
4.  **Be Factual and Precise:** Base your answer strictly on the details found in the `Retrieved Information`. Do not infer or add information not present in the text. If the context is insufficient, state that the answer cannot be found in the provided information.

--- EXAMPLES ---

**Good Answer Example (Direct and User-Friendly):**
*Query:* "What is the grace period for premium payment?"
*Good Answer:* "A grace period of thirty days is provided for premium payment after the due date. This allows you to renew the policy without losing continuity benefits."

**Bad Answer Example (Sounds like a machine):**
*Query:* "What is the grace period for premium payment?"
*Bad Answer:* "According to the text, a grace period of thirty days is mentioned for premium payment."

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
    
 