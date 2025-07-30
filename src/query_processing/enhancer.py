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

        prompt = f"""You are an expert AI assistant specializing in information extraction and synthesis. Your primary goal is to answer a user's query with complete accuracy, based strictly on the provided text chunks.

**User Query:** "{query}"

**Retrieved Information:**
{results_text}

**Instructions:**
1. **Analyze & Synthesize:** Carefully analyze all provided information to craft a complete, accurate, and natural-sounding response that directly answers the query.

2. **Natural, Direct Tone:** Provide answers in a clear, conversational tone as if explaining to a colleague. Avoid mechanical phrases like "the text states" or "according to the document." Simply present the information naturally.

3. **Accuracy & Completeness:** 
   - Base your answer strictly on the provided information
   - Synthesize multiple relevant pieces into a coherent response
   - Do not add information or make assumptions beyond what's given
   - If you cannot find a complete answer, acknowledge this clearly

4. **Verification:** Before responding, verify that your answer:
   - Addresses all aspects of the query
   - Maintains complete accuracy to the source material
   - Sounds natural and conversational
   - Includes all relevant details while remaining clear and focused

Example Query: "What is the grace period for premium payment?"

✅ Good Answer:
"You have 30 days after your premium due date to make the payment. During this grace period, your policy stays active and paying within this time ensures you keep your continuity benefits."

❌ Poor Answer:
"According to chunk 1, there is a grace period of 30 days mentioned for premium payments."

Your response:"""

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
    
 