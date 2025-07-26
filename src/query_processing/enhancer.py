import google.generativeai as genai
from typing import List, Dict, Any, Optional


class QueryEnhancer:
    """Uses Gemini to enhance search results by selecting most relevant items"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def get_most_relevant_chunk(self, query: str, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use Gemini to provide a direct, concise answer based on search results
        
        Args:
            query: Original user query
            search_results: List of search results with text, score, chunk_index
            
        Returns:
            Simple response with direct answer
        """
        if not search_results:
            return None
        
        # Format results for Gemini analysis
        results_text = ""
        for i, result in enumerate(search_results, 1):
            results_text += f"\n--- Chunk {result['chunk_index']} ---\n"
            results_text += f"{result['text']}\n"
            if 'source_query' in result:
                results_text += f"[Found via: {result['source_query']}]\n"
            results_text += "\n"

        # print(f"Results texttttt: {results_text}")
        
        prompt = f"""Based on the document information, provide a very brief answer to the user's query.

User Query: "{query}"

Retrieved Information:
{results_text}

Instructions:
- Answer in 1-2 lines maximum
- Be direct and factual
- Include only essential details (amounts, timeframes, yes/no)
- No explanations or elaborations

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text.strip()
            
            # Return simple response format
            result = {
                'answer': answer_text,
                'source_chunks': [result['chunk_index'] for result in search_results],
                'total_chunks_analyzed': len(search_results)
            }
            
            print(f"🧠 Generated concise answer based on {len(search_results)} chunks")
            
            return result
            
        except Exception as e:
            print(f"Error using Gemini for analysis: {e}")
            # Fallback to basic response
            return {
                'answer': f"Unable to analyze the query due to an error: {str(e)}",
                'source_chunks': [result['chunk_index'] for result in search_results],
                'total_chunks_analyzed': len(search_results)
            }
    
 