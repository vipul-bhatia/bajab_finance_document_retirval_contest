import google.generativeai as genai
from typing import List, Dict, Any, Optional


class QueryEnhancer:
    """Uses Gemini to enhance search results by selecting most relevant items"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def get_most_relevant_chunk(self, query: str, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use Gemini to pick the most relevant chunk from search results
        
        Args:
            query: Original user query
            search_results: List of search results with text, score, chunk_index
            
        Returns:
            Most relevant result or None if no results
        """
        print(f"Search results: {search_results}")
        if not search_results:
            return None
        
        # Format results for Gemini analysis
        results_text = ""
        for i, result in enumerate(search_results, 1):
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            results_text += f"\n{i}. [Chunk {result['chunk_index']}] (Score: {result['score']:.4f})\n"
            results_text += f"   Text: {text_preview}\n"
            if 'source_query' in result:
                results_text += f"   Found via: {result['source_query']}\n"
            results_text += "\n"
        
        prompt = f"""Given the user query: "{query}"

And these search results from a document:
{results_text}

Please return ONLY the number (1, 2, 3, 4, 5, etc.) of the most relevant chunk that best answers or relates to the user's query. Consider:

- How directly the content answers the user's question
- Relevance to the specific details mentioned in the query
- Completeness of information provided
- Context appropriateness

For insurance/medical queries like "{query}", prioritize chunks that contain:
- Specific eligibility criteria
- Coverage details for the mentioned condition
- Policy terms relevant to the situation
- Location-specific information if mentioned

Return only the number, nothing else."""

        try:
            response = self.model.generate_content(prompt)
            choice_text = response.text.strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', choice_text)
            if numbers:
                choice_num = int(numbers[0])
                if 1 <= choice_num <= len(search_results):
                    selected = search_results[choice_num - 1]
                    print(f"🎯 Gemini selected chunk {selected['chunk_index']} as most relevant")
                    return selected
            
            # Fallback to highest score if Gemini returns invalid number
            print("🔄 Using highest-scored result as fallback")
            return search_results[0]
            
        except Exception as e:
            print(f"Error using Gemini for result selection: {e}")
            return search_results[0]
    
 