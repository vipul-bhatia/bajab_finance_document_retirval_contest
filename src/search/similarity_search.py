from thefuzz import fuzz
import json
import aiofiles
import os
from typing import List, Tuple, Optional, Dict, Any

async def find_similar_questions(new_questions: List[str], document_url: str, qa_file_path: str = "questions_answers.jsonl", threshold: int = 100) -> Dict[str, Any]:
    """
    Find similar questions in the stored Q&A data for a specific document.
    
    Args:
        new_questions (List[str]): List of new questions to search for
        document_url (str): URL of the document being processed
        qa_file_path (str): Path to the questions_answers.jsonl file
        threshold (int): Minimum similarity score (0-100) to consider a match
        
    Returns:
        Dict containing:
        - 'found_similar': bool - Whether similar questions were found
        - 'matched_questions': List[Dict] - List of matched questions with their answers
        - 'unmatched_questions': List[str] - List of questions that didn't find matches
        - 'similarity_scores': Dict[str, int] - Similarity scores for each question
        - 'document_matched': bool - Whether the document was found in stored data
    """
    
    if not os.path.exists(qa_file_path):
        return {
            'found_similar': False,
            'matched_questions': [],
            'unmatched_questions': new_questions,
            'similarity_scores': {},
            'document_matched': False
        }
    
    try:
        # Read all stored Q&A data
        stored_qa_data = []
        async with aiofiles.open(qa_file_path, mode='r') as f:
            content = await f.read()
            lines = content.strip().split('\n')
            
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    try:
                        qa_entry = json.loads(line)
                        stored_qa_data.append(qa_entry)
                    except json.JSONDecodeError:
                        continue
        
        # Step 1: Find the document in stored data
        document_matched = False
        document_qa_entries = []
        
        for qa_entry in stored_qa_data:
            stored_doc_url = qa_entry.get('document_url', '')
            stored_doc_name = qa_entry.get('document_name', '')
            
            # Check if this is the same document (exact URL match or document name match)
            if stored_doc_url == document_url or stored_doc_name in document_url or document_url in stored_doc_url:
                document_matched = True
                document_qa_entries.append(qa_entry)
        
        if not document_matched:
            print(f"📄 Document not found in stored data: {document_url}")
            return {
                'found_similar': False,
                'matched_questions': [],
                'unmatched_questions': new_questions,
                'similarity_scores': {},
                'document_matched': False
            }
        
        print(f"📄 Found {len(document_qa_entries)} Q&A entries for document: {document_url}")
        
        # Step 2: Find similar questions within the matched document
        matched_questions = []
        unmatched_questions = []
        similarity_scores = {}
        
        # For each new question, find the best match in the document's stored data
        for new_question in new_questions:
            best_match = None
            best_score = 0
            best_answer = None
            best_document = None
            
            # Search through all Q&A entries for this specific document
            for qa_entry in document_qa_entries:
                stored_questions = qa_entry.get('questions', [])
                stored_answers = qa_entry.get('answers', [])
                
                # Check each stored question against the new question
                for i, stored_question in enumerate(stored_questions):
                    score = fuzz.ratio(new_question.lower(), stored_question.lower())
                    
                    if score > best_score:
                        best_score = score
                        best_match = stored_question
                        best_answer = stored_answers[i] if i < len(stored_answers) else None
                        best_document = qa_entry.get('document_name', 'Unknown')
            
            similarity_scores[new_question] = best_score
            
            if best_score >= threshold and best_match and best_answer:
                matched_questions.append({
                    'new_question': new_question,
                    'matched_question': best_match,
                    'answer': best_answer,
                    'document_name': best_document,
                    'document_url': document_url,
                    'similarity_score': best_score
                })
            else:
                unmatched_questions.append(new_question)
        
        return {
            'found_similar': len(matched_questions) > 0,
            'matched_questions': matched_questions,
            'unmatched_questions': unmatched_questions,
            'similarity_scores': similarity_scores,
            'document_matched': document_matched
        }
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return {
            'found_similar': False,
            'matched_questions': [],
            'unmatched_questions': new_questions,
            'similarity_scores': {},
            'document_matched': False
        }

def get_answers_for_matched_questions(matched_questions: List[Dict[str, Any]]) -> List[str]:
    """
    Extract answers from matched questions in the correct order.
    
    Args:
        matched_questions (List[Dict]): List of matched question dictionaries
        
    Returns:
        List[str]: List of answers in the order of the original questions
    """
    # Sort by similarity score (highest first) and return answers
    sorted_matches = sorted(matched_questions, key=lambda x: x['similarity_score'], reverse=True)
    return [match['answer'] for match in sorted_matches]

# Test function
async def test_similarity_search():
    """Test the similarity search functionality"""
    
    # Sample test questions and document URL
    test_questions = [
        "What is the main topic of this document?",
        "What are the key features mentioned?",
        "How does the process work?",
        "What is Newton's definition of quantity of motion?",  # Should match existing data
        "If my car is stolen, what case will it be in law?"   # Should match existing data
    ]
    
    # Test with a document URL that exists in the stored data
    test_document_url = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
    
    print("🔍 Testing similarity search with document context...")
    result = await find_similar_questions(test_questions, test_document_url, threshold=100)
    
    print(f"\n📊 Similarity Search Results:")
    print(f"Document matched: {result['document_matched']}")
    print(f"Found similar questions: {result['found_similar']}")
    print(f"Matched questions: {len(result['matched_questions'])}")
    print(f"Unmatched questions: {len(result['unmatched_questions'])}")
    
    if result['matched_questions']:
        print(f"\n✅ Matched Questions:")
        for match in result['matched_questions']:
            print(f"  New: '{match['new_question']}'")
            print(f"  Matched: '{match['matched_question']}'")
            print(f"  Score: {match['similarity_score']}%")
            print(f"  Document: {match['document_name']}")
            print(f"  Answer: {match['answer'][:100]}...")
            print()
    
    if result['unmatched_questions']:
        print(f"\n❌ Unmatched Questions:")
        for question in result['unmatched_questions']:
            score = result['similarity_scores'].get(question, 0)
            print(f"  '{question}' (Best score: {score}%)")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_similarity_search()) 