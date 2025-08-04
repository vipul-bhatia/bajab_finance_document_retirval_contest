from thefuzz import fuzz
import json
import aiofiles
import os
from typing import List, Tuple, Optional, Dict, Any

async def find_similar_questions(new_questions: List[str], qa_file_path: str = "questions_answers.jsonl", threshold: int = 90) -> Dict[str, Any]:
    """
    Find similar questions in the stored Q&A data.
    
    Args:
        new_questions (List[str]): List of new questions to search for
        qa_file_path (str): Path to the questions_answers.jsonl file
        threshold (int): Minimum similarity score (0-100) to consider a match
        
    Returns:
        Dict containing:
        - 'found_similar': bool - Whether similar questions were found
        - 'matched_questions': List[Dict] - List of matched questions with their answers
        - 'unmatched_questions': List[str] - List of questions that didn't find matches
        - 'similarity_scores': Dict[str, int] - Similarity scores for each question
    """
    
    if not os.path.exists(qa_file_path):
        return {
            'found_similar': False,
            'matched_questions': [],
            'unmatched_questions': new_questions,
            'similarity_scores': {}
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
        
        matched_questions = []
        unmatched_questions = []
        similarity_scores = {}
        
        # For each new question, find the best match in stored data
        for new_question in new_questions:
            best_match = None
            best_score = 0
            best_answer = None
            best_document = None
            
            # Search through all stored Q&A entries
            for qa_entry in stored_qa_data:
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
                    'similarity_score': best_score
                })
            else:
                unmatched_questions.append(new_question)
        
        return {
            'found_similar': len(matched_questions) > 0,
            'matched_questions': matched_questions,
            'unmatched_questions': unmatched_questions,
            'similarity_scores': similarity_scores
        }
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return {
            'found_similar': False,
            'matched_questions': [],
            'unmatched_questions': new_questions,
            'similarity_scores': {}
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
    
    # Sample test questions
    test_questions = [
        "What is the main topic of this document?",
        "What are the key features mentioned?",
        "How does the process work?",
        "What is Newton's definition of quantity of motion?",  # Should match existing data
        "If my car is stolen, what case will it be in law?"   # Should match existing data
    ]
    
    print("🔍 Testing similarity search...")
    result = await find_similar_questions(test_questions, threshold=90)
    
    print(f"\n📊 Similarity Search Results:")
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