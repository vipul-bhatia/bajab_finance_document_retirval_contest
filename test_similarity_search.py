#!/usr/bin/env python3
"""
Test script to demonstrate the similarity search functionality
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.search.similarity_search import find_similar_questions, test_similarity_search

async def main():
    """Main test function"""
    
    print("🧪 Testing Similarity Search Functionality")
    print("=" * 50)
    
    # Test 1: Basic similarity search
    print("\n📋 Test 1: Basic Similarity Search")
    print("-" * 30)
    
    test_questions = [
        "What is Newton's definition of quantity of motion?",
        "If my car is stolen, what case will it be in law?",
        "What is the main topic of this document?",
        "How does the process work?",
        "What are the key features mentioned?"
    ]
    
    result = await find_similar_questions(test_questions, threshold=90)
    
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
            print()
    
    if result['unmatched_questions']:
        print(f"\n❌ Unmatched Questions:")
        for question in result['unmatched_questions']:
            score = result['similarity_scores'].get(question, 0)
            print(f"  '{question}' (Best score: {score}%)")
    
    # Test 2: Lower threshold test
    print("\n📋 Test 2: Lower Threshold (70%)")
    print("-" * 30)
    
    result_lower = await find_similar_questions(test_questions, threshold=70)
    
    print(f"Found similar questions: {result_lower['found_similar']}")
    print(f"Matched questions: {len(result_lower['matched_questions'])}")
    print(f"Unmatched questions: {len(result_lower['unmatched_questions'])}")
    
    if result_lower['matched_questions']:
        print(f"\n✅ Matched Questions (70% threshold):")
        for match in result_lower['matched_questions']:
            print(f"  New: '{match['new_question']}'")
            print(f"  Matched: '{match['matched_question']}'")
            print(f"  Score: {match['similarity_score']}%")
            print()
    
    # Test 3: Exact match variations
    print("\n📋 Test 3: Exact Match Variations")
    print("-" * 30)
    
    exact_variations = [
        "Newton definition of 'quantity of motion",  # Slightly different from stored
        "If my car is stolen, what case will it be in law?",  # Exact match
        "What happens if someone steals my car?",  # Similar meaning
        "Newton's quantity of motion definition",  # Reordered words
        "Car theft legal case"  # Very different wording
    ]
    
    result_exact = await find_similar_questions(exact_variations, threshold=90)
    
    print(f"Found similar questions: {result_exact['found_similar']}")
    print(f"Matched questions: {len(result_exact['matched_questions'])}")
    
    if result_exact['matched_questions']:
        print(f"\n✅ Exact Match Variations:")
        for match in result_exact['matched_questions']:
            print(f"  New: '{match['new_question']}'")
            print(f"  Matched: '{match['matched_question']}'")
            print(f"  Score: {match['similarity_score']}%")
            print()
    
    # Test 4: Performance test
    print("\n📋 Test 4: Performance Test")
    print("-" * 30)
    
    import time
    
    performance_questions = [
        "What is Newton's definition of quantity of motion?",
        "If my car is stolen, what case will it be in law?",
        "What is the main topic of this document?",
        "How does the process work?",
        "What are the key features mentioned?"
    ] * 10  # Repeat 10 times for performance testing
    
    start_time = time.perf_counter()
    result_perf = await find_similar_questions(performance_questions, threshold=90)
    end_time = time.perf_counter()
    
    duration_ms = (end_time - start_time) * 1000
    print(f"Processed {len(performance_questions)} questions in {duration_ms:.2f} ms")
    print(f"Average time per question: {duration_ms/len(performance_questions):.2f} ms")
    print(f"Matched questions: {len(result_perf['matched_questions'])}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 