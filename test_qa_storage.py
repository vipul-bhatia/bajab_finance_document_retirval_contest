#!/usr/bin/env python3
"""
Test script to demonstrate the new questions and answers storage functionality
"""

import json
import asyncio
from datetime import datetime

async def test_qa_storage():
    """Test the questions and answers storage functionality"""
    
    # Sample data
    sample_questions = [
        "What is the main topic of this document?",
        "What are the key features mentioned?",
        "How does the process work?"
    ]
    
    sample_answers = [
        "The main topic is document processing and question answering.",
        "Key features include semantic search, document embedding, and intelligent query processing.",
        "The process involves downloading documents, generating embeddings, and using semantic search to find relevant answers."
    ]
    
    document_name = "test_document"
    document_url = "https://example.com/test-document.pdf"
    
    # Create a sample entry
    qa_entry = {
        "timestamp": datetime.now().isoformat(),
        "document_name": document_name,
        "document_url": document_url,
        "questions": sample_questions,
        "answers": sample_answers
    }
    
    # Save to questions_answers.jsonl
    qa_file = "questions_answers.jsonl"
    
    try:
        import aiofiles
        
        async with aiofiles.open(qa_file, mode='a') as f:
            await f.write(json.dumps(qa_entry) + '\n')
        
        print(f"✅ Sample questions and answers saved to {qa_file}")
        
        # Read and display the saved data
        async with aiofiles.open(qa_file, mode='r') as f:
            content = await f.read()
            lines = content.strip().split('\n')
            
            print(f"\n📊 Total entries in {qa_file}: {len([line for line in lines if line.strip() and not line.startswith('#')])}")
            
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    try:
                        entry = json.loads(line)
                        print(f"\n📝 Entry from {entry['timestamp']}")
                        print(f"   Document: {entry['document_name']}")
                        print(f"   Questions: {len(entry['questions'])}")
                        print(f"   Answers: {len(entry['answers'])}")
                    except json.JSONDecodeError:
                        continue
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_qa_storage()) 