#!/usr/bin/env python3
"""
Test script for the Document Retrieval API
"""

import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8000/process-document"

# Test data in the exact format you specified
test_data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_api():
    """Test the API with sample data"""
    print("🚀 Testing Document Retrieval API")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Make API request
        print(f"📤 Sending request to: {API_URL}")
        print(f"📄 Document URL: {test_data['documents'][:80]}...")
        print(f"❓ Questions: {len(test_data['questions'])}")
        
        response = requests.post(
            API_URL,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        total_time = time.time() - start_time
        
        print(f"\n✅ SUCCESS! API Response received in {total_time:.2f}s")
        print(f"📊 Processing Time: {result.get('processing_time', 'N/A')}s")
        print(f"📄 Document Chunks: {result.get('document_chunks', 'N/A')}")
        print(f"❓ Questions Processed: {result.get('total_questions', 'N/A')}")
        
        print(f"\n📋 ANSWERS:")
        print("=" * 80)
        
        for i, (question, answer) in enumerate(zip(test_data['questions'], result['answers']), 1):
            print(f"\n{i}. Q: {question}")
            print(f"   A: {answer}")
        
        print("=" * 80)
        
        # Save response to file
        with open('api_response.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full response saved to 'api_response.json'")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Status: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                print(f"Error Detail: {error_detail}")
            except:
                print(f"Response Text: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_api() 