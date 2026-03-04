"""
Test better query strategies for finding Lê Thái Tông ascension year
"""
import requests
import json

url = "http://localhost:8000/api/v1/chat"

# Test different question phrasings
questions = [
    "Lê Thái Tông lên ngôi năm nào?",
    "Lê Nguyên Long lên ngôi năm nào?",
    "Năm nào Lê Thái Tông kế vị?",
    "Lê Thái Tổ mất năm nào và ai kế vị?",
]

for question in questions:
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    
    data = {"question": question, "verbose": False}
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📝 ANSWER: {result['answer']}")
        
        metadata = result.get('metadata', {})
        variant_results = metadata.get('variant_results', [])
        
        if variant_results:
            print(f"\n🎯 TOP CHUNK IDS FOR EACH VARIANT:")
            for i, vr in enumerate(variant_results, 1):
                variant = vr.get('variant', 'N/A')
                top_3 = vr.get('top_3_chunks', [])
                chunk_ids = [c.get('id') for c in top_3]
                print(f"   Variant {i}: {variant}")
                print(f"   Top 3 IDs: {chunk_ids}")
                
                # Check if chunk 1190 is in results
                if 1190 in chunk_ids:
                    print(f"   ✅ CHUNK 1190 FOUND!")
    else:
        print(f"❌ Error: {response.status_code}")
