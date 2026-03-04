"""
Test script for RAG chat endpoint with variant analysis
"""
import requests
import json

# Make request to chat endpoint
url = "http://localhost:8000/api/v1/chat"
data = {
    "question": "Lê Thái Tông lên ngôi năm nào?",
    "verbose": False
}

print("="*80)
print("🔍 TESTING RAG CHAT ENDPOINT")
print("="*80)
print(f"Question: {data['question']}")
print()

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    
    print("="*80)
    print("📝 ANSWER:")
    print("="*80)
    print(result['answer'])
    print()
    
    metadata = result.get('metadata', {})
    
    print("="*80)
    print("📊 METADATA:")
    print("="*80)
    print(f"Retrieval Method: {metadata.get('retrieval_method', 'N/A')}")
    print(f"Source: {metadata.get('source', 'N/A')}")
    print(f"Variants Count: {metadata.get('variants_count', 0)}")
    print(f"Chunks Used: {metadata.get('chunks_used', 0)}")
    print()
    
    # Display variants
    variants = metadata.get('variants', [])
    if variants:
        print("="*80)
        print("🧩 QUERY VARIANTS:")
        print("="*80)
        for i, variant in enumerate(variants, 1):
            print(f"{i}. {variant}")
        print()
    
    # Display top 3 chunks for each variant
    variant_results = metadata.get('variant_results', [])
    if variant_results:
        print("="*80)
        print("🎯 TOP 3 CHILD CHUNKS PER VARIANT:")
        print("="*80)
        
        for i, vr in enumerate(variant_results, 1):
            variant = vr.get('variant', 'N/A')
            top_3 = vr.get('top_3_chunks', [])
            
            print(f"\n{'─'*80}")
            print(f"VARIANT {i}: {variant}")
            print(f"{'─'*80}")
            
            for j, chunk in enumerate(top_3, 1):
                print(f"\n  📄 Chunk {j}:")
                print(f"     ID: {chunk.get('id', 'N/A')}")
                print(f"     Score: {chunk.get('fused_score', 0):.4f}")
                
                # Headers
                headers = []
                if chunk.get('h1'):
                    headers.append(chunk['h1'])
                if chunk.get('h2'):
                    headers.append(chunk['h2'])
                if chunk.get('h3'):
                    headers.append(chunk['h3'])
                
                if headers:
                    print(f"     Headers: {' / '.join(headers)}")
                
                # Content preview
                content = chunk.get('content', '')
                preview = content[:200].replace('\n', ' ')
                print(f"     Content: {preview}...")
                print()
        
        print("="*80)
    else:
        print("⚠️ No variant results found in metadata")
    
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
