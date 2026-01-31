"""
Test script for hierarchical RAG retrieval workflow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.database.connection import get_db
from app.services.rag_service import get_rag_service


def test_hierarchical_retrieval():
    """Test the hierarchical retrieval workflow"""
    
    print("="*70)
    print("TEST HIERARCHICAL RAG RETRIEVAL")
    print("="*70)
    
    # Initialize
    db = next(get_db())
    rag_service = get_rag_service(db)
    
    # Test questions
    test_questions = [
        "Hồ Chí Minh sinh năm nao?",
        "Ai là người sáng lập Đảng Cộng sản Việt Nam?",
        "Chiến dịch Điện Biên Phủ diễn ra khi nào?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*70}")
        print(f"Question: {question}")
        
        try:
            # Test with hierarchical retrieval
            print("\n🔹 Testing HIERARCHICAL retrieval...")
            result = rag_service.chat(
                question=question,
                use_hierarchical=True,
                verbose=True
            )
            
            print(f"\n{'='*70}")
            print("RESULT")
            print(f"{'='*70}")
            print(f"Answer: {result['answer']}")
            print(f"\nMetadata:")
            for key, value in result['metadata'].items():
                print(f"  - {key}: {value}")
            
            # Compare with old method
            print(f"\n{'='*70}")
            print("🔹 Testing LEGACY retrieval (for comparison)...")
            result_old = rag_service.chat(
                question=question,
                use_hierarchical=False,
                verbose=False
            )
            
            print(f"\nLegacy Answer: {result_old['answer']}")
            print(f"Legacy Metadata: {result_old['metadata']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("TEST COMPLETED")
    print(f"{'='*70}")


def test_summary_search():
    """Test summary document search"""
    
    print("="*70)
    print("TEST SUMMARY DOCUMENT SEARCH")
    print("="*70)
    
    db = next(get_db())
    rag_service = get_rag_service(db)
    
    test_query = "Hồ Chí Minh"
    
    print(f"\nQuery: {test_query}")
    
    # Search summaries
    summary_docs = rag_service.search_service.hybrid_search_summaries(
        query=test_query,
        k=5
    )
    
    print(f"\nFound {len(summary_docs)} summary documents:")
    for i, doc in enumerate(summary_docs, 1):
        print(f"\n📄 Summary {i}:")
        print(f"   Score: {doc.get('fused_score', 0):.4f}")
        print(f"   Preview: {doc.get('summary_content', '')[:200]}...")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hierarchical RAG")
    parser.add_argument(
        "--mode",
        choices=["full", "summary"],
        default="full",
        help="Test mode: full (complete workflow) or summary (only summary search)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        test_hierarchical_retrieval()
    else:
        test_summary_search()
