#!/usr/bin/env python3
"""
Debug BM25 ranking for chunk 1190
"""
import sys
sys.path.append('.')

from app.database.connection import SessionLocal
from sqlalchemy import text

def debug_chunk_1190():
    db = SessionLocal()
    
    # 1. Kiểm tra chunk 1190 có tồn tại không
    chunk = db.execute(text("""
        SELECT id, content, parent_id,
               LEFT(content, 200) as preview
        FROM child_chunks 
        WHERE id = 1190
    """)).fetchone()
    
    print("=" * 80)
    print("📄 CHUNK 1190 INFO:")
    print("=" * 80)
    if chunk:
        print(f"ID: {chunk.id}")
        print(f"Parent ID: {chunk.parent_id}")
        print(f"Preview: {chunk.preview}...")
        print(f"\nFull content contains:")
        print(f"  - 'Lê Thái Tông': {'YES' if 'Lê Thái Tông' in chunk.content else 'NO'}")
        print(f"  - 'lên ngôi': {'YES' if 'lên ngôi' in chunk.content else 'NO'}")
        print(f"  - '1433': {'YES' if '1433' in chunk.content else 'NO'}")
    else:
        print("❌ Chunk 1190 NOT FOUND!")
        db.close()
        return
    
    # 2. Kiểm tra chunk 1190 thuộc summary nào
    summaries = db.execute(text("""
        SELECT summary_id 
        FROM child_chunk_summary_association 
        WHERE child_chunk_id = 1190
    """)).fetchall()
    
    print(f"\n📌 Summary associations: {[str(s.summary_id) for s in summaries]}")
    
    # 3. Test BM25 với nhiều query variants
    queries = [
        "Lê Thái Tông lên ngôi năm nào",
        "Lê Thái Tông 1433",
        "Thái tử Lê Nguyên Long lên nối ngôi",
        "Lê Thái Tổ qua đời Lê Thái Tông",
        "năm Quý Sửu 1433 Lê Thái Tông",
        "Lê Nguyên Long lên nối ngôi"
    ]
    
    print("\n" + "=" * 80)
    print("🔍 BM25 RANKING TEST:")
    print("=" * 80)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Search không filter summary
        results = db.execute(text("""
            SELECT 
                c.id,
                LEFT(c.content, 100) as preview,
                paradedb.score(c.id) as rank
            FROM child_chunks c
            WHERE content @@@ :query_text
            ORDER BY rank DESC
            LIMIT 20
        """), {"query_text": query}).fetchall()
        
        # Tìm vị trí của chunk 1190
        ranks = [(i+1, r.id, r.rank) for i, r in enumerate(results)]
        chunk_1190_rank = next((r for r in ranks if r[1] == 1190), None)
        
        if chunk_1190_rank:
            print(f"  ✅ Chunk 1190 at rank {chunk_1190_rank[0]} (score: {chunk_1190_rank[2]:.4f})")
        else:
            print(f"  ❌ Chunk 1190 NOT in top 20")
        
        # Show top 5
        print(f"  Top 5:")
        for i, r in enumerate(results[:5], 1):
            marker = "👉" if r.id == 1190 else "  "
            print(f"    {marker} #{i} - Chunk {r.id} (score: {r.rank:.4f})")
            print(f"       {r.preview}...")
    
    # 4. So sánh chunk 1190 vs chunk 876 (chunk được rank cao)
    print("\n" + "=" * 80)
    print("⚖️ COMPARISON: Chunk 1190 vs Chunk 876")
    print("=" * 80)
    
    for chunk_id in [1190, 876]:
        chunk_data = db.execute(text("""
            SELECT id, content
            FROM child_chunks 
            WHERE id = :chunk_id
        """), {"chunk_id": chunk_id}).fetchone()
        
        print(f"\n📄 Chunk {chunk_id}:")
        print(f"Length: {len(chunk_data.content)} chars")
        print(f"Contains 'Lê Thái Tông': {chunk_data.content.count('Lê Thái Tông')} times")
        print(f"Contains 'lên ngôi': {chunk_data.content.count('lên ngôi')} times")
        print(f"Contains 'năm': {chunk_data.content.count('năm')} times")
        print(f"Preview: {chunk_data.content[:200]}...")
    
    # 5. Test với query chính xác hơn
    print("\n" + "=" * 80)
    print("🎯 TEST WITH EXACT PHRASES:")
    print("=" * 80)
    
    exact_queries = [
        '"Lê Thái Tông" AND "lên ngôi"',
        '"Lê Thái Tông" AND 1433',
        '"Lê Nguyên Long" AND "lên nối ngôi"'
    ]
    
    for query in exact_queries:
        print(f"\nExact Query: {query}")
        
        results = db.execute(text("""
            SELECT 
                c.id,
                LEFT(c.content, 100) as preview,
                paradedb.score(c.id) as rank
            FROM child_chunks c
            WHERE content @@@ :query_text
            ORDER BY rank DESC
            LIMIT 10
        """), {"query_text": query}).fetchall()
        
        chunk_1190_rank = next(((i+1, r.rank) for i, r in enumerate(results) if r.id == 1190), None)
        
        if chunk_1190_rank:
            print(f"  ✅ Chunk 1190 at rank {chunk_1190_rank[0]} (score: {chunk_1190_rank[1]:.4f})")
        else:
            print(f"  ❌ Chunk 1190 NOT in top 10")
        
        print(f"  Top 3: {[(r.id, round(r.rank, 2)) for r in results[:3]]}")
    
    db.close()

if __name__ == "__main__":
    debug_chunk_1190()
