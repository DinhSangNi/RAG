#!/usr/bin/env python3
"""
Debug script to check why Lê Thái Tông search doesn't work
"""
import sys
sys.path.append('.')

from app.database.connection import SessionLocal
from app.database.models import ChildChunk, ParentChunk
from sqlalchemy import text

db = SessionLocal()

print("="*80)
print("🔍 CHECKING CHUNKS RELATED TO LÊ THÁI TÔNG + LÊN NGÔI")
print("="*80)

# 1. Check child chunks with "Lê Thái Tông" and "lên ngôi"
query1 = db.query(ChildChunk).filter(
    ChildChunk.content.ilike('%Lê Thái Tông%')
).all()

print(f"\n1️⃣ Total chunks mentioning 'Lê Thái Tông': {len(query1)}")

relevant = []
for chunk in query1:
    content_lower = chunk.content.lower() if chunk.content else ""
    if 'lên ngôi' in content_lower:
        relevant.append(chunk)

print(f"   → {len(relevant)} chunks also mention 'lên ngôi'\n")

for i, chunk in enumerate(relevant[:5], 1):
    print(f"{'─'*80}")
    print(f"CHUNK {i} (ID: {chunk.id})")
    print(f"Headers: {chunk.h1 or ''} / {chunk.h2 or ''} / {chunk.h3 or ''}")
    print(f"Content:")
    print(chunk.content)
    print()

# 2. Check for chunks with specific years
print("\n" + "="*80)
print("2️⃣ CHECKING FOR YEAR INFORMATION")
print("="*80)

years_to_check = [1433, 1434, 1442, 1460]  # Possible years for Lê Thái Tông
for year in years_to_check:
    chunks_with_year = db.query(ChildChunk).filter(
        ChildChunk.content.like(f'%{year}%')
    ).filter(
        ChildChunk.content.ilike('%Lê Thái Tông%')
    ).all()
    
    if chunks_with_year:
        print(f"\n📅 Year {year}: Found {len(chunks_with_year)} chunks")
        for chunk in chunks_with_year[:2]:
            print(f"   ID {chunk.id}: {chunk.content[:150]}...")

# 3. Check parent chunks
print("\n" + "="*80)
print("3️⃣ CHECKING PARENT CHUNKS")
print("="*80)

parent_chunks = db.query(ParentChunk).filter(
    ParentChunk.content.ilike('%Lê Thái Tông%')
).all()

print(f"\nTotal parent chunks mentioning 'Lê Thái Tông': {len(parent_chunks)}")

relevant_parents = []
for chunk in parent_chunks:
    content_lower = chunk.content.lower() if chunk.content else ""
    if 'lên ngôi' in content_lower:
        relevant_parents.append(chunk)

print(f"→ {len(relevant_parents)} parent chunks also mention 'lên ngôi'\n")

for i, chunk in enumerate(relevant_parents[:3], 1):
    print(f"{'─'*80}")
    print(f"PARENT CHUNK {i} (ID: {chunk.id})")
    print(f"Headers: {chunk.h1 or ''} / {chunk.h2 or ''} / {chunk.h3 or ''}")
    print(f"Content preview:")
    print(chunk.content[:400] + ("..." if len(chunk.content) > 400 else ""))
    print()

# 4. Test BM25 directly
print("\n" + "="*80)
print("4️⃣ TESTING BM25 SEARCH")
print("="*80)

test_queries = [
    "Lê Thái Tông lên ngôi năm nào",
    "Lê Thái Tông lên ngôi",
    "Thái Tông lên ngôi"
]

for query_text in test_queries:
    print(f"\n🔍 Query: '{query_text}'")
    
    try:
        bm25_query = text(f"""
            SELECT
                id,
                content,
                h1,
                h2,
                paradedb.score(id) as rank
            FROM child_chunks
            WHERE content @@@ :query_text
            ORDER BY rank DESC
            LIMIT 5
        """)
        
        results = db.execute(bm25_query, {"query_text": query_text}).fetchall()
        print(f"   Found {len(results)} results")
        
        for i, row in enumerate(results[:3], 1):
            print(f"   {i}. ID {row[0]}, Score: {row[4]:.4f}")
            print(f"      H2: {row[3] or 'N/A'}")
            print(f"      Preview: {row[1][:100]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

db.close()
