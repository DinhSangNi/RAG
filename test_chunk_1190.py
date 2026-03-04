#!/usr/bin/env python3
"""
Test why chunk 1190 is not returned in search
"""
import sys
sys.path.append('.')

from app.database.connection import SessionLocal
from app.database.models import ChildChunk, child_chunk_summary_association
from sqlalchemy import text

db = SessionLocal()

print("="*80)
print("WHY IS CHUNK 1190 NOT IN TOP RESULTS?")
print("="*80)

# Check which summary this chunk belongs to
chunk_1190 = db.query(ChildChunk).filter(ChildChunk.id == 1190).first()

print(f"\n📄 CHUNK 1190 INFO:")
print(f"   Document ID: {chunk_1190.document_id}")
print(f"   Parent ID: {chunk_1190.parent_id}")
print(f"   H2: {chunk_1190.h2}")

# Check summary associations
assoc = db.query(child_chunk_summary_association).filter(
    child_chunk_summary_association.c.child_chunk_id == 1190
).all()

print(f"\n🔗 Summary associations: {len(assoc)}")
for a in assoc:
    print(f"   Summary ID: {a.summary_id}")

# Direct BM25 test for chunk 1190
print("\n" + "="*80)
print("TESTING BM25 SEARCH")
print("="*80)

queries = [
    "Lê Thái Tông lên ngôi năm nào",
    "Lê Thái Tông 1433",
    "Thái tử Lê Nguyên Long lên nối ngôi"
]

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    
    bm25_query = text("""
        SELECT
            id,
            h2,
            paradedb.score(id) as rank
        FROM child_chunks
        WHERE content @@@ :query_text
        ORDER BY rank DESC
        LIMIT 20
    """)
    
    results = db.execute(bm25_query, {"query_text": query}).fetchall()
    
    # Check if 1190 is in results
    found = False
    for i, row in enumerate(results, 1):
        if row[0] == 1190:
            print(f"   ✅ FOUND at position {i}, Score: {row[2]:.4f}")
            found = True
            break
    
    if not found:
        print(f"   ❌ Chunk 1190 NOT in top 20")

# Test with summary scoping
print("\n" + "="*80)
print("TESTING WITH SUMMARY SCOPING")
print("="*80)

# Get summary IDs from the search in the logs
summary_ids_used = db.execute(text("""
    SELECT DISTINCT ccsa.summary_id
    FROM child_chunks c
    INNER JOIN child_chunk_summary_association ccsa ON c.id = ccsa.child_chunk_id
    WHERE c.content @@@ 'Lê Thái Tông'
    ORDER BY ccsa.summary_id
    LIMIT 10
""")).fetchall()

print(f"\n📋 Summaries containing 'Lê Thái Tông': {len(summary_ids_used)}")
for sid in summary_ids_used[:5]:
    print(f"   Summary ID: {sid[0]}")

# Check if chunk 1190's summary is in this list
if assoc:
    chunk_summary = str(assoc[0].summary_id)
    is_in_scope = any(str(sid[0]) == chunk_summary for sid in summary_ids_used)
    print(f"\n🎯 Chunk 1190's summary ({chunk_summary}) is in scope: {is_in_scope}")

# Test semantic search
print("\n" + "="*80)
print("TESTING SEMANTIC SEARCH")
print("="*80)

print("\nChecking if chunk 1190 has embeddings...")
has_embedding = chunk_1190.embedding is not None
print(f"   Has embedding: {has_embedding}")
if has_embedding:
    print(f"   Embedding dimension: {len(chunk_1190.embedding)}")

db.close()
