"""
Debug script to find chunks related to Lê Thái Tông using SQLAlchemy
"""
import sys
sys.path.append('.')

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.database.models import ChildChunk, ParentChunk, Document

# Create database connection
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("="*80)
print("🔍 SEARCHING FOR LÊ THÁI TÔNG CHUNKS IN DATABASE")
print("="*80)

# Search for child chunks containing "Lê Thái Tông"
chunks = db.query(ChildChunk).filter(
    ChildChunk.content.ilike('%Lê Thái Tông%')
).limit(50).all()

print(f"\n📊 Found {len(chunks)} child chunks mentioning 'Lê Thái Tông'\n")

# Filter for chunks that also mention "lên ngôi" or year numbers
relevant_chunks = []
for chunk in chunks:
    content = chunk.content.lower() if chunk.content else ""
    if 'lên ngôi' in content or any(str(year) in content for year in range(1400, 1500)):
        relevant_chunks.append(chunk)

print(f"📌 {len(relevant_chunks)} chunks mention both 'Lê Thái Tông' AND ('lên ngôi' OR years 1400-1500)\n")

# Display relevant chunks
for i, chunk in enumerate(relevant_chunks[:10], 1):
    print(f"{'─'*80}")
    print(f"CHILD CHUNK {i} (ID: {chunk.id})")
    print(f"{'─'*80}")
    print(f"Document ID: {chunk.document_id}")
    print(f"Headers: {chunk.h1 or ''} / {chunk.h2 or ''} / {chunk.h3 or ''}")
    print(f"Content Length: {len(chunk.content)} chars")
    print(f"\nContent:")
    print(chunk.content)
    print()

# Now search specifically for "1433" (known year when Lê Thái Tông might have ascended)
print("\n" + "="*80)
print("🔍 SEARCHING FOR CHUNKS WITH YEAR 1433")
print("="*80)

chunks_1433 = db.query(ChildChunk).filter(
    ChildChunk.content.like('%1433%')
).limit(20).all()

print(f"\n📊 Found {len(chunks_1433)} child chunks mentioning '1433'\n")

for i, chunk in enumerate(chunks_1433[:5], 1):
    print(f"{'─'*80}")
    print(f"CHUNK {i} (ID: {chunk.id})")
    print(f"{'─'*80}")
    print(f"Document ID: {chunk.document_id}")
    print(f"Headers: {chunk.h1 or ''} / {chunk.h2 or ''} / {chunk.h3 or ''}")
    print(f"\nContent preview:")
    preview = chunk.content[:500] if chunk.content else ""
    print(preview + ("..." if len(chunk.content) > 500 else ""))
    print()

# Check parent chunks too
print("\n" + "="*80)
print("🔍 SEARCHING PARENT CHUNKS FOR LÊ THÁI TÔNG")
print("="*80)

parent_chunks = db.query(ParentChunk).filter(
    ParentChunk.content.ilike('%Lê Thái Tông%')
).limit(50).all()

relevant_parents = []
for chunk in parent_chunks:
    content = chunk.content.lower() if chunk.content else ""
    if 'lên ngôi' in content or '1433' in content:
        relevant_parents.append(chunk)

print(f"\n📊 Found {len(relevant_parents)} parent chunks with 'Lê Thái Tông' AND ('lên ngôi' OR '1433')\n")

for i, chunk in enumerate(relevant_parents[:5], 1):
    print(f"{'─'*80}")
    print(f"PARENT CHUNK {i} (ID: {chunk.id})")
    print(f"{'─'*80}")
    print(f"Document ID: {chunk.document_id}")
    print(f"Headers: {chunk.h1 or ''} / {chunk.h2 or ''} / {chunk.h3 or ''}")
    print(f"\nContent:")
    print(chunk.content[:500] + ("..." if len(chunk.content) > 500 else ""))
    print()

# Test BM25 search directly
print("\n" + "="*80)
print("🔍 TESTING BM25 SEARCH FOR 'Lê Thái Tông lên ngôi'")
print("="*80)

bm25_query = text("""
    SELECT
        c.id,
        c.content,
        c.h1,
        c.h2,
        c.h3,
        paradedb.score(c.id) as rank
    FROM child_chunks c
    WHERE content @@@ 'Lê Thái Tông lên ngôi'
    ORDER BY rank DESC
    LIMIT 10
""")

try:
    results = db.execute(bm25_query).fetchall()
    print(f"\n📊 BM25 found {len(results)} results\n")
    
    for i, row in enumerate(results, 1):
        print(f"{'─'*80}")
        print(f"BM25 RESULT {i} (ID: {row.id}, Score: {row.rank:.4f})")
        print(f"{'─'*80}")
        print(f"Headers: {row.h1 or ''} / {row.h2 or ''} / {row.h3 or ''}")
        print(f"\nContent preview:")
        print(row.content[:300] + ("..." if len(row.content) > 300 else ""))
        print()
except Exception as e:
    print(f"❌ Error running BM25 query: {e}")

db.close()
