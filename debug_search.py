"""
Debug script to find chunks related to Lê Thái Tông
"""
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port=5434,
    database="rag_db",
    user="rag_user",
    password="rag_password"
)

cursor = conn.cursor(cursor_factory=RealDictCursor)

print("="*80)
print("🔍 SEARCHING FOR LÊ THÁI TÔNG CHUNKS IN DATABASE")
print("="*80)

# Search for chunks containing "Lê Thái Tông"
query = """
SELECT 
    id,
    content,
    h1,
    h2,
    h3,
    document_id,
    chunk_index,
    LENGTH(content) as content_length
FROM child_chunks
WHERE content ILIKE '%Lê Thái Tông%'
ORDER BY id
LIMIT 50
"""

cursor.execute(query)
results = cursor.fetchall()

print(f"\n📊 Found {len(results)} chunks mentioning 'Lê Thái Tông'\n")

# Filter for chunks that also mention "lên ngôi" or year numbers
relevant_chunks = []
for chunk in results:
    content = chunk['content'].lower()
    if 'lên ngôi' in content or any(str(year) in content for year in range(1400, 1500)):
        relevant_chunks.append(chunk)

print(f"📌 {len(relevant_chunks)} chunks mention both 'Lê Thái Tông' AND ('lên ngôi' OR years 1400-1500)\n")

# Display relevant chunks
for i, chunk in enumerate(relevant_chunks[:10], 1):
    print(f"{'─'*80}")
    print(f"CHUNK {i} (ID: {chunk['id']})")
    print(f"{'─'*80}")
    print(f"Document ID: {chunk['document_id']}")
    print(f"Headers: {chunk.get('h1', '')} / {chunk.get('h2', '')} / {chunk.get('h3', '')}")
    print(f"Content Length: {chunk['content_length']} chars")
    print(f"\nContent:")
    print(chunk['content'])
    print()

# Now search specifically for "1433" (known year when Lê Thái Tông ascended)
print("\n" + "="*80)
print("🔍 SEARCHING FOR CHUNKS WITH YEAR 1433")
print("="*80)

query_1433 = """
SELECT 
    id,
    content,
    h1,
    h2,
    h3,
    document_id
FROM child_chunks
WHERE content LIKE '%1433%'
ORDER BY id
LIMIT 20
"""

cursor.execute(query_1433)
results_1433 = cursor.fetchall()

print(f"\n📊 Found {len(results_1433)} chunks mentioning '1433'\n")

for i, chunk in enumerate(results_1433[:5], 1):
    print(f"{'─'*80}")
    print(f"CHUNK {i} (ID: {chunk['id']})")
    print(f"{'─'*80}")
    print(f"Document ID: {chunk['document_id']}")
    print(f"Headers: {chunk.get('h1', '')} / {chunk.get('h2', '')} / {chunk.get('h3', '')}")
    print(f"\nContent preview:")
    preview = chunk['content'][:300]
    print(preview + "...")
    print()

# Check parent chunks too
print("\n" + "="*80)
print("🔍 SEARCHING PARENT CHUNKS FOR LÊ THÁI TÔNG")
print("="*80)

query_parent = """
SELECT 
    id,
    content,
    h1,
    h2,
    h3,
    document_id
FROM parent_chunks
WHERE content ILIKE '%Lê Thái Tông%'
AND (content ILIKE '%lên ngôi%' OR content LIKE '%1433%')
ORDER BY id
LIMIT 10
"""

cursor.execute(query_parent)
results_parent = cursor.fetchall()

print(f"\n📊 Found {len(results_parent)} parent chunks with 'Lê Thái Tông' AND ('lên ngôi' OR '1433')\n")

for i, chunk in enumerate(results_parent, 1):
    print(f"{'─'*80}")
    print(f"PARENT CHUNK {i} (ID: {chunk['id']})")
    print(f"{'─'*80}")
    print(f"Document ID: {chunk['document_id']}")
    print(f"Headers: {chunk.get('h1', '')} / {chunk.get('h2', '')} / {chunk.get('h3', '')}")
    print(f"\nContent:")
    print(chunk['content'][:500] + ("..." if len(chunk['content']) > 500 else ""))
    print()

cursor.close()
conn.close()
