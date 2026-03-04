#!/usr/bin/env python3
"""
Check parent chunks that were actually retrieved
"""
import sys
sys.path.append('.')

from app.database.connection import SessionLocal
from app.database.models import ChildChunk, ParentChunk
from sqlalchemy import text

db = SessionLocal()

print("="*80)
print("CHECKING PARENT CHUNKS")
print("="*80)

# 1. Get parent of chunk 1190
chunk_1190 = db.query(ChildChunk).filter(ChildChunk.id == 1190).first()
parent_id_1190 = chunk_1190.parent_id

print(f"\n📄 CHUNK 1190:")
print(f"   Parent ID: {parent_id_1190}")

# Get parent chunk
parent_199 = db.query(ParentChunk).filter(ParentChunk.id == parent_id_1190).first()

print(f"\n📋 PARENT CHUNK {parent_id_1190}:")
print(f"   Document ID: {parent_199.document_id}")
print(f"   H1: {parent_199.h1}")
print(f"   H2: {parent_199.h2}")
print(f"   Content Length: {len(parent_199.content)} chars")
print(f"\nContent:")
print(parent_199.content)
print()

# 2. Check the actual child chunks that were retrieved (from logs)
# Top 3 from variant 1: 876, 5652, 5800
retrieved_child_ids = [876, 5652, 5800]

print("\n" + "="*80)
print("PARENT CHUNKS OF RETRIEVED CHILD CHUNKS")
print("="*80)

retrieved_parent_ids = set()

for child_id in retrieved_child_ids:
    child = db.query(ChildChunk).filter(ChildChunk.id == child_id).first()
    if child and child.parent_id:
        retrieved_parent_ids.add(child.parent_id)
        print(f"\nChild {child_id} → Parent {child.parent_id}")
        print(f"   H2: {child.h2}")

# Get those parent chunks
print(f"\n📋 {len(retrieved_parent_ids)} UNIQUE PARENT CHUNKS RETRIEVED:")
print(f"   Parent IDs: {sorted(retrieved_parent_ids)}")

# Check if parent 199 is in the retrieved set
print(f"\n🎯 Is parent chunk 199 (containing year 1433) in retrieved set?")
print(f"   {parent_id_1190 in retrieved_parent_ids}")

# Get full content of retrieved parents
for parent_id in sorted(retrieved_parent_ids)[:5]:
    parent = db.query(ParentChunk).filter(ParentChunk.id == parent_id).first()
    if parent:
        print(f"\n{'─'*80}")
        print(f"PARENT {parent_id}")
        print(f"{'─'*80}")
        print(f"H2: {parent.h2}")
        print(f"Content preview:")
        print(parent.content[:500] + ("..." if len(parent.content) > 500 else ""))
        
        # Check if contains "1433"
        if "1433" in parent.content:
            print("\n   ✅ Contains '1433'")
        if "Lê Nguyên Long" in parent.content:
            print("   ✅ Contains 'Lê Nguyên Long'")

# 3. Test: what child chunks link to parent 199?
print("\n" + "="*80)
print("ALL CHILD CHUNKS UNDER PARENT 199")
print("="*80)

children_of_199 = db.query(ChildChunk).filter(ChildChunk.parent_id == parent_id_1190).all()
print(f"\nFound {len(children_of_199)} child chunks under parent 199:")

for child in children_of_199:
    print(f"\n   Child {child.id}:")
    print(f"      H3: {child.h3 or 'N/A'}")
    print(f"      Preview: {child.content[:100]}...")
    
    # Check if this child was retrieved
    if child.id in [876, 5652, 5800]:
        print(f"      ✅ WAS RETRIEVED!")

db.close()
