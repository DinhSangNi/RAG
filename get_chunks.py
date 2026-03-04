#!/usr/bin/env python3
"""
Get full content of specific chunks
"""
import sys
sys.path.append('.')

from app.database.connection import SessionLocal
from app.database.models import ChildChunk

db = SessionLocal()

# Get chunks that have year 1433 info
chunk_ids = [1189, 1190, 1037, 876, 1269, 1271]

print("="*80)
print("EXAMINING KEY CHUNKS")
print("="*80)

for cid in chunk_ids:
    chunk = db.query(ChildChunk).filter(ChildChunk.id == cid).first()
    if chunk:
        print(f'\n{"="*80}')
        print(f'CHUNK ID: {cid}')
        print(f'{"="*80}')
        print(f'H1: {chunk.h1}')
        print(f'H2: {chunk.h2}')
        print(f'H3: {chunk.h3}')
        print(f'\nContent:')
        print(chunk.content)
        print()

db.close()
