#!/bin/bash
# Script to run database migration

echo "Running database migration: add_file_size_and_hash.sql"

# Run migration SQL in Docker container
docker exec -i rag_postgres psql -U rag_user -d rag_db < migrations/add_file_size_and_hash.sql

if [ $? -eq 0 ]; then
    echo "✓ Migration completed successfully!"
else
    echo "✗ Migration failed!"
    exit 1
fi
