# PowerShell script to run database migration

Write-Host "Running database migration: add_file_size_and_hash.sql" -ForegroundColor Cyan

# Run migration SQL in Docker container
Get-Content migrations\add_file_size_and_hash.sql | docker exec -i rag_postgres psql -U rag_user -d rag_db

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Migration completed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Migration failed!" -ForegroundColor Red
    exit 1
}
