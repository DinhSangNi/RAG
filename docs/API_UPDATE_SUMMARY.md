# Test Update Summary API

## Endpoint

```
POST /api/v1/update-summary/{summary_id}
```

## Example Usage

### 1. Update summary document

```bash
curl -X POST "http://localhost:8000/api/v1/update-summary/550e8400-e29b-41d4-a716-446655440000" \
  -F "file=@updated_summary.html"
```

### 2. Response - Content Updated

```json
{
  "status": "updated",
  "summary_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "660e8400-e29b-41d4-a716-446655440001",
  "job_id": "update_abc123",
  "message": "Summary document đang được cập nhật (re-processing)"
}
```

### 3. Response - Content Unchanged

```json
{
  "status": "unchanged",
  "summary_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "660e8400-e29b-41d4-a716-446655440001",
  "message": "Content không thay đổi, không cần update"
}
```

### 4. Response - Duplicate Content

```json
{
  "status": "duplicate",
  "summary_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "660e8400-e29b-41d4-a716-446655440001",
  "message": "Content trùng với summary khác (summary_id: 770e8400-e29b-41d4-a716-446655440002)"
}
```

## Features

- ✅ Check duplicate với content_hash
- ✅ Detect unchanged content (không re-process nếu content giống hệt)
- ✅ Prevent duplicate với summary documents khác
- ✅ Re-generate embedding khi content thay đổi
- ✅ Update document metadata (file_path, file_size, content_hash)
- ✅ Queue background job để xử lý bất đồng bộ
