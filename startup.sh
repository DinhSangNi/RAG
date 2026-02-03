# startup.sh
#!/bin/bash
# Chạy migration trước khi start app
alembic upgrade head

# Sau đó start Gunicorn/Uvicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000