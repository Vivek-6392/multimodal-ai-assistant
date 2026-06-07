#!/usr/bin/env sh
set -eu

cd /app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

cd /app/frontend
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --server.enableCORS=false