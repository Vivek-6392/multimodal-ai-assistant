FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install -r /app/backend/requirements.txt -r /app/frontend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY scripts /app/scripts

RUN chmod +x /app/scripts/start.sh

EXPOSE 8501
CMD ["/app/scripts/start.sh"]
