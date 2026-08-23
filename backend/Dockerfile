# Image de production de l'API FastAPI (backend/app/).
# Build depuis la racine du repo : docker build -f backend/Dockerfile -t foyer-rag-backend .
FROM python:3.11-slim

WORKDIR /app

# Dependances systeme minimales (aucune dependance PDF native requise :
# pypdf est pur Python).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif + corpus PDF de demo (deja genere et commite).
COPY backend/app ./app
COPY backend/data ./data

ENV RAW_DOCS_DIR=/app/data/raw_docs
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Render fournit $PORT dynamiquement ; on retombe sur 8000 en local
# (docker run -p 8000:8000 ...).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
