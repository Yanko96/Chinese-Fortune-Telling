FROM python:3.10-slim

WORKDIR /app

# Build context is repo root; copy only what the API needs
COPY api/requirements.txt .
# Install CPU-only torch first to avoid pulling 2GB CUDA libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so ECS tasks don't hit HuggingFace at runtime
# all-MiniLM-L6-v2: ~90MB  (embedding model for ChromaDB)
# bge-reranker-base: ~280MB (cross-encoder for HyDE rerank)
RUN python -c "
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
SentenceTransformer('all-MiniLM-L6-v2')
AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
print('Models pre-downloaded OK')
"

COPY api/ .
# Bake the vectorstore into the image (13 MB) so ECS has data without EFS
COPY chroma_db /app/chroma_db

# Set a default root path to work with ALB path-based routing; can be overridden at deploy time
ENV API_ROOT_PATH="/api"

# Use shell form to inject --root-path from env
CMD ["/bin/sh", "-c", "uvicorn fortune_main:app --host 0.0.0.0 --port 8000 --root-path ${API_ROOT_PATH}"]
