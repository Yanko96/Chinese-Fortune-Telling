FROM python:3.10-slim

WORKDIR /app

# Build context is repo root; copy only what the API needs
COPY api/requirements.txt .
# Install CPU-only torch first to avoid pulling 2GB CUDA libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so ECS tasks don't hit HuggingFace at runtime.
# Two separate RUN lines so each model gets its own docker layer cache entry —
# a single multiline `python -c "..."` was getting silently truncated by the
# build shell, so the second model never downloaded (see DEPLOYMENT_NOTES sec.4).
#   bge-small-zh-v1.5: ~100MB (embedding for the Chinese corpus index)
#   bge-reranker-base: ~280MB (cross-encoder for HyDE rerank)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5'); print('embedding model OK')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base'); print('reranker model OK')"

# Copy source + the PDF corpus, then build the vector index at image build time
# (replaces the previous approach of committing chroma_db/ — that was 13MB of
# binary data in the repo just so the image could COPY it). Building from the
# PDFs makes the image reproducible from source.
COPY api/ .
COPY fortune_books/ /app/fortune_books/
COPY scripts/build_index_bge.py /app/scripts/build_index_bge.py
# build_index_bge.py writes to ./chroma_db_bge (relative to CWD), which is
# /app at this point — i.e. /app/chroma_db_bge, matching the CHROMA_DIR env var
# set below. The script reads fortune_books/ relative to CWD too.
RUN python /app/scripts/build_index_bge.py

# Production retrieval reads from /app/chroma_db_bge via chroma_utils.get_vectorstore()
ENV CHROMA_DIR=/app/chroma_db_bge

# Default root path to work with ALB path-based routing; overridable at deploy time
ENV API_ROOT_PATH="/api"

# Shell form to inject --root-path from env
CMD ["/bin/sh", "-c", "uvicorn fortune_main:app --host 0.0.0.0 --port 8000 --root-path ${API_ROOT_PATH}"]
