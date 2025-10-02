FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set a default root path to work with ALB path-based routing; can be overridden at deploy time
ENV API_ROOT_PATH="/api"

# Use shell form to inject --root-path from env
CMD ["/bin/sh", "-c", "uvicorn fortune_main:app --host 0.0.0.0 --port 8000 --root-path ${API_ROOT_PATH}"]
