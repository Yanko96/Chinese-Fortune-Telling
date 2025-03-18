#!/bin/bash
# setup_fortune_teller.sh
# This script sets up the Chinese Fortune Teller application

echo "Setting up Chinese Fortune Teller application..."

# Create necessary directories
mkdir -p fortune_books
mkdir -p app/static/images

# Download sample fortune telling books
echo "You'll need to add your own fortune telling books in PDF format to the fortune_books directory."
echo "Books should include: san_ming_tong_hui.pdf, di_tian_sui.pdf, zi_ping_zhen_quan.pdf"
0
# Update requirements.txt
echo "Updating requirements.txt..."
cat > api/requirements.txt << EOF
langchain
langchain-google-genai
langchain-openai
langchain-core
langchain_community
docx2txt
pypdf
langchain_chroma
python-multipart
streamlit
streamlit-chat
streamlit-extras
chinese-calendar
lunardate
EOF

# Create Dockerfiles for easier deployment
echo "Creating Dockerfiles..."

# API Dockerfile
cat > api/Dockerfile << EOF
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "fortune_main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Frontend Dockerfile
cat > app/Dockerfile << EOF
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "fortune_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << EOF
version: '3'

services:
  fortune-api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ./fortune_books:/app/fortune_books
      - ./chroma_db:/app/chroma_db

  fortune-app:
    build: ./app
    ports:
      - "8501:8501"
    depends_on:
      - fortune-api
    environment:
      - API_URL=http://fortune-api:8000

volumes:
  chroma_db:
EOF

echo "Setup complete! Now you can:"
echo "1. Add fortune telling books to the fortune_books directory"
echo "2. Run 'python fortune_setup.py' to load the books into the RAG system"
echo "3. Run 'docker-compose up' to start the application"
echo "4. Access the Fortune Teller at http://localhost:8501"