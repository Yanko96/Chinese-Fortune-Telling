FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV API_URL="http://fortune-api:8000"

CMD ["streamlit", "run", "fortune_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
