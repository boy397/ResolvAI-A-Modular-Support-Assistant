FROM python:3.11-slim

WORKDIR /app

# system deps for sentence-transformers / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Render sets $PORT at runtime; gunicorn reads it via shell form CMD
CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 120 --workers 2 app:app
