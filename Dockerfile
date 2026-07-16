# Container for the pKa Predictor API.
# Works on Hugging Face Spaces (Docker SDK), Render, Fly.io, Railway, etc.
FROM python:3.11-slim

# RDKit's wheel needs a couple of shared libraries on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects port 7860; Render/Fly inject $PORT.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
