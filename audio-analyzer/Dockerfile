FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install BirdNET-Analyzer from source (lightweight model download on first run)
RUN git clone --depth 1 https://github.com/kahst/BirdNET-Analyzer.git /opt/birdnet && \
    pip install --no-cache-dir -e /opt/birdnet

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 analyzer && \
    mkdir -p /tmp/analyzer && \
    chown -R analyzer:analyzer /app /tmp/analyzer
USER analyzer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]
