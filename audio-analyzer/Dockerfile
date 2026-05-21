FROM python:3.11-slim

# Configurer pip pour éviter les timeouts sur les gros packages (TensorFlow, etc.)
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install BirdNET-Analyzer from source (lightweight model download on first run)
# Utiliser --prefer-binary pour éviter de compiler depuis les sources
RUN git clone --depth 1 https://github.com/kahst/BirdNET-Analyzer.git /opt/birdnet && \
    pip install --no-cache-dir --prefer-binary -e /opt/birdnet

WORKDIR /app

# Railway build depuis le root du repo, donc on utilise les chemins relatifs
COPY audio-analyzer/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY audio-analyzer/app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 analyzer && \
    mkdir -p /tmp/analyzer && \
    chown -R analyzer:analyzer /app /tmp/analyzer
USER analyzer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]
