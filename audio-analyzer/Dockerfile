FROM python:3.11-slim

# Configurer pip pour éviter les timeouts
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

# Install BirdNET-Analyzer
RUN git clone --depth 1 https://github.com/kahst/BirdNET-Analyzer.git /opt/birdnet && \
    pip install --no-cache-dir --prefer-binary -e /opt/birdnet

WORKDIR /app

# Copy application files
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app/ ./app/
COPY start.sh ./start.sh
RUN chmod +x start.sh

# Utiliser root pour éviter les problèmes de permission sur Railway
# (le plan Hobby a des limitations)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["./start.sh"]
