# =============================================================================
# Production Dockerfile for Customer Analytics Platform
# Compatible with Railway, Render, Fly.io, and Hugging Face Spaces
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies (libgomp1 for LightGBM/XGBoost, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy platform code, configs, trained models, processed data, and reports
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY reports/ /app/reports/
COPY src/ /app/src/
COPY app/ /app/app/

# Expose default port (dynamically overridden by cloud platform via $PORT)
EXPOSE 8501

# Dynamic port binding for Render / Railway / Hugging Face Spaces / local Docker
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
