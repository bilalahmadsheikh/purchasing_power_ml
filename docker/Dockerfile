# ============================================================================
# PPP-Q Investment Classifier - Production Dockerfile
# Optimized for CI/CD (handles missing data gracefully)
# ============================================================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directory structure (ensures dirs exist even if COPY sources don't)
RUN mkdir -p /app/models/pppq /app/data/processed/pppq/train /app/data/processed/pppq/test /app/data/processed/pppq/val /app/logs

# Copy application code
COPY src/ /app/src/

# Copy models (CI workflow creates mock files before build)
COPY models/ /app/models/

# Copy processed data (CI workflow creates mock files before build)
COPY data/processed/ /app/data/processed/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models/pppq/ \
    DATA_DIR=/app/data/processed/pppq/ \
    LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]