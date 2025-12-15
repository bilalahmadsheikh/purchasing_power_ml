# ============================================================================
# PPP-Q Investment Classifier - Production Dockerfile
# Multi-stage build for optimized image size
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser data/processed/ /app/data/processed/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models/pppq/ \
    DATA_DIR=/app/data/processed/pppq/ \
    LOG_LEVEL=INFO

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run FastAPI with Gunicorn for production
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# -----------------------------------------------------------------------------
# Stage 3: Development - With dev tools
# -----------------------------------------------------------------------------
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    ipython

# Copy test files
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser pytest.ini /app/

USER appuser

# Override CMD for development (with reload)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]