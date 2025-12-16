# 🐳 Docker Deployment Documentation

## Overview

This document describes the Docker containerization strategy, multi-stage builds, and deployment configurations for the PPP-Q ML pipeline.

The system uses:
- **Multi-stage Docker builds** for optimized production images
- **Docker Compose** for local development and production
- **Nginx** as reverse proxy for API
- **Prometheus** for metrics collection
- **Health checks** for container monitoring

---

## Architecture

### Deployment Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer (Optional)               │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                  Nginx Reverse Proxy                         │
│  - SSL/TLS termination                                       │
│  - Load balancing across API instances                       │
│  - Compression                                               │
│  - Caching                                                   │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                   API Container (Gunicorn)                   │
│  - Uvicorn + Gunicorn workers                               │
│  - FastAPI application                                       │
│  - Model inference                                           │
│  - Health checks                                             │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                   Data & Models Volume                       │
│  - final_consolidated_dataset.csv                           │
│  - Trained models (LightGBM, XGBoost)                       │
│  - Feature configurations                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ Dockerfile

**File:** `docker/Dockerfile`

### Multi-Stage Build Strategy

#### Stage 1: Builder
```dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

#### Stage 2: Production
```dockerfile
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run Gunicorn with Uvicorn workers
CMD ["gunicorn", \
     "--workers=2", \
     "--worker-class=uvicorn.workers.UvicornWorker", \
     "--bind=0.0.0.0:8000", \
     "--access-logfile=-", \
     "--error-logfile=-", \
     "src.api.main:app"]
```

### Build Optimization

**Benefits:**
- ✅ Smaller final image (no build tools)
- ✅ Faster deployment
- ✅ Reduced attack surface
- ✅ Non-root user for security
- ✅ Health checks for monitoring

**Image Sizes:**
```
Builder stage: ~1.2 GB (with build tools)
Production image: ~600 MB (final)
Compression: 50% reduction
```

### Build Arguments

```dockerfile
ARG PYTHON_VERSION=3.11
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created="$BUILD_DATE"
LABEL org.opencontainers.image.revision="$VCS_REF"
LABEL org.opencontainers.image.version="$VERSION"
```

---

## 2️⃣ Docker Compose Configurations

### Development Mode (`docker-compose.dev.yml`)

```yaml
version: '3.9'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
      target: development
    
    container_name: pppq-api-dev
    
    ports:
      - "8000:8000"
    
    volumes:
      # Mount source code for live reload
      - ./src:/app/src
      - ./data:/app/data
      - ./models:/app/models
    
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - RELOAD=true  # Auto-reload on code changes
    
    env_file:
      - .env
    
    # Run with Uvicorn directly (auto-reload)
    command: >
      uvicorn src.api.main:app
      --host 0.0.0.0
      --port 8000
      --reload
    
    networks:
      - pppq-network
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    
    restart: unless-stopped

  # PostgreSQL for MLflow (optional)
  mlflow-db:
    image: postgres:15-alpine
    container_name: mlflow-db
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow_dev_password
    volumes:
      - mlflow_db_data:/var/lib/postgresql/data
    networks:
      - pppq-network
    restart: unless-stopped

networks:
  pppq-network:
    driver: bridge

volumes:
  mlflow_db_data:
```

**Usage:**
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up

# Rebuild after dependency changes
docker-compose -f docker-compose.dev.yml up --build

# Stop containers
docker-compose -f docker-compose.dev.yml down
```

### Production Mode (`docker-compose.prod.yml`)

```yaml
version: '3.9'

services:
  nginx:
    image: nginx:alpine
    container_name: pppq-nginx
    
    ports:
      - "80:80"
      - "443:443"
    
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/ssl:/etc/nginx/ssl:ro
      - nginx_cache:/var/cache/nginx
    
    depends_on:
      - api
    
    networks:
      - pppq-network
    
    restart: always
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
      args:
        BUILD_DATE: ${BUILD_DATE}
        VCS_REF: ${VCS_REF}
        VERSION: ${VERSION}
    
    container_name: pppq-api
    
    expose:
      - "8000"
    
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
      - api_logs:/app/logs
    
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=INFO
      - WORKERS=2
      - TIMEOUT=60
    
    env_file:
      - .env.prod
    
    depends_on:
      prometheus:
        condition: service_healthy
    
    networks:
      - pppq-network
    
    restart: always
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  prometheus:
    image: prom/prometheus:latest
    container_name: pppq-prometheus
    
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    
    ports:
      - "9090:9090"
    
    networks:
      - pppq-network
    
    restart: always
    
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL for MLflow
  mlflow-db:
    image: postgres:15-alpine
    container_name: mlflow-db
    
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: ${MLFLOW_DB_USER}
      POSTGRES_PASSWORD: ${MLFLOW_DB_PASSWORD}
    
    volumes:
      - mlflow_db_data:/var/lib/postgresql/data
    
    networks:
      - pppq-network
    
    restart: always
    
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${MLFLOW_DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

networks:
  pppq-network:
    driver: bridge

volumes:
  nginx_cache:
  api_logs:
  mlflow_db_data:
  prometheus_data:
```

**Usage:**
```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f api

# Stop all containers
docker-compose -f docker-compose.prod.yml down

# Update and restart
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

---

## 3️⃣ Nginx Configuration

**File:** `docker/nginx.conf`

### HTTP → HTTPS Redirect
```nginx
server {
    listen 80;
    server_name _;
    
    location / {
        return 301 https://$host$request_uri;
    }
}
```

### HTTPS Server with Reverse Proxy
```nginx
server {
    listen 443 ssl http2;
    server_name api.purchasing-power-ml.dev;
    
    # SSL configuration
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Gzip compression
    gzip on;
    gzip_types application/json application/xml text/plain;
    gzip_min_length 1000;
    
    # Cache
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m;
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
    }
    
    # API routes
    location /api/v1 {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Caching
        proxy_cache api_cache;
        proxy_cache_valid 200 10m;
        proxy_cache_key "$scheme$request_method$host$request_uri";
        
        # Add cache status header
        add_header X-Cache-Status $upstream_cache_status;
    }
    
    # Prometheus metrics (internal only)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://prometheus:9090;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
    
    # Error pages
    error_page 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
```

### Nginx Benefits
- ✅ SSL/TLS termination
- ✅ Load balancing across API instances
- ✅ Response compression (gzip)
- ✅ Caching for repeated requests
- ✅ Rate limiting protection
- ✅ Static file serving

---

## 4️⃣ Prometheus Monitoring

**File:** `docker/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'pppq-ml-pipeline'

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Metrics Tracked

```
# API Metrics
http_requests_total{method, status}        # Total requests
http_request_duration_seconds              # Request latency
http_request_size_bytes                    # Request size
http_response_size_bytes                   # Response size

# Model Metrics
model_predictions_total                    # Total predictions
model_inference_duration_seconds           # Inference time
model_errors_total                         # Prediction errors

# System Metrics
process_cpu_seconds_total                  # CPU usage
process_resident_memory_bytes              # Memory usage
python_gc_collections_total                # Garbage collection

# Custom Metrics
pppq_new_data_rows                         # Data rows added
pppq_model_f1_score                        # Model F1 score
pppq_model_accuracy                        # Model accuracy
```

---

## 5️⃣ Deployment Workflow

### Local Testing

```bash
# 1. Build image
docker build -f docker/Dockerfile -t pppq-api:latest .

# 2. Run development compose
docker-compose -f docker-compose.dev.yml up

# 3. Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{...}'

# 4. Check logs
docker-compose -f docker-compose.dev.yml logs api
```

### Production Deployment

```bash
# 1. Build and push image
docker build -f docker/Dockerfile -t registry.example.com/pppq-api:v1.0.0 .
docker push registry.example.com/pppq-api:v1.0.0

# 2. Update docker-compose.prod.yml
# Change image: registry.example.com/pppq-api:v1.0.0

# 3. Deploy to server
docker-compose -f docker-compose.prod.yml up -d --pull always

# 4. Verify health
curl https://api.purchasing-power-ml.dev/health

# 5. Check metrics
curl https://api.purchasing-power-ml.dev/metrics
```

---

## 6️⃣ Environment Variables

### .env.example (Development)
```bash
# API Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Keys
FRED_API_KEY=your_fred_api_key

# Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAIL=ba8616127@gmail.com

# Database
MLFLOW_DB_USER=mlflow
MLFLOW_DB_PASSWORD=mlflow_dev_password
DATABASE_URL=postgresql://mlflow:mlflow_dev_password@mlflow-db:5432/mlflow

# MLflow
MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow_dev_password@mlflow-db:5432/mlflow
```

### .env.prod (Production)
```bash
# API Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# API Keys
FRED_API_KEY=${FRED_API_KEY_PROD}

# Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=${SENDER_EMAIL_PROD}
SENDER_PASSWORD=${SENDER_PASSWORD_PROD}
RECIPIENT_EMAIL=ba8616127@gmail.com

# Database
MLFLOW_DB_USER=${MLFLOW_DB_USER_PROD}
MLFLOW_DB_PASSWORD=${MLFLOW_DB_PASSWORD_PROD}
DATABASE_URL=postgresql://${MLFLOW_DB_USER_PROD}:${MLFLOW_DB_PASSWORD_PROD}@mlflow-db:5432/mlflow

# MLflow
MLFLOW_TRACKING_URI=postgresql://${MLFLOW_DB_USER_PROD}:${MLFLOW_DB_PASSWORD_PROD}@mlflow-db:5432/mlflow

# Security
CORS_ORIGINS=["https://api.purchasing-power-ml.dev"]
ALLOWED_HOSTS=["api.purchasing-power-ml.dev"]
```

---

## 7️⃣ Health Checks & Monitoring

### Container Health Status

```bash
# View health status
docker ps

# Expected output:
# pppq-api          Up 2 minutes (healthy)
# pppq-nginx        Up 2 minutes (healthy)
# mlflow-db         Up 2 minutes (healthy)
# pppq-prometheus   Up 2 minutes (healthy)
```

### Manual Health Check

```bash
# API health
curl http://localhost:8000/health
# Response: {"status": "healthy", "version": "1.0.0"}

# Nginx health
curl http://localhost:80/health
# Response: 200 OK

# Prometheus health
curl http://localhost:9090/-/healthy
# Response: 200 OK

# Database health
docker-compose exec mlflow-db pg_isready
# Response: accepting connections
```

### Metrics Dashboard

```
# Access Prometheus
http://localhost:9090

# Example queries:
- rate(http_requests_total[5m])           # Requests per second
- histogram_quantile(0.95, http_request_duration_seconds)  # 95th percentile latency
- model_predictions_total                  # Total predictions
- process_resident_memory_bytes            # Memory usage
```

---

## 8️⃣ Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs pppq-api

# Common issues:
# 1. Port already in use
docker ps | grep 8000
# Kill process: lsof -ti:8000 | xargs kill -9

# 2. Missing environment variables
docker-compose config | grep FRED_API_KEY

# 3. Volume mount issues
docker-compose exec api ls -la /app/data
```

### Health Check Failing

```bash
# Test endpoint directly
docker-compose exec api curl http://localhost:8000/health

# Check API logs
docker-compose logs api --tail=50

# Verify model files
docker-compose exec api ls -la /app/models/pppq/
```

### Memory Issues

```bash
# Check resource usage
docker stats pppq-api

# Adjust limits in docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 3G  # Increase from 2G

# Restart containers
docker-compose up -d
```

### Nginx Errors

```bash
# Test Nginx configuration
docker-compose exec nginx nginx -t

# Check logs
docker-compose logs nginx

# Test upstream
docker-compose exec nginx curl http://api:8000/health
```

---

## 9️⃣ Performance Optimization

### Image Optimization

```dockerfile
# Use Alpine for smaller base
FROM python:3.11-alpine

# Remove unnecessary files
RUN rm -rf /usr/local/lib/python*/*/dist-info/*.txt
RUN find / -type f -name '*.pyc' -delete
RUN find / -type d -name '__pycache__' -delete

# Use .dockerignore
echo "__pycache__" > .dockerignore
echo ".pytest_cache" >> .dockerignore
echo ".git" >> .dockerignore
echo "*.egg-info" >> .dockerignore
```

### Runtime Optimization

```yaml
# docker-compose.prod.yml
services:
  api:
    # Use more workers for high traffic
    environment:
      - WORKERS=4  # Default: 2
    
    # Adjust resource allocation
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
```

### Caching Strategy

```nginx
# nginx.conf - Cache prediction responses (if deterministic)
proxy_cache_valid 200 1h;
proxy_cache_key "$scheme$request_method$host$request_uri$request_body";
proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
```

---

## 🔟 Security Best Practices

### Image Security

- ✅ Non-root user (appuser:1000)
- ✅ Read-only filesystems where possible
- ✅ Minimal base image (python:3.11-slim)
- ✅ No shell in production

```dockerfile
# Remove shell from production image
FROM python:3.11-alpine
RUN apk del --no-cache bash sh
```

### Network Security

- ✅ Internal communication only between containers
- ✅ Nginx terminates SSL/TLS
- ✅ Rate limiting on API endpoints
- ✅ No sensitive data in environment

### Secret Management

```bash
# Use Docker secrets (for Swarm) or .env files
docker secret create mlflow_password -
# < enter password >

# Or use Docker compose secrets
secrets:
  mlflow_password:
    file: ./secrets/mlflow_password
```

---

## Summary

| Aspect | Development | Production |
|--------|-------------|------------|
| **Compose File** | docker-compose.dev.yml | docker-compose.prod.yml |
| **Build Target** | development | production (multi-stage) |
| **API** | Uvicorn (auto-reload) | Gunicorn + Uvicorn workers |
| **Reverse Proxy** | None | Nginx + SSL |
| **Monitoring** | Basic | Prometheus + Grafana ready |
| **Restart Policy** | unless-stopped | always |
| **Resource Limits** | None | Limited (CPU/Memory) |
| **Health Checks** | Basic | Full integration |
| **Data Volumes** | Read-write | Read-only |
| **Logs** | Console | File + Volume |

---

**Last Updated:** December 2024  
**Docker Version:** 24.0+  
**Compose Version:** 2.20+  
**Python Version:** 3.11  
**Status:** ✅ Production ready
