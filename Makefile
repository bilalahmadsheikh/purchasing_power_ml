# ============================================================================
# PPP-Q Investment Classifier - Makefile
# Quick commands for Docker operations
# ============================================================================

.PHONY: help build dev prod up down logs test clean

# Default target
help:
	@echo "PPP-Q Docker Commands:"
	@echo ""
	@echo "  make build      - Build Docker images"
	@echo "  make dev        - Start development environment"
	@echo "  make prod       - Start production environment"
	@echo "  make full       - Start full stack with monitoring"
	@echo "  make up         - Start default (production lite)"
	@echo "  make down       - Stop all containers"
	@echo "  make logs       - View API logs"
	@echo "  make test       - Run tests in container"
	@echo "  make shell      - Open shell in API container"
	@echo "  make clean      - Remove all containers and volumes"
	@echo "  make rebuild    - Force rebuild images"
	@echo ""

# Build production image
build:
	docker-compose -f docker/docker-compose.prod.yml build

# Development environment (with hot reload)
dev:
	docker-compose -f docker/docker-compose.dev.yml up --build

# Production environment (lite - API + Redis + Nginx)
prod:
	docker-compose -f docker/docker-compose.prod.yml up -d --build

# Full production stack (with PostgreSQL + Prometheus + Grafana)
full:
	docker-compose -f docker/docker-compose.yml up -d --build

# Start default (production lite)
up: prod

# Stop all containers
down:
	docker-compose -f docker/docker-compose.dev.yml down 2>/dev/null || true
	docker-compose -f docker/docker-compose.prod.yml down 2>/dev/null || true
	docker-compose -f docker/docker-compose.yml down 2>/dev/null || true

# View logs
logs:
	docker-compose -f docker/docker-compose.prod.yml logs -f pppq-api

# Run tests
test:
	docker-compose -f docker/docker-compose.dev.yml run --rm pppq-api python -m pytest tests/ -v

# Open shell in container
shell:
	docker-compose -f docker/docker-compose.prod.yml exec pppq-api /bin/bash

# Clean up everything
clean:
	docker-compose -f docker/docker-compose.dev.yml down -v --rmi all 2>/dev/null || true
	docker-compose -f docker/docker-compose.prod.yml down -v --rmi all 2>/dev/null || true
	docker-compose -f docker/docker-compose.yml down -v --rmi all 2>/dev/null || true
	docker system prune -f

# Force rebuild
rebuild:
	docker-compose -f docker/docker-compose.prod.yml build --no-cache

# Health check
health:
	curl -s http://localhost:8000/ | python -m json.tool

# API status
status:
	docker-compose -f docker/docker-compose.prod.yml ps
