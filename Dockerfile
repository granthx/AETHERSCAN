# Multi-stage Dockerfile for COVID-19 CT Scanner Inference Service
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY src/ ./src/
COPY app/ ./app/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/synthetic/ ./data/synthetic/

# Initialize reference model checkpoint and assets
RUN python scripts/init_model_checkpoint.py

EXPOSE 8000 8501

# Default command: run FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
