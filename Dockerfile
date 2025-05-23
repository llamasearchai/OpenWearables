# Multi-stage build for OpenWearables
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/config

# Set permissions
RUN chmod -R 755 /app

# Expose ports
EXPOSE 5000 8000

# Development command
CMD ["python", "-m", "openwearables.ui.app"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r openwearables && useradd -r -g openwearables openwearables

# Copy only necessary files
COPY openwearables/ ./openwearables/
COPY pyproject.toml setup.py README.md ./

# Install package
RUN pip install .

# Create directories and set ownership
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R openwearables:openwearables /app

# Switch to non-root user
USER openwearables

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/system/status || exit 1

# Expose port
EXPOSE 5000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "openwearables.ui.app:app"]

# GPU-enabled stage for CUDA support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python commands
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install dependencies with CUDA support
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source code
COPY openwearables/ ./openwearables/
COPY pyproject.toml setup.py README.md ./

# Install package
RUN pip install .

# Create directories
RUN mkdir -p /app/data /app/logs /app/config

# Expose port
EXPOSE 5000

# GPU-enabled command
CMD ["python", "-m", "openwearables.cli.openwearables_cli", "start", "--port", "5000"] 