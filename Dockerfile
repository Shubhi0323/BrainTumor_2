FROM python:3.11-slim

# System dependencies for SimpleITK, pyradiomics, matplotlib, and HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libhdf5-dev \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install numpy first (required by pyradiomics setup.py)
RUN pip install --no-cache-dir numpy

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /app/outputs /workspace/nnUNet_results /workspace/nnUNet_raw /workspace/nnUNet_preprocessed

# nnU-Net environment
ENV nnUNet_results=/workspace/nnUNet_results
ENV nnUNet_raw=/workspace/nnUNet_raw
ENV nnUNet_preprocessed=/workspace/nnUNet_preprocessed

# Ollama URL (points to the Ollama service in Docker Compose)
ENV OLLAMA_URL=http://ollama:11434

# Non-interactive matplotlib backend
ENV MPLBACKEND=Agg

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
