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
RUN mkdir -p /app/outputs /workspace/dynunet_weights

# DynUNet weights location
ENV DYNUNET_WEIGHTS_DIR=/workspace/dynunet_weights
ENV DYNUNET_WEIGHTS_FILE=model_brats_mri_segmentation.pt

# Ollama URL (points to the Ollama service in Docker Compose)
ENV OLLAMA_URL=http://ollama:11434

# Non-interactive matplotlib backend
ENV MPLBACKEND=Agg

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
