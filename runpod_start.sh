#!/bin/bash
# RunPod GPU Pod startup script for Brain Tumor Analysis Pipeline
# Usage: bash runpod_start.sh
set -e

REPO_DIR="/workspace/brain-tumor"
cd "$REPO_DIR"

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgl1 libglib2.0-0 libhdf5-dev curl wget unzip

echo "=== Installing Python dependencies ==="
pip install --no-cache-dir numpy
pip install --no-cache-dir --ignore-installed blinker
pip install --no-cache-dir streamlit -r requirements.txt

echo "=== Installing & starting Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull llama3

echo "=== Installing & starting Weaviate ==="
# Use the embedded Weaviate via the weaviate Python client (already in requirements)
# If a standalone Weaviate is needed, uncomment below:
# wget -qO weaviate https://github.com/weaviate/weaviate/releases/download/v1.28.2/weaviate-v1.28.2-linux-amd64
# chmod +x weaviate
# ./weaviate --host 0.0.0.0 --port 8080 --scheme http &

echo "=== Creating output directories ==="
mkdir -p outputs/{preprocessed,segmentation,radiomics,clinical_features,embeddings,reports/cap,reconstructed,visualizations,memory,chromadb,.jobs}

echo "=== Setting environment variables ==="
export OUTPUT_DIR="$REPO_DIR/outputs"
export DATA_DIR="$REPO_DIR/BraTS2020_training_data/content/data"
export OLLAMA_URL="http://localhost:11434"
export DYNUNET_WEIGHTS_DIR="/workspace/dynunet_weights"
export DYNUNET_WEIGHTS_FILE="model_brats_mri_segmentation.pt"
export MPLBACKEND=Agg

mkdir -p "$DYNUNET_WEIGHTS_DIR"

echo "=== Starting Streamlit UI ==="
echo "Access via RunPod proxy on port 8501"
streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --browser.gatherUsageStats=false
