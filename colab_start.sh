#!/bin/bash
# ============================================================
# NeuroAgent – Google Colab Startup Script
# ============================================================
# HOW TO USE IN COLAB (run these cells in order):
#
#  Cell 1 – Clone your repo:
#    !git clone https://github.com/<your-username>/brain-tumor.git /content/brain-tumor
#    %cd /content/brain-tumor
#
#  Cell 2 – Point to your BraTS data (already added via Kaggle):
#    # BraTS data should be at /content/brats2020 after Kaggle import
#    # The script will read DATA_DIR from environment or default below
#
#  Cell 3 – Run this script:
#    !bash colab_start.sh
#
#  Cell 4 – Open the UI (printed at end of this script):
#    # Click the public ngrok URL printed at the bottom
# ============================================================

set -e  # Exit on any error

# ── 1. PATHS (Colab uses /content/, not /workspace/) ─────────────────
REPO_DIR="/content/brain-tumor"
DATA_DIR="${DATA_DIR:-/content/brats2020}"        # Override with your Kaggle path
OUTPUT_DIR="$REPO_DIR/outputs"
WEIGHTS_DIR="/content/dynunet_weights"

cd "$REPO_DIR"
echo "=== Working directory: $REPO_DIR ==="

# ── 2. SYSTEM DEPENDENCIES ───────────────────────────────────────────
echo ""
echo "=== [1/7] Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    build-essential gcc g++ libgl1 libglib2.0-0 curl wget unzip zstd

# ── 3. PYTHON DEPENDENCIES ───────────────────────────────────────────
echo ""
echo "=== [2/7] Installing Python dependencies ==="

# Install numpy first to avoid build conflicts
pip install --no-cache-dir -q numpy

# Avoid blinker conflict (Colab pre-installs an older version)
pip install --no-cache-dir -q --ignore-installed blinker

# PyTorch with CUDA 12.1 (matches Colab T4/A100 GPU)
pip install --no-cache-dir -q torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# All other project dependencies
pip install --no-cache-dir -q -r requirements.txt

# pyngrok to expose Streamlit UI from Colab
pip install --no-cache-dir -q pyngrok

echo "✓ Python dependencies installed."

# ── 4. OLLAMA (LLM Server) ───────────────────────────────────────────
echo ""
echo "=== [3/7] Installing and starting Ollama ==="

if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server in background
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "  Ollama server starting (PID: $OLLAMA_PID)..."

# Wait until Ollama is actually ready (up to 30 seconds)
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  ✓ Ollama is ready (took ${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ⚠ Ollama did not respond in 30s — pipeline will use rule-based fallback"
    fi
    sleep 1
done

# Pull Llama 3 model (~4GB — this will take a few minutes)
echo "  Pulling Llama 3 model (this may take 5–10 min on first run)..."
ollama pull llama3 || echo "  ⚠ Llama 3 pull failed — pipeline will use rule-based fallback"

# ── 5. MONAI MODEL WEIGHTS ───────────────────────────────────────────
echo ""
echo "=== [4/7] Downloading MONAI SegResNet weights ==="

mkdir -p "$WEIGHTS_DIR"

# Check if weights already downloaded (avoid re-downloading)
if [ ! -f "$WEIGHTS_DIR/model_mri_segmentation.pt" ]; then
    python - <<'EOF'
import os
weights_dir = os.environ.get("WEIGHTS_DIR", "/content/dynunet_weights")
try:
    from monai.bundle import download
    download("brats_mri_segmentation", bundle_dir=weights_dir)
    print("  ✓ MONAI weights downloaded to:", weights_dir)
except Exception as e:
    print(f"  ⚠ MONAI bundle download failed: {e}")
    print("  Pipeline will fall back to ground-truth masks if available.")
EOF
else
    echo "  ✓ Weights already present — skipping download."
fi

# ── 6. OUTPUT DIRECTORIES ────────────────────────────────────────────
echo ""
echo "=== [5/7] Creating output directories ==="
mkdir -p "$OUTPUT_DIR"/{preprocessed,segmentation,radiomics,clinical_features,\
embeddings,reports/cap,reconstructed,visualizations,memory,chromadb,.jobs}
echo "  ✓ Output directories created at: $OUTPUT_DIR"

# ── 7. ENVIRONMENT VARIABLES ─────────────────────────────────────────
echo ""
echo "=== [6/7] Setting environment variables ==="
export OUTPUT_DIR="$OUTPUT_DIR"
export DATA_DIR="$DATA_DIR"
export OLLAMA_URL="http://localhost:11434"
export DYNUNET_WEIGHTS_DIR="$WEIGHTS_DIR"
export DYNUNET_WEIGHTS_FILE="model_mri_segmentation.pt"
export MPLBACKEND=Agg   # Prevent matplotlib GUI errors in headless Colab

echo "  DATA_DIR      = $DATA_DIR"
echo "  OUTPUT_DIR    = $OUTPUT_DIR"
echo "  WEIGHTS_DIR   = $WEIGHTS_DIR"
echo "  OLLAMA_URL    = $OLLAMA_URL"

# Validate that BraTS data actually exists
if [ ! -d "$DATA_DIR" ]; then
    echo ""
    echo "  ⚠ WARNING: BraTS data not found at $DATA_DIR"
    echo "  Please set the DATA_DIR environment variable to your Kaggle dataset path."
    echo "  Example: export DATA_DIR=/content/brats2020/BraTS2020_TrainingData"
    echo "  The pipeline will still start but will find 0 patients."
fi

# ── 8. STREAMLIT UI via ngrok tunnel ─────────────────────────────────
echo ""
echo "=== [7/7] Starting Streamlit UI ==="

# Start Streamlit in background
streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --browser.gatherUsageStats=false \
    > /tmp/streamlit.log 2>&1 &

STREAMLIT_PID=$!
echo "  Streamlit starting (PID: $STREAMLIT_PID)..."
sleep 5

# Open a public URL via ngrok
python - <<'EOF'
from pyngrok import ngrok
import time

try:
    # Open tunnel on port 8501
    public_url = ngrok.connect(8501)
    print("\n" + "="*60)
    print("  ✓ NeuroAgent is LIVE!")
    print(f"  🌐 Open this URL in your browser:")
    print(f"  👉  {public_url}")
    print("="*60 + "\n")
    print("  Tip: Keep this Colab cell running to maintain the tunnel.")
    print("  DATA_DIR must contain BraTS patient folders for the pipeline to run.")
except Exception as e:
    print(f"  ⚠ ngrok tunnel failed: {e}")
    print("  Try: !pip install pyngrok and set your ngrok auth token.")
    print("  Or access Streamlit directly at http://localhost:8501 if running locally.")
EOF
