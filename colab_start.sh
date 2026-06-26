#!/bin/bash
# ============================================================
# NeuroAgent – Google Colab Startup Script (v3 – fully fixed)
# ============================================================
# HOW TO USE IN COLAB (run these cells in order):
#
#  Cell 1 – Download BraTS dataset via Kaggle:
#    import kagglehub
#    path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
#    print("Dataset path:", path)
#
#  Cell 2 – Clone repo and set data path:
#    !git clone https://github.com/Shubhi0323/BrainTumor_2.git /content/brain-tumor
#    %cd /content/brain-tumor
#    import os
#    os.environ["DATA_DIR"] = path   # path from Cell 1
#
#  Cell 3 – Run this script:
#    !bash colab_start.sh
#
#  Cell 4 – Get the public URL (no password needed):
#    from google.colab.output import eval_js
#    print(eval_js("google.colab.kernel.proxyPort(8501)"))
# ============================================================

# Do NOT use set -e — pip version conflicts would abort the script
set +e

# ── 0. PATHS ──────────────────────────────────────────────────
# NOTE: kagglehub is installed AFTER all LangChain packages are pinned
# (see step 2 below) so it cannot pull in langchain-core>=1.0 before the
# ceiling constraint is in place.
REPO_DIR="/content/brain-tumor"
OUTPUT_DIR="$REPO_DIR/outputs"
WEIGHTS_DIR="/content/dynunet_weights"

cd "$REPO_DIR"
echo "=== NeuroAgent Colab Setup ==="
echo "    Working dir : $REPO_DIR"
echo "    Data dir    : $DATA_DIR"

# ── 2. SYSTEM DEPENDENCIES ───────────────────────────────────────────
echo ""
echo "=== [1/7] Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential gcc g++ libgl1 libglib2.0-0 curl wget unzip zstd \
    pciutils lshw
echo "  ✓ System packages ready."

# ── 3. PYTHON DEPENDENCIES (pinned to avoid conflicts) ───────────────
echo ""
echo "=== [2/7] Installing Python dependencies ==="

# FIX: Pin torch to exact version to avoid torchaudio conflict
# Colab T4 supports CUDA 12.1
pip install --no-cache-dir -q \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# FIX: Pin starlette/fastapi to compatible versions (resolves Gradio + FastAPI conflict)
pip install --no-cache-dir -q \
    "starlette==0.41.3" \
    "fastapi==0.115.5"

# FIX: Pin the full LangChain ecosystem with a hard <0.4 ceiling BEFORE
# requirements.txt runs. The ceiling stops any package (kagglehub,
# mcp-cli, langgraph-sdk, etc.) from silently upgrading langchain-core
# to the 1.x series which breaks langchain 0.3.x.
pip install --no-cache-dir -q \
    "langchain==0.3.7" \
    "langchain-core>=0.3.43,<0.4" \
    "langchain-community==0.3.7" \
    "langchain-text-splitters==0.3.2" \
    "langgraph==0.2.45"

# Avoid blinker conflict (Colab pre-installs an older version)
pip install --no-cache-dir -q --ignore-installed blinker

# Install numpy first (must be <2.0 for MONAI compatibility)
pip install --no-cache-dir -q "numpy<2.0"

# All other project dependencies (langchain-core ceiling in requirements.txt
# acts as a second guard during this install)
pip install --no-cache-dir -q -r requirements.txt

# Install kagglehub AFTER all LangChain packages are locked so pip cannot
# use kagglehub's transitive deps to break the ceiling.
echo ""
echo "=== [2b/7] Downloading BraTS Dataset ==="
pip install -q kagglehub
export DATA_DIR=$(python3 -c "import kagglehub; print(kagglehub.dataset_download('awsaf49/brats20-dataset-training-validation'))")
echo "  ✓ Dataset downloaded to: $DATA_DIR"

echo "  ✓ Python dependencies installed."

# ── 4. OLLAMA + GPU FIX ──────────────────────────────────────────────
echo ""
echo "=== [3/7] Installing and starting Ollama ==="

if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# FIX: Tell Ollama to use the Colab GPU explicitly
# Check if NVIDIA GPU is available
if nvidia-smi &>/dev/null; then
    echo "  ✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    export CUDA_VISIBLE_DEVICES=0
    export OLLAMA_NUM_GPU=999   # Use all GPU layers
else
    echo "  ⚠ No NVIDIA GPU found — Ollama will run on CPU (slower)"
fi

# Start Ollama server in background
OLLAMA_HOST=0.0.0.0 ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "  Ollama server starting (PID: $OLLAMA_PID)..."

# Wait until Ollama is actually ready (up to 40 seconds)
for i in $(seq 1 40); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  ✓ Ollama is ready (took ${i}s)"
        break
    fi
    if [ "$i" -eq 40 ]; then
        echo "  ⚠ Ollama did not respond in 40s — pipeline will use rule-based fallback"
    fi
    sleep 1
done

# Pull Llama 3 model (~4GB — takes a few minutes on first run)
echo "  Pulling Llama 3 model (5–10 min on first run)..."
ollama pull llama3 && echo "  ✓ Llama 3 ready." || \
    echo "  ⚠ Llama 3 pull failed — pipeline will use rule-based fallback"

# ── 5. MONAI MODEL WEIGHTS ───────────────────────────────────────────
echo ""
echo "=== [4/7] Downloading MONAI SegResNet weights ==="

mkdir -p "$WEIGHTS_DIR"

if [ ! -f "$WEIGHTS_DIR/model_mri_segmentation.pt" ]; then
    # FIX: Suppress TF/CUDA warnings from MONAI download (cosmetic only)
    TF_CPP_MIN_LOG_LEVEL=3 python - <<EOF
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
weights_dir = "$WEIGHTS_DIR"
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
echo "=== [6/7] Configuring environment ==="
export OUTPUT_DIR="$OUTPUT_DIR"
export DATA_DIR="$DATA_DIR"
export OLLAMA_URL="http://localhost:11434"
export DYNUNET_WEIGHTS_DIR="$WEIGHTS_DIR"
export DYNUNET_WEIGHTS_FILE="model_mri_segmentation.pt"
export MPLBACKEND=Agg        # Prevent matplotlib GUI errors in headless Colab
export TF_CPP_MIN_LOG_LEVEL=3   # Suppress TF/CUDA registration warnings

echo "  DATA_DIR    = $DATA_DIR"
echo "  OUTPUT_DIR  = $OUTPUT_DIR"
echo "  WEIGHTS_DIR = $WEIGHTS_DIR"
echo "  OLLAMA_URL  = $OLLAMA_URL"

# Validate BraTS data
if [ ! -d "$DATA_DIR" ]; then
    echo ""
    echo "  ⚠ WARNING: BraTS data not found at $DATA_DIR"
    echo "  Set DATA_DIR before running, e.g.:"
    echo "    import os; os.environ['DATA_DIR'] = path  # path from kagglehub"
    echo "  Pipeline will start but find 0 patients."
fi

# ── 8. STREAMLIT UI ──────────────────────────────────────────────────
echo ""
echo "=== [7/7] Starting Streamlit UI ==="

streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false \
    > /tmp/streamlit.log 2>&1 &

STREAMLIT_PID=$!
echo "  Streamlit starting (PID: $STREAMLIT_PID)..."

# Wait until Streamlit is actually up
for i in $(seq 1 30); do
    if curl -s http://localhost:8501 >/dev/null 2>&1; then
        echo "  ✓ Streamlit is ready (took ${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ⚠ Streamlit did not start in 30s. Check: cat /tmp/streamlit.log"
    fi
    sleep 1
done

# ── 9. PUBLIC URL (Cloudflare Tunnel — most reliable, no auth needed) ──
echo ""
echo "  Setting up public URL via Cloudflare Tunnel..."

# Download cloudflared if not present
if ! command -v cloudflared &>/dev/null; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# Start cloudflare tunnel and capture the URL
cloudflared tunnel --url http://localhost:8501 > /tmp/cf.log 2>&1 &
CF_PID=$!

# Wait for cloudflare to print the URL (up to 15 seconds)
CF_URL=""
for i in $(seq 1 15); do
    CF_URL=$(grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' /tmp/cf.log | head -1)
    if [ -n "$CF_URL" ]; then
        break
    fi
    sleep 1
done

echo ""
echo "============================================================"
if [ -n "$CF_URL" ]; then
    echo "  ✓ NeuroAgent is LIVE!"
    echo ""
    echo "  🌐 Public URL (share this link):"
    echo "  👉  $CF_URL"
    echo ""
    echo "  No password, no account needed — click and it opens."
else
    echo "  ⚠ Cloudflare tunnel URL not detected."
    echo "  Use Colab's built-in proxy instead — run in a new cell:"
    echo ""
    echo "  from google.colab.output import eval_js"
    echo "  print(eval_js('google.colab.kernel.proxyPort(8501)'))"
    cat /tmp/cf.log 2>/dev/null | tail -5
fi
echo "============================================================"
echo ""
echo "  Keep this Colab session alive to maintain the tunnel."
echo "  DATA_DIR must point to BraTS patient folders."
echo "  Streamlit log : /tmp/streamlit.log"
echo "  Ollama log    : /tmp/ollama.log"
