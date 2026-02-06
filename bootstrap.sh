#!/bin/bash

# IndexTTS Runpod Serverless Bootstrap Script
# This script sets up the environment for IndexTTS serverless on Runpod
#
# First boot: clones source, creates venv, installs all Python packages,
#             downloads model checkpoints (~15-20 min)
# Subsequent boots: activates existing venv, copies latest handler files (~seconds)

set -e  # Exit on any error

echo "=== IndexTTS Runpod Bootstrap Starting ==="

# Configuration
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/indextts}"
DOCKER_SRC="/opt/index-tts"
SRC_DIR="${SRC_DIR:-$INSTALL_DIR/src}"
VENV_DIR="${VENV_DIR:-$INSTALL_DIR/venv}"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$INSTALL_DIR/checkpoints}"

# Git clone settings
INDEXTTS_REPO="${INDEXTTS_REPO:-https://github.com/index-tts/index-tts.git}"
INDEXTTS_REF="${INDEXTTS_REF:-main}"

# Logging
LOG_FILE="$INSTALL_DIR/bootstrap.log"
mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Log file: $LOG_FILE"
echo "Install directory: $INSTALL_DIR"
echo "Source directory: $SRC_DIR"
echo "Venv directory: $VENV_DIR"
echo "Docker source: $DOCKER_SRC"
echo "Checkpoints directory: $CHECKPOINTS_DIR"

# Function to print with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Ensure required directories exist
log "Ensuring required directories exist..."
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$CHECKPOINTS_DIR"

# Clone IndexTTS source to network volume (first time only)
if [ ! -d "$SRC_DIR/.git" ]; then
    log "Cloning IndexTTS source to $SRC_DIR..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$INDEXTTS_REF" "$INDEXTTS_REPO" "$SRC_DIR"
    log "IndexTTS source cloned successfully"
else
    log "IndexTTS source already exists at $SRC_DIR"
fi

# Always copy latest handler files from Docker image
log "Copying handler files from Docker image..."
cp "$DOCKER_SRC/handler.py" "$SRC_DIR/"
cp "$DOCKER_SRC/config.py" "$SRC_DIR/"
cp "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

# Create Python virtual environment and install dependencies (first time only)
if [ ! -d "$VENV_DIR/bin/activate" ]; then
    log "=== First-time setup: creating virtual environment ==="
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    log "Installing uv package manager..."
    pip install --no-cache-dir uv
    export UV_LINK_MODE=copy

    log "Installing PyTorch (CUDA 12.8)..."
    uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128 || \
        pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cu128

    log "Installing flash-attention..."
    pip install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

    log "Installing IndexTTS (--no-deps, pins are incompatible with Python 3.12)..."
    (cd "$SRC_DIR" && pip install --no-cache-dir --no-deps -e .)

    log "Installing IndexTTS runtime dependencies..."
    pip install --no-cache-dir \
        numpy safetensors einops huggingface-hub modelscope \
        pyyaml tqdm transformers accelerate \
        "numba>=0.59" "llvmlite>=0.42" \
        librosa soundfile pysoundfile \
        whisper-timestamped

    log "Installing RunPod and serverless dependencies..."
    pip install --no-cache-dir \
        runpod==1.6.1 \
        "uvicorn[standard]" \
        pydantic \
        python-multipart \
        tqdm \
        boto3

    log "Installing HuggingFace CLI..."
    pip install --no-cache-dir "huggingface-hub[cli,hf_xet]"

    log "=== Virtual environment setup complete ==="
else
    log "Activating existing virtual environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Make sure the source is importable
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# Check if checkpoints exist, if not download them
if [ ! -f "$CHECKPOINTS_DIR/config.yaml" ]; then
    log "Checkpoints not found. Downloading models..."

    # Download models using hf CLI
    log "Downloading IndexTTS-2 models from HuggingFace..."
    if [ -n "$HF_TOKEN" ]; then
        hf download IndexTeam/IndexTTS-2 --local-dir="$CHECKPOINTS_DIR" --token="$HF_TOKEN" || {
            log "WARNING: Failed to download with token, trying without..."
            hf download IndexTeam/IndexTTS-2 --local-dir="$CHECKPOINTS_DIR"
        }
    else
        log "HF_TOKEN not set, downloading without authentication..."
        hf download IndexTeam/IndexTTS-2 --local-dir="$CHECKPOINTS_DIR"
    fi

    log "Model download complete"
else
    log "Checkpoints already exist at $CHECKPOINTS_DIR"
fi

# Verify checkpoints
if [ -f "$CHECKPOINTS_DIR/config.yaml" ]; then
    log "✓ Config file found"
else
    log "ERROR: Config file not found after download"
    ls -la "$CHECKPOINTS_DIR" || true
fi

# Start handler (runpod serverless mode)
log "Starting RunPod handler..."
log "Container ready for requests"
exec python "$SRC_DIR/handler.py"
