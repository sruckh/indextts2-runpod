#!/bin/bash

# IndexTTS Runpod Serverless Bootstrap Script
# This script sets up the environment for IndexTTS serverless on Runpod

set -e  # Exit on any error

echo "=== IndexTTS Runpod Bootstrap Starting ==="

# Configuration
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/indextts}"
DOCKER_SRC="/opt/index-tts"
SRC_DIR="${SRC_DIR:-$INSTALL_DIR/src}"
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
    git clone --depth 1 --branch "$INDEXTTS_REF" "$INDEXTTS_REPO" "$SRC_DIR"
    (cd "$SRC_DIR" && git lfs pull)
    log "IndexTTS source cloned successfully"
else
    log "IndexTTS source already exists at $SRC_DIR"
fi

# Always copy latest handler files from Docker image
log "Copying handler files from Docker image..."
cp "$DOCKER_SRC/handler.py" "$SRC_DIR/"
cp "$DOCKER_SRC/config.py" "$SRC_DIR/"
cp "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

# Make sure the source is importable
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# Check if checkpoints exist, if not download them
if [ ! -f "$CHECKPOINTS_DIR/config.yaml" ]; then
    log "Checkpoints not found. Downloading models..."

    # Install huggingface-hub if needed
    if ! command -v hf &> /dev/null; then
        log "Installing HuggingFace CLI..."
        pip install --no-cache-dir "huggingface-hub[cli,hf_xet]"
    fi

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
