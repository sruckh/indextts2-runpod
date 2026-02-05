#!/bin/bash

# IndexTTS Runpod Serverless Bootstrap Script
# This script sets up the environment for IndexTTS serverless on Runpod

set -e  # Exit on any error

echo "=== IndexTTS Runpod Bootstrap Starting ==="

# Configuration
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/indextts}"
SRC_DIR="${SRC_DIR:-/opt/index-tts}"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$INSTALL_DIR/checkpoints}"

# Logging
LOG_FILE="$INSTALL_DIR/bootstrap.log"
mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Log file: $LOG_FILE"
echo "Install directory: $INSTALL_DIR"
echo "Source directory: $SRC_DIR"
echo "Checkpoints directory: $CHECKPOINTS_DIR"

# Function to print with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Ensure required directories exist
log "Ensuring required directories exist..."
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$CHECKPOINTS_DIR"

# Make sure the shipped source is importable
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
