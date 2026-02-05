# coding=utf-8
# IndexTTS RunPod Serverless Configuration
# SPDX-License-Identifier: MIT

"""
Configuration module for IndexTTS RunPod Serverless.

Centralizes all environment variables, model paths, and configuration constants.
Following Fish Audio/Echo-TTS pattern for consistency across TTS services.
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# =============================================================================
# HuggingFace Configuration
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

# =============================================================================
# S3 Configuration (required for production audio output storage)
# =============================================================================

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# =============================================================================
# RunPod Volume Structure
# =============================================================================

RUNPOD_VOLUME = "/runpod-volume"
INDEXTTS_DIR = f"{RUNPOD_VOLUME}/indextts"
AUDIO_VOICES_DIR = Path(os.environ.get("AUDIO_VOICES_DIR", f"{INDEXTTS_DIR}/audio_voices"))
OUTPUT_AUDIO_DIR = Path(os.environ.get("OUTPUT_AUDIO_DIR", f"{INDEXTTS_DIR}/output_audio"))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", f"{INDEXTTS_DIR}/checkpoints"))

# =============================================================================
# Audio Configuration
# =============================================================================

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}

# Default output sample rate (IndexTTS outputs at 24kHz)
DEFAULT_SAMPLE_RATE = 24000

# =============================================================================
# Model Configuration
# =============================================================================

# Device configuration
DEVICE = "cuda" if os.environ.get("DEVICE") != "cpu" else "cpu"

# Model settings
DEFAULT_USE_FP16 = os.environ.get("USE_FP16", "false").lower() == "true"
DEFAULT_USE_DEEPSPEED = os.environ.get("USE_DEEPSPEED", "false").lower() == "true"
DEFAULT_USE_CUDA_KERNEL = os.environ.get("USE_CUDA_KERNEL", "false").lower() == "true"

# Max tokens per generation segment (controls VRAM usage)
DEFAULT_MAX_TOKENS_PER_SEGMENT = int(os.environ.get("MAX_TOKENS_PER_SEGMENT", "800"))

# Model source - HuggingFace or ModelScope
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "huggingface")  # or "modelscope"
MODEL_REPO = os.environ.get("MODEL_REPO", "IndexTeam/IndexTTS-2")

# =============================================================================
# Generation Parameter Defaults
# =============================================================================

# Emotion vector defaults [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
DEFAULT_EMO_VECTOR = None
DEFAULT_EMO_ALPHA = 1.0
DEFAULT_USE_RANDOM = False
DEFAULT_USE_EMO_TEXT = False

# =============================================================================
# File Cleanup Configuration
# =============================================================================

CLEANUP_DAYS = 2  # Delete output files older than this many days

# =============================================================================
# Config Class (for runtime validation and logging)
# =============================================================================

class Config:
    """
    Configuration validation and storage.

    Validates required environment variables and creates necessary directories.
    """

    def __init__(self):
        self.validation_errors = []

        # Basic hardware detection
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            log.info(f"GPU detected: {self.gpu_name} with {self.gpu_memory:.1f}GB memory")

        # Required environment variables
        self.HF_TOKEN = HF_TOKEN
        
        # S3 Configuration (required for production)
        self.S3_ENDPOINT_URL = S3_ENDPOINT_URL
        self.S3_ACCESS_KEY_ID = S3_ACCESS_KEY_ID
        self.S3_SECRET_ACCESS_KEY = S3_SECRET_ACCESS_KEY
        self.S3_BUCKET_NAME = S3_BUCKET_NAME
        self.S3_REGION = S3_REGION

        # Check if S3 is properly configured
        s3_missing = [
            var for var in ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
            if not getattr(self, var)
        ]
        if s3_missing:
            self.validation_errors.append(f"S3 configuration missing: {', '.join(s3_missing)}")

        # Directory configuration
        self.AUDIO_VOICES_DIR = AUDIO_VOICES_DIR
        self.OUTPUT_AUDIO_DIR = OUTPUT_AUDIO_DIR
        self.CHECKPOINTS_DIR = CHECKPOINTS_DIR

        # Ensure directories exist
        try:
            self.AUDIO_VOICES_DIR.mkdir(parents=True, exist_ok=True)
            self.OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            self.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            log.info(f"Audio directories: {self.AUDIO_VOICES_DIR}, {self.OUTPUT_AUDIO_DIR}")
            log.info(f"Checkpoints directory: {self.CHECKPOINTS_DIR}")
        except Exception as e:
            self.validation_errors.append(f"Failed to create directories: {e}")

        # Model configuration
        self.use_fp16 = DEFAULT_USE_FP16
        self.use_deepspeed = DEFAULT_USE_DEEPSPEED
        self.use_cuda_kernel = DEFAULT_USE_CUDA_KERNEL
        self.max_tokens_per_segment = DEFAULT_MAX_TOKENS_PER_SEGMENT
        self.model_source = MODEL_SOURCE
        self.model_repo = MODEL_REPO

        # Additional configuration
        self.AUDIO_EXTS = AUDIO_EXTS
        self.sample_rate = DEFAULT_SAMPLE_RATE

        # Log all environment variables (without sensitive data)
        log.info(f"Device: {self.device}")
        log.info(f"FP16: {self.use_fp16}, DeepSpeed: {self.use_deepspeed}, CUDA Kernel: {self.use_cuda_kernel}")
        log.debug(f"AUDIO_VOICES_DIR: {self.AUDIO_VOICES_DIR}")
        log.debug(f"OUTPUT_AUDIO_DIR: {'SET' if self.OUTPUT_AUDIO_DIR else 'NOT SET'}")
        log.debug(f"CHECKPOINTS_DIR: {self.CHECKPOINTS_DIR}")
        log.debug(f"S3_ENDPOINT_URL: {'SET' if self.S3_ENDPOINT_URL else 'NOT SET'}")
        log.info(f"S3_BUCKET_NAME: {'SET' if self.S3_BUCKET_NAME else 'NOT SET'}")
        log.info(f"HF_TOKEN: {'SET' if self.HF_TOKEN else 'NOT SET'}")

        # Check audio files in voices directory
        try:
            audio_files = list(self.AUDIO_VOICES_DIR.glob("*"))
            audio_files = [f for f in audio_files if f.suffix.lower() in self.AUDIO_EXTS]
            log.debug(f"Found {len(audio_files)} audio files")
            for f in audio_files[:5]:  # Log first 5
                log.debug(f"  - {f.name}")
            if len(audio_files) > 5:
                log.debug(f"  ... and {len(audio_files) - 5} more")
        except Exception as e:
            log.warning(f"Could not scan audio directory: {e}")

    def validate(self) -> bool:
        """Return True if configuration is valid."""
        if self.validation_errors:
            log.error("Configuration validation failed:")
            for error in self.validation_errors:
                log.error(f"  - {error}")
            return False
        return True


# Global configuration instance (initialized at module load)
config = Config()
