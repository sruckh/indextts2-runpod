# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IndexTTS RunPod Serverless is a cloud-based Text-to-Speech API worker that runs on RunPod's serverless GPU infrastructure. It provides high-quality speech synthesis with voice cloning and emotion control capabilities using the IndexTTS-2 model.

## Architecture

### Core Modules

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless entry point - validates requests, manages S3 uploads, health checks |
| `config.py` | Centralized configuration with environment variables and validation |
| `serverless_engine.py` | Inference wrapper with lazy model loading, text chunking, and crossfade |
| `bootstrap.sh` | Container initialization - creates directories, downloads models from HuggingFace |
| `Dockerfile` | Multi-stage CUDA 12.8 build with IndexTTS installed from GitHub |

### Request Flow

1. RunPod calls `handler(job)` with `{"input": {...}}`
2. `extract_and_validate_params()` validates and sanitizes inputs
3. `get_inference_engine()` returns lazy-loaded singleton `IndexTTSInference`
4. For long texts: `chunk_text()` splits at sentence/clause boundaries
5. Each chunk is synthesized via `IndexTTS2.infer()` from the `indextts` package
6. Chunks are combined with optional crossfade
7. Audio is encoded to Opus via ffmpeg and uploaded to S3
8. Returns presigned URL and metadata

### Key Patterns

- **Lazy Loading**: IndexTTS model loads on first inference request via `_load_model()`
- **Global Singleton**: `get_inference_engine()` returns cached `_inference_engine` instance
- **Session-based**: Each request generates unique filename (from `session_id` or UUID)
- **Path Security**: Speaker/emotion audio files are validated to stay within `AUDIO_VOICES_DIR`

## Common Commands

### Build and Run Locally

```bash
# Build Docker image
docker build -t indextts-runpod .

# Run handler locally (requires S3 env vars)
export S3_ENDPOINT_URL="..."
export S3_ACCESS_KEY_ID="..."
export S3_SECRET_ACCESS_KEY="..."
export S3_BUCKET_NAME="..."
python handler.py

# Warmup models (load to cache, then exit)
python handler.py --warmup
```

### Local Testing (Manual)

```python
# Test inference directly
from serverless_engine import get_inference_engine

engine = get_inference_engine()
audio, sr = engine.generate_speech(
    text="Hello world",
    speaker_voice="my_voice.wav",
    enable_chunking=True
)
```

## Environment Variables

### Required (Production)
- `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME` - For audio uploads

### Optional
- `HF_TOKEN` - HuggingFace token for gated model access
- `AUDIO_VOICES_DIR` - Voice reference files (default: `/runpod-volume/indextts/audio_voices`)
- `OUTPUT_AUDIO_DIR` - Temporary output (default: `/runpod-volume/indextts/output_audio`)
- `CHECKPOINTS_DIR` - Model files (default: `/runpod-volume/indextts/checkpoints`)
- `USE_FP16=true` - Enable FP16 for 2x speed, ~50% less VRAM
- `USE_DEEPSPEED=true` - Experimental DeepSpeed optimization
- `USE_CUDA_KERNEL=true` - CUDA kernel optimization
- `MAX_TOKENS_PER_SEGMENT` - VRAM control (default: 800, lower for 8GB GPUs)

## Directory Structure (RunPod Volume)

```
/runpod-volume/indextts/
├── audio_voices/          # Reference audio for voice cloning (my_voice.wav, etc.)
├── output_audio/          # Generated audio (temporary, auto-cleanup)
├── checkpoints/           # Model files downloaded from HuggingFace
│   ├── config.yaml
│   ├── bigvgan.bin
│   └── ...
└── bootstrap.log          # Startup logs
```

## Model Source

IndexTTS is installed from the upstream GitHub repo during Docker build:
- Repo: `https://github.com/index-tts/index-tts.git`
- Branch: `main` (configurable via `INDEXTTS_REF` build arg)
- Models downloaded from: `IndexTeam/IndexTTS-2` on HuggingFace

## API Reference

### Input Parameters (handler.py:278)

- `text` (required): Text to synthesize, max 10000 chars
- `speaker_voice`: Filename in `AUDIO_VOICES_DIR` for voice cloning
- `emo_audio_prompt`: Filename for emotion reference
- `emo_vector`: 8 floats `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`
- `emo_alpha`: Emotion strength 0.0-1.0
- `enable_chunking`: Split long text (default: true)
- `max_chars_per_chunk`: 50-1000 chars (default: 300)
- `enable_crossfade`: Smooth chunk transitions (default: true)
- `crossfade_ms`: Crossfade duration in ms (default: 100)

### Response (handler.py:491)

```json
{
    "status": "completed",
    "url": "https://s3...",
    "filename": "session_id.ogg",
    "s3_key": "session_id.ogg",
    "metadata": {
        "sample_rate": 24000,
        "codec": "opus",
        "bitrate": "128k",
        "duration": 3.45,
        "device": "cuda"
    }
}
```

## Important Notes

- Audio output is always Opus in OGG container at 24kHz/128kbps
- Voice files auto-detect extension if not specified
- Path traversal protection on audio file inputs
- Old output files auto-deleted after 2 days (`CLEANUP_DAYS`)
- Model is NOT thread-safe - uses global singleton
