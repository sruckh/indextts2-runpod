FROM runpod/base:1.0.3-cuda1281-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub

# Install system dependencies including ffmpeg and git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git git-lfs ca-certificates curl build-essential cmake ninja-build pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && git lfs install

WORKDIR /opt

# Download IndexTTS source via tarball (avoids git dependency during build)
# Full git clone happens at runtime in bootstrap.sh on the network volume
ARG INDEXTTS_REPO="https://github.com/index-tts/index-tts"
ARG INDEXTTS_REF="main"

RUN curl -fsSL "$INDEXTTS_REPO/archive/$INDEXTTS_REF.tar.gz" | tar xz -C /opt \
    && mv /opt/index-tts-$INDEXTTS_REF /opt/index-tts

# Install uv package manager for faster installs
RUN pip install --no-cache-dir uv

# Install IndexTTS dependencies using uv
# Using --system since we're in a container
WORKDIR /opt/index-tts
RUN uv pip install --system torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || \
    pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install flash-attention
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Install remaining dependencies
RUN uv pip install --system -e . || \
    (pip install --no-cache-dir -e . && pip install --no-cache-dir \
    numpy safetensors einops huggingface-hub modelscope \
    pyyaml tqdm transformers accelerate)

# Install RunPod and additional serverless dependencies
RUN pip install --no-cache-dir \
    runpod==1.6.1 \
    uvicorn[standard] \
    pydantic \
    python-multipart \
    tqdm \
    boto3

# Copy serverless handler files into the image
COPY handler.py /opt/index-tts/handler.py
COPY config.py /opt/index-tts/config.py
COPY serverless_engine.py /opt/index-tts/serverless_engine.py
COPY bootstrap.sh /opt/bootstrap.sh

WORKDIR /opt/index-tts

CMD ["bash", "/opt/bootstrap.sh"]
