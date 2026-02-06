# coding=utf-8
# IndexTTS RunPod Serverless Inference Engine
# SPDX-License-Identifier: MIT

"""
IndexTTS Inference Engine for RunPod Serverless

Wraps IndexTTS2 model with a clean API for serverless inference.
Supports voice cloning with speaker reference audio and emotion control.
"""

import gc
import os
import re
import tempfile
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Generator
from pathlib import Path

import torch
import torchaudio
import numpy as np

import config

log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_WHITESPACE_RE = re.compile(r"\s+")

# =============================================================================
# TEXT CHUNKING UTILITIES
# =============================================================================

def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    """Split input text into <= max_chars character chunks, preferring sentence/clause/word boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    normalized = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    sentence_enders = {".", "!", "?", "。", "！", "？"}
    clause_enders = {",", ";", ":", "，", "；", "："}
    closers = {'"', "'", ")", "]", "}", "\u201d", "\u2019", "」", "』"}

    chunks: List[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[: max_chars + 1]
        candidate_sentence = None
        candidate_clause = None
        candidate_space = None

        for i in range(1, len(window)):
            if not window[i].isspace():
                continue

            candidate_space = i
            prev = window[i - 1]
            prev2 = window[i - 2] if i >= 2 else ""

            if prev in sentence_enders or (prev in closers and prev2 in sentence_enders):
                candidate_sentence = i
            elif prev in clause_enders or (prev in closers and prev2 in clause_enders):
                candidate_clause = i

        split_at = candidate_sentence or candidate_clause or candidate_space
        if split_at is None:
            split_at = max_chars

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return chunks


def crossfade_chunks(audio_chunks: List[torch.Tensor], crossfade_ms: int = 50, sample_rate: int = 24000) -> torch.Tensor:
    """Crossfade audio chunks for smoother transitions."""
    if len(audio_chunks) == 0:
        return torch.tensor([])
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
    result = audio_chunks[0]
    for i, chunk in enumerate(audio_chunks[1:], 1):
        if result.dim() > 1:
            result = result.squeeze()
        if chunk.dim() > 1:
            chunk = chunk.squeeze()

        if len(result) < crossfade_samples or len(chunk) < crossfade_samples:
            result = torch.cat([result, chunk])
            continue

        fade_out = torch.linspace(1, 0, crossfade_samples, device=result.device)
        fade_in = torch.linspace(0, 1, crossfade_samples, device=chunk.device)

        result_tail = result[-crossfade_samples:] * fade_out
        chunk_head = chunk[:crossfade_samples] * fade_in
        crossfaded = result_tail + chunk_head

        result = torch.cat([result[:-crossfade_samples], crossfaded, chunk[crossfade_samples:]])

    return result


# =============================================================================
# INDEXTTS INFERENCE ENGINE
# =============================================================================

class IndexTTSInference:
    """IndexTTS Inference Engine. Supports voice cloning with speaker reference and emotion control."""

    def __init__(
        self,
        cfg_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        use_deepspeed: bool = False,
        use_cuda_kernel: bool = False,
        max_tokens_per_segment: int = 800,
    ):
        self.device = device or config.DEVICE
        self.use_fp16 = use_fp16
        self.use_deepspeed = use_deepspeed
        self.use_cuda_kernel = use_cuda_kernel
        self.max_tokens_per_segment = max_tokens_per_segment
        
        # Set paths
        self.cfg_path = cfg_path or str(config.CHECKPOINTS_DIR / "config.yaml")
        self.model_dir = model_dir or str(config.CHECKPOINTS_DIR)
        
        self._tts = None
        log.info(f"IndexTTSInference initialized: device={self.device}, fp16={use_fp16}")

    def _load_model(self):
        """Lazy load the IndexTTS model."""
        if self._tts is not None:
            return

        log.info("Loading IndexTTS2 model...")
        start_time = time.time()

        try:
            # Import here to avoid loading on module import
            from indextts.infer_v2 import IndexTTS2
            
            self._tts = IndexTTS2(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                use_fp16=self.use_fp16,
                use_cuda_kernel=self.use_cuda_kernel,
                use_deepspeed=self.use_deepspeed,
            )
            
            # Set max tokens per segment if the model supports it
            if hasattr(self._tts, 'max_tokens_per_segment'):
                self._tts.max_tokens_per_segment = self.max_tokens_per_segment
                log.info(f"Set max_tokens_per_segment to {self.max_tokens_per_segment}")

            load_time = time.time() - start_time
            log.info(f"IndexTTS2 model loaded in {load_time:.2f}s")

        except Exception as e:
            log.error(f"Failed to load IndexTTS2 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _resolve_voice_path(self, speaker_voice: str) -> Optional[Path]:
        """Resolve speaker voice to a valid audio file path."""
        if not speaker_voice:
            return None

        # Check if it's already a full path
        voice_path = Path(speaker_voice)
        if voice_path.is_absolute() and voice_path.exists():
            return voice_path

        # Look in AUDIO_VOICES_DIR
        candidate_path = (config.AUDIO_VOICES_DIR / speaker_voice).resolve()
        if candidate_path.exists() and candidate_path.suffix.lower() in config.AUDIO_EXTS:
            return candidate_path

        # Try with auto-detected extension
        for ext in config.AUDIO_EXTS:
            test_path = config.AUDIO_VOICES_DIR / f"{speaker_voice}{ext}"
            if test_path.exists():
                return test_path.resolve()

        return None

    def generate_speech(
        self,
        text: str,
        speaker_voice: Optional[str] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_vector: Optional[List[float]] = None,
        emo_alpha: float = 1.0,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,
        verbose: bool = False,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 300,
        enable_crossfade: bool = True,
        crossfade_ms: int = 140,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech from text with optional voice cloning and emotion control.

        Args:
            text: Text to synthesize
            speaker_voice: Filename of reference audio for voice cloning (in AUDIO_VOICES_DIR)
            emo_audio_prompt: Filename of emotion reference audio (optional)
            emo_vector: Emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_alpha: Emotion strength (0.0-1.0)
            use_emo_text: Use text for emotion guidance
            emo_text: Emotion text description (when use_emo_text=True)
            use_random: Enable randomness in generation
            verbose: Verbose output
            enable_chunking: Split long text into chunks
            max_chars_per_chunk: Maximum characters per chunk
            enable_crossfade: Apply crossfade between chunks
            crossfade_ms: Crossfade duration in milliseconds

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        self._load_model()

        # Resolve voice paths
        spk_audio_path = self._resolve_voice_path(speaker_voice) if speaker_voice else None
        emo_audio_path = self._resolve_voice_path(emo_audio_prompt) if emo_audio_prompt else None

        if speaker_voice and spk_audio_path is None:
            available = [f.stem for f in config.AUDIO_VOICES_DIR.glob("*")
                        if f.suffix.lower() in config.AUDIO_EXTS]
            raise ValueError(f"Speaker voice '{speaker_voice}' not found. Available: {available}")

        log.info(f"Generating speech for text ({len(text)} chars)")
        if spk_audio_path:
            log.info(f"Using speaker voice: {spk_audio_path.name}")
        if emo_audio_path:
            log.info(f"Using emotion audio: {emo_audio_path.name}")

        # Handle chunking for long text
        if enable_chunking and len(text) > max_chars_per_chunk:
            log.info(f"Chunking text (max {max_chars_per_chunk} chars per chunk)")
            chunks = chunk_text(text, max_chars_per_chunk)
            log.info(f"Text split into {len(chunks)} chunks")

            audio_chunks = []
            for i, chunk_text_content in enumerate(chunks):
                log.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Generate audio for this chunk
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    self._tts.infer(
                        spk_audio_prompt=str(spk_audio_path) if spk_audio_path else None,
                        text=chunk_text_content,
                        output_path=tmp_path,
                        emo_audio_prompt=str(emo_audio_path) if emo_audio_path else None,
                        emo_vector=emo_vector,
                        emo_alpha=emo_alpha,
                        use_emo_text=use_emo_text,
                        emo_text=emo_text,
                        use_random=use_random,
                        verbose=verbose,
                    )

                    # Load the generated audio
                    audio_tensor, sr = torchaudio.load(tmp_path)
                    audio_chunks.append(audio_tensor.to(self.device))
                    actual_sample_rate = sr  # capture model's actual sample rate

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # Use the actual sample rate from the model output
            sr = actual_sample_rate if audio_chunks else config.DEFAULT_SAMPLE_RATE
            log.info(f"Model output sample rate: {sr}")

            # Combine chunks with crossfade
            if enable_crossfade and len(audio_chunks) > 1:
                log.info(f"Applying crossfade ({crossfade_ms}ms)")
                combined = crossfade_chunks(audio_chunks, crossfade_ms=crossfade_ms, sample_rate=sr)
            else:
                combined = torch.cat(audio_chunks, dim=-1)

            return combined, sr

        else:
            # Single chunk generation
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                self._tts.infer(
                    spk_audio_prompt=str(spk_audio_path) if spk_audio_path else None,
                    text=text,
                    output_path=tmp_path,
                    emo_audio_prompt=str(emo_audio_path) if emo_audio_path else None,
                    emo_vector=emo_vector,
                    emo_alpha=emo_alpha,
                    use_emo_text=use_emo_text,
                    emo_text=emo_text,
                    use_random=use_random,
                    verbose=verbose,
                )

                # Load the generated audio
                audio_tensor, sample_rate = torchaudio.load(tmp_path)
                return audio_tensor.to(self.device), sample_rate

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def generate_audio_stream_decoded(
        self,
        text: str,
        speaker_voice: Optional[str] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_vector: Optional[List[float]] = None,
        emo_alpha: float = 1.0,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,
        max_chars_per_chunk: int = 300,
        enable_crossfade: bool = True,
        crossfade_ms: int = 140,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming audio chunks as base64-encoded signed int16 PCM.

        Each text chunk is synthesized and yielded immediately so the client
        receives audio as soon as the first chunk finishes inference, rather
        than waiting for all chunks to complete.

        When crossfade is enabled, only the last `crossfade_ms` of each chunk
        is held back and blended with the start of the next chunk.
        """
        import base64
        import traceback

        self._load_model()

        spk_audio_path = self._resolve_voice_path(speaker_voice) if speaker_voice else None
        emo_audio_path = self._resolve_voice_path(emo_audio_prompt) if emo_audio_prompt else None

        if speaker_voice and spk_audio_path is None:
            available = [f.stem for f in config.AUDIO_VOICES_DIR.glob("*")
                         if f.suffix.lower() in config.AUDIO_EXTS]
            yield {"error": f"Speaker voice '{speaker_voice}' not found. Available: {available}"}
            return

        try:
            chunks = chunk_text(text, max_chars=max_chars_per_chunk) if max_chars_per_chunk and max_chars_per_chunk > 0 else [text]
            if not chunks:
                yield {"error": "Text is empty after normalization"}
                return

            output_sample_rate = None
            crossfade_samples = 0
            # Only holds the trailing crossfade overlap from the previous chunk
            crossfade_tail = None
            chunk_num = 0

            def _to_int16_bytes(tensor):
                """Convert float tensor to base64-encoded int16 PCM."""
                arr = tensor.detach().cpu().numpy()
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = np.clip(arr, -1.0, 1.0)
                    arr = (arr * 32767).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
                return base64.b64encode(arr.tobytes()).decode("utf-8")

            for i, chunk_text_content in enumerate(chunks):
                log.info(f"Streaming chunk {i + 1}/{len(chunks)}")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    self._tts.infer(
                        spk_audio_prompt=str(spk_audio_path) if spk_audio_path else None,
                        text=chunk_text_content,
                        output_path=tmp_path,
                        emo_audio_prompt=str(emo_audio_path) if emo_audio_path else None,
                        emo_vector=emo_vector,
                        emo_alpha=emo_alpha,
                        use_emo_text=use_emo_text,
                        emo_text=emo_text,
                        use_random=use_random,
                        verbose=False,
                    )

                    audio_tensor, sr = torchaudio.load(tmp_path)

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                chunk_tensor = audio_tensor.to(self.device)
                if chunk_tensor.dim() > 1:
                    chunk_tensor = chunk_tensor[0]
                if chunk_tensor.dim() > 1:
                    chunk_tensor = chunk_tensor.reshape(-1)

                if output_sample_rate is None:
                    output_sample_rate = sr
                    crossfade_samples = (
                        int(output_sample_rate * (float(crossfade_ms) / 1000.0))
                        if crossfade_ms and enable_crossfade
                        else 0
                    )
                    log.info(f"Model output sample rate: {output_sample_rate}")
                elif sr != output_sample_rate:
                    chunk_tensor = torchaudio.functional.resample(chunk_tensor.unsqueeze(0), sr, output_sample_rate).squeeze(0)

                is_last_chunk = (i == len(chunks) - 1)

                # Blend crossfade tail from previous chunk with start of this chunk
                if crossfade_tail is not None:
                    if crossfade_samples > 0:
                        cf = min(crossfade_samples, len(crossfade_tail), len(chunk_tensor))
                        if cf > 0:
                            fade_out = torch.linspace(1.0, 0.0, cf, device=chunk_tensor.device, dtype=chunk_tensor.dtype)
                            fade_in = 1.0 - fade_out
                            blended = crossfade_tail * fade_out + chunk_tensor[:cf] * fade_in
                            audio_to_emit = torch.cat([blended, chunk_tensor[cf:]], dim=-1)
                        else:
                            audio_to_emit = torch.cat([crossfade_tail, chunk_tensor], dim=-1)
                    else:
                        audio_to_emit = torch.cat([crossfade_tail, chunk_tensor], dim=-1)
                else:
                    audio_to_emit = chunk_tensor

                # Hold back crossfade overlap for next chunk (unless last chunk)
                if crossfade_samples > 0 and not is_last_chunk and len(audio_to_emit) > crossfade_samples:
                    crossfade_tail = audio_to_emit[-crossfade_samples:]
                    audio_to_emit = audio_to_emit[:-crossfade_samples]
                else:
                    crossfade_tail = None

                # Yield this chunk's audio immediately
                if audio_to_emit is not None and len(audio_to_emit) > 0:
                    chunk_num += 1
                    yield {
                        "status": "streaming",
                        "chunk": chunk_num,
                        "format": "pcm_16",
                        "audio_chunk": _to_int16_bytes(audio_to_emit),
                        "sample_rate": output_sample_rate,
                    }

            # Flush any remaining crossfade tail
            if crossfade_tail is not None and len(crossfade_tail) > 0:
                chunk_num += 1
                yield {
                    "status": "streaming",
                    "chunk": chunk_num,
                    "format": "pcm_16",
                    "audio_chunk": _to_int16_bytes(crossfade_tail),
                    "sample_rate": output_sample_rate or config.DEFAULT_SAMPLE_RATE,
                }

            yield {
                "status": "complete",
                "format": "pcm_16",
                "message": "All chunks streamed",
                "total_chunks": chunk_num,
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            log.error(f"Streaming mode failed: {str(e)}")
            log.error(f"Traceback: {error_trace}")
            yield {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_trace
            }

    def cleanup(self):
        """Clean up model and free memory."""
        if self._tts is not None:
            log.info("Cleaning up IndexTTS model...")
            del self._tts
            self._tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Model cleanup complete")


# =============================================================================
# GLOBAL INFERENCE ENGINE INSTANCE
# =============================================================================

_inference_engine: Optional[IndexTTSInference] = None


def get_inference_engine(
    cfg_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    use_fp16: Optional[bool] = None,
    use_deepspeed: Optional[bool] = None,
    use_cuda_kernel: Optional[bool] = None,
    max_tokens_per_segment: Optional[int] = None,
) -> IndexTTSInference:
    """
    Get or create the global inference engine instance.
    
    Args:
        cfg_path: Path to config.yaml (default: CHECKPOINTS_DIR/config.yaml)
        model_dir: Path to model directory (default: CHECKPOINTS_DIR)
        device: Device to use (default: cuda if available)
        use_fp16: Use FP16 inference (default: from config)
        use_deepspeed: Use DeepSpeed (default: from config)
        use_cuda_kernel: Use CUDA kernel (default: from config)
        max_tokens_per_segment: Max tokens per segment (default: from config)
    
    Returns:
        IndexTTSInference instance
    """
    global _inference_engine
    
    if _inference_engine is None:
        _inference_engine = IndexTTSInference(
            cfg_path=cfg_path,
            model_dir=model_dir,
            device=device or config.DEVICE,
            use_fp16=use_fp16 if use_fp16 is not None else config.config.use_fp16,
            use_deepspeed=use_deepspeed if use_deepspeed is not None else config.config.use_deepspeed,
            use_cuda_kernel=use_cuda_kernel if use_cuda_kernel is not None else config.config.use_cuda_kernel,
            max_tokens_per_segment=max_tokens_per_segment or config.config.max_tokens_per_segment,
        )
    
    return _inference_engine
