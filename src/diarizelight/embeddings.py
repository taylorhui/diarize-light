"""Speaker embedding extraction using sherpa-onnx.

Extracts speaker embeddings from audio segments detected
by VAD. Long segments are split with a sliding window so that each
window produces its own embedding, improving clustering granularity.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx

from .utils import SpeechSegment, SubSegment

logger = logging.getLogger(__name__)

__all__ = ["extract_embeddings"]

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum segment duration for embedding extraction (seconds).
MIN_SEGMENT_DURATION: float = 0.4

#: Sliding window length for splitting long segments (seconds).
EMBEDDING_WINDOW: float = 1.2

#: Sliding window step size (seconds).  Overlap = WINDOW − STEP.
EMBEDDING_STEP: float = 0.6

# Use the proven 3dspeaker model
EMBEDDING_MODEL_PATH = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMBEDDING_MODEL_DL_URL = "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

def extract_embeddings(
    audio_path: str | Path,
    speech_segments: list[SpeechSegment],
) -> tuple[np.ndarray, list[SubSegment]]:
    """Extract speaker embeddings using sherpa-onnx purely in RAM."""

    logger.info("Extracting speaker embeddings (sherpa-onnx)...")

    # NEW: Download WeSpeaker if missing
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        logger.info(f"Downloading {EMBEDDING_MODEL_PATH}...")
        urllib.request.urlretrieve(EMBEDDING_MODEL_DL_URL, EMBEDDING_MODEL_PATH)
        
    # Initialize the lightweight Sherpa-ONNX Extractor
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        wespeaker=EMBEDDING_MODEL_PATH,  # <-- CRITICAL: Use 'wespeaker=', not 'model='
        num_threads=4
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    # Load full audio directly into memory
    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # stereo → mono

    embeddings: list[np.ndarray] = []
    subsegments: list[SubSegment] = []

    for idx, seg in enumerate(speech_segments):
        seg_duration = seg.duration

        if seg_duration < MIN_SEGMENT_DURATION:
            continue

        # Split long segments with a sliding window
        if seg_duration <= EMBEDDING_WINDOW * 1.5:
            windows = [(seg.start, seg.end)]
        else:
            windows: list[tuple[float, float]] = []
            win_start = seg.start
            while win_start + MIN_SEGMENT_DURATION < seg.end:
                win_end = min(win_start + EMBEDDING_WINDOW, seg.end)
                windows.append((win_start, win_end))
                win_start += EMBEDDING_STEP

        for win_start, win_end in windows:
            start_sample = int(win_start * sr)
            end_sample = int(win_end * sr)
            
            # FIX 1: Force a contiguous memory block for the C++ backend
            chunk = np.ascontiguousarray(audio_data[start_sample:end_sample])

            try:
                stream = extractor.create_stream()
                stream.accept_waveform(sr, chunk)
                
                # FIX 2: Explicitly flush the stream buffer so the model evaluates the whole chunk
                stream.input_finished()
                
                emb = extractor.compute(stream)
                
                # FIX 3: Multiply by 15.0 to "Un-normalize" the vectors. 
                # This artificially restores the magnitude to match raw WeSpeaker outputs, 
                # allowing the clustering algorithm's distance thresholds to work properly again.
                emb_array = np.array(emb) * 15.0
                logger.info(f"Embedding Variance Check: {np.sum(emb_array):.2f}")
                
            except Exception as e:
                logger.debug("Embedding extraction failed for window %.2f-%.2f: %s", win_start, win_end, e)
                continue

            if emb_array is not None and len(emb_array) > 0:
                embeddings.append(emb_array)
                subsegments.append(SubSegment(start=win_start, end=win_end, parent_idx=idx))

   # Downstream clustering algorithms dynamically read the shape, so both work perfectly.
    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), []

    X = np.stack(embeddings)  # (N, D)
    logger.info("Extracted %d embeddings (dim=%d)", X.shape[0], X.shape[1])
    return X, subsegments