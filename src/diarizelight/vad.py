"""Voice Activity Detection using Silero VAD (Pure ONNX).

Detects speech segments in audio files. Returns a list of
:class:`~diarize.utils.SpeechSegment` objects with start/end timestamps.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from .utils import SpeechSegment

logger = logging.getLogger(__name__)

__all__ = ["run_vad"]

# Fallback: Auto-download the ONNX model if it isn't in the directory
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SILERO_MODEL_PATH = "silero_vad.onnx"

def run_vad(
    audio_path: str | Path,
    *,
    threshold: float = 0.45,
    min_speech_duration_ms: int = 200,
    min_silence_duration_ms: int = 50,
    speech_pad_ms: int = 20,
) -> list[SpeechSegment]:
    """Detect speech segments using pure ONNX Silero VAD."""
    
    logger.info("Running Voice Activity Detection (Silero VAD - ONNX)...")
    
    if not os.path.exists(SILERO_MODEL_PATH):
        logger.info(f"Downloading Silero ONNX model to {SILERO_MODEL_PATH}...")
        urllib.request.urlretrieve(SILERO_ONNX_URL, SILERO_MODEL_PATH)

    # 1. Read Audio via soundfile
    waveform, sample_rate = sf.read(str(audio_path), dtype="float32")
    
    # Convert stereo to mono if necessary
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    total_audio_sec = len(waveform) / sample_rate

    # 2. Setup ONNX Session dynamically supporting v3, v4, and v5
    session = ort.InferenceSession(SILERO_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    audio_input_name = 'x' if 'x' in input_names else 'input'
    
    window_size_samples = 512 if sample_rate == 16000 else 256
    sr = np.array([sample_rate], dtype=np.int64)
    
    if 'state' in input_names:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        has_hc = False
    else:
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        has_hc = True

    segments = []
    is_speaking = False
    start_time = 0.0
    silence_start = None
    
    # Convert kwargs arguments to seconds
    min_silence_sec = min_silence_duration_ms / 1000.0
    min_speech_sec = min_speech_duration_ms / 1000.0
    pad_sec = speech_pad_ms / 1000.0

    # 3. Processing Loop
    for i in range(0, len(waveform), window_size_samples):
        chunk = waveform[i:i+window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, window_size_samples - len(chunk)), 'constant')
            
        ort_inputs = {audio_input_name: np.expand_dims(chunk, axis=0).astype(np.float32)}
        if 'sr' in input_names: 
            ort_inputs['sr'] = sr
            
        if has_hc:
            ort_inputs['h'] = h; ort_inputs['c'] = c
            out, h, c = session.run(None, ort_inputs)
        else:
            ort_inputs['state'] = state
            out, state = session.run(None, ort_inputs)
            
        prob = out[0][0]
        time_sec = i / sample_rate
        
        if prob >= threshold:
            if not is_speaking:
                is_speaking = True
                # Apply front padding, ensuring it doesn't go below 0
                start_time = max(0.0, time_sec - pad_sec)
            silence_start = None  
        elif prob < 0.15:
            if is_speaking:
                if silence_start is None:
                    silence_start = time_sec
                elif (time_sec - silence_start) > min_silence_sec:
                    is_speaking = False
                    # Apply back padding, ensuring it doesn't exceed total audio length
                    end_time = min(total_audio_sec, silence_start + pad_sec)
                    if (end_time - start_time) > min_speech_sec:
                        segments.append(SpeechSegment(start=start_time, end=end_time))
                    silence_start = None
            
    if is_speaking:
        end_time = silence_start if silence_start is not None else total_audio_sec
        end_time = min(total_audio_sec, end_time + pad_sec)
        if (end_time - start_time) > min_speech_sec:
            segments.append(SpeechSegment(start=start_time, end=end_time))

    total_speech = sum((seg.end - seg.start) for seg in segments)
    logger.info(
        "ONNX VAD complete: %d speech segments, %.1f seconds of speech",
        len(segments),
        total_speech,
    )

    return segments