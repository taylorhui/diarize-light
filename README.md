# diarize-light

**Fork of FoxNoseTech's diarize: Remove dependency on torch, torchaudio and silero_vad. Use onnxruntime directly.**

Speaker diarization for Python — answers "who spoke when?" in audio file.

Runs on CPU. No GPU, no API keys, no account signup. Apache 2.0 licensed. 

```bash
pip install git+https://github.com/taylorhui/diarize-light.git
```

```python
from diarizelight import diarize

result = diarize("meeting.wav")
for seg in result.segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")
```

## Quick Start (Similar to FoxNoseTech's diarize)

```python
from diarizelight import diarize

result = diarize("meeting.wav")

print(f"Found {result.num_speakers} speakers")
for seg in result.segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")

# Export to RTTM format
result.to_rttm("meeting.rttm")
```

Requires Python 3.9+. Supports WAV (or formats supported by soundfile).
`diarize-light` does not use `torch/torchaudio` for simplicity.

📖 **[Full documentation from FoxNoseTech](https://foxnosetech.github.io/diarize/)** — API reference, architecture, benchmarks.


## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

All dependencies are permissively licensed:
- Silero VAD: MIT
- WeSpeaker: Apache 2.0
- scikit-learn: BSD

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/FoxNoseTech/diarize).
