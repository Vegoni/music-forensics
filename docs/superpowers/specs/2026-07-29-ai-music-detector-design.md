# AI Music Forensics Tool — Design Spec
**Date:** 2026-07-29
**Status:** Approved

## Overview

A global CLI command (`music-forensics`) that downloads audio from a YouTube URL (or accepts a local file), runs a multi-stage forensic analysis to detect signs of AI generation, renders a color-coded terminal report, then deletes the temporary audio file.

The tool runs entirely locally with no external API dependencies. An optional `--deep` flag enables a HuggingFace ML classifier for probabilistic AI detection scoring.

## Architecture

Modular pipeline: each stage is an independent analyzer that returns structured `Finding` objects. A thin CLI entry point orchestrates the pipeline and passes findings to a `Reporter` for rendering.

```
music-forensics/
├── music_forensics/
│   ├── cli.py              # entry point, argument parsing
│   ├── downloader.py       # yt-dlp wrapper, temp WAV/FLAC download
│   ├── analyzers/
│   │   ├── metadata.py     # ID3/EXIF tags, encoder strings
│   │   ├── spectral.py     # librosa: frequency distribution, artifacts
│   │   ├── waveform.py     # timing regularity, dynamic range, silence
│   │   └── ml.py           # HuggingFace classifier (--deep only)
│   └── reporter.py         # rich-based terminal rendering
├── pyproject.toml
└── README.md
```

## Invocation

```bash
music-forensics "https://youtube.com/watch?v=..."           # standard analysis
music-forensics "https://youtube.com/watch?v=..." --deep    # includes ML stage
music-forensics /path/to/local.mp3                          # local file input
```

URL quoting is recommended — YouTube URLs frequently contain `&` which shells interpret as a background operator.

## Data Model

Each analyzer returns one or more `Finding` objects:

```python
@dataclass
class Finding:
    label: str          # short name, e.g. "Spectral Cutoff"
    score: float        # 0.0 (likely real) to 1.0 (likely AI)
    evidence: list[str] # human-readable observations
```

The reporter aggregates scores across all findings and renders a final verdict with color coding:
- **Green** (0.0–0.35): Likely human-made
- **Yellow** (0.35–0.65): Uncertain — mixed signals
- **Red** (0.65–1.0): Likely AI-generated

## Analyzers

### MetadataAnalyzer (`mutagen`)
- Scans ID3/EXIF tags for known AI generator strings (Suno, Udio, Stable Audio, MusicGen, etc.)
- Flags missing fields expected in real recordings (artist, album, recording date)
- Detects suspiciously generic or templated tag values
- Identifies encoding tool fingerprints left by AI platforms

### SpectralAnalyzer (`librosa`)
- Detects unnaturally clean frequency cutoffs at known AI model boundaries
- Measures spectral flatness for anomalously high/low values
- Checks for absence of room/mic noise floor in sub-100Hz range
- Analyzes harmonic-to-noise ratio patterns

### WaveformAnalyzer (`librosa`, `numpy`)
- Detects rhythmic timing that is too metronomically precise (no human micro-timing variation)
- Measures dynamic range compression — AI output tends toward narrow loudness variation
- Flags abrupt or unnatural silence at track start/end
- Analyzes onset regularity — beats locking perfectly to a grid

### MLAnalyzer (`transformers`, `torch`) — optional `--deep`
- Downloads and caches a HuggingFace audio classifier to `~/.cache/music-forensics/` on first use
- Preferred: a music-specific AI detector (e.g., trained on Suno/Udio samples)
- Fallback: a general synthetic audio classifier, clearly labeled as such in output
- Returns a probability score with confidence interval
- `torch` is a lazy import — not required unless `--deep` is passed

## Audio Download

- `yt-dlp` downloads audio-only in WAV or FLAC (lossless) for cleaner spectral analysis
- MP3 lossy compression creates its own spectral artifacts that would pollute AI detection signals
- Local MP3/FLAC/WAV files are accepted as input and analyzed as-is
- Temp file stored in system temp dir, always cleaned up via `try/finally`

## Error Handling

| Scenario | Behavior |
|---|---|
| Bad or private YouTube URL | Clear error message, temp file cleaned up, non-zero exit |
| `ffmpeg` not installed | Detected at startup, install instructions printed, exit |
| HuggingFace model download fails | Graceful message, remaining analysis stages still run |
| Local file missing or unsupported format | Early exit with helpful message |
| Analyzer crash mid-pipeline | Logged as warning, other analyzers still run, temp file cleaned |

## Dependencies

| Package | Purpose |
|---|---|
| `yt-dlp` | YouTube audio download |
| `librosa` | Spectral and waveform analysis |
| `mutagen` | Audio metadata parsing |
| `numpy` | Numerical signal processing |
| `scipy` | Signal processing utilities |
| `rich` | Colorful terminal output |
| `click` | CLI argument parsing |
| `transformers` | HuggingFace model interface (lazy, `--deep` only) |
| `torch` | ML inference (lazy, `--deep` only) |

## Testing

- Unit tests per analyzer using small fixture audio files (a few seconds each of real and known-AI-generated samples)
- No live network calls in tests — yt-dlp and HuggingFace model are mocked
- One integration test running the full pipeline on a local file end-to-end
- Test framework: `pytest`
