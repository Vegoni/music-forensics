# music-forensics

**A Claude skill and standalone CLI for detecting AI-generated music.**

Drop in a YouTube URL or a local audio file and get a forensic analysis report in your terminal — color-coded, scored, and broken down by signal type. Built to work entirely offline at runtime, with no Claude dependency once installed.

---

## What It Does

music-forensics downloads the first 90 seconds of audio from a YouTube URL (or reads a local file) and runs a multi-layer forensic analysis looking for patterns common in AI-generated music:

| Signal | What's checked |
|--------|----------------|
| **Metadata** | Missing or generic artist/title tags, suspicious generator strings |
| **Spectral** | Hard frequency cutoffs (AI models often cap at ≤16 kHz), unnaturally tonal spectrum, missing low-frequency room noise |
| **Waveform** | Metronomic beat timing (CV < 2%), over-compressed dynamic range, abrupt start/end silence |
| **ML Classifier** *(optional)* | HuggingFace deepfake-audio model — runs locally, no API key needed |

Results are displayed as a rich terminal report with a scored verdict: **Likely Human**, **Uncertain**, or **Likely AI-Generated**.

---

## As a Claude Skill (Slash Command)

This tool is designed to be invoked directly from [Claude Code](https://claude.com/code) as a `/music-forensics` slash command. Claude runs the analysis and then interprets the results for you — explaining which signals are most suspicious and why.

### Install the slash command

```bash
mkdir -p ~/.claude/commands
cat > ~/.claude/commands/music-forensics.md << 'EOF'
Run the music-forensics CLI tool to analyze audio from a YouTube URL or local file for signs of AI generation.

The user has provided: $ARGUMENTS

Steps:
1. Run the command: `/path/to/music-forensics/.venv/bin/music-forensics $ARGUMENTS`
2. The tool will download the audio, run analysis, and print a color-coded report directly in the terminal.
3. If the user passed `--deep` as well as a URL, the ML classifier will also run (may take a moment on first use).
4. After the tool finishes, briefly summarize what the verdict was and which signals were most suspicious (if any).

If the command fails because ffmpeg is not installed, tell the user to run: `brew install ffmpeg`
If the command fails because the URL is private or unavailable, tell the user and suggest they try a local file instead.
EOF
```

Update the path in that file to match where you cloned this repo.

### Usage in Claude Code

```
/music-forensics https://www.youtube.com/watch?v=...
/music-forensics https://www.youtube.com/watch?v=... --deep
/music-forensics /path/to/local/file.mp3
```

Claude will run the analysis and give you a plain-English breakdown of the findings.

---

## Standalone CLI Usage

You can also run the tool directly without Claude.

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) — required for audio extraction

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Install

```bash
git clone https://github.com/Vegoni/music-forensics
cd music-forensics
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the optional ML classifier:

```bash
pip install -e ".[deep]"
```

### Run

```bash
# Analyze a YouTube URL (downloads first 90 seconds)
music-forensics "https://www.youtube.com/watch?v=..."

# Analyze a local file
music-forensics /path/to/song.mp3

# Include the ML deepfake classifier
music-forensics "https://www.youtube.com/watch?v=..." --deep
```

---

## How the Scoring Works

Each analyzer returns one or more **findings**, each with:

- A **label** describing what was detected
- A **score** from 0.0 (very human) to 1.0 (very AI)
- **Evidence** — a plain-English explanation of the specific measurement

The overall verdict aggregates all findings:

| Score | Verdict |
|-------|---------|
| < 0.35 | Likely Human |
| 0.35 – 0.65 | Uncertain |
| > 0.65 | Likely AI-Generated |

Scores are color-coded in the terminal: green → yellow → red.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| YouTube download | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Audio analysis | [librosa](https://librosa.org/) |
| Signal processing | [scipy](https://scipy.org/) |
| Metadata parsing | [mutagen](https://mutagen.readthedocs.io/) |
| Terminal output | [rich](https://github.com/Textualize/rich) |
| CLI | [click](https://click.palletsprojects.com/) |
| ML classifier *(optional)* | [transformers](https://huggingface.co/docs/transformers) + [torch](https://pytorch.org/) |
| ML model *(optional)* | [MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) |

---

## Known Limitations

**90-second sample window**
Only the first 90 seconds of audio are analyzed. This is intentional — it keeps analysis fast (under 30 seconds on most machines) and prevents timeouts on long videos. AI artifacts tend to be consistent throughout a track, so 90 seconds is generally sufficient.

**ML model is voice-trained, not music-specific**
The `--deep` classifier (`MelodyMachine/Deepfake-audio-detection-V2`) was trained primarily on speech deepfakes. It can detect some AI music artifacts but may produce false positives or false negatives on purely instrumental tracks. Treat its output as one signal among many, not a definitive verdict.

**The ML classifier dominates the verdict when enabled**
Stage weights live in `STAGE_WEIGHTS` in `music_forensics/reporter.py`. The ML classifier is weighted above the three heuristic stages combined, so with `--deep` enabled it can carry a verdict over them. Without `--deep`, the verdict rests entirely on heuristics that are far weaker — treat the two modes as different tools.

**Spectral cutoff thresholds are heuristic**
The 16 kHz frequency cutoff check reflects current AI music generation limitations, but this will improve over time as models get better. A clean high-frequency response doesn't prove the track is human-made.

**YouTube throttling**
yt-dlp downloads can be throttled by YouTube, especially for longer videos. The 90-second limit dramatically reduces the amount of data downloaded, but speeds can still vary.

**No provenance verification**
This tool analyzes audio signal characteristics only. It cannot verify upload history, detect re-uploads of AI content, or authenticate ownership claims.

**False positives on heavily mastered music**
Professionally mastered tracks are often highly compressed and EQ'd in ways that resemble AI signatures (e.g., narrow dynamic range, frequency shaping). The tool may flag these. Context matters.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT
