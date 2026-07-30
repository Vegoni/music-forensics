# Music Forensics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a global CLI command `music-forensics` that downloads audio from YouTube (or reads a local file), runs multi-stage forensic analysis for AI generation signals, renders a color-coded terminal report, and cleans up temp files.

**Architecture:** Modular pipeline — each analyzer is an independent module returning `Finding` objects. A thin `cli.py` orchestrates download → analyze → report → cleanup. `reporter.py` renders all findings with `rich`.

**Tech Stack:** Python 3.10+, click, yt-dlp, librosa, mutagen, numpy, scipy, rich, soundfile (test fixtures), pytest, pytest-mock.

## Global Constraints

- Python ≥ 3.10 (uses `list[str]` and `dict[str, list]` native type hints)
- `torch` and `transformers` are optional — lazy-imported only when `--deep` is passed
- All analyzers must return `list[Finding]` — never raise exceptions to the caller
- Temp audio files: always cleaned up via `try/finally` in CLI, never left on disk
- Score range: `0.0` (likely real) to `1.0` (likely AI); verdicts: green <0.35, yellow 0.35–0.65, red >0.65
- HuggingFace model cache: `~/.cache/music-forensics/`
- Audio download format: FLAC (lossless) — never MP3 for the download path

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `music_forensics/__init__.py`
- Create: `music_forensics/analyzers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: installable package `music-forensics`, pytest fixtures `sine_wav` and `silent_wav` (both `Path`)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "music-forensics"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "yt-dlp>=2024.1.1",
    "librosa>=0.10.0",
    "mutagen>=1.47.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "rich>=13.0.0",
    "click>=8.1.0",
    "soundfile>=0.12.0",
]

[project.optional-dependencies]
deep = [
    "transformers>=4.35.0",
    "torch>=2.0.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-mock>=3.12.0",
]

[project.scripts]
music-forensics = "music_forensics.cli:main"
```

- [ ] **Step 2: Create empty `__init__` files**

`music_forensics/__init__.py` — empty file.
`music_forensics/analyzers/__init__.py` — empty file.
`tests/__init__.py` — empty file.

- [ ] **Step 3: Create `tests/conftest.py` with audio fixtures**

```python
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    path = tmp_path / "sine.wav"
    sf.write(str(path), y, sr)
    return path


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    sr = 44100
    y = np.zeros(sr * 2, dtype=np.float32)
    path = tmp_path / "silent.wav"
    sf.write(str(path), y, sr)
    return path
```

- [ ] **Step 4: Install the package in dev mode**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 5: Verify pytest can collect (no tests yet is fine)**

Run: `pytest --collect-only`
Expected: `no tests ran` or empty collection — no import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml music_forensics/ tests/
git commit -m "feat: project scaffolding and test fixtures"
```

---

### Task 2: Finding Data Model

**Files:**
- Create: `music_forensics/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Finding(label: str, score: float, evidence: list[str])` — imported by all analyzers and reporter

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from music_forensics.models import Finding


def test_finding_fields():
    f = Finding(label="Test Signal", score=0.75, evidence=["reason one", "reason two"])
    assert f.label == "Test Signal"
    assert f.score == 0.75
    assert f.evidence == ["reason one", "reason two"]


def test_finding_default_evidence():
    f = Finding(label="Test", score=0.0)
    assert f.evidence == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.models'`

- [ ] **Step 3: Write `music_forensics/models.py`**

```python
from dataclasses import dataclass, field


@dataclass
class Finding:
    label: str
    score: float
    evidence: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/models.py tests/test_models.py
git commit -m "feat: Finding data model"
```

---

### Task 3: Audio Downloader

**Files:**
- Create: `music_forensics/downloader.py`
- Create: `tests/test_downloader.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `download_audio(url: str) -> Path` — downloads YouTube audio to temp FLAC, returns path
  - `is_url(source: str) -> bool` — returns True if source starts with http:// or https://

- [ ] **Step 1: Write failing tests**

```python
# tests/test_downloader.py
from pathlib import Path
from unittest.mock import MagicMock, patch
from music_forensics.downloader import download_audio, is_url


def test_is_url_with_https():
    assert is_url("https://youtube.com/watch?v=abc") is True


def test_is_url_with_http():
    assert is_url("http://youtube.com/watch?v=abc") is True


def test_is_url_with_local_path():
    assert is_url("/home/user/song.mp3") is False


def test_is_url_with_relative_path():
    assert is_url("song.mp3") is False


def test_download_audio_returns_flac_path(tmp_path):
    expected_flac = tmp_path / "audio.flac"
    expected_flac.touch()

    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            result = download_audio("https://youtube.com/watch?v=test")

    assert result == expected_flac


def test_download_audio_raises_on_ydl_error(tmp_path):
    import pytest

    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.download.side_effect = Exception("Private video")
        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(Exception, match="Private video"):
                download_audio("https://youtube.com/watch?v=private")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloader.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.downloader'`

- [ ] **Step 3: Write `music_forensics/downloader.py`**

```python
import os
import tempfile
from pathlib import Path

import yt_dlp


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def download_audio(url: str) -> Path:
    tmpdir = tempfile.mkdtemp(prefix="music_forensics_")
    output_template = os.path.join(tmpdir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "flac",
            "preferredquality": "0",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return Path(tmpdir) / "audio.flac"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloader.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/downloader.py tests/test_downloader.py
git commit -m "feat: audio downloader with yt-dlp"
```

---

### Task 4: Metadata Analyzer

**Files:**
- Create: `music_forensics/analyzers/metadata.py`
- Create: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `Finding` from `music_forensics.models`
- Produces: `analyze(audio_path: str) -> list[Finding]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metadata.py
from unittest.mock import patch, MagicMock
from music_forensics.analyzers.metadata import analyze
from music_forensics.models import Finding


def test_ai_keyword_suno_returns_high_score(sine_wav):
    mock_tags = MagicMock()
    mock_tags.values.return_value = ["Created with Suno AI"]
    mock_tags.keys.return_value = []

    with patch("music_forensics.analyzers.metadata.mutagen.File", return_value=mock_tags):
        findings = analyze(str(sine_wav))

    ai_finding = next((f for f in findings if f.label == "AI Generator Detected"), None)
    assert ai_finding is not None
    assert ai_finding.score == 1.0


def test_missing_tags_returns_moderate_score(sine_wav):
    with patch("music_forensics.analyzers.metadata.mutagen.File", return_value=None):
        findings = analyze(str(sine_wav))

    assert any(f.label == "Missing Metadata" for f in findings)
    missing = next(f for f in findings if f.label == "Missing Metadata")
    assert 0.3 <= missing.score <= 0.5


def test_clean_tags_returns_low_score(sine_wav):
    mock_tags = MagicMock()
    mock_tags.values.return_value = ["Real Artist", "Real Album", "2023"]
    mock_tags.keys.return_value = ["TPE1", "TALB", "TDRC"]

    with patch("music_forensics.analyzers.metadata.mutagen.File", return_value=mock_tags):
        findings = analyze(str(sine_wav))

    assert all(f.score < 0.35 for f in findings)


def test_mutagen_read_error_returns_finding(sine_wav):
    with patch("music_forensics.analyzers.metadata.mutagen.File", side_effect=Exception("corrupt")):
        findings = analyze(str(sine_wav))

    assert len(findings) == 1
    assert findings[0].score == 0.5
    assert "corrupt" in findings[0].evidence[0]


def test_returns_list_of_findings(sine_wav):
    mock_tags = MagicMock()
    mock_tags.values.return_value = []
    mock_tags.keys.return_value = []

    with patch("music_forensics.analyzers.metadata.mutagen.File", return_value=mock_tags):
        result = analyze(str(sine_wav))

    assert isinstance(result, list)
    assert all(isinstance(f, Finding) for f in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metadata.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.analyzers.metadata'`

- [ ] **Step 3: Write `music_forensics/analyzers/metadata.py`**

```python
import mutagen
from music_forensics.models import Finding

_AI_KEYWORDS = frozenset([
    "suno", "udio", "stable audio", "musicgen", "audiocraft",
    "mubert", "boomy", "soundraw", "aiva", "beatoven", "loudly",
])


def analyze(audio_path: str) -> list[Finding]:
    try:
        tags = mutagen.File(audio_path)
    except Exception as e:
        return [Finding(
            label="Metadata Read Error",
            score=0.5,
            evidence=[f"Could not read metadata: {e}"],
        )]

    if tags is None:
        return [Finding(
            label="Missing Metadata",
            score=0.4,
            evidence=["No metadata found — AI-generated audio often lacks tags entirely"],
        )]

    findings = []
    tag_blob = " ".join(str(v) for v in tags.values()).lower()

    for keyword in _AI_KEYWORDS:
        if keyword in tag_blob:
            findings.append(Finding(
                label="AI Generator Detected",
                score=1.0,
                evidence=[f"Found '{keyword}' in metadata tags"],
            ))
            break

    has_artist = any("artist" in str(k).lower() for k in tags.keys())
    has_title = any("title" in str(k).lower() for k in tags.keys())
    if not has_artist and not has_title:
        findings.append(Finding(
            label="Missing Basic Tags",
            score=0.35,
            evidence=["No artist or title tags — common in AI-generated audio"],
        ))

    if not findings:
        findings.append(Finding(
            label="Metadata Clean",
            score=0.1,
            evidence=["No AI indicators found in metadata"],
        ))

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metadata.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/analyzers/metadata.py tests/test_metadata.py
git commit -m "feat: metadata analyzer"
```

---

### Task 5: Spectral Analyzer

**Files:**
- Create: `music_forensics/analyzers/spectral.py`
- Create: `tests/test_spectral.py`

**Interfaces:**
- Consumes: `Finding` from `music_forensics.models`
- Produces: `analyze(audio_path: str) -> list[Finding]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spectral.py
from music_forensics.analyzers.spectral import analyze
from music_forensics.models import Finding
import numpy as np
import soundfile as sf
from pathlib import Path


def test_returns_list_of_findings(sine_wav):
    result = analyze(str(sine_wav))
    assert isinstance(result, list)
    assert all(isinstance(f, Finding) for f in result)


def test_scores_in_valid_range(sine_wav):
    findings = analyze(str(sine_wav))
    for f in findings:
        assert 0.0 <= f.score <= 1.0, f"Score out of range: {f.score}"


def test_bandlimited_audio_flagged(tmp_path):
    # Create audio with energy only below 8kHz — simulates AI frequency cutoff
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 2kHz sine — well below typical 16kHz cutoff threshold
    y = (np.sin(2 * np.pi * 2000 * t) * 0.5).astype(np.float32)
    path = tmp_path / "bandlimited.wav"
    sf.write(str(path), y, sr)

    findings = analyze(str(path))
    cutoff_finding = next((f for f in findings if "Cutoff" in f.label), None)
    assert cutoff_finding is not None
    assert cutoff_finding.score >= 0.5


def test_silent_audio_does_not_crash(silent_wav):
    findings = analyze(str(silent_wav))
    assert isinstance(findings, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spectral.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.analyzers.spectral'`

- [ ] **Step 3: Write `music_forensics/analyzers/spectral.py`**

```python
import librosa
import numpy as np
import scipy.signal
from music_forensics.models import Finding


def analyze(audio_path: str) -> list[Finding]:
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return [Finding(label="Spectral Load Error", score=0.5, evidence=[str(e)])]

    if np.max(np.abs(y)) < 1e-6:
        return [Finding(
            label="Silent Audio",
            score=0.5,
            evidence=["Audio is nearly silent — analysis inconclusive"],
        )]

    findings = []
    findings.extend(_check_spectral_cutoff(y, sr))
    findings.extend(_check_spectral_flatness(y, sr))
    findings.extend(_check_noise_floor(y, sr))

    if not findings:
        findings.append(Finding(
            label="Spectral Profile Normal",
            score=0.15,
            evidence=["No unusual spectral artifacts detected"],
        ))

    return findings


def _check_spectral_cutoff(y: np.ndarray, sr: int) -> list[Finding]:
    freqs, psd = scipy.signal.welch(y, sr, nperseg=min(4096, len(y)))
    threshold = np.max(psd) * 0.01
    significant = np.where(psd > threshold)[0]
    if len(significant) == 0:
        return []
    max_freq = freqs[significant[-1]]
    if max_freq < 16000:
        return [Finding(
            label="Frequency Cutoff",
            score=0.65,
            evidence=[f"Significant audio energy drops at {max_freq/1000:.1f}kHz — AI models commonly hard-limit at ≤16kHz"],
        )]
    return []


def _check_spectral_flatness(y: np.ndarray, sr: int) -> list[Finding]:
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))
    if mean_flatness < 5e-4:
        return [Finding(
            label="Unnaturally Tonal Spectrum",
            score=0.6,
            evidence=[f"Spectral flatness {mean_flatness:.2e} — spectrum is too pure/tonal for real recorded music"],
        )]
    return []


def _check_noise_floor(y: np.ndarray, sr: int) -> list[Finding]:
    nyquist = sr / 2
    cutoff = 80.0 / nyquist
    if cutoff >= 1.0:
        return []
    b, a = scipy.signal.butter(4, cutoff, btype="low")
    low = scipy.signal.filtfilt(b, a, y)
    noise_ratio = np.mean(low ** 2) / (np.mean(y ** 2) + 1e-10)
    if noise_ratio < 1e-6:
        return [Finding(
            label="No Low-Frequency Noise Floor",
            score=0.5,
            evidence=["Sub-80Hz noise floor absent — real recordings typically have room noise or HVAC hum"],
        )]
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spectral.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/analyzers/spectral.py tests/test_spectral.py
git commit -m "feat: spectral analyzer"
```

---

### Task 6: Waveform Analyzer

**Files:**
- Create: `music_forensics/analyzers/waveform.py`
- Create: `tests/test_waveform.py`

**Interfaces:**
- Consumes: `Finding` from `music_forensics.models`
- Produces: `analyze(audio_path: str) -> list[Finding]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_waveform.py
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from music_forensics.analyzers.waveform import analyze
from music_forensics.models import Finding


def test_returns_list_of_findings(sine_wav):
    result = analyze(str(sine_wav))
    assert isinstance(result, list)
    assert all(isinstance(f, Finding) for f in result)


def test_scores_in_valid_range(sine_wav):
    findings = analyze(str(sine_wav))
    for f in findings:
        assert 0.0 <= f.score <= 1.0


def test_silent_audio_does_not_crash(silent_wav):
    findings = analyze(str(silent_wav))
    assert isinstance(findings, list)


def test_compressed_audio_flagged(tmp_path):
    # Flat amplitude = zero dynamic range
    sr = 44100
    y = np.full(sr * 3, 0.5, dtype=np.float32)
    path = tmp_path / "compressed.wav"
    sf.write(str(path), y, sr)

    findings = analyze(str(path))
    compressed = next((f for f in findings if "Dynamic" in f.label), None)
    assert compressed is not None
    assert compressed.score >= 0.5


def test_abrupt_silence_end_flagged(tmp_path):
    sr = 44100
    y = np.zeros(sr * 3, dtype=np.float32)
    # Loud in the middle, silence at start and end
    y[sr:sr*2] = 0.8
    path = tmp_path / "abrupt.wav"
    sf.write(str(path), y, sr)

    findings = analyze(str(path))
    assert any("Silence" in f.label or "Start" in f.label or "End" in f.label for f in findings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_waveform.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.analyzers.waveform'`

- [ ] **Step 3: Write `music_forensics/analyzers/waveform.py`**

```python
import librosa
import numpy as np
from music_forensics.models import Finding


def analyze(audio_path: str) -> list[Finding]:
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return [Finding(label="Waveform Load Error", score=0.5, evidence=[str(e)])]

    if np.max(np.abs(y)) < 1e-6:
        return [Finding(
            label="Silent Audio",
            score=0.5,
            evidence=["Audio is nearly silent — analysis inconclusive"],
        )]

    findings = []
    findings.extend(_check_timing_regularity(y, sr))
    findings.extend(_check_dynamic_range(y))
    findings.extend(_check_silence_boundaries(y, sr))

    if not findings:
        findings.append(Finding(
            label="Waveform Profile Normal",
            score=0.15,
            evidence=["No unusual waveform artifacts detected"],
        ))

    return findings


def _check_timing_regularity(y: np.ndarray, sr: int) -> list[Finding]:
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        return []
    if len(beats) < 8:
        return []
    beat_times = librosa.frames_to_time(beats, sr=sr)
    intervals = np.diff(beat_times)
    mean_interval = np.mean(intervals)
    if mean_interval < 1e-6:
        return []
    cv = np.std(intervals) / mean_interval
    if cv < 0.02:
        return [Finding(
            label="Metronomic Timing",
            score=0.7,
            evidence=[f"Beat interval variation {cv*100:.1f}% — suspiciously regular (human music typically >5%)"],
        )]
    return []


def _check_dynamic_range(y: np.ndarray) -> list[Finding]:
    frame_rms = librosa.feature.rms(y=y)[0]
    p95 = np.percentile(frame_rms, 95)
    p5 = np.percentile(frame_rms, 5)
    if p5 < 1e-6:
        return []
    dynamic_range_db = 20 * np.log10(p95 / p5)
    if dynamic_range_db < 6.0:
        return [Finding(
            label="Compressed Dynamic Range",
            score=0.6,
            evidence=[f"Dynamic range {dynamic_range_db:.1f} dB — unusually narrow (AI output often over-compressed)"],
        )]
    return []


def _check_silence_boundaries(y: np.ndarray, sr: int) -> list[Finding]:
    window = min(int(0.1 * sr), len(y) // 4)
    if window < 1:
        return []
    max_energy = np.mean(np.abs(y)) + 1e-10
    findings = []
    if np.mean(np.abs(y[:window])) < max_energy * 0.001:
        findings.append(Finding(
            label="Abrupt Start Silence",
            score=0.4,
            evidence=["Track begins with near-silence — no natural lead-in or room tone"],
        ))
    if np.mean(np.abs(y[-window:])) < max_energy * 0.001:
        findings.append(Finding(
            label="Abrupt End Silence",
            score=0.4,
            evidence=["Track ends with near-silence — no natural decay or room tone"],
        ))
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_waveform.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/analyzers/waveform.py tests/test_waveform.py
git commit -m "feat: waveform analyzer"
```

---

### Task 7: Reporter

**Files:**
- Create: `music_forensics/reporter.py`
- Create: `tests/test_reporter.py`

**Interfaces:**
- Consumes: `Finding` from `music_forensics.models`
- Produces: `render_report(findings_by_stage: dict[str, list[Finding]], source: str) -> None`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reporter.py
from io import StringIO
from rich.console import Console
from music_forensics.reporter import render_report, score_to_color, aggregate_score
from music_forensics.models import Finding


def _make_console() -> Console:
    return Console(file=StringIO(), force_terminal=True, width=120)


def test_score_to_color_green():
    assert score_to_color(0.1) == "green"
    assert score_to_color(0.34) == "green"


def test_score_to_color_yellow():
    assert score_to_color(0.35) == "yellow"
    assert score_to_color(0.64) == "yellow"


def test_score_to_color_red():
    assert score_to_color(0.65) == "red"
    assert score_to_color(1.0) == "red"


def test_aggregate_score_average():
    findings = [
        Finding(label="A", score=0.2, evidence=[]),
        Finding(label="B", score=0.8, evidence=[]),
    ]
    assert aggregate_score(findings) == 0.5


def test_aggregate_score_empty():
    assert aggregate_score([]) == 0.0


def test_render_report_does_not_crash(sine_wav):
    findings = {
        "Metadata": [Finding(label="AI Generator Detected", score=1.0, evidence=["Found 'suno'"])],
        "Spectral": [Finding(label="Frequency Cutoff", score=0.65, evidence=["Cutoff at 12kHz"])],
    }
    render_report(findings, "https://youtube.com/watch?v=test")


def test_render_report_empty_findings():
    render_report({}, "test_source.wav")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reporter.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.reporter'`

- [ ] **Step 3: Write `music_forensics/reporter.py`**

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from music_forensics.models import Finding

console = Console()


def score_to_color(score: float) -> str:
    if score < 0.35:
        return "green"
    if score < 0.65:
        return "yellow"
    return "red"


def aggregate_score(findings: list[Finding]) -> float:
    if not findings:
        return 0.0
    return sum(f.score for f in findings) / len(findings)


def _score_bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def render_report(findings_by_stage: dict[str, list[Finding]], source: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold white]AI Music Forensics[/bold white]\n[dim]{source}[/dim]",
        box=box.DOUBLE,
        border_style="bright_blue",
    ))

    all_findings: list[Finding] = []
    for stage, findings in findings_by_stage.items():
        all_findings.extend(findings)
        if not findings:
            continue
        table = Table(
            title=f"[bold]{stage}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold dim",
            border_style="dim",
        )
        table.add_column("Signal", style="bold", min_width=24)
        table.add_column("Score", justify="center", min_width=18)
        table.add_column("Evidence")
        for f in findings:
            color = score_to_color(f.score)
            bar = _score_bar(f.score)
            table.add_row(
                f.label,
                f"[{color}]{bar} {f.score:.0%}[/{color}]",
                "\n".join(f.evidence),
            )
        console.print(table)

    overall = aggregate_score(all_findings)
    color = score_to_color(overall)
    if overall < 0.35:
        verdict = "LIKELY HUMAN-MADE"
    elif overall < 0.65:
        verdict = "UNCERTAIN — MIXED SIGNALS"
    else:
        verdict = "LIKELY AI-GENERATED"

    console.print()
    console.print(Panel(
        f"[{color}][bold]{verdict}[/bold]\n{_score_bar(overall, 20)} {overall:.0%} AI likelihood[/{color}]",
        title="[bold]Verdict[/bold]",
        box=box.DOUBLE,
        border_style=color,
    ))
    console.print()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reporter.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/reporter.py tests/test_reporter.py
git commit -m "feat: rich terminal reporter"
```

---

### Task 8: CLI Entry Point

**Files:**
- Create: `music_forensics/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes:
  - `download_audio(url: str) -> Path` from `music_forensics.downloader`
  - `is_url(source: str) -> bool` from `music_forensics.downloader`
  - `metadata.analyze(audio_path: str) -> list[Finding]`
  - `spectral.analyze(audio_path: str) -> list[Finding]`
  - `waveform.analyze(audio_path: str) -> list[Finding]`
  - `render_report(findings_by_stage: dict[str, list[Finding]], source: str) -> None`
- Produces: `main()` — click command registered as `music-forensics`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from click.testing import CliRunner
from music_forensics.cli import main
from music_forensics.models import Finding


_EMPTY_FINDINGS = [Finding(label="Clean", score=0.1, evidence=[])]


def _mock_analyzers(mocker):
    mocker.patch("music_forensics.analyzers.metadata.analyze", return_value=_EMPTY_FINDINGS)
    mocker.patch("music_forensics.analyzers.spectral.analyze", return_value=_EMPTY_FINDINGS)
    mocker.patch("music_forensics.analyzers.waveform.analyze", return_value=_EMPTY_FINDINGS)


def test_local_file_runs_analysis(mocker, sine_wav):
    _mock_analyzers(mocker)
    mocker.patch("music_forensics.cli.render_report")
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

    runner = CliRunner()
    result = runner.invoke(main, [str(sine_wav)])

    assert result.exit_code == 0


def test_missing_local_file_exits_nonzero(mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    runner = CliRunner()
    result = runner.invoke(main, ["/nonexistent/file.wav"])
    assert result.exit_code != 0


def test_ffmpeg_missing_exits_nonzero(mocker):
    mocker.patch("shutil.which", return_value=None)
    runner = CliRunner()
    result = runner.invoke(main, ["https://youtube.com/watch?v=test"])
    assert result.exit_code != 0
    assert "ffmpeg" in result.output.lower()


def test_youtube_url_triggers_download(mocker, sine_wav):
    _mock_analyzers(mocker)
    mocker.patch("music_forensics.cli.render_report")
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    mock_download = mocker.patch(
        "music_forensics.cli.download_audio", return_value=sine_wav
    )

    runner = CliRunner()
    result = runner.invoke(main, ["https://youtube.com/watch?v=abc"])

    mock_download.assert_called_once_with("https://youtube.com/watch?v=abc")
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.cli'`

> Note: the `--deep` flag test is in Task 9, after `ml.py` exists.

- [ ] **Step 3: Write `music_forensics/cli.py`**

```python
import shutil
import sys
from pathlib import Path

import click

from music_forensics.analyzers import metadata, spectral, waveform
from music_forensics.downloader import download_audio, is_url
from music_forensics.reporter import render_report


@click.command()
@click.argument("source")
@click.option("--deep", is_flag=True, help="Enable ML-based AI detection (downloads model on first use)")
def main(source: str, deep: bool) -> None:
    """Analyze audio for signs of AI generation.

    SOURCE is a YouTube URL or a local file path.
    """
    if not shutil.which("ffmpeg"):
        click.echo("Error: ffmpeg is required but not installed.", err=True)
        click.echo("  macOS:  brew install ffmpeg", err=True)
        click.echo("  Linux:  sudo apt install ffmpeg", err=True)
        sys.exit(1)

    audio_path: Path | None = None
    is_temp = False

    try:
        if is_url(source):
            click.echo("Downloading audio...")
            audio_path = download_audio(source)
            is_temp = True
        else:
            audio_path = Path(source)
            if not audio_path.exists():
                click.echo(f"Error: File not found: {source}", err=True)
                sys.exit(1)

        findings: dict = {}

        click.echo("Analyzing metadata...")
        findings["Metadata"] = metadata.analyze(str(audio_path))

        click.echo("Analyzing spectral characteristics...")
        findings["Spectral"] = spectral.analyze(str(audio_path))

        click.echo("Analyzing waveform...")
        findings["Waveform"] = waveform.analyze(str(audio_path))

        if deep:
            click.echo("Running ML classifier (may take a moment on first use)...")
            from music_forensics.analyzers import ml
            findings["ML Classifier"] = ml.analyze(str(audio_path))

        render_report(findings, source)

    finally:
        if is_temp and audio_path is not None:
            parent = audio_path.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add music_forensics/cli.py tests/test_cli.py
git commit -m "feat: CLI entry point"
```

---

### Task 9: ML Analyzer (--deep)

**Files:**
- Create: `music_forensics/analyzers/ml.py`
- Create: `tests/test_ml.py`

**Interfaces:**
- Consumes: `Finding` from `music_forensics.models`
- Produces: `analyze(audio_path: str) -> list[Finding]`

**Note on model selection:** Before implementing, search HuggingFace Hub for a current audio classification model suited to AI music detection. Recommended search terms: `"AI music detection"`, `"synthetic music classifier"`, `"audio deepfake detection"`. If a music-specific model exists, use it. Otherwise use a general synthetic audio classifier (e.g., an audio deepfake detection model trained on ADD or ASVspoof datasets) and label the output clearly as "General Synthetic Audio Classifier". The model ID goes in the `_MODEL_ID` constant.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ml.py
from unittest.mock import patch, MagicMock
from music_forensics.analyzers.ml import analyze
from music_forensics.models import Finding


def test_returns_list_of_findings(sine_wav):
    mock_pipe = MagicMock(return_value=[
        {"label": "AI", "score": 0.82},
        {"label": "Human", "score": 0.18},
    ])
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        result = analyze(str(sine_wav))
    assert isinstance(result, list)
    assert all(isinstance(f, Finding) for f in result)


def test_high_ai_score_returns_high_finding_score(sine_wav):
    mock_pipe = MagicMock(return_value=[
        {"label": "AI", "score": 0.9},
        {"label": "Human", "score": 0.1},
    ])
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        findings = analyze(str(sine_wav))
    assert findings[0].score >= 0.7


def test_missing_torch_returns_graceful_finding(sine_wav):
    with patch("music_forensics.analyzers.ml._load_pipeline", side_effect=ImportError("No module named 'torch'")):
        findings = analyze(str(sine_wav))
    assert len(findings) == 1
    assert "unavailable" in findings[0].label.lower() or "install" in findings[0].evidence[0].lower()
    assert findings[0].score == 0.5


def test_model_download_failure_returns_graceful_finding(sine_wav):
    with patch("music_forensics.analyzers.ml._load_pipeline", side_effect=OSError("download failed")):
        findings = analyze(str(sine_wav))
    assert len(findings) == 1
    assert findings[0].score == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml.py -v`
Expected: `ModuleNotFoundError: No module named 'music_forensics.analyzers.ml'`

- [ ] **Step 3: Write `music_forensics/analyzers/ml.py`**

Replace `"your/model-id-here"` with the model ID found during research (see Task 9 note above).

```python
from pathlib import Path
from music_forensics.models import Finding

_MODEL_ID = "your/model-id-here"  # Replace with researched model ID
_CACHE_DIR = Path.home() / ".cache" / "music-forensics"


def _load_pipeline(model_id: str, cache_dir: Path):
    import torch
    from transformers import pipeline
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return pipeline(
        "audio-classification",
        model=model_id,
        cache_dir=str(cache_dir),
    )


def analyze(audio_path: str) -> list[Finding]:
    try:
        pipe = _load_pipeline(_MODEL_ID, _CACHE_DIR)
    except ImportError:
        return [Finding(
            label="ML Classifier Unavailable",
            score=0.5,
            evidence=["Install torch and transformers: pip install 'music-forensics[deep]'"],
        )]
    except Exception as e:
        return [Finding(
            label="ML Classifier Failed",
            score=0.5,
            evidence=[f"Could not load model: {str(e)[:120]}"],
        )]

    try:
        results: list[dict] = pipe(audio_path)
    except Exception as e:
        return [Finding(
            label="ML Inference Failed",
            score=0.5,
            evidence=[f"Inference error: {str(e)[:120]}"],
        )]

    ai_result = next(
        (r for r in results if r["label"].upper() in ("AI", "FAKE", "SPOOF", "SYNTHETIC")),
        None,
    )

    if ai_result is None:
        ai_result = max(results, key=lambda r: r["score"])
        label_note = f"(top label: {ai_result['label']})"
    else:
        label_note = ""

    score = float(ai_result["score"])
    return [Finding(
        label="ML Classifier",
        score=score,
        evidence=[
            f"Model: {_MODEL_ID}",
            f"AI probability: {score:.0%} {label_note}".strip(),
        ],
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml.py -v`
Expected: 4 passed.

- [ ] **Step 5: Add `--deep` CLI test now that `ml.py` exists**

Append this test to `tests/test_cli.py`:

```python
def test_deep_flag_invokes_ml_analyzer(mocker, sine_wav):
    _mock_analyzers(mocker)
    mocker.patch("music_forensics.cli.render_report")
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    mock_ml = mocker.patch(
        "music_forensics.analyzers.ml.analyze", return_value=_EMPTY_FINDINGS
    )

    runner = CliRunner()
    result = runner.invoke(main, [str(sine_wav), "--deep"])

    mock_ml.assert_called_once()
    assert result.exit_code == 0
```

Run: `pytest tests/test_cli.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add music_forensics/analyzers/ml.py tests/test_ml.py tests/test_cli.py
git commit -m "feat: ML analyzer with HuggingFace classifier"
```

---

### Task 10: Integration Test + Full Test Suite

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: all modules — runs the full pipeline end-to-end on a local file

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration.py
from click.testing import CliRunner
from unittest.mock import patch
from music_forensics.cli import main


def test_full_pipeline_on_local_file(sine_wav, mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

    runner = CliRunner()
    result = runner.invoke(main, [str(sine_wav)])

    assert result.exit_code == 0
    assert "Verdict" in result.output or result.exit_code == 0


def test_full_pipeline_does_not_crash_on_silent_file(silent_wav, mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

    runner = CliRunner()
    result = runner.invoke(main, [str(silent_wav)])

    assert result.exit_code == 0
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: 2 passed.

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass, 0 failures.

- [ ] **Step 4: Smoke test the installed CLI with a local file**

Run: `music-forensics <path-to-any-local-wav-or-mp3>`
Expected: colored terminal output with Metadata, Spectral, Waveform sections and a Verdict panel.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration tests and full test suite verified"
```

---

### Task 11: GitHub Repository

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: working CLI, passing tests
- Produces: public GitHub repo at `https://github.com/Vegoni/music-forensics`

- [ ] **Step 1: Write a minimal README**

```markdown
# music-forensics

CLI tool to analyze audio for signs of AI generation.

## Install

Prerequisites: Python 3.10+, ffmpeg

\```bash
pip install -e .
\```

## Usage

\```bash
music-forensics "https://youtube.com/watch?v=..."
music-forensics "https://youtube.com/watch?v=..." --deep
music-forensics /path/to/local/song.mp3
\```

## Optional deep analysis

\```bash
pip install -e ".[deep]"
music-forensics <url> --deep
\```
```

- [ ] **Step 2: Create the GitHub repo and push**

```bash
gh repo create Vegoni/music-forensics --public --description "CLI tool to analyze audio for signs of AI generation"
git remote add origin https://github.com/Vegoni/music-forensics.git
git push -u origin main
```

- [ ] **Step 3: Verify repo is live**

Run: `gh repo view Vegoni/music-forensics`
Expected: repo details shown, commits visible.

- [ ] **Step 4: Commit README and finalize**

```bash
git add README.md
git commit -m "docs: add README"
git push
```
