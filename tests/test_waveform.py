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
