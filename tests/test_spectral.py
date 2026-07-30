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
