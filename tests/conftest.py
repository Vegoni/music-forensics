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
