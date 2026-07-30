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
