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
