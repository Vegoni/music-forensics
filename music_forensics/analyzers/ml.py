from pathlib import Path
from music_forensics.models import Finding

# MelodyMachine/Deepfake-audio-detection-V2 is a wav2vec2-based audio classification
# model trained for deepfake/synthetic audio detection. It outputs labels containing
# "fake"/"real" variants, compatible with our FAKE label matching logic.
_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
_CACHE_DIR = Path.home() / ".cache" / "music-forensics"


def _load_pipeline(model_id: str, cache_dir: Path):
    import torch  # noqa: F401 — lazy import; presence check
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
