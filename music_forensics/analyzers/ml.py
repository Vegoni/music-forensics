import re
from pathlib import Path
from music_forensics.models import Finding

# AI-Music-Detection/ai_music_detection_large_60s is an Audio Spectrogram
# Transformer (fine-tuned from mit/ast-finetuned-audioset-10-10-0.4593) trained on
# SleepyJesse/ai_music_large — 10k human-composed and 10k AI-generated tracks. It
# is trained on the question this tool actually asks, unlike the general
# speech-oriented deepfake detectors it replaced.
_MODEL_ID = "AI-Music-Detection/ai_music_detection_large_60s"
_CACHE_DIR = Path.home() / ".cache" / "music-forensics"

# The model was trained with a 60-second window. Feeding it more buys nothing and
# costs inference time, and it keeps --deep consistent with the other analyzers,
# which each cap at 90 seconds.
_MAX_SECONDS = 60.0
_FALLBACK_SAMPLE_RATE = 16000

# Label vocabularies, matched per-token against the model's own label names. Token
# matching rather than substring matching so that "chair" does not read as "ai".
# Inflected forms are listed explicitly rather than matched by prefix: prefix
# matching on a token as short as "ai" would read "air" and "aim" as AI.
_AI_TERMS = frozenset({
    "ai", "fake", "faked", "deepfake", "spoof", "spoofed", "synthetic", "synth",
    "generated", "generative", "artificial", "machine",
})
_HUMAN_TERMS = frozenset({
    "human", "real", "bonafide", "authentic", "genuine", "natural", "organic",
})


def _load_pipeline(model_id: str, cache_dir: Path):
    import torch  # noqa: F401 — lazy import; presence check
    from transformers import pipeline
    cache_dir.mkdir(parents=True, exist_ok=True)
    return pipeline(
        "audio-classification",
        model=model_id,
        cache_dir=str(cache_dir),
    )


def _classify_label(label: str) -> str | None:
    """Map a model's label name onto "ai", "human", or None if unrecognized."""
    tokens = set(re.sub(r"[^a-z0-9]+", " ", label.lower()).split())
    if tokens & _AI_TERMS:
        return "ai"
    if tokens & _HUMAN_TERMS:
        return "human"
    return None


def _ai_probability(results: list[dict]) -> float | None:
    """Probability that the audio is AI-generated, or None if labels are unreadable.

    Returning None matters: guessing here would silently invert the verdict. The
    previous implementation fell back to whichever label scored highest and
    reported that as the AI probability, so a model answering "real, 95%" would
    have been reported as "AI probability: 95%".
    """
    for r in results:
        if _classify_label(r["label"]) == "ai":
            return float(r["score"])

    # No AI label. For a two-class model, the complement of the human class is
    # equivalent. Any other shape is ambiguous and must not be guessed at.
    if len(results) == 2:
        for r in results:
            if _classify_label(r["label"]) == "human":
                return 1.0 - float(r["score"])

    return None


def _load_audio(audio_path: str, sample_rate: int):
    import librosa
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=_MAX_SECONDS)
    return y


def analyze(audio_path: str) -> list[Finding]:
    try:
        pipe = _load_pipeline(_MODEL_ID, _CACHE_DIR)
    except ImportError:
        return [Finding(
            label="ML Classifier Unavailable",
            score=0.5,
            evidence=["Install torch and transformers: pip install 'music-forensics[deep]'"],
            counts_toward_verdict=False,
        )]
    except Exception as e:
        return [Finding(
            label="ML Classifier Failed",
            score=0.5,
            evidence=[f"Could not load model: {str(e)[:120]}"],
            counts_toward_verdict=False,
        )]

    sample_rate = getattr(getattr(pipe, "feature_extractor", None), "sampling_rate", None)
    if not isinstance(sample_rate, int):
        sample_rate = _FALLBACK_SAMPLE_RATE

    try:
        audio = _load_audio(audio_path, sample_rate)
        results: list[dict] = pipe(audio)
    except Exception as e:
        return [Finding(
            label="ML Inference Failed",
            score=0.5,
            evidence=[f"Inference error: {str(e)[:120]}"],
            counts_toward_verdict=False,
        )]

    score = _ai_probability(results)
    if score is None:
        seen = ", ".join(r["label"] for r in results) or "none"
        return [Finding(
            label="ML Classifier Labels Unrecognized",
            score=0.5,
            evidence=[
                f"Model: {_MODEL_ID}",
                f"Could not tell which label means AI-generated (saw: {seen})",
            ],
            counts_toward_verdict=False,
        )]

    return [Finding(
        label="ML Classifier",
        score=score,
        evidence=[
            f"Model: {_MODEL_ID} (Audio Spectrogram Transformer, trained on AI vs. human music)",
            f"AI probability: {score:.0%}",
            f"Analyzed first {_MAX_SECONDS:.0f}s at {sample_rate} Hz",
        ],
    )]
