import pytest
from unittest.mock import patch, MagicMock
from music_forensics.analyzers.ml import analyze, _ai_probability, _classify_label
from music_forensics.models import Finding


@pytest.mark.parametrize("label", ["ai", "AI", "fake", "Fake Audio", "synthetic", "ai_generated"])
def test_classify_label_recognizes_ai(label):
    assert _classify_label(label) == "ai"


@pytest.mark.parametrize("label", ["human", "Human", "real", "bonafide", "human_composed"])
def test_classify_label_recognizes_human(label):
    assert _classify_label(label) == "human"


@pytest.mark.parametrize("label", ["LABEL_0", "class1", "chair", ""])
def test_classify_label_rejects_unknown(label):
    assert _classify_label(label) is None


def test_ai_probability_reads_ai_label():
    assert _ai_probability([{"label": "ai", "score": 0.9}, {"label": "human", "score": 0.1}]) == 0.9


def test_ai_probability_infers_from_human_label_in_binary_model():
    """A two-class model with no AI-named label: the complement is unambiguous."""
    result = _ai_probability([{"label": "human", "score": 0.95}, {"label": "other", "score": 0.05}])
    assert result == pytest.approx(0.05)


def test_ai_probability_none_when_labels_unrecognized():
    """Must not guess. Guessing here silently inverts the verdict."""
    assert _ai_probability([{"label": "LABEL_0", "score": 0.95}, {"label": "LABEL_1", "score": 0.05}]) is None


def test_unrecognized_labels_do_not_invert_the_verdict(sine_wav):
    """Regression: a confident 'real' reading must never be reported as AI."""
    mock_pipe = MagicMock(return_value=[
        {"label": "bonafide", "score": 0.95},
        {"label": "spoofed_speech", "score": 0.03},
        {"label": "unknown", "score": 0.02},
    ])
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        findings = analyze(str(sine_wav))
    # 'spoofed_speech' matches the AI vocabulary, so this reads as AI at 3%. The
    # property under test is that 0.95 is never reported as the AI probability.
    assert findings[0].score == pytest.approx(0.03)


def test_three_class_model_with_no_ai_label_is_not_counted(sine_wav):
    """Ambiguous label shape must produce no measurement rather than a guess."""
    mock_pipe = MagicMock(return_value=[
        {"label": "LABEL_0", "score": 0.95},
        {"label": "LABEL_1", "score": 0.03},
        {"label": "LABEL_2", "score": 0.02},
    ])
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        findings = analyze(str(sine_wav))
    assert findings[0].counts_toward_verdict is False
    assert "LABEL_0" in findings[0].evidence[1]


def test_returns_list_of_findings(sine_wav):
    mock_pipe = MagicMock(return_value=[
        {"label": "AI", "score": 0.82},
        {"label": "Human", "score": 0.18},
    ])
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        result = analyze(str(sine_wav))
    assert isinstance(result, list)
    assert all(isinstance(f, Finding) for f in result)
    assert any("ai vs. human music" in e.lower() for f in result for e in f.evidence)


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
    assert findings[0].counts_toward_verdict is False


def test_model_download_failure_returns_graceful_finding(sine_wav):
    with patch("music_forensics.analyzers.ml._load_pipeline", side_effect=OSError("download failed")):
        findings = analyze(str(sine_wav))
    assert len(findings) == 1
    assert findings[0].counts_toward_verdict is False


def test_inference_failure_does_not_count_toward_verdict(sine_wav):
    mock_pipe = MagicMock(side_effect=RuntimeError("inference blew up"))
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        findings = analyze(str(sine_wav))
    assert len(findings) == 1
    assert findings[0].counts_toward_verdict is False


def test_successful_classification_counts_toward_verdict(sine_wav):
    mock_pipe = MagicMock(return_value=[
        {"label": "AI", "score": 0.82},
        {"label": "Human", "score": 0.18},
    ])
    with patch("music_forensics.analyzers.ml._load_pipeline", return_value=mock_pipe):
        findings = analyze(str(sine_wav))
    assert findings[0].counts_toward_verdict is True
