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
    assert any("general synthetic" in e.lower() for f in result for e in f.evidence)


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
