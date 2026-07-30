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
