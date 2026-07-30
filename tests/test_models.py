from music_forensics.models import Finding


def test_finding_fields():
    f = Finding(label="Test Signal", score=0.75, evidence=["reason one", "reason two"])
    assert f.label == "Test Signal"
    assert f.score == 0.75
    assert f.evidence == ["reason one", "reason two"]


def test_finding_default_evidence():
    f = Finding(label="Test", score=0.0)
    assert f.evidence == []
