from io import StringIO
from rich.console import Console
from music_forensics.reporter import render_report, score_to_color, aggregate_score
from music_forensics.models import Finding


def _make_console() -> Console:
    return Console(file=StringIO(), force_terminal=True, width=120)


def test_score_to_color_green():
    assert score_to_color(0.1) == "green"
    assert score_to_color(0.34) == "green"


def test_score_to_color_yellow():
    assert score_to_color(0.35) == "yellow"
    assert score_to_color(0.64) == "yellow"


def test_score_to_color_red():
    assert score_to_color(0.65) == "red"
    assert score_to_color(1.0) == "red"


def test_aggregate_score_average():
    findings = [
        Finding(label="A", score=0.2, evidence=[]),
        Finding(label="B", score=0.8, evidence=[]),
    ]
    assert aggregate_score(findings) == 0.5


def test_aggregate_score_empty():
    assert aggregate_score([]) == 0.0


def test_render_report_does_not_crash(sine_wav):
    findings = {
        "Metadata": [Finding(label="AI Generator Detected", score=1.0, evidence=["Found 'suno'"])],
        "Spectral": [Finding(label="Frequency Cutoff", score=0.65, evidence=["Cutoff at 12kHz"])],
    }
    render_report(findings, "https://youtube.com/watch?v=test")


def test_render_report_empty_findings():
    render_report({}, "test_source.wav")
