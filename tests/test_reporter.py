from io import StringIO
from rich.console import Console
from music_forensics.reporter import (
    render_report,
    score_to_color,
    aggregate_score,
    stage_score,
)
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


def test_stage_score_averages_findings():
    findings = [
        Finding(label="A", score=0.2, evidence=[]),
        Finding(label="B", score=0.8, evidence=[]),
    ]
    assert stage_score(findings) == 0.5


def test_stage_score_ignores_uncounted_findings():
    findings = [
        Finding(label="A", score=0.2, evidence=[]),
        Finding(label="Failed", score=0.5, evidence=[], counts_toward_verdict=False),
    ]
    assert stage_score(findings) == 0.2


def test_stage_score_none_when_no_measurements():
    findings = [Finding(label="Failed", score=0.5, evidence=[], counts_toward_verdict=False)]
    assert stage_score(findings) is None


def test_aggregate_score_weighted_average():
    findings = {
        "Metadata": [Finding(label="A", score=0.0, evidence=[])],
        "ML Classifier": [Finding(label="B", score=1.0, evidence=[])],
    }
    # Metadata weight 1.0, ML weight 4.0 -> (0.0*1 + 1.0*4) / 5
    assert aggregate_score(findings) == 0.8


def test_aggregate_score_is_not_diluted_by_finding_count():
    """A stage emitting many rows must not outweigh one emitting a single row."""
    many = {
        "Metadata": [Finding(label=f"m{i}", score=0.0, evidence=[]) for i in range(10)],
        "Spectral": [Finding(label="s", score=0.0, evidence=[])],
        "Waveform": [Finding(label="w", score=0.0, evidence=[])],
        "ML Classifier": [Finding(label="ml", score=1.0, evidence=[])],
    }
    # ML is 4.0 of 7.0 total weight regardless of how many rows Metadata emitted.
    assert aggregate_score(many) == 4 / 7


def test_failed_ml_stage_does_not_drag_verdict():
    """A broken analyzer must be ignored, not scored as 0.5."""
    human = [Finding(label="h", score=0.1, evidence=[])]
    without_ml = {"Metadata": human, "Spectral": human, "Waveform": human}
    with_failed_ml = {
        **without_ml,
        "ML Classifier": [
            Finding(label="Failed", score=0.5, evidence=[], counts_toward_verdict=False)
        ],
    }
    assert aggregate_score(with_failed_ml) == aggregate_score(without_ml)


def test_aggregate_score_empty():
    assert aggregate_score({}) is None


def test_aggregate_score_none_when_every_stage_failed():
    findings = {
        "ML Classifier": [
            Finding(label="Failed", score=0.5, evidence=[], counts_toward_verdict=False)
        ],
    }
    assert aggregate_score(findings) is None


def test_render_report_handles_no_verdict():
    findings = {
        "ML Classifier": [
            Finding(label="Failed", score=0.5, evidence=["boom"], counts_toward_verdict=False)
        ],
    }
    render_report(findings, "test_source.wav")


def test_render_report_does_not_crash(sine_wav):
    findings = {
        "Metadata": [Finding(label="AI Generator Detected", score=1.0, evidence=["Found 'suno'"])],
        "Spectral": [Finding(label="Frequency Cutoff", score=0.65, evidence=["Cutoff at 12kHz"])],
    }
    render_report(findings, "https://youtube.com/watch?v=test")


def test_render_report_empty_findings():
    render_report({}, "test_source.wav")
