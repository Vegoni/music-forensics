from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from music_forensics.models import Finding

console = Console()


def score_to_color(score: float) -> str:
    if score < 0.35:
        return "green"
    if score < 0.65:
        return "yellow"
    return "red"


# Relative influence of each stage on the final verdict.
#
# Scoring is per-stage rather than per-finding so that a stage's weight does not
# depend on how many rows it happens to emit. Under a flat per-finding average the
# ML classifier — a single row — was worth under 10% of the verdict while the
# heuristic stages contributed a dozen rows between them, so enabling --deep
# barely moved the result.
#
# The ML classifier outweighs any single heuristic stage because it is the only
# signal trained directly on the question being asked. It is deliberately held
# below a majority (2.0 against 3.0 combined = 40%) because the pinned model is a
# general synthetic-audio detector, not a music-specific one.
STAGE_WEIGHTS = {
    "Metadata": 1.0,
    "Spectral": 1.0,
    "Waveform": 1.0,
    "ML Classifier": 2.0,
}
DEFAULT_STAGE_WEIGHT = 1.0


def stage_score(findings: list[Finding]) -> float | None:
    """Mean of a stage's scored findings, or None if it produced no measurement."""
    scored = [f for f in findings if f.counts_toward_verdict]
    if not scored:
        return None
    return sum(f.score for f in scored) / len(scored)


def aggregate_score(findings_by_stage: dict[str, list[Finding]]) -> float | None:
    """Weighted mean of per-stage scores.

    Returns None when no stage produced a usable measurement, which the caller
    must report as "no verdict" rather than as a score of zero.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for stage, findings in findings_by_stage.items():
        score = stage_score(findings)
        if score is None:
            continue
        weight = STAGE_WEIGHTS.get(stage, DEFAULT_STAGE_WEIGHT)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def _score_bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def render_report(findings_by_stage: dict[str, list[Finding]], source: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold white]AI Music Forensics[/bold white]\n[dim]{source}[/dim]",
        box=box.DOUBLE,
        border_style="bright_blue",
    ))

    for stage, findings in findings_by_stage.items():
        if not findings:
            continue
        table = Table(
            title=f"[bold]{stage}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold dim",
            border_style="dim",
        )
        table.add_column("Signal", style="bold", min_width=24)
        table.add_column("Score", justify="center", min_width=18)
        table.add_column("Evidence")
        for f in findings:
            if f.counts_toward_verdict:
                color = score_to_color(f.score)
                score_cell = f"[{color}]{_score_bar(f.score)} {f.score:.0%}[/{color}]"
            else:
                score_cell = "[dim]— not counted[/dim]"
            table.add_row(
                f.label,
                score_cell,
                "\n".join(f.evidence),
            )
        console.print(table)

    overall = aggregate_score(findings_by_stage)

    console.print()
    if overall is None:
        console.print(Panel(
            "[dim][bold]NO VERDICT[/bold]\nNo analyzer produced a usable measurement.[/dim]",
            title="[bold]Verdict[/bold]",
            box=box.DOUBLE,
            border_style="dim",
        ))
        console.print()
        return

    color = score_to_color(overall)
    if overall < 0.35:
        verdict = "LIKELY HUMAN-MADE"
    elif overall < 0.65:
        verdict = "UNCERTAIN — MIXED SIGNALS"
    else:
        verdict = "LIKELY AI-GENERATED"

    console.print(Panel(
        f"[{color}][bold]{verdict}[/bold]\n{_score_bar(overall, 20)} {overall:.0%} AI likelihood[/{color}]",
        title="[bold]Verdict[/bold]",
        box=box.DOUBLE,
        border_style=color,
    ))
    console.print()
