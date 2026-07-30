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


def aggregate_score(findings: list[Finding]) -> float:
    if not findings:
        return 0.0
    return sum(f.score for f in findings) / len(findings)


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

    all_findings: list[Finding] = []
    for stage, findings in findings_by_stage.items():
        all_findings.extend(findings)
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
            color = score_to_color(f.score)
            bar = _score_bar(f.score)
            table.add_row(
                f.label,
                f"[{color}]{bar} {f.score:.0%}[/{color}]",
                "\n".join(f.evidence),
            )
        console.print(table)

    overall = aggregate_score(all_findings)
    color = score_to_color(overall)
    if overall < 0.35:
        verdict = "LIKELY HUMAN-MADE"
    elif overall < 0.65:
        verdict = "UNCERTAIN — MIXED SIGNALS"
    else:
        verdict = "LIKELY AI-GENERATED"

    console.print()
    console.print(Panel(
        f"[{color}][bold]{verdict}[/bold]\n{_score_bar(overall, 20)} {overall:.0%} AI likelihood[/{color}]",
        title="[bold]Verdict[/bold]",
        box=box.DOUBLE,
        border_style=color,
    ))
    console.print()
