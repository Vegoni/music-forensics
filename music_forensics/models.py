from dataclasses import dataclass, field


@dataclass
class Finding:
    label: str
    score: float
    evidence: list[str] = field(default_factory=list)
    # False when the finding reports a failure rather than a measurement. Such a
    # finding is still shown in the report, but must not influence the verdict —
    # an analyzer that could not run is not evidence of anything.
    counts_toward_verdict: bool = True
