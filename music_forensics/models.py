from dataclasses import dataclass, field


@dataclass
class Finding:
    label: str
    score: float
    evidence: list[str] = field(default_factory=list)
