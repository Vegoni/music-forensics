import mutagen
from music_forensics.models import Finding

_AI_KEYWORDS = frozenset([
    "suno", "udio", "stable audio", "musicgen", "audiocraft",
    "mubert", "boomy", "soundraw", "aiva", "beatoven", "loudly",
])

_ARTIST_KEYS = frozenset(["tpe1", "tpe2", "tcom", "artist", "author"])
_TITLE_KEYS = frozenset(["tit2", "tit1", "title"])


def analyze(audio_path: str) -> list[Finding]:
    try:
        tags = mutagen.File(audio_path)
    except Exception as e:
        return [Finding(
            label="Metadata Read Error",
            score=0.5,
            evidence=[f"Could not read metadata: {e}"],
        )]

    if tags is None:
        return [Finding(
            label="Missing Metadata",
            score=0.4,
            evidence=["No metadata found — AI-generated audio often lacks tags entirely"],
        )]

    findings = []
    tag_blob = " ".join(str(v) for v in tags.values()).lower()

    for keyword in _AI_KEYWORDS:
        if keyword in tag_blob:
            findings.append(Finding(
                label="AI Generator Detected",
                score=1.0,
                evidence=[f"Found '{keyword}' in metadata tags"],
            ))
            break

    has_artist = any(
        str(k).lower() in _ARTIST_KEYS or "artist" in str(k).lower()
        for k in tags.keys()
    )
    has_title = any(
        str(k).lower() in _TITLE_KEYS or "title" in str(k).lower()
        for k in tags.keys()
    )
    if not has_artist and not has_title:
        findings.append(Finding(
            label="Missing Basic Tags",
            score=0.35,
            evidence=["No artist or title tags — common in AI-generated audio"],
        ))

    if not findings:
        findings.append(Finding(
            label="Metadata Clean",
            score=0.1,
            evidence=["No AI indicators found in metadata"],
        ))

    return findings
