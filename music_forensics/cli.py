import shutil
import sys
from pathlib import Path

import click

from music_forensics.analyzers import metadata, spectral, waveform
from music_forensics.downloader import download_audio, is_url
from music_forensics.reporter import render_report


@click.command()
@click.argument("source")
@click.option("--deep", is_flag=True, help="Enable ML-based AI detection (downloads model on first use)")
def main(source: str, deep: bool) -> None:
    """Analyze audio for signs of AI generation.

    SOURCE is a YouTube URL or a local file path.
    """
    if not shutil.which("ffmpeg"):
        click.echo("Error: ffmpeg is required but not installed.", err=True)
        click.echo("  macOS:  brew install ffmpeg", err=True)
        click.echo("  Linux:  sudo apt install ffmpeg", err=True)
        sys.exit(1)

    audio_path: Path | None = None
    is_temp = False

    try:
        if is_url(source):
            click.echo("Downloading audio...")
            audio_path = download_audio(source)
            is_temp = True
        else:
            audio_path = Path(source)
            if not audio_path.exists():
                click.echo(f"Error: File not found: {source}", err=True)
                sys.exit(1)

        findings: dict = {}

        click.echo("Analyzing metadata...")
        findings["Metadata"] = metadata.analyze(str(audio_path))

        click.echo("Analyzing spectral characteristics...")
        findings["Spectral"] = spectral.analyze(str(audio_path))

        click.echo("Analyzing waveform...")
        findings["Waveform"] = waveform.analyze(str(audio_path))

        if deep:
            click.echo("Running ML classifier (may take a moment on first use)...")
            from music_forensics.analyzers import ml
            findings["ML Classifier"] = ml.analyze(str(audio_path))

        render_report(findings, source)

    finally:
        if is_temp and audio_path is not None:
            parent = audio_path.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)
