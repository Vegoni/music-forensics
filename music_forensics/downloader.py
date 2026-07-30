import os
import tempfile
from pathlib import Path

import yt_dlp


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def download_audio(url: str) -> Path:
    tmpdir = tempfile.mkdtemp(prefix="music_forensics_")
    output_template = os.path.join(tmpdir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "flac",
            "preferredquality": "0",
        }],
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, 90)]),
        "force_keyframes_at_cuts": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return Path(tmpdir) / "audio.flac"
