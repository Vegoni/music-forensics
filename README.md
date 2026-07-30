# music-forensics

CLI tool to analyze audio for signs of AI generation.

## Install

Prerequisites: Python 3.10+, ffmpeg

```bash
pip install -e .
```

## Usage

```bash
music-forensics "https://youtube.com/watch?v=..."
music-forensics "https://youtube.com/watch?v=..." --deep
music-forensics /path/to/local/song.mp3
```

## Optional deep analysis

```bash
pip install -e ".[deep]"
music-forensics <url> --deep
```
