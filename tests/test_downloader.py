from pathlib import Path
from unittest.mock import MagicMock, patch
from music_forensics.downloader import download_audio, is_url


def test_is_url_with_https():
    assert is_url("https://youtube.com/watch?v=abc") is True


def test_is_url_with_http():
    assert is_url("http://youtube.com/watch?v=abc") is True


def test_is_url_with_local_path():
    assert is_url("/home/user/song.mp3") is False


def test_is_url_with_relative_path():
    assert is_url("song.mp3") is False


def test_download_audio_returns_flac_path(tmp_path):
    expected_flac = tmp_path / "audio.flac"
    expected_flac.touch()

    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            result = download_audio("https://youtube.com/watch?v=test")

    assert result == expected_flac


def test_download_audio_raises_on_ydl_error(tmp_path):
    import pytest

    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.download.side_effect = Exception("Private video")
        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(Exception, match="Private video"):
                download_audio("https://youtube.com/watch?v=private")
