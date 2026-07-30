from click.testing import CliRunner
from music_forensics.cli import main
from music_forensics.models import Finding


_EMPTY_FINDINGS = [Finding(label="Clean", score=0.1, evidence=[])]


def _mock_analyzers(mocker):
    mocker.patch("music_forensics.analyzers.metadata.analyze", return_value=_EMPTY_FINDINGS)
    mocker.patch("music_forensics.analyzers.spectral.analyze", return_value=_EMPTY_FINDINGS)
    mocker.patch("music_forensics.analyzers.waveform.analyze", return_value=_EMPTY_FINDINGS)


def test_local_file_runs_analysis(mocker, sine_wav):
    _mock_analyzers(mocker)
    mocker.patch("music_forensics.cli.render_report")
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

    runner = CliRunner()
    result = runner.invoke(main, [str(sine_wav)])

    assert result.exit_code == 0


def test_missing_local_file_exits_nonzero(mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    runner = CliRunner()
    result = runner.invoke(main, ["/nonexistent/file.wav"])
    assert result.exit_code != 0


def test_ffmpeg_missing_exits_nonzero(mocker):
    mocker.patch("shutil.which", return_value=None)
    runner = CliRunner()
    result = runner.invoke(main, ["https://youtube.com/watch?v=test"])
    assert result.exit_code != 0
    assert "ffmpeg" in result.output.lower()


def test_youtube_url_triggers_download(mocker, sine_wav):
    _mock_analyzers(mocker)
    mocker.patch("music_forensics.cli.render_report")
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    mock_download = mocker.patch(
        "music_forensics.cli.download_audio", return_value=sine_wav
    )

    runner = CliRunner()
    result = runner.invoke(main, ["https://youtube.com/watch?v=abc"])

    mock_download.assert_called_once_with("https://youtube.com/watch?v=abc")
    assert result.exit_code == 0


def test_deep_flag_invokes_ml_analyzer(mocker, sine_wav):
    _mock_analyzers(mocker)
    mocker.patch("music_forensics.cli.render_report")
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    mock_ml = mocker.patch(
        "music_forensics.analyzers.ml.analyze", return_value=_EMPTY_FINDINGS
    )

    runner = CliRunner()
    result = runner.invoke(main, [str(sine_wav), "--deep"])

    mock_ml.assert_called_once()
    assert result.exit_code == 0
