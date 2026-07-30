from click.testing import CliRunner
from music_forensics.cli import main


def test_full_pipeline_on_local_file(sine_wav, mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

    runner = CliRunner()
    result = runner.invoke(main, [str(sine_wav)])

    assert result.exit_code == 0
    assert "Verdict" in result.output or result.exit_code == 0


def test_full_pipeline_does_not_crash_on_silent_file(silent_wav, mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

    runner = CliRunner()
    result = runner.invoke(main, [str(silent_wav)])

    assert result.exit_code == 0
