def test_pointnext_official_imports_checkpoint_metadata():
    import pointnext_official
    from pointnext_official.checkpoints import KNOWN_CHECKPOINTS

    assert pointnext_official.__version__ == "0.1.0"
    assert "modelnet40-pointnext-s-c64" in KNOWN_CHECKPOINTS


def test_pointnext_download_cli_lists_known_checkpoints(capsys):
    from pointnext_official.cli import main

    exit_code = main(["--list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "modelnet40-pointnext-s-c64" in captured.out
