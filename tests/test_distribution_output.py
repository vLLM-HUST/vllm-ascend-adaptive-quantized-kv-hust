import json
import os
import stat
import sys

import pytest

from scripts import validate_distribution as distribution


@pytest.fixture
def invoke(monkeypatch, tmp_path):
    receipt = {"status": "PASS", "runtime_activation_present": False}
    monkeypatch.setattr(distribution, "validate_distribution", lambda _: receipt)

    def run(output=None):
        argv = ["validate_distribution", "--dist", str(tmp_path / "dist")]
        if output is not None:
            argv += ["--output", str(output)]
        monkeypatch.setattr(sys, "argv", argv)
        return distribution.main()

    return run, receipt


def test_report_created_private_under_permissive_umask(invoke, tmp_path):
    run, receipt = invoke
    output = tmp_path / "receipt.json"
    old_umask = os.umask(0)
    try:
        assert run(output) == 0
    finally:
        os.umask(old_umask)
    assert json.loads(output.read_text()) == receipt
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize("linked", [False, True])
def test_existing_report_or_link_target_not_overwritten(invoke, tmp_path, linked):
    run, _ = invoke
    preserved = tmp_path / "preserved.json"
    preserved.write_bytes(b"historical receipt\n")
    preserved.chmod(0o640)
    output = tmp_path / "receipt.json" if linked else preserved
    if linked:
        output.symlink_to(preserved)
    with pytest.raises(SystemExit) as error:
        run(output)
    assert error.value.code == 2
    assert preserved.read_bytes() == b"historical receipt\n"
    assert stat.S_IMODE(preserved.stat().st_mode) == 0o640
    if linked:
        assert output.is_symlink()


def test_dangling_link_does_not_create_target(invoke, tmp_path):
    run, _ = invoke
    target = tmp_path / "missing.json"
    output = tmp_path / "receipt.json"
    output.symlink_to(target)
    with pytest.raises(SystemExit) as error:
        run(output)
    assert error.value.code == 2
    assert not target.exists()
    assert output.is_symlink()


def test_stdout_only_still_returns_receipt(invoke, capsys, tmp_path):
    run, receipt = invoke
    assert run() == 0
    assert json.loads(capsys.readouterr().out) == receipt
    assert not list(tmp_path.iterdir())
