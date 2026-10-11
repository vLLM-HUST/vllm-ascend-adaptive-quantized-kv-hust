import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import pujiang_shared_source as m


def sources(tmp_path, dependency, target):
    records = {}
    for name, raw in (("inputs", dependency), ("target", target)):
        file = tmp_path / (name + ".py")
        file.write_bytes(raw)
        records[name] = (file.name, hashlib.sha256(raw).hexdigest())
    return records


def exact_git(tmp_path, monkeypatch):
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=str(tmp_path)
            if cmd[-1] == "--show-toplevel"
            else m.BENCHMARK_REVISION,
        ),
    )


@pytest.mark.parametrize("present", [False, True])
def test_loader_restores_dependency_namespace_after_verified_target_failure(
    tmp_path, monkeypatch, present
):
    names = ("synthetic_pujiang", "synthetic_pujiang.inputs")
    old = {name: ModuleType(name) for name in names}
    for name in names:
        if present:
            monkeypatch.setitem(sys.modules, name, old[name])
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)
    records = sources(
        tmp_path, b"VALUE=1\n", b"raise ValueError('synthetic target failure')\n"
    )
    exact_git(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="synthetic target failure"):
        m.load_verified_module(tmp_path, records, "target", ("inputs", names[1]))
    for name in names:
        assert (
            (sys.modules[name] is old[name]) if present else (name not in sys.modules)
        )


@pytest.mark.parametrize("mismatch", ["root", "HEAD"])
def test_git_identity_checked_before_source_read(tmp_path, monkeypatch, mismatch):
    def git(cmd, **kwargs):
        value = str(tmp_path) if cmd[-1] == "--show-toplevel" else m.BENCHMARK_REVISION
        if (mismatch == "root" and cmd[-1] == "--show-toplevel") or (
            mismatch == "HEAD" and cmd[-1] == "HEAD"
        ):
            value = "not-the-pinned-value"
        return SimpleNamespace(returncode=0, stdout=value)

    monkeypatch.setattr(m.subprocess, "run", git)
    with pytest.raises(ValueError, match="exact pinned repository"):
        m.load_verified_module(
            tmp_path, {"target": ("nonexistent.py", "0" * 64)}, "target"
        )


def test_verified_target_keeps_verified_dependency_after_namespace_restore(
    tmp_path, monkeypatch
):
    name = "synthetic_pujiang_inputs"
    monkeypatch.delitem(sys.modules, name, raising=False)
    records = sources(
        tmp_path,
        b"VALUE=1\n",
        b"from synthetic_pujiang_inputs import VALUE\nRESULT=VALUE+1\n",
    )
    exact_git(tmp_path, monkeypatch)
    target = m.load_verified_module(tmp_path, records, "target", ("inputs", name))
    assert target.RESULT == 2
    assert name not in sys.modules
