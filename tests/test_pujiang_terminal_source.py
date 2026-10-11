import argparse
import copy
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_pujiang_terminal_source as m


@pytest.fixture
def task(tmp_path):
    pytest.importorskip("tomllib")
    identity = "source-fixture"
    raw = b"""schema_version = "1.1"
[task]
name = "terminal-bench/source-fixture"
[environment]
docker_image = "alexgshaw/source-fixture:fixed"
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
build_timeout_sec = 600.0
allow_internet = true
mcp_servers = []
env = {}
[agent]
timeout_sec = 900.0
[verifier]
timeout_sec = 600.0
env = {}
"""
    import tomllib

    policy = tomllib.loads(raw.decode())
    local, files, entries = {}, {}, {}
    for name in m.REQUIRED:
        data = raw if name == "task.toml" else b"synthetic input; never executed\n"
        local[name] = {
            "bytes": len(data),
            "sha256": m.sha(data),
            "git_blob_sha": m.git_object("blob", data),
        }
        full = "tasks/" + identity + "/" + name
        files[full] = local[name]
        entries[full] = {"mode": "100755" if name.endswith(".sh") else "100644"}
        path = tmp_path / full
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    image = {
        "tag_digest": "sha256:" + "a" * 64,
        "manifest_digest": "sha256:" + "b" * 64,
        "config_digest": "sha256:" + "c" * 64,
        "platform": "linux/amd64",
        "layers": {"sha256:" + "d" * 64: 42},
        "compressed_layer_bytes": 42,
    }
    row = {
        "task_id": identity,
        "task_name": policy["task"]["name"],
        "source_revision": "e" * 40,
        "files": local,
        "image": policy["environment"]["docker_image"],
        "image_audit": image,
        "resources": {
            key: policy["environment"][key]
            for key in (
                "cpus",
                "memory_mb",
                "storage_mb",
                "gpus",
                "build_timeout_sec",
                "allow_internet",
            )
        },
        "agent_timeout_sec": 900.0,
        "verifier_timeout_sec": 600.0,
        "mcp_server_count": 0,
        "rights_review_complete": False,
        "image_layers_pulled": False,
        "task_executed": False,
    }
    receipt = {
        "task_id": identity,
        "task_toml_sha256": m.sha(raw),
        "environment": policy["environment"],
        "agent": policy["agent"],
        "verifier": policy["verifier"],
        "os": "linux",
        "architecture": "amd64",
        "image": row["image"],
        **{key: value for key, value in image.items() if key != "platform"},
    }
    calls = []

    def pin(text, original, pinned):
        calls.append((original, pinned))
        return text.replace(original, pinned)

    plan = SimpleNamespace(
        source_audit=SimpleNamespace(EXPECTED_TASKS=1, REVISION="e" * 40),
        pin_task_image=pin,
    )
    return tmp_path, entries, files, [row], [receipt], plan, calls


def test_source_policy_binding_is_offline_and_does_not_stage(task):
    result = m.audit_tasks(*task[:6])
    assert result["task_files_verified"] == 5
    assert result["source_git_mode_counts"] == {"100644": 3, "100755": 2}
    assert result["published_unique_compressed_layer_bytes"] == 42
    assert result["declared_agent_timeout_sum_sec"] == 900
    assert result["declared_verifier_timeout_sum_sec"] == 600
    assert result["task_bindings"][0]["pinned_image"].endswith("@sha256:" + "b" * 64)
    assert len(task[-1]) == 1
    assert b"@sha256" not in (task[0] / "tasks/source-fixture/task.toml").read_bytes()
    assert not any(
        key in json.dumps(result)
        for key in ("instruction", "synthetic input", "docker_image")
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing-required",
        "file-digest",
        "task-toml",
        "source-revision",
        "task-name",
        "duplicate-row",
        "duplicate-receipt",
        "receipt-toml",
        "receipt-policy",
        "resources",
        "agent-timeout",
        "verifier-timeout",
        "mcp-count",
        "rights",
        "executed",
        "image-layers",
        "architecture",
        "image-platform",
        "image-name",
        "image-digest",
        "image-total",
        "invalid-layer",
        "bool-layer",
        "bool-total",
        "understated-total",
        "unsafe-id",
    ],
)
def test_tampered_source_policy_or_mapping_fails_closed(task, change):
    snapshot, entries, files, rows, receipts, plan = task[:6]
    row, receipt = rows[0], receipts[0]
    if change == "missing-required":
        del files["tasks/source-fixture/tests/test.sh"]
        del row["files"]["tests/test.sh"]
    elif change == "file-digest":
        row["files"] = copy.deepcopy(row["files"])
        row["files"]["task.toml"]["sha256"] = "f" * 64
    elif change == "task-toml":
        (snapshot / "tasks/source-fixture/task.toml").write_bytes(b"changed")
    elif change in ("source-revision", "task-name"):
        row[change.replace("-", "_")] = "changed"
    elif change == "duplicate-row":
        rows.append(copy.deepcopy(row))
    elif change == "duplicate-receipt":
        receipts.append(copy.deepcopy(receipt))
    elif change == "receipt-toml":
        receipt["task_toml_sha256"] = "f" * 64
    elif change == "receipt-policy":
        receipt["environment"]["cpus"] = 4
    elif change == "resources":
        row["resources"]["cpus"] = 4
    elif change == "agent-timeout":
        row["agent_timeout_sec"] = 1200
    elif change == "verifier-timeout":
        row["verifier_timeout_sec"] = 1
    elif change == "mcp-count":
        row["mcp_server_count"] = 1
    elif change in ("rights", "executed", "image-layers"):
        row[
            {
                "rights": "rights_review_complete",
                "executed": "task_executed",
                "image-layers": "image_layers_pulled",
            }[change]
        ] = 0
    elif change == "architecture":
        receipt["architecture"] = "arm64"
    elif change == "image-platform":
        row["image_audit"]["platform"] = "linux/arm64"
    elif change == "image-name":
        row["image"] = "different:tag"
    elif change == "image-digest":
        row["image_audit"]["manifest_digest"] = "mutable"
    elif change == "image-total":
        row["image_audit"]["compressed_layer_bytes"] = 43
    elif change in ("invalid-layer", "bool-layer"):
        # Both published records agree: validation must still reject bad types.
        row["image_audit"]["layers"]["sha256:" + "d" * 64] = (
            True if change == "bool-layer" else -1
        )
    elif change in ("bool-total", "understated-total"):
        value = True if change == "bool-total" else 41
        row["image_audit"]["compressed_layer_bytes"] = value
        receipt["compressed_layer_bytes"] = value
    else:
        new = "../escape"
        for obj in (row, receipt):
            obj["task_id"] = new
        files = {
            name.replace("source-fixture", new): data for name, data in files.items()
        }
    with pytest.raises(ValueError):
        m.audit_tasks(snapshot, entries, files, rows, receipts, plan)


def test_descriptor_occurrence_total_is_not_confused_with_unique_layers(task):
    task[3][0]["image_audit"]["compressed_layer_bytes"] = 74
    task[4][0]["compressed_layer_bytes"] = 74
    result = m.audit_tasks(*task[:6])
    assert result["published_unique_compressed_layer_bytes"] == 42
    assert result["published_layer_total_excesses"] == [
        {
            "task_id": "source-fixture",
            "unique_layer_bytes": 42,
            "published_descriptor_bytes": 74,
            "difference_bytes": 32,
        }
    ]
    assert not result["layer_descriptor_multiplicity_reverified"]


def test_preflight_rejects_symlink(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.preflight_snapshot(tmp_path)


def test_preflight_caps_source_bytes(tmp_path, monkeypatch):
    (tmp_path / "raw").write_bytes(b"12345")
    monkeypatch.setattr(m, "SOURCE_BYTE_CAP", 4)
    with pytest.raises(ValueError, match="byte or file cap"):
        m.preflight_snapshot(tmp_path)


def test_preflight_rejects_non_directory(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"input")
    with pytest.raises(ValueError, match="root"):
        m.preflight_snapshot(path)


def test_cli_refuses_to_overwrite_prior_receipt(tmp_path, monkeypatch):
    target = tmp_path / "prior.json"
    target.write_bytes(b"keep original receipt")
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            benchmark_checkout=tmp_path,
            published_root=tmp_path,
            snapshot=tmp_path,
            output=target,
        ),
    )
    monkeypatch.setattr(m, "audit", lambda *args: {"fixture": True})
    with pytest.raises(FileExistsError):
        m.main()
    assert target.read_bytes() == b"keep original receipt"


def test_unpack_caps_metadata(monkeypatch):
    monkeypatch.setattr(m, "METADATA_BYTE_CAP", 4)
    with pytest.raises(ValueError, match="byte cap"):
        m.unpack(gzip.compress(b"12345"))


@pytest.mark.parametrize(
    "mode,body",
    [("100644", b"original"), ("120000", b"original"), ("100644", b"tamper")],
)
def test_report_uses_git_blobs_not_ambient_contents(tmp_path, monkeypatch, mode, body):
    path = m.REPORT + "/summary.json"
    local = tmp_path / path
    local.parent.mkdir(parents=True)
    local.write_bytes(body)
    commands = []

    def output(argv, **kwargs):
        commands.append(argv)
        return f"{mode} blob {m.git_object('blob', b'original')}\t{path}\n".encode()

    monkeypatch.setattr(m.subprocess, "check_output", output)
    if mode == "100644" and body == b"original":
        assert m.verify_report(tmp_path, tmp_path) == {"summary.json": body}
        assert all("--long" not in cmd for cmd in commands)
    else:
        with pytest.raises(ValueError):
            m.verify_report(tmp_path, tmp_path)


def test_real_terminal_source_opt_in():
    root = os.environ.get("AQKV_TEST_TERMINAL_SOURCE_ROOT")
    checkout = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not root or not checkout:
        pytest.skip("requires explicitly supplied pinned Terminal source and benchmark")
    pytest.importorskip("tomllib")
    root = Path(root)
    report = m.audit(
        Path(checkout), root / "published", root / "terminal-assets/snapshot"
    )
    assert report["task_count"] == 89
    assert report["source_files_verified"] == 1101
    assert report["source_bytes_verified"] == 59252655
    assert report["task_source_bytes_verified"] and report["source_membership_verified"]
    assert (
        report["source_git_modes_verified"]
        and not report["local_staged_modes_verified"]
    )
    assert not report["registry_metadata_bytes_reverified"]
    assert (
        not report["ready_for_scored_execution"] and not report["execution_authorized"]
    )
    assert report["task_quality"] is None
    assert report["tasks_executed"] == report["model_requests_sent"] == 0
    assert all(value is False for value in report["claim_boundary"].values())
    assert len(report["task_bindings"]) == 89
    assert "task.toml" not in json.dumps(report["task_bindings"])
