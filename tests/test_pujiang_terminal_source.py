import argparse
import copy
import gzip
import json
import os
import shutil
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
            staged_tasks=None,
            candidate_inventory=None,
        ),
    )
    monkeypatch.setattr(m, "audit", lambda *args, **kwargs: {"fixture": True})
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


def stage_rows(snapshot, staged, entries, rows, plan):
    import tomllib

    inventory = []
    for row in rows:
        identity = row["task_id"]
        shutil.copytree(snapshot / "tasks" / identity, staged / identity)
        path = staged / identity / "task.toml"
        before = path.read_bytes()
        policy = tomllib.loads(before.decode())
        pinned = (
            row["image"].rsplit(":", 1)[0] + "@" + row["image_audit"]["manifest_digest"]
        )
        path.write_bytes(
            plan.pin_task_image(before.decode(), row["image"], pinned).encode()
        )
        files, modes = {}, {}
        for name in row["files"]:
            mode = entries["tasks/" + identity + "/" + name]["mode"]
            path = staged / identity / name
            path.chmod(int(mode, 8) & 0o777)
            raw = path.read_bytes()
            modes[name] = mode
            files[name] = {
                "bytes": len(raw),
                "sha256": m.sha(raw),
                "git_blob_sha": m.git_object("blob", raw),
            }
        inventory.append(
            {
                "task_id": identity,
                "pinned_image": pinned,
                "source_row_sha256": m.sha(m.encoded(row)),
                "staged_files_sha256": m.sha(m.encoded(files)),
                "file_modes_sha256": m.sha(m.encoded(modes)),
                "task_policy_sha256": m.sha(
                    m.encoded(
                        {
                            key: policy[key]
                            for key in ("environment", "agent", "verifier")
                        }
                    )
                ),
            }
        )
    return inventory


@pytest.fixture
def staged_task(task):
    snapshot, entries, files, rows, receipts, plan = task[:6]
    staged = snapshot / "candidate"
    inventory = stage_rows(snapshot, staged, entries, rows, plan)
    return snapshot, staged, entries, rows, inventory, plan


def test_staged_assets_bind_actual_bytes_permissions_and_pin(staged_task):
    report = m.audit_staged_tasks(*staged_task)
    assert report["candidate_task_count"] == 1
    assert report["staged_files_verified"] == 5
    assert report["task_assets_verified"] and report["local_staged_modes_verified"]
    assert report["staged_task_bindings"] == staged_task[4]
    assert (
        b"@sha256"
        not in (staged_task[0] / "tasks/source-fixture/task.toml").read_bytes()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "traversal",
        "duplicate",
        "empty",
        "non-object",
        "non-list",
        "missing-file",
        "extra-file",
        "extra-dir",
        "missing-task",
        "extra-task",
        "symlink-file",
        "symlink-dir",
        "symlink-root",
        "changed-instruction",
        "changed-test",
        "changed-solution",
        "unpinned",
        "changed-pin",
        "changed-policy",
        "chmod",
        "lost-executable",
        "setuid",
        "false-source-row",
        "false-files",
        "false-modes",
        "false-policy",
        "false-image",
        "size-cap",
    ],
)
def test_staged_asset_tampering_fails_even_if_candidate_digest_is_updated(
    staged_task, mutation, monkeypatch
):
    snapshot, staged, entries, rows, inventory, plan = staged_task
    task = staged / "source-fixture"
    if mutation in ("unknown", "traversal"):
        inventory[0]["task_id"] = (
            "not-in-source" if mutation == "unknown" else "../escape"
        )
    elif mutation == "duplicate":
        inventory.append(copy.deepcopy(inventory[0]))
    elif mutation == "empty":
        inventory.clear()
    elif mutation == "non-object":
        inventory[0] = []
    elif mutation == "non-list":
        inventory = {}
    elif mutation == "missing-file":
        (task / "instruction.md").unlink()
    elif mutation == "extra-file":
        (task / "added.txt").write_bytes(b"extra")
    elif mutation in ("extra-dir", "extra-task"):
        (task / "empty" if mutation == "extra-dir" else staged / "extra-task").mkdir()
    elif mutation == "missing-task":
        task.rename(staged.parent / "removed")
    elif mutation == "symlink-file":
        path = task / "instruction.md"
        path.unlink()
        path.symlink_to(snapshot / "tasks/source-fixture/instruction.md")
    elif mutation == "symlink-dir":
        (task / "extra").symlink_to(task / "tests", target_is_directory=True)
    elif mutation == "symlink-root":
        link = staged.parent / "link"
        link.symlink_to(staged, target_is_directory=True)
        staged = link
    elif mutation in ("changed-instruction", "changed-test", "changed-solution"):
        name = {
            "changed-instruction": "instruction.md",
            "changed-test": "tests/test.sh",
            "changed-solution": "solution/solve.sh",
        }[mutation]
        (task / name).write_bytes(b"replaced content")
    elif mutation in ("unpinned", "changed-pin", "changed-policy"):
        path = task / "task.toml"
        raw = path.read_bytes()
        path.write_bytes(
            (snapshot / "tasks/source-fixture/task.toml").read_bytes()
            if mutation == "unpinned"
            else raw.replace(b"cpus = 1", b"cpus = 4")
            if mutation == "changed-policy"
            else raw.replace(b"sha256:" + b"b" * 64, b"sha256:" + b"f" * 64)
        )
    elif mutation in ("chmod", "lost-executable", "setuid"):
        path = task / (
            "tests/test.sh" if mutation == "lost-executable" else "instruction.md"
        )
        path.chmod(
            {"chmod": 0o666, "lost-executable": 0o644, "setuid": 0o4644}[mutation]
        )
    elif mutation.startswith("false-"):
        key = {
            "false-source-row": "source_row_sha256",
            "false-files": "staged_files_sha256",
            "false-modes": "file_modes_sha256",
            "false-policy": "task_policy_sha256",
            "false-image": "pinned_image",
        }[mutation]
        inventory[0][key] = "f" * 64
    else:
        monkeypatch.setattr(m, "SOURCE_BYTE_CAP", 4)
    # Rehash tampered files to ensure self-consistent declarations cannot bypass
    # comparison with the independently fixed original bytes and permissions.
    if mutation.startswith("changed-") or mutation in (
        "unpinned",
        "chmod",
        "lost-executable",
        "setuid",
    ):
        files, modes = {}, {}
        for name in rows[0]["files"]:
            path = task / name
            raw = path.read_bytes()
            files[name] = {
                "bytes": len(raw),
                "sha256": m.sha(raw),
                "git_blob_sha": m.git_object("blob", raw),
            }
            modes[name] = "100755" if path.stat().st_mode & 0o111 else "100644"
        inventory[0]["staged_files_sha256"] = m.sha(m.encoded(files))
        inventory[0]["file_modes_sha256"] = m.sha(m.encoded(modes))
    with pytest.raises(ValueError):
        m.audit_staged_tasks(snapshot, staged, entries, rows, inventory, plan)


@pytest.mark.parametrize("staged,inventory", [(Path("unused"), None), (None, [])])
def test_incomplete_asset_binding_rejected_before_source_reads(
    tmp_path, staged, inventory
):
    with pytest.raises(ValueError, match="must be supplied together"):
        m.audit(
            tmp_path,
            tmp_path,
            tmp_path,
            staged_tasks=staged,
            candidate_inventory=inventory,
        )


def test_real_all_terminal_task_copies_opt_in(tmp_path):
    from scripts import validate_pujiang_terminal_plan as candidate

    root = os.environ.get("AQKV_TEST_TERMINAL_SOURCE_ROOT")
    checkout = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not root or not checkout:
        pytest.skip("requires explicitly supplied pinned Terminal source and benchmark")
    pytest.importorskip("tomllib")
    root, checkout = Path(root), Path(checkout)
    snapshot, published = root / "terminal-assets/snapshot", root / "published"
    plan = m.load_plan(checkout)
    reports = m.verify_report(checkout, published)
    entries = m.verify_git_tree(
        m.parse_json(m.unpack(reports["tree.json.gz"])),
        m.TREE_ROOT,
        query_commit=plan.source_audit.REVISION,
    )
    rows = []
    for name in sorted(name for name in reports if name.startswith("inventory-")):
        rows.extend(m.parse_json(m.unpack(reports[name])))
    staged = tmp_path / "tasks"
    inventory = stage_rows(snapshot, staged, entries, rows, plan)
    report = m.audit(
        checkout,
        published,
        snapshot,
        staged_tasks=staged,
        candidate_inventory=inventory,
    )
    assert report["status"] == "terminal_staged_assets_bound_execution_blocked"
    assert (
        report["candidate_task_count"] == 89 and report["staged_files_verified"] == 1035
    )
    assert report["task_assets_verified"] and report["local_staged_modes_verified"]
    assert (
        not report["execution_authorized"] and not report["ready_for_scored_execution"]
    )
    assert (
        not report["harbor_schema_validated_locally"]
        and not report["image_layers_pulled"]
    )
    assert report["tasks_executed"] == report["model_requests_sent"] == 0
    assert report["task_quality"] is None
    assert all(value is False for value in report["claim_boundary"].values())
    endpoints = {
        "B0": "https://b0.example.invalid/v1",
        "B1": "https://b1.example.invalid/v1",
    }
    paths = {"inventory": tmp_path / "inventory.json"}
    paths["inventory"].write_bytes(m.encoded(inventory))
    for arm, endpoint in endpoints.items():
        job = plan.job_config(
            arm.lower(), [row["task_id"] for row in inventory], endpoint
        )
        job["job_name"] = "aqkv-terminal-" + arm.lower()
        job["agents"][0]["model_name"] = "openai/synthetic-service-intent"
        paths[arm] = tmp_path / (arm + ".json")
        paths[arm].write_bytes(m.encoded(job))
    envelope = {
        "schema_version": candidate.SCHEMA,
        "evidence_kind": "development-fixture",
        "scope": "supplied-task-plan-only",
        "dataset": "terminal-bench-2.1",
        "dataset_revision": candidate.DATASET_REVISION,
        "benchmark_revision": candidate.BENCHMARK_REVISION,
        "source_sha256": {
            key: value[1] for key, value in candidate.SOURCE_FILES.items()
        },
        "harbor_version": plan.HARBOR_VERSION,
        "harbor_revision": plan.HARBOR_REVISION,
        "retry_policy": candidate.RETRY_POLICY.copy(),
        "task_count": len(inventory),
        "serving_intent": {
            "model_name": "openai/synthetic-service-intent",
            "api_base": endpoints,
            "aqkv_enabled": {"B0": False, "B1": True},
        },
        "runtime_receipts": {"B0": None, "B1": None},
        "files_sha256": {key: m.sha(path.read_bytes()) for key, path in paths.items()},
    }
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_bytes(m.encoded(envelope))
    pair = candidate.validate_files(
        envelope_path,
        paths["inventory"],
        paths["B0"],
        paths["B1"],
        checkout,
        published_root=published,
        snapshot=snapshot,
        staged_tasks=staged,
    )
    assert pair["status"] == "candidate_plan_assets_bound_execution_blocked"
    assert pair["source_asset_binding"]["staged_files_verified"] == 1035
    assert pair["source_membership_verified"] and pair["task_assets_verified"]
    assert not pair["actual_backend_connection_verified"]
    assert not pair["actual_service_switches_verified"]
    assert (
        not pair["harbor_schema_validated_locally"]
        and not pair["ready_for_scored_execution"]
    )
    assert (
        pair["task_quality"] is None
        and pair["model_calls"] == pair["tasks_executed"] == 0
    )
