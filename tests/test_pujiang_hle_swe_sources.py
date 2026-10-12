import argparse
import copy
import gzip
import json
import os
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from scripts import audit_pujiang_hle_swe_sources as m


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def tree_fixture(files):
    entries = {
        p: {
            "path": p,
            "mode": "100644",
            "type": "blob",
            "sha": m.git_object("blob", data),
            "size": len(data),
        }
        for p, data in files.items()
    }
    for path in files:
        for parent in PurePosixPath(path).parents:
            if str(parent) != ".":
                entries.setdefault(
                    str(parent), {"path": str(parent), "mode": "040000", "type": "tree"}
                )
    folders = [p for p, e in entries.items() if e["type"] == "tree"] + [""]
    root = None
    for folder in sorted(folders, key=lambda p: p.count("/") + bool(p), reverse=True):
        children = [
            e
            for p, e in entries.items()
            if str(PurePosixPath(p).parent) == (folder or ".")
        ]
        children.sort(
            key=lambda e: (
                PurePosixPath(e["path"]).name + ("/" if e["type"] == "tree" else "")
            ).encode()
        )
        raw = b"".join(
            ("40000" if e["type"] == "tree" else e["mode"]).encode()
            + b" "
            + PurePosixPath(e["path"]).name.encode()
            + b"\0"
            + bytes.fromhex(e["sha"])
            for e in children
        )
        digest = m.git_object("tree", raw)
        if folder:
            entries[folder]["sha"] = digest
        else:
            root = digest
    return {"sha": root, "truncated": False, "tree": list(entries.values())}


def test_git_tree_reconstructs_nested_content_and_git_sort_order():
    value = tree_fixture({"a.c": b"one", "a/z": b"two", "a/nested/t": b"three"})
    actual = m.verify_git_tree(value, value["sha"])
    assert actual["a/nested/t"]["sha"] == m.git_object("blob", b"three")


def test_commit_alias_does_not_replace_tree_content_verification():
    value = tree_fixture({"file": b"fixture"})
    root = value["sha"]
    value["sha"] = "a" * 40
    assert m.verify_git_tree(value, root, query_commit="a" * 40)
    value["tree"][0]["sha"] = "b" * 40
    with pytest.raises(ValueError, match="content identity"):
        m.verify_git_tree(value, root, query_commit="a" * 40)


@pytest.mark.parametrize(
    "change", ["truncated", "zero", "duplicate", "sha", "mode", "parent", "root"]
)
def test_git_tree_rejects_incomplete_or_forged_content(change):
    value = tree_fixture({"dir/file": b"synthetic"})
    root = value["sha"]
    if change == "truncated":
        value["truncated"] = True
    elif change == "zero":
        value["truncated"] = 0
    elif change == "duplicate":
        value["tree"].append(copy.deepcopy(value["tree"][0]))
    elif change == "sha":
        value["tree"][0]["sha"] = "f" * 40
    elif change == "mode":
        value["tree"][0]["mode"] = "040000"
    elif change == "parent":
        value["tree"] = [e for e in value["tree"] if e["type"] != "tree"]
    else:
        value["sha"] = "a" * 40
    with pytest.raises(ValueError):
        m.verify_git_tree(value, root)


@pytest.mark.parametrize(
    "path", ["../file", "/file", "a//b", "a/./b", "a/../b", "bad\0file"]
)
def test_git_tree_rejects_unsafe_paths(path):
    value = tree_fixture({"file": b"fixture"})
    value["tree"][0]["path"] = path
    with pytest.raises(ValueError, match="unsafe tree path"):
        m.verify_git_tree(value, value["sha"])


def test_inventory_preserves_part_order_and_rejects_changed_digest():
    raw = b'{"id":"first"}\n{"id":"second"}\n'
    parts = [gzip.compress(line + b"\n") for line in raw.splitlines()]
    assert [r["id"] for r in m.inventory(parts, m.sha(raw))] == ["first", "second"]
    with pytest.raises(ValueError, match="identity differs"):
        m.inventory(list(reversed(parts)), m.sha(raw))


@pytest.mark.parametrize("raw", [b'{"id":1,"id":2}\n', b'{"n":NaN}\n', b"[]\n"])
def test_inventory_rejects_ambiguous_json_even_with_matching_digest(raw):
    with pytest.raises(ValueError):
        m.inventory([gzip.compress(raw)], m.sha(raw))


def test_inventory_decompression_cap_is_enforced(monkeypatch):
    raw = b'{"id":"too-long"}\n'
    monkeypatch.setattr(m, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="byte cap"):
        m.inventory([gzip.compress(raw)], m.sha(raw))


def test_regular_input_rejects_symlink_and_oversize(tmp_path):
    file = tmp_path / "raw"
    file.write_bytes(b"12345")
    link = tmp_path / "link"
    link.symlink_to(file)
    with pytest.raises(ValueError, match="regular file"):
        m.read_bounded(link)
    with pytest.raises(ValueError, match="byte cap"):
        m.read_bounded(file, 4)


@pytest.mark.parametrize("change", [None, "head", "root", "bytes", "mode"])
def test_reports_bind_checkout_identity_and_blob_bytes(tmp_path, monkeypatch, change):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    path = m.REPORTS[0] + "/summary.json"
    data = b'{"fixture":true}\n'
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(data if change != "bytes" else b"changed")
    commands = []

    def output(argv, **kwargs):
        commands.append(argv)
        if argv[-1] == "--show-toplevel":
            return str(checkout if change != "root" else tmp_path).encode()
        if argv[-1] == "HEAD":
            return (m.BENCHMARK_REVISION if change != "head" else "a" * 40).encode()
        mode = "100644" if change != "mode" else "120000"
        return f"{mode} blob {m.git_object('blob', data)}\t{path}\n".encode()

    monkeypatch.setattr(m.subprocess, "check_output", output)
    if change is None:
        assert m.verify_reports(checkout, tmp_path) == {path: data}
        assert all("--long" not in cmd for cmd in commands)
    else:
        with pytest.raises(ValueError):
            m.verify_reports(checkout, tmp_path)


def hle_fixture(tmp_path, monkeypatch):
    rows = [
        {
            "id": "synthetic-boolean",
            "source_subset": "Gold",
            "source_index": 0,
            "problem_image_present": True,
            "image_preview_present": True,
            "requires_image_handling": True,
            "rationale_image_present": False,
            "answer_type": "exactMatch",
            "answer_value_type": "bool",
            "verified_class": "Gold",
        },
        {
            "id": "synthetic-zero",
            "source_subset": "Gold",
            "source_index": 1,
            "problem_image_present": False,
            "image_preview_present": False,
            "requires_image_handling": False,
            "rationale_image_present": True,
            "answer_type": "exactMatch",
            "answer_value_type": "int",
            "verified_class": "Gold",
        },
    ]
    source_digest = "a" * 64
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{source_digest}\nsize 123\n"
    ).encode()
    readme = b"Synthetic source metadata, not a rights grant.\n"
    tree = tree_fixture({"README.md": readme, "data/Gold.jsonl": pointer})
    monkeypatch.setattr(m, "HLE_TREE", tree["sha"])
    (tmp_path / "data").mkdir()
    (tmp_path / "README.md").write_bytes(readme)
    (tmp_path / "data/Gold.jsonl").write_bytes(pointer)
    summary = {
        "revision": "b" * 40,
        "solver_calls": 0,
        "judge_calls": 0,
        "scores": None,
        "execution_subset": None,
        "source_readme_sha256": m.sha(readme),
        "subsets": {
            "Gold": {
                "tasks": 2,
                "image_bearing": 1,
                "image_field": 1,
                "preview_only": 0,
                "answer_types": {"exactMatch": 2},
                "answer_value_types": {"bool": 1, "int": 1},
                "classes": {"Gold": 2},
            }
        },
    }
    return (
        rows,
        summary,
        tree,
        SimpleNamespace(
            REVISION=summary["revision"], SOURCES={"Gold": (2, source_digest)}
        ),
    )


def hle_files(rows, summary, tree):
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    summary["subsets"]["Gold"]["inventory_sha256"] = m.sha(raw)
    base = m.REPORTS[0]
    return {
        base + "/summary.json": canonical(summary),
        base + "/source-tree.json": canonical(tree),
        base + "/Gold-inventory.jsonl.gz": gzip.compress(raw),
    }


def test_hle_preserves_nonstring_answer_types_and_excludes_rationale_only_images(
    tmp_path, monkeypatch
):
    rows, summary, tree, hle = hle_fixture(tmp_path, monkeypatch)
    result = m.audit_hle(hle_files(rows, summary, tree), tmp_path, hle)
    assert result["subsets"]["Gold"]["answer_value_types"] == {"bool": 1, "int": 1}
    assert result["subsets"]["Gold"]["image_bearing"] == 1
    assert result["declared_source_bytes"] == 123
    assert result["raw_source_rows_reverified"] is False
    assert result["execution_scope_frozen"] is False
    assert result["rights_clearance"] is False
    assert result["solver_or_judge_executed"] is False and result["scores"] is None


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "position",
        "modality_type",
        "rationale_input",
        "question",
        "answer",
        "summary",
        "readme",
        "pointer",
        "solver",
        "scope",
        "answer_type",
    ],
)
def test_hle_rejects_ambiguous_or_changed_metadata(tmp_path, monkeypatch, change):
    rows, summary, tree, hle = hle_fixture(tmp_path, monkeypatch)
    if change == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif change == "position":
        rows[1]["source_index"] = True
    elif change == "modality_type":
        rows[0]["problem_image_present"] = 1
    elif change == "rationale_input":
        rows[1]["requires_image_handling"] = True
    elif change in {"question", "answer"}:
        rows[0][change] = "must not publish source content"
    elif change == "summary":
        summary["subsets"]["Gold"]["image_bearing"] = 2
    elif change == "readme":
        (tmp_path / "README.md").write_bytes(b"changed")
    elif change == "pointer":
        (tmp_path / "data/Gold.jsonl").write_bytes(b"changed")
    elif change == "solver":
        summary["solver_calls"] = 1
    elif change == "scope":
        summary["execution_subset"] = "Gold"
    else:
        rows[0]["answer_value_type"] = "str"
    with pytest.raises(ValueError):
        m.audit_hle(hle_files(rows, summary, tree), tmp_path, hle)


def test_cli_never_overwrites_an_existing_receipt(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    target.write_text("prior receipt\n")
    args = ["audit"]
    for option in (
        "benchmark-checkout",
        "published-root",
        "hle-pointer-root",
        "swe-data",
        "harness-tree",
        "original-test-patch",
        "output",
    ):
        args.extend(["--" + option, str(target)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(m, "audit", lambda args: {"model_requests_sent": 0})
    with pytest.raises(FileExistsError):
        m.main()
    assert target.read_text() == "prior receipt\n"


def swe_fixture(crlf=False):
    row = {
        "instance_id": "fixture",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "hard": False,
        "version": "2.0.0",
        "problem_statement": "synthetic problem",
        "patch": "synthetic gold patch",
        "test_patch": "first\nsecond\n",
        "fail_to_pass": '["test_a"]',
        "pass_to_pass": "[]",
        "selected_test_files_to_run": '["test.py"]',
        "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:fixture",
    }
    original = (
        row["test_patch"].encode().replace(b"\n", b"\r\n")
        if crlf
        else row["test_patch"].encode()
    )
    contents = {
        name: b"synthetic metadata"
        for name in (
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "tests/config.json",
            "tests/test.sh",
            "tests/run_script.sh",
            "tests/parser.py",
        )
    }
    contents["solution/gold_patch.diff"] = row["patch"].encode()
    contents["tests/test_patch.patch"] = original
    files = {
        name: {"git_blob_sha": m.git_object("blob", raw), "bytes": len(raw)}
        for name, raw in contents.items()
    }
    tree = {
        f"v2/tasks/fixture/{name}": {
            "type": "blob",
            "sha": entry["git_blob_sha"],
            "size": entry["bytes"],
        }
        for name, entry in files.items()
    }
    record = {key: row[key] for key in ("instance_id", "repo", "base_commit", "hard")}
    record.update(
        source_row_sha256=m.sha(canonical(row)),
        release=row["version"],
        image_tag=row["docker_image"],
        harness_files=files,
        problem_sha256=m.sha(row["problem_statement"].encode()),
        patch_sha256=m.sha(row["patch"].encode()),
        test_patch_sha256=m.sha(row["test_patch"].encode()),
        harness_test_patch_sha256=m.sha(original),
        test_patch_byte_identical=not crlf,
        verifier_patch_authority="pinned harness bytes",
        runtime_repo_head_verified=False,
        tests_executed=False,
        test_counts={
            "fail_to_pass": 1,
            "pass_to_pass": 0,
            "selected_test_files_to_run": 1,
        },
        image_layers={"sha256:" + "a" * 64: 123},
    )

    def check_blob(raw, expected):
        m.require(m.git_object("blob", raw) == expected, "Git blob mismatch")

    return (
        row,
        record,
        tree,
        original,
        SimpleNamespace(
            canonical=canonical, check_blob=check_blob, test_list=json.loads
        ),
    )


@pytest.mark.parametrize("crlf", [False, True])
def test_swe_source_binding_preserves_verifier_bytes_without_execution(crlf):
    row, record, tree, original, swe = swe_fixture(crlf)
    result = m.audit_swe_rows([row], [record], tree, original, swe)
    assert result["source_rows_verified"] == 1
    assert result["test_patch_byte_differences"] == (["fixture"] if crlf else [])
    assert result["sandbox_tasks_executed"] == 0 and result["scores"] is None
    assert result["registry_manifest_bytes_reverified"] is False


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "repo",
        "test_counts",
        "authority",
        "patch_flag",
        "runtime",
        "blob",
        "size",
        "required",
        "field",
        "layer",
    ],
)
def test_swe_source_binding_rejects_changed_independent_mappings(change):
    row, record, tree, original, swe = swe_fixture()
    if change == "source":
        row["problem_statement"] = "changed"
    elif change == "repo":
        record["repo"] = "other/repo"
    elif change == "test_counts":
        record["test_counts"]["fail_to_pass"] = True
    elif change == "authority":
        record["verifier_patch_authority"] = "normalized HF text"
    elif change == "patch_flag":
        record["test_patch_byte_identical"] = 1
    elif change == "runtime":
        record["tests_executed"] = True
    elif change == "blob":
        tree["v2/tasks/fixture/tests/parser.py"]["sha"] = "b" * 40
    elif change == "size":
        tree["v2/tasks/fixture/tests/parser.py"]["size"] += 1
    elif change == "required":
        del record["harness_files"]["instruction.md"]
    elif change == "field":
        record["problem_sha256"] = "c" * 64
    else:
        record["image_layers"]["sha256:" + "a" * 64] = False
    with pytest.raises(ValueError):
        m.audit_swe_rows([row], [record], tree, original, swe)


@pytest.mark.parametrize("which", ["source", "inventory", "missing"])
def test_swe_source_binding_rejects_missing_or_duplicate_tasks(which):
    row, record, tree, original, swe = swe_fixture()
    rows = [row, copy.deepcopy(row)] if which == "source" else [row]
    records = [record, copy.deepcopy(record)] if which == "inventory" else [record]
    if which == "missing":
        records = []
    with pytest.raises(ValueError):
        m.audit_swe_rows(rows, records, tree, original, swe)


def test_swe_crlf_patch_cannot_be_replaced_by_lf_or_other_bytes():
    row, record, tree, original, swe = swe_fixture(True)
    for wrong in (original.replace(b"\r\n", b"\n"), b"different patch"):
        with pytest.raises(ValueError, match="Git blob mismatch"):
            m.audit_swe_rows([row], [record], tree, wrong, swe)


def test_fixed_sources_full_local_integration_when_supplied():
    root = os.environ.get("AQKV_TEST_HLE_SWE_AUDIT_ROOT")
    checkout = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not root or not checkout:
        pytest.skip("Set explicit pinned checkout and HLE/SWE source metadata inputs")
    root = Path(root)
    result = m.audit(
        argparse.Namespace(
            benchmark_checkout=Path(checkout),
            published_root=root / "published",
            hle_pointer_root=root / "hle-source",
            swe_data=root / "swe-source/default-test.parquet",
            harness_tree=root / "swe-source/harness-tree.json",
            original_test_patch=root / "swe-source/original-test-patch.patch",
        )
    )
    assert result["hle"]["published_inventory_rows_verified"] == 2500
    assert result["hle"]["raw_source_rows_reverified"] is False
    assert result["swe"]["source_rows_verified"] == 642
    assert all(v is False for v in result["claim_boundary"].values())
