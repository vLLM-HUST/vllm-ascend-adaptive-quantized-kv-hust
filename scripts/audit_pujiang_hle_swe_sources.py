#!/usr/bin/env python3
"""Bind pinned public HLE/SWE inventories to sources; never run or grade tasks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts.pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from scripts.validate_pujiang_readiness import parse_json
except ModuleNotFoundError:
    from pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from validate_pujiang_readiness import parse_json

REPORTS = (
    "reports/pyramidkv-hle-source-audit-20261009",
    "reports/pyramidkv-swe-source-audit-20261009",
)
HLE_TREE = "471981d04bf02180e3a4c53da8da2f12fa989c4d"
SWE_TREE = "beb15336e37995a2dc2e5908e2bde931d59cef1e"
MAX_BYTES = 32 * 1024 * 1024
SOURCES = {
    "hle": (
        "scripts/audit_pyramidkv_hle.py",
        "2190113809026f35e847de9ec460b77e294a51d5f1a42a8c79d78bc3822d99e2",
    ),
    "swe": (
        "scripts/audit_pyramidkv_swe.py",
        "64296056ab78423ee7e5a0d9fb243dc905ba4f904b923a65661771eedc18ebf2",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_object(kind: str, raw: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(raw)}\0".encode() + raw).hexdigest()


def read_bounded(path: Path, limit: int = MAX_BYTES) -> bytes:
    require(path.is_file() and not path.is_symlink(), "input must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "input exceeds byte cap")
    return raw


def verify_git_tree(
    value: Any, expected_root: str, *, query_commit: str | None = None
) -> dict[str, dict]:
    require(isinstance(value, dict), "tree must be an object")
    require(value.get("truncated") is False, "tree must be complete")
    entries = value.get("tree")
    require(isinstance(entries, list) and len(entries) <= 50000, "invalid tree entries")
    paths, children = {}, defaultdict(list)
    for entry in entries:
        require(isinstance(entry, dict), "invalid tree entry")
        path = entry.get("path")
        require(isinstance(path, str) and bool(path), "invalid tree path")
        parts = path.split("/")
        require(
            all(p not in ("", ".", "..") and "\0" not in p for p in parts),
            "unsafe tree path",
        )
        require(path not in paths, "duplicate tree path")
        mode, kind = entry.get("mode"), entry.get("type")
        require(
            (mode, kind)
            in {
                ("040000", "tree"),
                ("100644", "blob"),
                ("100755", "blob"),
                ("120000", "blob"),
                ("160000", "commit"),
            },
            "invalid Git mode/type",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", str(entry.get("sha"))) is not None,
            "invalid Git object identity",
        )
        paths[path] = entry
        parent = str(PurePosixPath(path).parent)
        children["" if parent == "." else parent].append(entry)
    for path in paths:
        parent = str(PurePosixPath(path).parent)
        if parent != ".":
            require(
                parent in paths and paths[parent]["type"] == "tree",
                "missing parent tree",
            )
    folders = [p for p, e in paths.items() if e["type"] == "tree"] + [""]
    for folder in sorted(folders, key=lambda p: p.count("/") + bool(p), reverse=True):
        nodes = children[folder]
        nodes.sort(
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
            for e in nodes
        )
        expected = paths[folder]["sha"] if folder else expected_root
        require(
            git_object("tree", raw) == expected, "Git tree content identity mismatch"
        )
    # A commit-addressed API receipt may retain the query commit in its sha field.
    # The independently pinned tree root is still reconstructed above.
    require(
        value.get("sha") == expected_root
        or (query_commit is not None and value.get("sha") == query_commit),
        "unexpected root tree identity",
    )
    return paths


def verify_reports(checkout: Path, root: Path) -> dict[str, bytes]:
    for arg, expected in [
        ("--show-toplevel", str(checkout.resolve())),
        ("HEAD", BENCHMARK_REVISION),
    ]:
        actual = (
            subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", arg], timeout=10
            )
            .decode()
            .strip()
        )
        require(actual == expected, "benchmark checkout identity differs")
    listing = (
        subprocess.check_output(
            ["git", "-C", str(checkout), "ls-tree", "-r", "HEAD", *REPORTS], timeout=10
        )
        .decode()
        .splitlines()
    )
    require(bool(listing), "missing pinned report tree")
    verified = {}
    total = 0
    for line in listing:
        header, path = line.split("\t", 1)
        mode, kind, digest = header.split()
        require(mode == "100644" and kind == "blob", "unexpected report file mode")
        raw = read_bounded(root / path)
        total += len(raw)
        require(total <= MAX_BYTES, "report bundle exceeds byte cap")
        require(git_object("blob", raw) == digest, "pinned report bytes differ")
        verified[path] = raw
    return verified


def inventory(parts: list[bytes], expected: str) -> list[dict]:
    chunks, size = [], 0
    for part in parts:
        with gzip.GzipFile(fileobj=io.BytesIO(part)) as stream:
            raw = stream.read(MAX_BYTES - size + 1)
        size += len(raw)
        require(size <= MAX_BYTES, "uncompressed inventory exceeds byte cap")
        chunks.append(raw)
    raw = b"".join(chunks)
    require(sha(raw) == expected, "uncompressed inventory identity differs")
    rows = [parse_json(line) for line in raw.splitlines()]
    require(all(isinstance(row, dict) for row in rows), "invalid inventory row")
    return rows


def audit_hle(files: dict[str, bytes], source: Path, hle: Any) -> dict:
    base = REPORTS[0]
    summary = parse_json(files[base + "/summary.json"])
    tree = verify_git_tree(
        parse_json(files[base + "/source-tree.json"]),
        HLE_TREE,
        query_commit=hle.REVISION,
    )
    require(
        summary["revision"] == hle.REVISION, "HLE source namespace/revision differs"
    )
    require(
        summary["solver_calls"] == 0
        and summary["judge_calls"] == 0
        and summary["scores"] is None
        and summary["execution_subset"] is None,
        "published HLE inventory is not input-only",
    )
    readme = read_bounded(source / "README.md")
    require(
        sha(readme) == summary["source_readme_sha256"]
        and git_object("blob", readme) == tree["README.md"]["sha"],
        "HLE README identity differs",
    )
    seen, subsets, asset_bytes = set(), {}, 0
    for subset, (count, digest) in hle.SOURCES.items():
        pointer = read_bounded(source / "data" / (subset + ".jsonl"), 512)
        require(
            git_object("blob", pointer) == tree[f"data/{subset}.jsonl"]["sha"],
            "HLE pointer Git identity differs",
        )
        lines = pointer.decode().splitlines()
        require(
            len(lines) == 3
            and lines[0] == "version https://git-lfs.github.com/spec/v1"
            and lines[1] == "oid sha256:" + digest
            and re.fullmatch(r"size [1-9][0-9]*", lines[2]) is not None,
            "HLE LFS identity differs",
        )
        asset_bytes += int(lines[2].split()[1])
        declared = summary["subsets"][subset]
        rows = inventory(
            [files[f"{base}/{subset}-inventory.jsonl.gz"]], declared["inventory_sha256"]
        )
        require(len(rows) == count, "HLE subset cardinality differs")
        for index, row in enumerate(rows):
            require(
                row["source_subset"] == subset
                and type(row["source_index"]) is int
                and row["source_index"] == index,
                "HLE source position differs",
            )
            require(
                isinstance(row["id"], str) and row["id"] and row["id"] not in seen,
                "duplicate or missing HLE task ID",
            )
            seen.add(row["id"])
            for key in (
                "problem_image_present",
                "image_preview_present",
                "requires_image_handling",
                "rationale_image_present",
            ):
                require(type(row[key]) is bool, "HLE modality must be boolean")
            require(
                row["requires_image_handling"]
                is (row["problem_image_present"] or row["image_preview_present"]),
                "HLE problem modality differs",
            )
            require(
                "question" not in row and "answer" not in row,
                "inventory must not expose question or answer contents",
            )
        computed = {
            "tasks": len(rows),
            "image_bearing": sum(r["requires_image_handling"] for r in rows),
            "image_field": sum(r["problem_image_present"] for r in rows),
            "preview_only": sum(
                r["image_preview_present"] and not r["problem_image_present"]
                for r in rows
            ),
            "answer_types": dict(Counter(r["answer_type"] for r in rows)),
            "answer_value_types": dict(Counter(r["answer_value_type"] for r in rows)),
            "classes": dict(Counter(r["verified_class"] for r in rows)),
            "inventory_sha256": declared["inventory_sha256"],
        }
        require(computed == declared, "HLE inventory/summary differs")
        subsets[subset] = computed
    return {
        "published_inventory_rows_verified": len(seen),
        "subsets": subsets,
        "lfs_pointer_identities_verified": True,
        "declared_source_bytes": asset_bytes,
        "raw_source_rows_reverified": False,
        "execution_scope_frozen": False,
        "rights_clearance": False,
        "solver_or_judge_executed": False,
        "scores": None,
    }


def audit_swe_rows(
    rows: list[dict], records: list[dict], tree: dict, original_patch: bytes, swe: Any
) -> dict:
    require(
        len({r["instance_id"] for r in rows}) == len(rows), "duplicate SWE source task"
    )
    require(
        len({r["instance_id"] for r in records}) == len(records),
        "duplicate SWE inventory task",
    )
    source = {r["instance_id"]: r for r in rows}
    require(
        set(source) == {r["instance_id"] for r in records}, "SWE task coverage differs"
    )
    mismatches, layers = [], {}
    required = {
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/config.json",
        "tests/test.sh",
        "tests/run_script.sh",
        "tests/parser.py",
        "tests/test_patch.patch",
        "solution/gold_patch.diff",
    }
    for record in records:
        row = source[record["instance_id"]]
        require(
            record["source_row_sha256"] == sha(swe.canonical(row)),
            "SWE source row differs",
        )
        require(row["version"] == record["release"] == "2.0.0", "SWE release differs")
        for key in ("repo", "base_commit", "hard"):
            require(
                type(record[key]) is type(row[key]) and record[key] == row[key],
                "SWE task identity differs",
            )
        require(
            record["image_tag"]
            == row["docker_image"]
            == f"ghcr.io/scaleapi/swe-bench_pro-v2:{row['instance_id']}",
            "SWE image tag differs",
        )
        require(
            required <= record["harness_files"].keys(),
            "missing required harness mapping",
        )
        for name, metadata in record["harness_files"].items():
            entry = tree[f"v2/tasks/{row['instance_id']}/{name}"]
            require(
                entry["type"] == "blob"
                and entry["sha"] == metadata["git_blob_sha"]
                and entry["size"] == metadata["bytes"],
                "SWE harness mapping differs",
            )
        for field, target in [
            ("problem_statement", "problem_sha256"),
            ("patch", "patch_sha256"),
            ("test_patch", "test_patch_sha256"),
        ]:
            require(
                sha(row[field].encode()) == record[target], "SWE source field differs"
            )
        swe.check_blob(
            row["patch"].encode(),
            record["harness_files"]["solution/gold_patch.diff"]["git_blob_sha"],
        )
        patch = row["test_patch"].encode()
        digest = record["harness_files"]["tests/test_patch.patch"]["git_blob_sha"]
        exact = git_object("blob", patch) == digest
        require(
            type(record["test_patch_byte_identical"]) is bool
            and record["test_patch_byte_identical"] is exact,
            "SWE patch equality flag differs",
        )
        if not exact:
            swe.check_blob(original_patch, digest)
            require(
                original_patch.replace(b"\r\n", b"\n") == patch,
                "unexplained verifier patch difference",
            )
            mismatches.append(row["instance_id"])
            patch = original_patch
        require(
            sha(patch) == record["harness_test_patch_sha256"],
            "verifier patch SHA differs",
        )
        require(
            record["verifier_patch_authority"] == "pinned harness bytes",
            "verifier authority differs",
        )
        for key in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run"):
            require(
                type(record["test_counts"][key]) is int
                and record["test_counts"][key] == len(swe.test_list(row[key])),
                "SWE test count differs",
            )
        require(
            record["runtime_repo_head_verified"] is False
            and record["tests_executed"] is False,
            "source inventory cannot assert runtime verification",
        )
        for digest, size in record["image_layers"].items():
            require(type(size) is int and size >= 0, "invalid layer metadata size")
            require(
                digest not in layers or layers[digest] == size,
                "conflicting layer metadata",
            )
            layers[digest] = size
    return {
        "source_rows_verified": len(rows),
        "harness_blob_mappings_verified": sum(len(r["harness_files"]) for r in records),
        "test_patch_byte_differences": mismatches,
        "unique_layer_metadata_count": len(layers),
        "declared_compressed_layer_bytes": sum(layers.values()),
        "repositories": dict(Counter(r["repo"] for r in records)),
        "registry_manifest_bytes_reverified": False,
        "image_layers_downloaded": False,
        "runtime_repo_heads_verified": False,
        "rights_clearance": False,
        "sandbox_tasks_executed": 0,
        "scores": None,
    }


def audit(args: argparse.Namespace) -> dict:
    hle = load_verified_module(args.benchmark_checkout, SOURCES, "hle")
    swe = load_verified_module(args.benchmark_checkout, SOURCES, "swe")
    files = verify_reports(args.benchmark_checkout, args.published_root)
    base = REPORTS[1]
    receipt = parse_json(files[base + "/audit-receipt.json"])
    summary = parse_json(files[base + "/summary.json"])
    parts = [
        files[p] for p in sorted(files) if p.startswith(base + "/task-inventory.part-")
    ]
    records = inventory(parts, receipt["inventory_sha256"])
    require(
        len(records) == receipt["inventory_rows"] == summary["rows"] == 642,
        "SWE published cardinality differs",
    )
    data = read_bounded(args.swe_data, 8 * 1024 * 1024)
    require(
        sha(data) == swe.DATA_SHA256 == summary["data_sha256"],
        "SWE source checksum differs",
    )
    tree_raw = read_bounded(args.harness_tree)
    tree = verify_git_tree(
        parse_json(tree_raw), SWE_TREE, query_commit=swe.HARNESS_REVISION
    )
    import pyarrow.parquet as pq

    rows = pq.read_table(io.BytesIO(data)).to_pylist()
    require(len(rows) == 642, "SWE source cardinality differs")
    result = audit_swe_rows(
        rows, records, tree, read_bounded(args.original_test_patch), swe
    )
    require(
        result["test_patch_byte_differences"] == summary["test_patch_byte_differences"]
        and result["repositories"] == summary["repositories"]
        and result["declared_compressed_layer_bytes"]
        == summary["unique_compressed_layers_bytes"]
        and result["unique_layer_metadata_count"] == summary["unique_layers"],
        "SWE inventory aggregate differs",
    )
    return {
        "schema_version": "aqkv-hle-swe-source-linkage/v1",
        "status": "offline-source-linkage-verified-runtime-unverified",
        "benchmark_revision": BENCHMARK_REVISION,
        "hle": audit_hle(files, args.hle_pointer_root, hle),
        "swe": result,
        "input_sha256": {"swe_data": sha(data), "harness_tree": sha(tree_raw)},
        "model_requests_sent": 0,
        "claim_boundary": {
            k: False
            for k in (
                "execution_authorized",
                "runtime_ready",
                "task_quality_claim",
                "performance_claim",
            )
        },
    }


def main() -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "benchmark-checkout",
        "published-root",
        "hle-pointer-root",
        "swe-data",
        "harness-tree",
        "original-test-patch",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o600)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
