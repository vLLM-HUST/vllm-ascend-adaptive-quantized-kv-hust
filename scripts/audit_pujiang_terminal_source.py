#!/usr/bin/env python3
"""Bind Terminal source files and policies offline; never run tasks or fetch images."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.audit_pujiang_hle_swe_sources import (
        git_object,
        read_bounded,
        require,
        sha,
        verify_git_tree,
    )
    from scripts.pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from scripts.validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json
    from scripts.validate_pujiang_terminal_plan import SOURCE_FILES, load_plan
except ModuleNotFoundError:
    from audit_pujiang_hle_swe_sources import (
        git_object,
        read_bounded,
        require,
        sha,
        verify_git_tree,
    )
    from pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json
    from validate_pujiang_terminal_plan import SOURCE_FILES, load_plan

REPORT = "reports/pyramidkv-terminal-source-audit-20261009"
TREE_ROOT = "ddbd9031e59804a04e24019fc408d51b56a4e773"
SOURCE_BYTE_CAP = 64 * 1024 * 1024
METADATA_BYTE_CAP = 8 * 1024 * 1024
REQUIRED = {
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "solution/solve.sh",
}


def encoded(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def unpack(raw: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        value = stream.read(METADATA_BYTE_CAP + 1)
    require(len(value) <= METADATA_BYTE_CAP, "uncompressed metadata exceeds byte cap")
    return value


def verify_report(checkout: Path, root: Path) -> dict[str, bytes]:
    listing = (
        subprocess.check_output(
            ["git", "-C", str(checkout), "ls-tree", "-r", "HEAD", REPORT], timeout=10
        )
        .decode()
        .splitlines()
    )
    require(bool(listing), "missing pinned Terminal report")
    files, total = {}, 0
    for line in listing:
        header, path = line.split("\t", 1)
        mode, kind, digest = header.split()
        require(
            mode == "100644" and kind == "blob" and path.startswith(REPORT + "/"),
            "unsupported report entry",
        )
        raw = read_bounded(root / path, METADATA_BYTE_CAP)
        total += len(raw)
        require(total <= METADATA_BYTE_CAP, "report exceeds byte cap")
        require(git_object("blob", raw) == digest, "published report blob differs")
        files[path.removeprefix(REPORT + "/")] = raw
    return files


def preflight_snapshot(snapshot: Path) -> None:
    require(snapshot.is_dir() and not snapshot.is_symlink(), "invalid snapshot root")
    total, count = 0, 0
    for path in snapshot.rglob("*"):
        require(not path.is_symlink(), "symlink in source snapshot")
        if path.is_dir():
            continue
        require(path.is_file(), "source must contain regular files")
        total += path.stat().st_size
        count += 1
        require(
            total <= SOURCE_BYTE_CAP and count <= 1101,
            "source snapshot exceeds byte or file cap",
        )


def audit_tasks(
    snapshot: Path, entries: dict, files: dict, rows: list, receipts: list, plan: Any
) -> dict:
    import tomllib

    ids = sorted(
        path.split("/")[1]
        for path in files
        if path.startswith("tasks/")
        and path.count("/") == 2
        and path.endswith("/task.toml")
    )
    expected = plan.source_audit.EXPECTED_TASKS
    require(len(ids) == len(set(ids)) == expected, "source task cardinality differs")
    require(
        all(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name) for name in ids),
        "unsafe task identity",
    )
    require(
        isinstance(rows, list)
        and len(rows) == expected
        and [row["task_id"] for row in rows] == ids,
        "published task inventory differs",
    )
    require(isinstance(receipts, list), "invalid image receipt inventory")
    images = {item["task_id"]: item for item in receipts}
    require(
        len(images) == len(receipts) == expected and set(images) == set(ids),
        "image receipt membership differs",
    )
    verified, modes, layers, layer_excesses = [], {}, {}, []
    for row in rows:
        identity = row["task_id"]
        base = "tasks/" + identity + "/"
        task_files = {
            name.removeprefix(base): value
            for name, value in files.items()
            if name.startswith(base)
        }
        require(task_files.keys() >= REQUIRED, "required task source file missing")
        require(row["files"] == task_files, "source file inventory differs")
        raw = read_bounded(snapshot / base / "task.toml")
        require(sha(raw) == task_files["task.toml"]["sha256"], "task TOML changed")
        task = tomllib.loads(raw.decode())
        require(
            task["schema_version"] == "1.1"
            and task["task"]["name"] == row["task_name"] == "terminal-bench/" + identity
            and row["source_revision"] == plan.source_audit.REVISION,
            "task source identity differs",
        )
        receipt = images[identity]
        require(
            not receipt.get("error") and receipt["task_toml_sha256"] == sha(raw),
            "image mapping identity differs",
        )
        for key in ("environment", "agent", "verifier"):
            require(task[key] == receipt[key], "task policy differs: " + key)
        environment = task["environment"]
        require(
            row["resources"] == {key: environment[key] for key in row["resources"]}
            and set(row["resources"])
            == {
                "cpus",
                "memory_mb",
                "storage_mb",
                "gpus",
                "build_timeout_sec",
                "allow_internet",
            }
            and row["agent_timeout_sec"] == task["agent"]["timeout_sec"]
            and row["verifier_timeout_sec"] == task["verifier"]["timeout_sec"]
            and row["mcp_server_count"] == len(environment["mcp_servers"]),
            "published resource or timeout policy differs",
        )
        require(
            all(
                row.get(key) is False
                for key in (
                    "rights_review_complete",
                    "image_layers_pulled",
                    "task_executed",
                )
            ),
            "published evidence must remain input-only",
        )
        image = row["image_audit"]
        require(
            (receipt["os"], receipt["architecture"]) == ("linux", "amd64")
            and image["platform"] == "linux/amd64"
            and row["image"] == receipt["image"] == environment["docker_image"],
            "published image platform or source mapping differs",
        )
        for key in ("tag_digest", "manifest_digest", "config_digest"):
            require(
                re.fullmatch(r"sha256:[0-9a-f]{64}", image[key]) is not None
                and image[key] == receipt[key],
                "image digest mapping differs",
            )
        require(
            image["layers"] == receipt["layers"]
            and image["compressed_layer_bytes"] == receipt["compressed_layer_bytes"],
            "image layer metadata differs",
        )
        for digest, size in image["layers"].items():
            require(
                re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
                and type(size) is int
                and size >= 0,
                "invalid layer metadata",
            )
            require(
                digest not in layers or layers[digest] == size, "conflicting layer size"
            )
            layers[digest] = size
        unique_bytes = sum(image["layers"].values())
        descriptor_bytes = image["compressed_layer_bytes"]
        # Upstream sums descriptor occurrences but stores layers in a digest-keyed
        # map. Without manifest bytes, duplicate occurrences cannot be reverified.
        require(
            type(descriptor_bytes) is int and descriptor_bytes >= unique_bytes,
            "invalid compressed descriptor total",
        )
        if descriptor_bytes != unique_bytes:
            layer_excesses.append(
                {
                    "task_id": identity,
                    "unique_layer_bytes": unique_bytes,
                    "published_descriptor_bytes": descriptor_bytes,
                    "difference_bytes": descriptor_bytes - unique_bytes,
                }
            )
        pinned = row["image"].rsplit(":", 1)[0] + "@" + image["manifest_digest"]
        preview = plan.pin_task_image(raw.decode(), row["image"], pinned).encode()
        task_modes = {name: entries[base + name]["mode"] for name in task_files}
        modes.update({base + name: mode for name, mode in task_modes.items()})
        verified.append(
            {
                "task_id": identity,
                "source_row_sha256": sha(encoded(row)),
                "source_files_sha256": sha(encoded(task_files)),
                "git_file_modes_sha256": sha(encoded(task_modes)),
                "task_policy_sha256": sha(
                    encoded(
                        {key: task[key] for key in ("environment", "agent", "verifier")}
                    )
                ),
                "pinned_image": pinned,
                "pin_preview_sha256": sha(preview),
            }
        )
    return {
        "task_bindings": verified,
        "task_bindings_sha256": sha(encoded(verified)),
        "source_git_file_modes_sha256": sha(encoded(modes)),
        "source_git_mode_counts": dict(Counter(modes.values())),
        "task_files_verified": len(modes),
        "published_unique_layers": len(layers),
        "published_unique_compressed_layer_bytes": sum(layers.values()),
        "published_layer_total_excesses": layer_excesses,
        "layer_descriptor_multiplicity_reverified": False,
        "resource_values": {
            key: sorted({row["resources"][key] for row in rows})
            for key in rows[0]["resources"]
        },
        "agent_timeout_values_sec": sorted({row["agent_timeout_sec"] for row in rows}),
        "verifier_timeout_values_sec": sorted(
            {row["verifier_timeout_sec"] for row in rows}
        ),
        "declared_agent_timeout_sum_sec": sum(row["agent_timeout_sec"] for row in rows),
        "declared_verifier_timeout_sum_sec": sum(
            row["verifier_timeout_sec"] for row in rows
        ),
    }


def audit(checkout: Path, published_root: Path, snapshot: Path) -> dict:
    plan = load_plan(checkout)
    shared = load_verified_module(checkout, SOURCE_FILES, "audit")
    reports = verify_report(checkout, published_root)
    tree_raw = unpack(reports["tree.json.gz"])
    require(sha(tree_raw) == shared.TREE_SHA256, "source tree receipt differs")
    tree = parse_json(tree_raw)
    entries = verify_git_tree(tree, TREE_ROOT, query_commit=shared.REVISION)
    preflight_snapshot(snapshot)
    files = shared.verify_snapshot(snapshot, tree)
    rows = []
    for name in sorted(name for name in reports if name.startswith("inventory-")):
        part = parse_json(unpack(reports[name]))
        require(isinstance(part, list), "invalid published inventory part")
        rows.extend(part)
    reproducer = parse_json(reports["reproducibility.json"])
    require(
        sha(encoded(rows)) == reproducer["files"]["inventory.json"],
        "reconstructed inventory identity differs",
    )
    summary = parse_json(reports["summary.json"])
    require(
        sha(reports["summary.json"]) == reproducer["files"]["summary.json"],
        "published summary identity differs",
    )
    require(
        summary["source_repo"] == "harbor-framework/terminal-bench-2-1"
        and summary["source_revision"] == shared.REVISION
        and summary["root_license"] == files["LICENSE"]
        and all(
            summary.get(key) is False
            for key in (
                "rights_review_complete",
                "image_layers_pulled",
                "tasks_executed",
            )
        ),
        "source summary identity or evidence boundary differs",
    )
    result = audit_tasks(
        snapshot,
        entries,
        files,
        rows,
        parse_json(unpack(reports["image-receipts.json.gz"])),
        plan,
    )
    for key in (
        "resource_values",
        "agent_timeout_values_sec",
        "verifier_timeout_values_sec",
    ):
        require(result[key] == summary[key], "recomputed resource summary differs")
    require(
        result["published_unique_layers"] == summary["unique_compressed_layers"]
        and result["published_unique_compressed_layer_bytes"]
        == summary["unique_compressed_layer_bytes"]
        and summary["tasks"] == len(rows) == shared.EXPECTED_TASKS
        and summary["verified_source_files"] == len(files) == 1101,
        "published source/layer totals differ",
    )
    return {
        "status": "terminal_source_bound_execution_blocked",
        "benchmark_revision": BENCHMARK_REVISION,
        "source_revision": shared.REVISION,
        "source_tree_root": TREE_ROOT,
        "source_tree_sha256": sha(tree_raw),
        "source_script_sha256": {key: value[1] for key, value in SOURCE_FILES.items()},
        "source_files_verified": len(files),
        "source_bytes_verified": sum(value["bytes"] for value in files.values()),
        "task_count": len(rows),
        "report_files_sha256": {name: sha(raw) for name, raw in reports.items()},
        **result,
        "source_membership_verified": True,
        "task_source_bytes_verified": True,
        "source_git_modes_verified": True,
        "local_staged_modes_verified": False,
        "image_pin_semantics_verified": True,
        "registry_metadata_bytes_reverified": False,
        "image_layers_pulled": False,
        "rights_review_complete": False,
        "harbor_schema_validated_locally": False,
        "ready_for_scored_execution": False,
        "execution_authorized": False,
        "tasks_executed": 0,
        "model_requests_sent": 0,
        "task_quality": None,
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark-checkout", "published-root", "snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.benchmark_checkout, args.published_root, args.snapshot)
    with args.output.open("xb") as stream:
        stream.write(encoded(result))
    args.output.chmod(0o600)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "task_bindings"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
