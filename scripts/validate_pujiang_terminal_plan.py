#!/usr/bin/env python3
"""Check matched Terminal candidate plans offline, not services or task results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from scripts.pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from scripts.validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json
except ModuleNotFoundError:
    from pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json

SCHEMA = "aqkv-terminal-candidate-envelope/v1"
DATASET_REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"
SOURCE_FILES = {
    "audit": (
        "scripts/audit_pyramidkv_terminal.py",
        "9dcfbec30b1a75df2fca3e46b23b3aee12779912c1db8b15c6d0ae9e18a5951c",
    ),
    "plan": (
        "scripts/prepare_pyramidkv_terminal.py",
        "1a36a171be3db57964284bc2ff9fa26cd218ef9f766ed3bdaafa548e55780a95",
    ),
}
RETRY_POLICY = {
    "whole_trial_retries": 0,
    "litellm_sdk_retries": 0,
    "harbor_llm_layer_attempts": 3,
    "terminus_query_layer_attempts": 3,
}
PENDING_GATES = (
    "canonical contract and task/third-party rights review",
    "verified task files, executable modes, task policies and pinned image layers",
    "actual AMD64 Docker/Harbor environment and effective resource limits",
    "actual backend-to-model connection/authentication and matched service provenance",
    "accepted Host, calibrated profile, Manager activation and real AQKV correctness",
    "fresh-sandbox reference/failure verifier checks and complete task outcomes",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def obj(value: object, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value  # type: ignore[return-value]


def digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_plan(checkout: Path) -> ModuleType:
    return load_verified_module(
        checkout, SOURCE_FILES, "plan", ("audit", "audit_pyramidkv_terminal")
    )


def validate_inputs(
    envelope: object, inventory: object, jobs: dict[str, object], plan: ModuleType
) -> dict[str, Any]:
    record = obj(envelope, "envelope")
    require(record.get("schema_version") == SCHEMA, "Envelope schema mismatch")
    require(
        record.get("evidence_kind")
        in ("development-fixture", "unverified-candidate-plan"),
        "Candidate evidence only",
    )
    require(
        record.get("scope") == "supplied-task-plan-only", "No official execution scope"
    )
    require(
        record.get("dataset") == "terminal-bench-2.1"
        and record.get("dataset_revision") == DATASET_REVISION
        and record.get("benchmark_revision") == BENCHMARK_REVISION
        and record.get("source_sha256")
        == {name: value[1] for name, value in SOURCE_FILES.items()}
        and record.get("harbor_version") == plan.HARBOR_VERSION
        and record.get("harbor_revision") == plan.HARBOR_REVISION,
        "Dataset/planner/Harbor candidate provenance mismatch",
    )
    require(
        canonical(record.get("retry_policy")) == canonical(RETRY_POLICY),
        "Nested retries must remain explicit, not zero model-request retries",
    )
    require(
        isinstance(inventory, list) and 0 < len(inventory) <= 89,
        "Need a named task inventory, at most 89 tasks",
    )
    rows = [obj(row, "task") for row in inventory]
    require(
        type(record.get("task_count")) is int and record["task_count"] == len(rows),
        "Task count differs from supplied plan",
    )
    identities = []
    for row in rows:
        identity = row.get("task_id")
        require(
            isinstance(identity, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9.-]*", identity) is not None
            and identity not in identities,
            "Task IDs must be unique safe names",
        )
        identities.append(identity)
        image = row.get("pinned_image")
        require(
            isinstance(image, str)
            and re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image) is not None,
            "Need a digest-pinned image, not a mutable tag",
        )
        for key in (
            "source_row_sha256",
            "staged_files_sha256",
            "file_modes_sha256",
            "task_policy_sha256",
        ):
            require(digest(row.get(key)), f"Missing task metadata digest: {key}")
    serving = obj(record.get("serving_intent"), "serving intent")
    model = serving.get("model_name")
    require(
        isinstance(model, str)
        and re.fullmatch(r"openai/[A-Za-z0-9._-]+", model) is not None,
        "Need an explicit credential-free served model alias",
    )
    api = obj(serving.get("api_base"), "planned API endpoints")
    intended = obj(serving.get("aqkv_enabled"), "planned AQKV switches")
    require(
        set(api) == {"B0", "B1"}
        and all(isinstance(value, str) for value in api.values()),
        "Require both planned API endpoints",
    )
    require(
        set(intended) == {"B0", "B1"}
        and intended["B0"] is False
        and intended["B1"] is True,
        "Planned B0/B1 switches must be explicit bools",
    )
    require(
        record.get("runtime_receipts") == {"B0": None, "B1": None},
        "This validator cannot accept service activation receipts",
    )
    require(set(jobs) == {"B0", "B1"}, "Require exactly B0 and B1")
    for arm, value in jobs.items():
        job = obj(value, arm)
        expected = plan.job_config(arm.lower(), identities, api[arm])
        expected["job_name"] = "aqkv-terminal-" + arm.lower()
        expected["agents"][0]["model_name"] = model
        # Artifact labels and API destinations are intent, not server activation.
        require(
            canonical(job) == canonical(expected),
            f"{arm}: candidate task/policy/budget mismatch",
        )
    return {
        "status": "candidate_plan_valid_execution_blocked",
        "evidence_kind": record["evidence_kind"],
        "scope": "supplied-task-plan-only; no task, sandbox or service validation",
        "benchmark_revision": BENCHMARK_REVISION,
        "task_count": len(rows),
        "source_membership_verified": False,
        "task_assets_verified": False,
        "harbor_schema_validated_locally": False,
        "actual_backend_connection_verified": False,
        "actual_service_switches_verified": False,
        "ready_for_scored_execution": False,
        "model_calls": 0,
        "tasks_executed": 0,
        "task_quality": None,
        "retry_policy": RETRY_POLICY.copy(),
        "pending_gates": list(PENDING_GATES),
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }


def validate_files(
    envelope_path: Path,
    inventory_path: Path,
    b0_path: Path,
    b1_path: Path,
    benchmark_checkout: Path,
    *,
    published_root: Path | None = None,
    snapshot: Path | None = None,
    staged_tasks: Path | None = None,
) -> dict[str, Any]:
    assets = (published_root, snapshot, staged_tasks)
    require(
        all(path is None for path in assets)
        or all(path is not None for path in assets),
        "published root, snapshot and staged tasks must be supplied together",
    )
    envelope_raw = envelope_path.read_bytes()
    record = obj(parse_json(envelope_raw), "envelope")
    checksums = obj(record.get("files_sha256"), "files_sha256")
    paths = {"inventory": inventory_path, "B0": b0_path, "B1": b1_path}
    require(set(checksums) == set(paths), "Require exactly inventory/B0/B1 checksums")
    parsed = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        require(
            digest(checksums[name]) and sha(raw) == checksums[name],
            f"{name}: artifact checksum mismatch",
        )
        parsed[name] = parse_json(raw)
    report = validate_inputs(
        record,
        parsed["inventory"],
        {arm: parsed[arm] for arm in ("B0", "B1")},
        load_plan(benchmark_checkout),
    )
    report["envelope_sha256"] = sha(envelope_raw)
    report["artifact_sha256"] = dict(checksums)
    if published_root is not None and snapshot is not None and staged_tasks is not None:
        try:
            from scripts.audit_pujiang_terminal_source import audit
        except ModuleNotFoundError:
            from audit_pujiang_terminal_source import audit

        source = audit(
            benchmark_checkout,
            published_root,
            snapshot,
            staged_tasks=staged_tasks,
            candidate_inventory=parsed["inventory"],
        )
        report["status"] = "candidate_plan_assets_bound_execution_blocked"
        report["scope"] = (
            "supplied source members and local files; no sandbox or service validation"
        )
        report["source_membership_verified"] = True
        report["task_assets_verified"] = True
        report["local_staged_modes_verified"] = True
        report["source_asset_binding"] = {
            key: source[key]
            for key in (
                "benchmark_revision",
                "source_revision",
                "source_tree_root",
                "source_tree_sha256",
                "report_files_sha256",
                "candidate_task_count",
                "staged_files_verified",
                "staged_bytes_verified",
                "staged_task_bindings_sha256",
                "registry_metadata_bytes_reverified",
                "image_layers_pulled",
                "rights_review_complete",
            )
        }
        report["source_asset_report_sha256"] = sha(canonical(source))
        report["pending_gates"] = [
            gate.replace(
                "verified task files, executable modes, task policies "
                "and pinned image layers",
                "digest-pinned registry metadata and image layers",
            )
            for gate in PENDING_GATES
        ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("envelope", "inventory", "b0", "b1", "benchmark-checkout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("published-root", "snapshot", "staged-tasks"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        report = validate_files(
            args.envelope,
            args.inventory,
            args.b0,
            args.b1,
            args.benchmark_checkout,
            published_root=args.published_root,
            snapshot=args.snapshot,
            staged_tasks=args.staged_tasks,
        )
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2) + "\n")
        args.output.chmod(0o600)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
