#!/usr/bin/env python3
"""Check the five-dataset planning record; never authorize runtime execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATASET_IDS = (
    "mmlu-pro",
    "hle-verified",
    "swe-bench-pro",
    "frontierscience",
    "terminal-bench-2.1",
)
SCHEMA_VERSION = "adaptive-quantized-kv-pujiang-readiness/v1"
BENCHMARK_REVISION = "e3a7183206bd399447475b2d117fef897c0880b6"
PROGRAM_SHA256 = "de1347eb1557f7e82ea3314c118ee090df1d4a87dd18e474c801b4be3a8185a4"
SHARED_IMPLEMENTATION_REVISION = "8ed7ec9a16075c5bce5644a68a712139a6e5d403"
CLAIM_BOUNDARY = {
    "runtime_ready": False,
    "execution_authorized": False,
    "task_quality_claim": False,
    "performance_claim": False,
    "public_surface_restoration": False,
}
EXECUTION_FIELDS = (
    "task_manifest_sha256",
    "scorer_revision",
    "execution_environment_digest",
    "rights_review_receipt",
    "agent_contract_sha256",
    "b0_b1_pair_sha256",
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "docs/evidence/PUJIANG_AQKV_APPLICABILITY_20261009.json"
DEFAULT_PROGRAM = ROOT / "docs/evidence/PUJIANG_DATASET_PROGRAM_SOURCE_20261009.json"
DELIVERY_REVISION = "b32fcbdaf8d3237936b35f9f4ba441bf2d64b361"
DELIVERY_SHA256 = "0e4e14a2ad0b6a2fe351b817c03519ecdb638464152bb4e429933be9fcee4f11"
DELIVERY_RELATIVE = "docs/evidence/PUJIANG_V54_CONTRACT_SOURCE_20261010.json"
DEFAULT_DELIVERY = ROOT / DELIVERY_RELATIVE


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def parse_json(raw: bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def load_json(path: Path) -> Any:
    return parse_json(path.read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _ordered_datasets(value: object, label: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list), f"{label} must be a list")
    rows = [_object(row, label) for row in value]  # type: ignore[union-attr]
    _require(
        tuple(row.get("id") for row in rows) == DATASET_IDS,
        f"{label} dataset coverage or order drifted",
    )
    return rows


def delivery_projection(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == DELIVERY_SHA256,
        "pinned current delivery contract checksum mismatch",
    )
    contract = _object(parse_json(raw), "delivery contract")
    mandatory = contract["mandatory_delivery"]
    return {
        "test_plan_version": contract["test_plan_version"],
        "source_revision": DELIVERY_REVISION,
        "snapshot_path": DELIVERY_RELATIVE,
        "snapshot_sha256": DELIVERY_SHA256,
        "mandatory_model": mandatory["model"],
        "hardware": mandatory["hardware"],
        "chip_count": mandatory["chip_count"],
        "tensor_parallel_size": mandatory["tensor_parallel_size"],
        "precision": mandatory["precision"],
        "dtype": mandatory["dtype"],
        "baseline_roles": contract["baseline_roles"]["allowed"],
        "inactive_optional_deliveries": [
            item["id"] for item in contract["optional_deliveries"]
        ],
        "source_fidelity": contract["source_fidelity"]["status"],
        "canonical_final_document_verified": False,
        "aqkv_formal_inclusion_status": "unresolved-kv-precision-scope-and-b1-manifest",
        "optional_measurements_enqueued": 0,
        "execution_authorized": False,
    }


def validate_matrix(
    matrix: object, program: object, *, delivery_path: Path = DEFAULT_DELIVERY
) -> dict[str, Any]:
    record = _object(matrix, "matrix")
    source = _object(program, "program")
    _require(record.get("schema_version") == SCHEMA_VERSION, "matrix schema drifted")
    _require(record.get("record_kind") == "planning-only", "not a planning record")
    delivery = delivery_projection(delivery_path)
    _require(
        json.dumps(record.get("delivery_control"), sort_keys=True, allow_nan=False)
        == json.dumps(delivery, sort_keys=True, allow_nan=False),
        "current delivery control drifted; historical C8 plans cannot authorize V5.4",
    )
    _require(
        record.get("review_status") == "prepared-unreviewed",
        "this snapshot must not claim owner review",
    )
    _require(source.get("contract_version") == "dataset-program-v1", "program schema")
    designation = _object(source.get("designation"), "designation")
    _require(
        designation.get("id") == "pujiang-specified-dataset-scope",
        "dataset designation drifted",
    )
    claims = _object(record.get("claim_boundary"), "claim boundary")
    _require(
        set(claims) == set(CLAIM_BOUNDARY)
        and all(value is False for value in claims.values()),
        "planning validation cannot authorize claims or execution",
    )
    provenance = _object(record.get("sources"), "sources")
    _require(
        provenance.get("benchmark_revision") == BENCHMARK_REVISION
        and provenance.get("snapshot_sha256") == PROGRAM_SHA256,
        "canonical provenance drifted",
    )
    _require(
        provenance.get("shared_implementation_revision")
        == SHARED_IMPLEMENTATION_REVISION
        and provenance.get("program_snapshot_unchanged_at_shared_revision") is True,
        "shared implementation provenance drifted",
    )
    for field in (
        "plugin_revision",
        "host_main_revision",
        "host_provider_candidate_revision",
        "host_ci_candidate_revision",
    ):
        value = provenance.get(field)
        _require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
            f"{field} must be an exact commit",
        )
    runtime = _object(record.get("runtime"), "runtime")
    _require(
        runtime.get("manifest_schema") == "0.3-experimental"
        and runtime.get("implementation_status") == "import_only",
        "runtime state drifted; this validator is planning-only",
    )
    for field in (
        "accepted_host_revision",
        "calibrated_profile_receipt",
        "manager_activation_receipt",
        "npu_correctness_receipt",
        "assigned_execution_environment",
    ):
        _require(field in runtime and runtime[field] is None, f"unverified {field}")
    _require(
        runtime.get("asset_access_verified_locally") is False,
        "upstream asset verification is not a local-container receipt",
    )
    blockers = record.get("shared_blockers")
    _require(isinstance(blockers, list) and bool(blockers), "missing shared blockers")
    for blocker in blockers:
        item = _object(blocker, "shared blocker")
        _require(
            bool(item.get("required_input")) and "owner" in item,
            "blocker needs a concrete input and an owner or explicit null",
        )
    rows = _ordered_datasets(record.get("datasets"), "matrix")
    canonical = _ordered_datasets(source.get("primary_datasets"), "program")
    for row, target in zip(rows, canonical, strict=True):
        label = row["id"]
        readiness = _object(target.get("readiness"), label)
        _require(row.get("module_status") == "blocked", f"{label}: no runtime gate")
        _require(bool(row.get("path_hypothesis")), f"{label}: missing path rationale")
        _require(
            row.get("path_observations") == [], f"{label}: no runtime observations"
        )
        _require(
            row.get("data_readiness") == readiness.get("status") == "asset-frozen",
            f"{label}: asset readiness drifted",
        )
        for field in ("split", "sample_count", "sample_count_scope", "manifest_sha256"):
            _require(
                row.get(field) == readiness.get(field), f"{label}: {field} drifted"
            )
        _require(
            row.get("source_revision") == readiness.get("dataset_revision"),
            f"{label}: dataset revision drifted",
        )
        _require(
            row.get("primary_metric") == target.get("primary_metric"),
            f"{label}: primary metric drifted",
        )
        gaps = row.get("canonical_blockers")
        _require(isinstance(gaps, list), f"{label}: blockers must be a list")
        gap_objects = [_object(gap, label) for gap in gaps]
        _require(
            [gap.get("required_input") for gap in gap_objects]
            == readiness.get("blockers"),
            f"{label}: canonical blocker drifted",
        )
        _require(all("owner" in gap for gap in gap_objects), f"{label}: missing owners")
        progress = _object(row.get("shared_progress"), f"{label}: shared progress")
        _require(
            progress.get("aqkv_runtime_verified") is False,
            f"{label}: another MOD's evidence is not AQKV runtime verification",
        )
        _require(
            isinstance(progress.get("source_url"), str)
            and progress["source_url"].startswith(
                "https://github.com/vLLM-HUST/vllm-hust-benchmark/"
            )
            and bool(progress.get("implementation_path"))
            and bool(progress.get("evidence_kind"))
            and bool(progress.get("summary")),
            f"{label}: shared progress needs attributable evidence",
        )
        current = row.get("current_blockers")
        _require(
            isinstance(current, list) and bool(current),
            f"{label}: current blockers are separate from the frozen ledger",
        )
        for gap in current:
            item = _object(gap, f"{label}: current blocker")
            _require(
                "owner" in item and bool(item.get("required_input")),
                f"{label}: current blocker needs input and explicit owner",
            )
        contract = _object(row.get("execution_contract"), label)
        _require(
            set(contract) == set(EXECUTION_FIELDS)
            and all(value is None for value in contract.values()),
            f"{label}: an unreviewed contract cannot invent frozen inputs",
        )
        _require(
            row.get("formal_execution_complete") is False
            and row.get("raw_result_artifacts") == []
            and "task_quality" in row
            and row["task_quality"] is None
            and "supporting_telemetry" in row
            and row["supporting_telemetry"] is None,
            f"{label}: planning record contains a measured result",
        )
    return {
        "status": "planning_valid_execution_blocked",
        "evidence_kind": "off-device-contract-validation",
        "dataset_count": len(rows),
        "formal_run_count": 0,
        "review_status": record["review_status"],
        "benchmark_revision": BENCHMARK_REVISION,
        "shared_implementation_revision": SHARED_IMPLEMENTATION_REVISION,
        "current_delivery_contract": delivery,
        "canonical_snapshot_scope": "historical-asset-ledger-not-execution-authority",
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }


def validate_files(
    matrix_path: Path, program_path: Path, delivery_path: Path = DEFAULT_DELIVERY
) -> dict[str, Any]:
    digest = hashlib.sha256(program_path.read_bytes()).hexdigest()
    _require(digest == PROGRAM_SHA256, "pinned canonical source checksum mismatch")
    return validate_matrix(
        load_json(matrix_path), load_json(program_path), delivery_path=delivery_path
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--delivery-contract", type=Path, default=DEFAULT_DELIVERY)
    args = parser.parse_args()
    try:
        report = validate_files(args.matrix, args.program, args.delivery_contract)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
