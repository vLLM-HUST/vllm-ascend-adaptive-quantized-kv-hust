#!/usr/bin/env python3
"""Validate an immutable Qwen3.5 correctness evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from scripts.validate_qwen35_correctness import (
        CASE_IDS,
        MANAGER_PHASES,
        SHA256_RE,
        canonical_contract_sha256,
        validate_receipt,
    )
else:  # Direct execution puts scripts/ on sys.path.
    from validate_qwen35_correctness import (  # type: ignore[no-redef]
        CASE_IDS,
        MANAGER_PHASES,
        SHA256_RE,
        canonical_contract_sha256,
        validate_receipt,
    )

BUNDLE_SCHEMA = "adaptive-quantized-kv-qwen35-evidence-bundle/v1"


def _artifact_contracts() -> list[dict[str, str | None]]:
    fixed = [
        ("correctness-contract", "contract", None),
        ("correctness-receipt", "contract", None),
        ("execution-inputs", "inputs", None),
        ("runtime-environment", "inputs", None),
        ("launch-command", "inputs", None),
        ("resource-before", "resource", None),
        ("resource-after", "resource", None),
        ("service-stdout", "service", None),
        ("service-stderr", "service", None),
        ("worker-rank-0", "worker", "worker.rank.0"),
        ("worker-rank-1", "worker", "worker.rank.1"),
    ]
    manager = [
        (f"manager-{phase}", "manager", f"manager.{phase}") for phase in MANAGER_PHASES
    ]
    cases = [(f"case-{case_id}", "case", f"case.{case_id}") for case_id in CASE_IDS]
    lifecycle = [
        ("graph-lifecycle", "lifecycle", "graph_lifecycle"),
        ("native-fallback", "lifecycle", "fallback"),
        ("rollback-cleanup", "lifecycle", "rollback_cleanup"),
    ]
    return [
        {"id": item_id, "phase": phase, "receipt_binding": binding}
        for item_id, phase, binding in fixed + manager + cases + lifecycle
    ]


ARTIFACT_CONTRACTS = _artifact_contracts()
ARTIFACT_IDS = [entry["id"] for entry in ARTIFACT_CONTRACTS]


def _claim_boundary(complete: bool) -> dict[str, bool]:
    return {
        "correctness_evidence_complete": complete,
        "performance_authorized": False,
        "quality_claim_authorized": False,
        "public_surface_authorized": False,
    }


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase SHA256 digest")
    return value


def canonical_bundle_sha256(bundle: dict[str, Any]) -> str:
    encoded = json.dumps(
        bundle,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_bundle_contract(bundle: dict[str, Any]) -> dict[str, Any]:
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unexpected evidence bundle schema")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(artifact, dict) for artifact in artifacts
    ):
        raise ValueError("artifacts must be a list of objects")

    static_contract = [
        {
            "id": artifact.get("id"),
            "phase": artifact.get("phase"),
            "receipt_binding": artifact.get("receipt_binding"),
        }
        for artifact in artifacts
    ]
    if static_contract != ARTIFACT_CONTRACTS:
        raise ValueError("artifact order or coverage drifted")

    complete = bundle.get("bundle_complete") is True
    expected_status = (
        "correctness_evidence_complete" if complete else "preregistered_blocked_on_h3"
    )
    if bundle.get("status") != expected_status:
        raise ValueError("bundle status and completion disagree")

    claim_boundary = _claim_boundary(complete)
    if bundle.get("claim_boundary") != claim_boundary:
        raise ValueError("claim boundary drifted")

    if not complete:
        if bundle.get("correctness_contract_sha256") is not None:
            raise ValueError("blocked bundle must not pin a correctness contract")
        if bundle.get("correctness_receipt_sha256") is not None:
            raise ValueError("blocked bundle must not pin a correctness receipt")
        for artifact in artifacts:
            for field in ("path", "size_bytes", "sha256"):
                if artifact.get(field) is not None:
                    raise ValueError(f"blocked artifact {artifact['id']} has {field}")
    else:
        _require_digest(
            bundle.get("correctness_contract_sha256"),
            "correctness contract",
        )
        _require_digest(
            bundle.get("correctness_receipt_sha256"),
            "correctness receipt",
        )
        paths: list[str] = []
        for artifact in artifacts:
            path = artifact.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"artifact {artifact['id']} path must be non-empty")
            size = artifact.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"artifact {artifact['id']} size must be positive")
            _require_digest(artifact.get("sha256"), f"artifact {artifact['id']}")
            paths.append(path)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")

    return {
        "schema_version": BUNDLE_SCHEMA,
        "status": "PASS",
        "bundle_complete": complete,
        "artifact_count": len(ARTIFACT_IDS),
        "bundle_sha256": canonical_bundle_sha256(bundle),
        **claim_boundary,
    }


def _secure_artifact_path(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ValueError(f"artifact path is not POSIX: {relative_path}")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"artifact path escapes the bundle root: {relative_path}")

    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {relative_path}")
    try:
        resolved = cursor.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"artifact is missing: {relative_path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"artifact path escapes the bundle root: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {relative_path}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _receipt_hashes(receipt: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    manager = receipt["manager"]["phases"]
    for record in manager:
        hashes[f"manager.{record['id']}"] = record["raw_receipt_sha256"]
    for rank in receipt["worker_coverage"]["ranks"]:
        hashes[f"worker.rank.{rank['rank']}"] = rank["raw_receipt_sha256"]
    for record in receipt["cases"]:
        hashes[f"case.{record['id']}"] = record["raw_receipt_sha256"]
    hashes["graph_lifecycle"] = receipt["graph_lifecycle"]["raw_receipt_sha256"]
    hashes["fallback"] = receipt["fallback"]["raw_receipt_sha256"]
    hashes["rollback_cleanup"] = receipt["rollback_cleanup"]["raw_receipt_sha256"]
    return hashes


def validate_bundle(bundle: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    contract_result = validate_bundle_contract(bundle)
    if contract_result["bundle_complete"] is not True:
        raise ValueError("evidence bundle is not complete")
    if bundle_root.is_symlink():
        raise ValueError("bundle root must not be a symlink")
    root = bundle_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("bundle root must be a directory")

    paths: dict[str, Path] = {}
    physical_files: set[tuple[int, int]] = set()
    total_bytes = 0
    for artifact in bundle["artifacts"]:
        path = _secure_artifact_path(root, artifact["path"])
        stat_result = path.stat()
        physical_identity = (stat_result.st_dev, stat_result.st_ino)
        if physical_identity in physical_files:
            raise ValueError(f"artifact file is reused: {artifact['id']}")
        physical_files.add(physical_identity)
        size = stat_result.st_size
        if size != artifact["size_bytes"]:
            raise ValueError(f"artifact size drifted: {artifact['id']}")
        digest = _file_sha256(path)
        if digest != artifact["sha256"]:
            raise ValueError(f"artifact digest drifted: {artifact['id']}")
        paths[artifact["id"]] = path
        total_bytes += size

    artifact_by_id = {artifact["id"]: artifact for artifact in bundle["artifacts"]}
    contract_payload = paths["correctness-contract"].read_bytes()
    receipt_payload = paths["correctness-receipt"].read_bytes()
    if (
        _sha256_bytes(contract_payload)
        != artifact_by_id["correctness-contract"]["sha256"]
    ):
        raise ValueError("correctness contract changed during validation")
    if (
        _sha256_bytes(receipt_payload)
        != artifact_by_id["correctness-receipt"]["sha256"]
    ):
        raise ValueError("correctness receipt changed during validation")
    correctness_contract = json.loads(contract_payload)
    correctness_receipt = json.loads(receipt_payload)
    correctness_result = validate_receipt(correctness_contract, correctness_receipt)
    contract_digest = canonical_contract_sha256(correctness_contract)
    if contract_digest != bundle["correctness_contract_sha256"]:
        raise ValueError("bundle is not bound to the correctness contract")
    receipt_file_digest = _file_sha256(paths["correctness-receipt"])
    if receipt_file_digest != bundle["correctness_receipt_sha256"]:
        raise ValueError("bundle is not bound to the correctness receipt file")

    receipt_hashes = _receipt_hashes(correctness_receipt)
    for artifact in bundle["artifacts"]:
        binding = artifact["receipt_binding"]
        if binding is not None and artifact["sha256"] != receipt_hashes.get(binding):
            raise ValueError(f"receipt binding drifted: {artifact['id']}")

    return {
        **contract_result,
        "correctness_evidence_complete": correctness_result["correctness_complete"],
        "total_bytes": total_bytes,
        "correctness_contract_sha256": contract_digest,
        "correctness_receipt_sha256": receipt_file_digest,
        "performance_authorized": False,
        "quality_claim_authorized": False,
        "public_surface_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    result = validate_bundle_contract(bundle)
    if args.require_complete and result["bundle_complete"] is not True:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "error": "correctness evidence bundle is not complete",
                    "bundle_sha256": result["bundle_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if args.root is not None:
        result = validate_bundle(bundle, args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
