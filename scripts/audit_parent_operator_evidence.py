#!/usr/bin/env python3
"""Verify frozen parent operator summaries without executing their probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final = "adaptive-quantized-kv-parent-operator-audit/v1"
M0_REVISION: Final = "69137e4e6bd08333707bb9c74241b2501c885eb0"
M1_REVISION: Final = "c3d4c667b74479678e81a86598b3bd864b3d6955"
PINNED_PARENT_SOURCE: Final = "8be03304c1657de2f5eb75de6b859d899e92bef5"
PINNED_ASCEND_SOURCE: Final = "e2e1c96e3b2f040aaba06b8ba333070d19be4738"
RAW_NZ_RECEIPT_SHA256: Final = (
    "f2809988027d13b2428d7a742ae05c3829825a9145e6bcf9f4a7c74ca2dad9c9"
)
RAW_ND_RECEIPT_SHA256: Final = (
    "8bd782e0f127262ed130db86306ee51a479d85164630903fa871e87eb6f05a22"
)


@dataclass(frozen=True, slots=True)
class RecordSpec:
    identifier: str
    revision: str
    path: str
    sha256: str
    json_record: bool = True


RECORDS: Final = (
    RecordSpec(
        "probe_contract",
        M0_REVISION,
        "artifacts/m0/paged-int8-prefill-probe-contract-20260818.json",
        "5dedf67b28e68741f36f5a0e3124c34e702cc78599fcced6b32bef0a04fc6e76",
    ),
    RecordSpec(
        "raw_nz_pilot_summary",
        M0_REVISION,
        "artifacts/m0/paged-int8-prefill-nz-pilot-20260818.json",
        "97d11c02fb7d6294422903a4461ed6069781d646d1a671394be3e40782af20d1",
    ),
    RecordSpec(
        "nd_repack_pilot_summary",
        M0_REVISION,
        "artifacts/m0/paged-int8-prefill-nd-pilot-20260818.json",
        "9a4ff50b15b671b7038186bbb7fa1bda8cd5752077aba864ae7b12780671f3a9",
    ),
    RecordSpec(
        "probe_implementation",
        M0_REVISION,
        "scripts/probe_m0_paged_int8_prefill.py",
        "7ba1bab9a54649e258a0e407024536b0ce9092cdd04723edd4198b9d506086dd",
        json_record=False,
    ),
    RecordSpec(
        "v2_compatibility_audit",
        M1_REVISION,
        "artifacts/m1/v2-compatibility-audit-20260826.json",
        "2c5db68afe94a5e67ad242af991a4f8519fe159b0dc26af65f6d470f6bf0cdd0",
    ),
)


def _run_git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        capture_output=True,
        timeout=15,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(detail or f"git {' '.join(args)} failed")
    return result.stdout


def _resolve_revision(repo: Path, revision: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"revision is not an exact lowercase SHA: {revision}")
    resolved = _run_git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    resolved_text = resolved.decode().strip()
    if resolved_text != revision:
        raise ValueError(f"revision resolved to unexpected commit: {resolved_text}")
    return resolved_text


def _read_git_blob(repo: Path, revision: str, path: str) -> bytes:
    return _run_git(repo, "show", f"{revision}:{path}")


def _origin(repo: Path) -> str:
    return _run_git(repo, "remote", "get-url", "origin").decode().strip()


def _expect(
    record: dict[str, Any],
    path: tuple[str, ...],
    expected: Any,
    errors: list[str],
) -> None:
    value: Any = record
    for part in path:
        if not isinstance(value, dict) or part not in value:
            errors.append(f"missing semantic field: {'.'.join(path)}")
            return
        value = value[part]
    if value != expected:
        errors.append(
            f"semantic drift at {'.'.join(path)}: expected {expected!r}, got {value!r}"
        )


def _check_semantics(records: dict[str, dict[str, Any]], errors: list[str]) -> None:
    checks = {
        "probe_contract": (
            (
                ("schema_version",),
                "adaptive-quantized-kv-paged-int8-prefill-probe-contract/v1",
            ),
            (("status",), "PREREGISTERED"),
            (("revision",), 2),
            (("hardware", "device_type"), "Ascend 910B2"),
            (("hardware", "co_rental_allowed"), False),
        ),
        "raw_nz_pilot_summary": (
            (("status",), "PILOT_DIAGNOSTIC"),
            (("runtime", "device"), "Ascend 910B2"),
            (("native_status",), "UNSUPPORTED_OR_ERROR"),
            (("native_error_code",), 561002),
            (("raw_receipt", "sha256"), RAW_NZ_RECEIPT_SHA256),
        ),
        "nd_repack_pilot_summary": (
            (("status",), "PILOT_DIAGNOSTIC"),
            (("runtime", "device"), "Ascend 910B2"),
            (("native_status",), "UNSUPPORTED_OR_ERROR"),
            (("native_error_code",), 561002),
            (("raw_receipt", "sha256"), RAW_ND_RECEIPT_SHA256),
        ),
        "v2_compatibility_audit": (
            (("execution_performed",), False),
            (("npu_allocated",), False),
            (("source", "parent_commit"), PINNED_PARENT_SOURCE),
            (("source", "vllm_ascend_commit"), PINNED_ASCEND_SOURCE),
            (("direct_v2_coverage", "status"), "NO_DIRECT_COVERAGE"),
            (("direct_v2_coverage", "query_length_covered_dispatches"), 0),
            (("direct_v2_coverage", "query_length_total_dispatches"), 3840),
            (("direct_v2_coverage", "query_head_pair_supported"), False),
        ),
    }
    for identifier, expected_fields in checks.items():
        record = records.get(identifier)
        if record is None:
            continue
        for path, expected in expected_fields:
            _expect(record, path, expected, errors)

    nd_summary = records.get("nd_repack_pilot_summary")
    if nd_summary is not None and "antiquant scale is nullptr" not in str(
        nd_summary.get("native_error_summary", "")
    ):
        errors.append("ND pilot no longer retains the antiquant failure summary")


def audit_parent_operator_evidence(research_repo: Path) -> dict[str, Any]:
    errors: list[str] = []
    for revision in (M0_REVISION, M1_REVISION):
        try:
            _resolve_revision(research_repo, revision)
        except ValueError as exc:
            errors.append(f"cannot resolve {revision}: {exc}")

    parsed: dict[str, dict[str, Any]] = {}
    indexed: list[dict[str, Any]] = []
    for spec in RECORDS:
        try:
            raw = _read_git_blob(research_repo, spec.revision, spec.path)
        except ValueError as exc:
            errors.append(f"cannot read {spec.identifier}: {exc}")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        indexed.append(
            {
                "identifier": spec.identifier,
                "revision": spec.revision,
                "path": spec.path,
                "sha256": digest,
                "expected_sha256": spec.sha256,
            }
        )
        if digest != spec.sha256:
            errors.append(
                f"hash drift for {spec.identifier}: expected {spec.sha256}, "
                f"got {digest}"
            )
            continue
        if spec.json_record:
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON for {spec.identifier}: {exc}")
                continue
            if not isinstance(loaded, dict):
                errors.append(f"JSON record is not an object: {spec.identifier}")
                continue
            parsed[spec.identifier] = loaded

    _check_semantics(parsed, errors)
    try:
        origin = _origin(research_repo)
    except ValueError as exc:
        origin = "UNAVAILABLE"
        errors.append(f"cannot identify parent origin: {exc}")

    supported = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "read-only-pinned-summary-integrity-audit",
        "status": "SUPPORTED_BY_PINNED_SUMMARIES" if supported else "INCONCLUSIVE",
        "runtime_acceptance": False,
        "source": {
            "origin": origin,
            "m0_revision": M0_REVISION,
            "m1_revision": M1_REVISION,
        },
        "records": indexed,
        "findings": (
            [
                "The two 910B2 pilot summaries remain PILOT_DIAGNOSTIC and "
                "retain their raw remote receipt hashes.",
                "The raw-NZ and ND-repack candidates failed before numerical "
                "or timing comparison.",
                "The V2 record remains a no-execution audit with no direct "
                "documented coverage for the archived workload.",
            ]
            if supported
            else []
        ),
        "claim_limits": [
            "Raw remote receipts are not present in Git and are not "
            "reconstructed by this audit.",
            "No NPU, operator, service, correctness, graph, quality, or "
            "performance execution is performed.",
            "The records do not prove universal CANN or Ascend hardware incapability.",
            "The records do not provide or approve a host execution interface.",
        ],
        "errors": errors,
    }


def _write_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit_parent_operator_evidence(args.research_repo)
    _write_exclusive(args.output, receipt)
    return 0 if receipt["status"] == "SUPPORTED_BY_PINNED_SUMMARIES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
