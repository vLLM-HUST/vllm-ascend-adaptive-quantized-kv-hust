#!/usr/bin/env python3
"""Validate one immutable Qwen3.5 prefix-pilot result pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from scripts.validate_qwen35_prefix_pilot import (
        BENCHMARK_COMMIT,
        MODEL_REVISION,
        validate_contract,
    )
else:  # Direct execution puts scripts/ on sys.path.
    from validate_qwen35_prefix_pilot import (  # type: ignore[no-redef]
        BENCHMARK_COMMIT,
        MODEL_REVISION,
        validate_contract,
    )

RESULT_SCHEMA = "adaptive-quantized-kv-qwen35-prefix-pilot-result/v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ARM_IDS = ["baseline", "treatment"]
ARTIFACT_IDS = [
    "raw-benchmark-result",
    "request-records",
    "resolved-command",
    "environment",
    "service-stdout",
    "service-stderr",
    "memory-telemetry",
    "safety",
    "cleanup",
]
EXPECTED_REQUESTS = 200
EXPECTED_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 256
MAX_ARM_SECONDS = 30 * 60


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase SHA256 digest")
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def canonical_pilot_contract_sha256(contract: dict[str, Any]) -> str:
    validate_contract(contract)
    return _canonical_sha256(contract)


def _secure_artifact_path(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ValueError(f"artifact path is not POSIX: {relative_path}")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"artifact path escapes the result root: {relative_path}")
    cursor = root
    for part in pure.parts:
        cursor /= part
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
            f"artifact path escapes the result root: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {relative_path}")
    return resolved


def _validate_authorization(result: dict[str, Any]) -> dict[str, Any]:
    auth = _require_dict(result.get("authorization"), "authorization")
    approval_url = auth.get("owner_approval_url")
    if not isinstance(approval_url, str) or not approval_url.startswith("https://"):
        raise ValueError("owner approval URL is missing")
    approved_at = auth.get("approved_at")
    if not isinstance(approved_at, str) or "T" not in approved_at:
        raise ValueError("owner approval timestamp is missing")
    for field in ("host_revision", "plugin_revision", "manager_revision"):
        _require_commit(auth.get(field), field)
    if auth.get("benchmark_commit") != BENCHMARK_COMMIT:
        raise ValueError("benchmark commit drifted")
    if auth.get("model_revision") != MODEL_REVISION:
        raise ValueError("model revision drifted")
    for field in (
        "model_weight_sha256",
        "tokenizer_sha256",
        "profile_sha256",
        "request_set_sha256",
        "request_order_sha256",
    ):
        _require_digest(auth.get(field), field)
    lease_id = auth.get("lease_id")
    if not isinstance(lease_id, str) or not lease_id:
        raise ValueError("resource lease ID is missing")
    device_ids = auth.get("device_ids")
    if (
        not isinstance(device_ids, list)
        or len(device_ids) != 2
        or len(set(device_ids)) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in device_ids
        )
    ):
        raise ValueError("exactly two distinct device IDs are required")
    return auth


def _validate_result_contract(
    result: dict[str, Any], pilot_contract: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unexpected pilot result schema")
    if result.get("status") != "complete":
        raise ValueError("pilot result must be complete")
    contract_digest = canonical_pilot_contract_sha256(pilot_contract)
    if result.get("pilot_contract_sha256") != contract_digest:
        raise ValueError("pilot result is not bound to the frozen contract")

    correctness = _require_dict(
        result.get("correctness_evidence"), "correctness evidence"
    )
    if correctness.get("status") != "correctness_evidence_complete":
        raise ValueError("correctness evidence is incomplete")
    _require_digest(correctness.get("bundle_sha256"), "correctness bundle")

    auth = _validate_authorization(result)
    for field in ("host_revision", "plugin_revision", "manager_revision"):
        if correctness.get(field) != auth[field]:
            raise ValueError(f"correctness evidence {field} drifted")
    if correctness.get("model_revision") != MODEL_REVISION:
        raise ValueError("correctness evidence model revision drifted")
    if correctness.get("profile_sha256") != auth["profile_sha256"]:
        raise ValueError("correctness evidence profile drifted")

    execution = _require_dict(result.get("execution"), "execution")
    if execution != {
        "arm_order": ARM_IDS,
        "fresh_service_per_arm": True,
        "release_devices_after_each_arm": True,
    }:
        raise ValueError("execution contract drifted")

    arms = _require_list(result.get("arms"), "arms")
    if [arm.get("id") for arm in arms if isinstance(arm, dict)] != ARM_IDS:
        raise ValueError("arm order drifted")
    for arm in arms:
        arm_dict = _require_dict(arm, "arm")
        artifacts = _require_list(arm_dict.get("artifacts"), "arm artifacts")
        artifact_ids = [item.get("id") for item in artifacts if isinstance(item, dict)]
        if artifact_ids != ARTIFACT_IDS:
            raise ValueError(f"artifact order or coverage drifted: {arm_dict['id']}")
        paths: list[str] = []
        for artifact in artifacts:
            artifact_dict = _require_dict(artifact, "artifact")
            path = artifact_dict.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"artifact path is missing: {artifact_dict.get('id')}")
            size = artifact_dict.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"artifact size is invalid: {artifact_dict['id']}")
            _require_digest(artifact_dict.get("sha256"), artifact_dict["id"])
            paths.append(path)
        if len(paths) != len(set(paths)):
            raise ValueError(f"artifact paths are not unique: {arm_dict['id']}")

    if result.get("declared_verdict") not in {
        "BOUNDED_POSITIVE",
        "BOUNDED_NEGATIVE",
    }:
        raise ValueError("declared verdict is invalid")
    expected_boundary = {
        "general_performance_claim_authorized": False,
        "quality_claim_authorized": False,
        "public_surface_authorized": False,
    }
    if result.get("claim_boundary") != expected_boundary:
        raise ValueError("claim boundary drifted")
    return auth, [_require_dict(arm, "arm") for arm in arms]


def _load_artifacts(
    arms: list[dict[str, Any]], root: Path
) -> tuple[dict[str, dict[str, Any]], int]:
    loaded: dict[str, dict[str, Any]] = {}
    physical_files: set[tuple[int, int]] = set()
    total_bytes = 0
    for arm in arms:
        arm_id = arm["id"]
        loaded[arm_id] = {}
        for artifact in arm["artifacts"]:
            path = _secure_artifact_path(root, artifact["path"])
            stat_result = path.stat()
            identity = (stat_result.st_dev, stat_result.st_ino)
            if identity in physical_files:
                raise ValueError(f"artifact file is reused: {arm_id}/{artifact['id']}")
            physical_files.add(identity)
            if stat_result.st_size != artifact["size_bytes"]:
                raise ValueError(f"artifact size drifted: {arm_id}/{artifact['id']}")
            if _file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"artifact digest drifted: {arm_id}/{artifact['id']}")
            total_bytes += stat_result.st_size
            artifact_id = artifact["id"]
            if artifact_id in {"service-stdout", "service-stderr"}:
                loaded[arm_id][artifact_id] = {"path": str(path)}
            else:
                try:
                    loaded[arm_id][artifact_id] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"artifact is not valid JSON: {arm_id}/{artifact_id}"
                    ) from exc
    return loaded, total_bytes


def _validate_environment(
    arm_id: str, artifacts: dict[str, Any], auth: dict[str, Any]
) -> None:
    environment = _require_dict(artifacts["environment"], "environment")
    expected = {
        "host_revision": auth["host_revision"],
        "plugin_revision": auth["plugin_revision"],
        "manager_revision": auth["manager_revision"],
        "benchmark_commit": BENCHMARK_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": auth["model_weight_sha256"],
        "tokenizer_sha256": auth["tokenizer_sha256"],
        "profile_sha256": auth["profile_sha256"],
        "request_set_sha256": auth["request_set_sha256"],
        "request_order_sha256": auth["request_order_sha256"],
        "lease_id": auth["lease_id"],
        "device_ids": auth["device_ids"],
    }
    if environment != expected:
        raise ValueError(f"environment identity drifted: {arm_id}")

    command = _require_dict(artifacts["resolved-command"], "resolved command")
    if command.get("arm") != arm_id:
        raise ValueError(f"resolved command arm drifted: {arm_id}")
    if command.get("scenario") != "prefix-repetition-online-2chip":
        raise ValueError(f"resolved command scenario drifted: {arm_id}")
    if not isinstance(command.get("argv"), list) or not command["argv"]:
        raise ValueError(f"resolved command argv is missing: {arm_id}")


def _validate_requests(
    arm_id: str, payload: object, auth: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    records = _require_dict(payload, "request records")
    if records.get("request_set_sha256") != auth["request_set_sha256"]:
        raise ValueError(f"request set drifted: {arm_id}")
    if records.get("request_order_sha256") != auth["request_order_sha256"]:
        raise ValueError(f"request order drifted: {arm_id}")
    requests = _require_list(records.get("requests"), "requests")
    if len(requests) != EXPECTED_REQUESTS:
        raise ValueError(f"request count drifted: {arm_id}")
    normalized: list[dict[str, Any]] = []
    total_output_tokens = 0
    for index, item in enumerate(requests):
        request = _require_dict(item, "request")
        if request.get("index") != index or request.get("status") != "PASS":
            raise ValueError(f"request order or status failed: {arm_id}/{index}")
        if request.get("input_tokens") != EXPECTED_INPUT_TOKENS:
            raise ValueError(f"input token count drifted: {arm_id}/{index}")
        output_tokens = request.get("output_tokens")
        if (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or not 1 <= output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError(f"output token count is invalid: {arm_id}/{index}")
        _require_digest(
            request.get("output_token_ids_sha256"),
            f"output token IDs: {arm_id}/{index}",
        )
        _positive_number(request.get("ttft_ms"), f"TTFT: {arm_id}/{index}")
        _positive_number(request.get("tpot_ms"), f"TPOT: {arm_id}/{index}")
        total_output_tokens += output_tokens
        normalized.append(request)
    return normalized, total_output_tokens


def _validate_benchmark(
    arm_id: str, payload: object, total_output_tokens: int
) -> dict[str, float]:
    benchmark = _require_dict(payload, "raw benchmark result")
    expected_ints = {
        "num_prompts": EXPECTED_REQUESTS,
        "completed": EXPECTED_REQUESTS,
        "failed": 0,
        "total_input_tokens": EXPECTED_REQUESTS * EXPECTED_INPUT_TOKENS,
        "total_output_tokens": total_output_tokens,
    }
    for field, expected in expected_ints.items():
        if benchmark.get(field) != expected:
            raise ValueError(f"raw benchmark {field} drifted: {arm_id}")
    metrics = {
        field: _positive_number(benchmark.get(field), f"{field}: {arm_id}")
        for field in (
            "median_ttft_ms",
            "output_throughput",
            "mean_tpot_ms",
            "mean_itl_ms",
            "duration_s",
        )
    }
    if metrics["duration_s"] > MAX_ARM_SECONDS:
        raise ValueError(f"arm exceeded the time bound: {arm_id}")
    return metrics


def _validate_memory(arm_id: str, payload: object, device_ids: list[int]) -> None:
    memory = _require_dict(payload, "memory telemetry")
    if memory.get("device_ids") != device_ids:
        raise ValueError(f"memory telemetry device IDs drifted: {arm_id}")
    samples = _require_list(memory.get("samples"), "memory samples")
    if len(samples) < 2:
        raise ValueError(f"memory telemetry is incomplete: {arm_id}")
    seen: set[int] = set()
    observed_peaks = {device_id: 0.0 for device_id in device_ids}
    for item in samples:
        sample = _require_dict(item, "memory sample")
        device_id = sample.get("device_id")
        if device_id not in device_ids:
            raise ValueError(f"memory sample device drifted: {arm_id}")
        used = _positive_number(sample.get("used_memory_mb"), "used memory")
        observed_peaks[device_id] = max(observed_peaks[device_id], used)
        seen.add(device_id)
    if seen != set(device_ids):
        raise ValueError(f"memory telemetry missed a device: {arm_id}")
    declared = _require_dict(memory.get("peak_used_memory_mb"), "memory peaks")
    expected = {str(key): value for key, value in observed_peaks.items()}
    if declared != expected:
        raise ValueError(f"memory peak drifted: {arm_id}")


def _validate_safety_cleanup(arm_id: str, artifacts: dict[str, Any]) -> None:
    safety = _require_dict(artifacts["safety"], "safety")
    if safety != {
        "correctness_failures": 0,
        "graph_failures": 0,
        "fallback_failures": 0,
    }:
        raise ValueError(f"safety gate failed: {arm_id}")
    cleanup = _require_dict(artifacts["cleanup"], "cleanup")
    if cleanup != {
        "project_processes_after_exit": 0,
        "device_memory_released": True,
        "unknown_processes_signaled": 0,
    }:
        raise ValueError(f"cleanup gate failed: {arm_id}")


def validate_result(
    result: dict[str, Any], pilot_contract: dict[str, Any], result_root: Path
) -> dict[str, Any]:
    auth, arms = _validate_result_contract(result, pilot_contract)
    if result_root.is_symlink():
        raise ValueError("result root must not be a symlink")
    root = result_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("result root must be a directory")
    artifacts, total_bytes = _load_artifacts(arms, root)

    requests_by_arm: dict[str, list[dict[str, Any]]] = {}
    metrics_by_arm: dict[str, dict[str, float]] = {}
    for arm_id in ARM_IDS:
        arm_artifacts = artifacts[arm_id]
        _validate_environment(arm_id, arm_artifacts, auth)
        requests, total_output_tokens = _validate_requests(
            arm_id, arm_artifacts["request-records"], auth
        )
        requests_by_arm[arm_id] = requests
        metrics_by_arm[arm_id] = _validate_benchmark(
            arm_id,
            arm_artifacts["raw-benchmark-result"],
            total_output_tokens,
        )
        _validate_memory(arm_id, arm_artifacts["memory-telemetry"], auth["device_ids"])
        _validate_safety_cleanup(arm_id, arm_artifacts)

    baseline_requests = requests_by_arm["baseline"]
    treatment_requests = requests_by_arm["treatment"]
    for index, (baseline, treatment) in enumerate(
        zip(baseline_requests, treatment_requests, strict=True)
    ):
        for field in (
            "index",
            "input_tokens",
            "output_tokens",
            "output_token_ids_sha256",
        ):
            if baseline[field] != treatment[field]:
                raise ValueError(f"baseline/treatment output drifted: request {index}")

    baseline_metrics = metrics_by_arm["baseline"]
    treatment_metrics = metrics_by_arm["treatment"]
    ttft_improvement = (
        (baseline_metrics["median_ttft_ms"] - treatment_metrics["median_ttft_ms"])
        / baseline_metrics["median_ttft_ms"]
        * 100.0
    )
    throughput_regression = (
        (baseline_metrics["output_throughput"] - treatment_metrics["output_throughput"])
        / baseline_metrics["output_throughput"]
        * 100.0
    )
    computed_verdict = (
        "BOUNDED_POSITIVE"
        if ttft_improvement >= 5.0 and throughput_regression <= 3.0
        else "BOUNDED_NEGATIVE"
    )
    if result["declared_verdict"] != computed_verdict:
        raise ValueError("declared verdict does not match the frozen gates")

    return {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS",
        "verdict": computed_verdict,
        "pilot_contract_sha256": canonical_pilot_contract_sha256(pilot_contract),
        "result_sha256": _canonical_sha256(result),
        "correctness_bundle_sha256": result["correctness_evidence"]["bundle_sha256"],
        "artifact_count": len(ARM_IDS) * len(ARTIFACT_IDS),
        "total_bytes": total_bytes,
        "baseline_median_ttft_ms": baseline_metrics["median_ttft_ms"],
        "treatment_median_ttft_ms": treatment_metrics["median_ttft_ms"],
        "ttft_improvement_percent": ttft_improvement,
        "baseline_output_throughput": baseline_metrics["output_throughput"],
        "treatment_output_throughput": treatment_metrics["output_throughput"],
        "throughput_regression_percent": throughput_regression,
        "general_performance_claim_authorized": False,
        "quality_claim_authorized": False,
        "public_surface_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--pilot-contract", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    pilot_contract = json.loads(args.pilot_contract.read_text(encoding="utf-8"))
    print(
        json.dumps(
            validate_result(result, pilot_contract, args.root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
