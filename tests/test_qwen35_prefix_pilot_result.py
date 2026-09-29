import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.validate_qwen35_prefix_pilot_result import (
    ARTIFACT_IDS,
    BENCHMARK_COMMIT,
    MODEL_REVISION,
    RESULT_SCHEMA,
    canonical_pilot_contract_sha256,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_CONTRACT_PATH = (
    ROOT / "docs" / "evidence" / "QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pilot_contract() -> dict:
    return json.loads(PILOT_CONTRACT_PATH.read_text(encoding="utf-8"))


def _authorization() -> dict:
    return {
        "owner_approval_url": "https://example.invalid/owner-approval",
        "approved_at": "2026-09-29T12:00:00Z",
        "host_revision": "1" * 40,
        "plugin_revision": "2" * 40,
        "manager_revision": "3" * 40,
        "benchmark_commit": BENCHMARK_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "profile_sha256": "c" * 64,
        "request_set_sha256": "d" * 64,
        "request_order_sha256": "e" * 64,
        "lease_id": "lease-qwen35-test",
        "device_ids": [2, 3],
    }


def _environment(auth: dict) -> dict:
    return {
        field: auth[field]
        for field in (
            "host_revision",
            "plugin_revision",
            "manager_revision",
            "benchmark_commit",
            "model_revision",
            "model_weight_sha256",
            "tokenizer_sha256",
            "profile_sha256",
            "request_set_sha256",
            "request_order_sha256",
            "lease_id",
            "device_ids",
        )
    }


def _request_records(auth: dict) -> dict:
    return {
        "request_set_sha256": auth["request_set_sha256"],
        "request_order_sha256": auth["request_order_sha256"],
        "requests": [
            {
                "index": index,
                "status": "PASS",
                "input_tokens": 4096,
                "output_tokens": 256,
                "output_token_ids_sha256": hashlib.sha256(
                    f"tokens-{index}".encode()
                ).hexdigest(),
                "ttft_ms": 100.0 + index / 100,
                "tpot_ms": 8.0,
            }
            for index in range(200)
        ],
    }


def _artifact_payloads(
    arm: str, auth: dict, *, median_ttft_ms: float, output_throughput: float
) -> dict[str, bytes]:
    json_payloads = {
        "raw-benchmark-result": {
            "num_prompts": 200,
            "completed": 200,
            "failed": 0,
            "total_input_tokens": 819200,
            "total_output_tokens": 51200,
            "median_ttft_ms": median_ttft_ms,
            "output_throughput": output_throughput,
            "mean_tpot_ms": 8.0,
            "mean_itl_ms": 7.9,
            "duration_s": 600.0,
        },
        "request-records": _request_records(auth),
        "resolved-command": {
            "arm": arm,
            "scenario": "prefix-repetition-online-2chip",
            "argv": ["vllm-hust-benchmark", "run", arm],
        },
        "environment": _environment(auth),
        "memory-telemetry": {
            "device_ids": auth["device_ids"],
            "samples": [
                {"device_id": 2, "timestamp": "t0", "used_memory_mb": 1000.0},
                {"device_id": 3, "timestamp": "t0", "used_memory_mb": 1100.0},
                {"device_id": 2, "timestamp": "t1", "used_memory_mb": 1200.0},
                {"device_id": 3, "timestamp": "t1", "used_memory_mb": 1300.0},
            ],
            "peak_used_memory_mb": {"2": 1200.0, "3": 1300.0},
        },
        "safety": {
            "correctness_failures": 0,
            "graph_failures": 0,
            "fallback_failures": 0,
        },
        "cleanup": {
            "project_processes_after_exit": 0,
            "device_memory_released": True,
            "unknown_processes_signaled": 0,
        },
    }
    payloads = {
        artifact_id: (json.dumps(value, sort_keys=True) + "\n").encode()
        for artifact_id, value in json_payloads.items()
    }
    payloads["service-stdout"] = f"{arm} service stdout\n".encode()
    payloads["service-stderr"] = f"{arm} service stderr\n".encode()
    return payloads


def _complete_result(
    tmp_path: Path,
    *,
    treatment_ttft: float = 90.0,
    treatment_throughput: float = 98.0,
    declared_verdict: str = "BOUNDED_POSITIVE",
) -> tuple[dict, Path, dict]:
    pilot_contract = _pilot_contract()
    auth = _authorization()
    result_root = tmp_path / "result"
    result_root.mkdir()
    arms = []
    for arm_id, ttft, throughput in (
        ("baseline", 100.0, 100.0),
        ("treatment", treatment_ttft, treatment_throughput),
    ):
        payloads = _artifact_payloads(
            arm_id,
            auth,
            median_ttft_ms=ttft,
            output_throughput=throughput,
        )
        artifacts = []
        for artifact_id in ARTIFACT_IDS:
            relative_path = f"{arm_id}/{artifact_id}.data"
            path = result_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payloads[artifact_id])
            artifacts.append(
                {
                    "id": artifact_id,
                    "path": relative_path,
                    "size_bytes": len(payloads[artifact_id]),
                    "sha256": _sha256(payloads[artifact_id]),
                }
            )
        arms.append({"id": arm_id, "artifacts": artifacts})

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "pilot_contract_sha256": canonical_pilot_contract_sha256(pilot_contract),
        "correctness_evidence": {
            "status": "correctness_evidence_complete",
            "bundle_sha256": "f" * 64,
            "host_revision": auth["host_revision"],
            "plugin_revision": auth["plugin_revision"],
            "manager_revision": auth["manager_revision"],
            "model_revision": MODEL_REVISION,
            "profile_sha256": auth["profile_sha256"],
        },
        "authorization": auth,
        "execution": {
            "arm_order": ["baseline", "treatment"],
            "fresh_service_per_arm": True,
            "release_devices_after_each_arm": True,
        },
        "arms": arms,
        "declared_verdict": declared_verdict,
        "claim_boundary": {
            "general_performance_claim_authorized": False,
            "quality_claim_authorized": False,
            "public_surface_authorized": False,
        },
    }
    return result, result_root, pilot_contract


def _artifact(result: dict, arm_id: str, artifact_id: str) -> dict:
    arm = next(item for item in result["arms"] if item["id"] == arm_id)
    return next(item for item in arm["artifacts"] if item["id"] == artifact_id)


def _rewrite_json(
    result: dict,
    root: Path,
    arm_id: str,
    artifact_id: str,
    mutator,
) -> None:
    artifact = _artifact(result, arm_id, artifact_id)
    path = root / artifact["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    artifact["size_bytes"] = len(payload)
    artifact["sha256"] = _sha256(payload)


def test_complete_positive_pair_passes(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)

    receipt = validate_result(result, contract, result_root)

    assert receipt["status"] == "PASS"
    assert receipt["verdict"] == "BOUNDED_POSITIVE"
    assert receipt["artifact_count"] == 18
    assert receipt["ttft_improvement_percent"] == pytest.approx(10.0)
    assert receipt["throughput_regression_percent"] == pytest.approx(2.0)
    assert receipt["general_performance_claim_authorized"] is False


def test_complete_negative_pair_is_retained(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(
        tmp_path,
        treatment_ttft=96.0,
        declared_verdict="BOUNDED_NEGATIVE",
    )

    receipt = validate_result(result, contract, result_root)

    assert receipt["status"] == "PASS"
    assert receipt["verdict"] == "BOUNDED_NEGATIVE"
    assert receipt["ttft_improvement_percent"] == pytest.approx(4.0)


def test_false_declared_verdict_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(
        tmp_path,
        treatment_ttft=96.0,
    )

    with pytest.raises(ValueError, match="declared verdict"):
        validate_result(result, contract, result_root)


def test_environment_identity_drift_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)
    _rewrite_json(
        result,
        result_root,
        "treatment",
        "environment",
        lambda value: value.update({"host_revision": "9" * 40}),
    )

    with pytest.raises(ValueError, match="environment identity"):
        validate_result(result, contract, result_root)


def test_output_token_drift_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)

    def mutate(value: dict) -> None:
        value["requests"][17]["output_token_ids_sha256"] = "0" * 64

    _rewrite_json(result, result_root, "treatment", "request-records", mutate)

    with pytest.raises(ValueError, match="output drifted"):
        validate_result(result, contract, result_root)


def test_cleanup_failure_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)
    _rewrite_json(
        result,
        result_root,
        "treatment",
        "cleanup",
        lambda value: value.update({"project_processes_after_exit": 1}),
    )

    with pytest.raises(ValueError, match="cleanup gate"):
        validate_result(result, contract, result_root)


def test_raw_median_metric_is_required(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)
    _rewrite_json(
        result,
        result_root,
        "baseline",
        "raw-benchmark-result",
        lambda value: value.pop("median_ttft_ms"),
    )

    with pytest.raises(ValueError, match="median_ttft_ms"):
        validate_result(result, contract, result_root)


def test_artifact_digest_drift_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)
    artifact = _artifact(result, "baseline", "service-stdout")
    (result_root / artifact["path"]).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact (size|digest) drifted"):
        validate_result(result, contract, result_root)


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)
    _artifact(result, "baseline", "service-stdout")["path"] = "../outside.log"

    with pytest.raises(ValueError, match="escapes the result root"):
        validate_result(result, contract, result_root)


def test_hardlink_reuse_fails_closed(tmp_path: Path) -> None:
    result, result_root, contract = _complete_result(tmp_path)
    source_artifact = _artifact(result, "baseline", "service-stdout")
    target_artifact = _artifact(result, "treatment", "service-stderr")
    source = result_root / source_artifact["path"]
    target = result_root / target_artifact["path"]
    target.unlink()
    os.link(source, target)
    target_artifact["size_bytes"] = source.stat().st_size
    target_artifact["sha256"] = _file_digest(source)

    with pytest.raises(ValueError, match="artifact file is reused"):
        validate_result(result, contract, result_root)


def _file_digest(path: Path) -> str:
    return _sha256(path.read_bytes())
