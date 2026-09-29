import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_qwen35_correctness import (
    CASE_CONTRACTS,
    EXTENSION_ID,
    FULL_ATTENTION_LAYERS,
    MANAGER_PHASES,
    MODEL_REVISION,
    RECEIPT_SCHEMA,
    canonical_contract_sha256,
)
from scripts.validate_qwen35_evidence_bundle import (
    ARTIFACT_IDS,
    canonical_bundle_sha256,
    main,
    validate_bundle,
    validate_bundle_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "docs" / "evidence" / "QWEN35_CORRECTNESS_MATRIX_CONTRACT_20260929.json"
)
BUNDLE_PATH = (
    ROOT / "docs" / "evidence" / "QWEN35_P1_EVIDENCE_BUNDLE_CONTRACT_20260929.json"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authorized_contract() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["status"] = "authorized_for_correctness"
    contract["execution_authorized"] = True
    inputs = contract["runtime_inputs"]
    inputs.update(
        {
            "host_revision": "1" * 40,
            "plugin_revision": "2" * 40,
            "manager_revision": "3" * 40,
            "model_weight_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
        }
    )
    inputs["profile"] = {
        "logical_tensor_sha256": "c" * 64,
        "tp2_shard_sha256": ["d" * 64, "e" * 64],
        "provenance_url": "https://example.invalid/profile-receipt",
    }
    inputs["resource_lease"] = {
        "lease_id": "lease-1",
        "window_start": "2026-09-29T12:00:00Z",
        "window_end": "2026-09-29T12:45:00Z",
        "cleanup_owner": "test-owner",
    }
    inputs["numerical_oracle"] = {
        "absolute_tolerance": 0.01,
        "relative_tolerance": 0.01,
        "approval_source": "https://example.invalid/oracle-decision",
    }
    return contract


def _receipt(contract: dict, digests: dict[str, str]) -> dict:
    inputs = contract["runtime_inputs"]
    cases = []
    for case_contract in CASE_CONTRACTS:
        case_id = case_contract["id"]
        cases.append(
            {
                "id": case_id,
                "status": "PASS",
                "raw_receipt_sha256": digests[f"case-{case_id}"],
                "observed_path": case_contract["expected_path"],
                "numerical_oracle": {
                    "status": "PASS",
                    "output_token_ids_match": True,
                    "max_absolute_error": 0.0,
                    "max_relative_error": 0.0,
                },
            }
        )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "contract_sha256": canonical_contract_sha256(contract),
        "identities": {
            "host_revision": inputs["host_revision"],
            "plugin_revision": inputs["plugin_revision"],
            "manager_revision": inputs["manager_revision"],
            "model_revision": MODEL_REVISION,
            "model_weight_sha256": inputs["model_weight_sha256"],
            "tokenizer_sha256": inputs["tokenizer_sha256"],
            "profile_logical_tensor_sha256": inputs["profile"]["logical_tensor_sha256"],
        },
        "resource": {
            "lease_id": inputs["resource_lease"]["lease_id"],
            "exclusive_device_count": 2,
            "co_tenant_processes": 0,
            "exclusive": True,
            "session_minutes": 30,
        },
        "manager": {
            "extension_id": EXTENSION_ID,
            "phases": [
                {
                    "id": phase,
                    "status": "PASS",
                    "raw_receipt_sha256": digests[f"manager-{phase}"],
                }
                for phase in MANAGER_PHASES
            ],
        },
        "worker_coverage": {
            "ranks": [
                {
                    "rank": rank,
                    "status": "PASS",
                    "profile_shard_sha256": digest,
                    "full_attention_layers": FULL_ATTENTION_LAYERS,
                    "raw_receipt_sha256": digests[f"worker-rank-{rank}"],
                }
                for rank, digest in enumerate(inputs["profile"]["tp2_shard_sha256"])
            ],
            "linear_attention_layers_rejected": 30,
        },
        "cases": cases,
        "graph_lifecycle": {
            "capture": "PASS",
            "first_replay": "PASS",
            "subsequent_replay": "PASS",
            "provider_identity_stable": True,
            "raw_receipt_sha256": digests["graph-lifecycle"],
        },
        "fallback": {
            "status": "PASS",
            "native_path_restored": True,
            "raw_receipt_sha256": digests["native-fallback"],
        },
        "rollback_cleanup": {
            "rollback": "PASS",
            "uninstall": "PASS",
            "service_exit": "PASS",
            "device_release": "PASS",
            "project_processes_after_exit": 0,
            "device_memory_released": True,
            "raw_receipt_sha256": digests["rollback-cleanup"],
        },
    }


def _complete_bundle(tmp_path: Path) -> tuple[dict, Path]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    bundle["status"] = "correctness_evidence_complete"
    bundle["bundle_complete"] = True
    bundle["claim_boundary"]["correctness_evidence_complete"] = True

    payloads: dict[str, bytes] = {}
    for artifact_id in ARTIFACT_IDS:
        if artifact_id not in {"correctness-contract", "correctness-receipt"}:
            payloads[artifact_id] = f"raw evidence for {artifact_id}\n".encode()
    digests = {artifact_id: _sha256(data) for artifact_id, data in payloads.items()}

    contract = _authorized_contract()
    receipt = _receipt(contract, digests)
    payloads["correctness-contract"] = (
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    ).encode()
    payloads["correctness-receipt"] = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode()

    for artifact in bundle["artifacts"]:
        artifact_id = artifact["id"]
        relative_path = f"raw/{artifact_id}.txt"
        path = run_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[artifact_id])
        artifact["path"] = relative_path
        artifact["size_bytes"] = len(payloads[artifact_id])
        artifact["sha256"] = _sha256(payloads[artifact_id])

    bundle["correctness_contract_sha256"] = canonical_contract_sha256(contract)
    bundle["correctness_receipt_sha256"] = _sha256(payloads["correctness-receipt"])
    return bundle, run_root


def test_preregistered_bundle_is_valid_but_incomplete() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    result = validate_bundle_contract(bundle)

    assert result["status"] == "PASS"
    assert result["bundle_complete"] is False
    assert result["artifact_count"] == 26
    assert result["performance_authorized"] is False


def test_complete_bundle_passes_and_keeps_claims_closed(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)

    result = validate_bundle(bundle, run_root)

    assert result["correctness_evidence_complete"] is True
    assert result["artifact_count"] == 26
    assert result["total_bytes"] > 0
    assert result["performance_authorized"] is False
    assert result["quality_claim_authorized"] is False
    assert result["public_surface_authorized"] is False


def test_modified_raw_file_fails_digest_check(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)
    (run_root / "raw" / "manager-run.txt").write_text("modified\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="artifact size drifted|artifact digest drifted"
    ):
        validate_bundle(bundle, run_root)


def test_missing_raw_file_is_rejected(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)
    (run_root / "raw" / "manager-plan.txt").unlink()

    with pytest.raises(ValueError, match="artifact is missing: raw/manager-plan.txt"):
        validate_bundle(bundle, run_root)


def test_rehashed_replacement_still_fails_receipt_binding(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)
    replacement = b"replacement evidence\n"
    path = run_root / "raw" / "manager-run.txt"
    path.write_bytes(replacement)
    artifact = next(item for item in bundle["artifacts"] if item["id"] == "manager-run")
    artifact["size_bytes"] = len(replacement)
    artifact["sha256"] = _sha256(replacement)

    with pytest.raises(ValueError, match="receipt binding drifted: manager-run"):
        validate_bundle(bundle, run_root)


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)
    artifact = next(
        item for item in bundle["artifacts"] if item["id"] == "service-stdout"
    )
    original = run_root / artifact["path"]
    target = run_root / "actual-service-stdout.txt"
    original.rename(target)
    original.symlink_to(target)

    with pytest.raises(ValueError, match="contains a symlink"):
        validate_bundle(bundle, run_root)


def test_duplicate_artifact_path_is_rejected(tmp_path: Path) -> None:
    bundle, _ = _complete_bundle(tmp_path)
    bundle["artifacts"][1]["path"] = bundle["artifacts"][0]["path"]

    with pytest.raises(ValueError, match="paths must be unique"):
        validate_bundle_contract(bundle)


def test_hardlinked_artifact_file_is_rejected(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)
    source = run_root / bundle["artifacts"][0]["path"]
    target_artifact = bundle["artifacts"][1]
    target = run_root / target_artifact["path"]
    target.unlink()
    target.hardlink_to(source)
    target_artifact["size_bytes"] = source.stat().st_size
    target_artifact["sha256"] = _sha256(source.read_bytes())

    with pytest.raises(ValueError, match="artifact file is reused"):
        validate_bundle(bundle, run_root)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    bundle, run_root = _complete_bundle(tmp_path)
    artifact = bundle["artifacts"][0]
    artifact["path"] = "../outside.txt"

    with pytest.raises(ValueError, match="escapes the bundle root"):
        validate_bundle(bundle, run_root)


def test_cli_require_complete_rejects_checked_in_template(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_qwen35_evidence_bundle.py",
            str(BUNDLE_PATH),
            "--require-complete",
        ],
    )

    assert main() == 2
    assert "FAIL_CLOSED" in capsys.readouterr().out


def test_bundle_digest_is_deterministic() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert canonical_bundle_sha256(bundle) == canonical_bundle_sha256(
        copy.deepcopy(bundle)
    )
