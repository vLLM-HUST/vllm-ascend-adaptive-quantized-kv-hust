import copy
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
    main,
    validate_contract,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "docs" / "evidence" / "QWEN35_CORRECTNESS_MATRIX_CONTRACT_20260929.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _authorized_contract() -> dict:
    contract = _contract()
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


def _pass_record(record_id: str, digest: str = "f" * 64) -> dict:
    return {
        "id": record_id,
        "status": "PASS",
        "raw_receipt_sha256": digest,
    }


def _receipt(contract: dict) -> dict:
    inputs = contract["runtime_inputs"]
    cases = []
    for case_contract in CASE_CONTRACTS:
        case = _pass_record(case_contract["id"])
        case["observed_path"] = case_contract["expected_path"]
        case["numerical_oracle"] = {
            "status": "PASS",
            "output_token_ids_match": True,
            "max_absolute_error": 0.0,
            "max_relative_error": 0.0,
        }
        cases.append(case)
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
            "phases": [_pass_record(phase) for phase in MANAGER_PHASES],
        },
        "worker_coverage": {
            "ranks": [
                {
                    "rank": rank,
                    "status": "PASS",
                    "profile_shard_sha256": digest,
                    "full_attention_layers": FULL_ATTENTION_LAYERS,
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
            "raw_receipt_sha256": "f" * 64,
        },
        "fallback": {
            "status": "PASS",
            "native_path_restored": True,
            "raw_receipt_sha256": "f" * 64,
        },
        "rollback_cleanup": {
            "rollback": "PASS",
            "uninstall": "PASS",
            "service_exit": "PASS",
            "device_release": "PASS",
            "project_processes_after_exit": 0,
            "device_memory_released": True,
            "raw_receipt_sha256": "f" * 64,
        },
    }


def test_preregistered_contract_is_valid_but_not_authorized() -> None:
    result = validate_contract(_contract())

    assert result["status"] == "PASS"
    assert result["execution_authorized"] is False
    assert result["case_count"] == 8


def test_premature_authorization_fails_closed() -> None:
    contract = _contract()
    contract["status"] = "authorized_for_correctness"
    contract["execution_authorized"] = True

    with pytest.raises(ValueError, match="host_revision"):
        validate_contract(contract)


def test_complete_correctness_receipt_passes() -> None:
    contract = _authorized_contract()

    result = validate_receipt(contract, _receipt(contract))

    assert result["correctness_complete"] is True
    assert result["performance_authorized"] is False
    assert result["case_count"] == 8


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("identity", "runtime identities"),
        ("manager", "Manager record order"),
        ("case", "correctness case record order"),
        ("dispatch", "dispatch path"),
        ("numerical", "numerical tolerance"),
        ("graph", "first_replay"),
        ("cleanup", "project processes"),
    ],
)
def test_incomplete_or_drifted_receipt_fails_closed(mutation: str, match: str) -> None:
    contract = _authorized_contract()
    receipt = _receipt(contract)
    if mutation == "identity":
        receipt["identities"]["host_revision"] = "9" * 40
    elif mutation == "manager":
        receipt["manager"]["phases"].pop()
    elif mutation == "case":
        receipt["cases"].pop(3)
    elif mutation == "dispatch":
        receipt["cases"][2]["observed_path"] = "native"
    elif mutation == "numerical":
        receipt["cases"][2]["numerical_oracle"]["max_absolute_error"] = 1.0
    elif mutation == "graph":
        receipt["graph_lifecycle"]["first_replay"] = "FAIL"
    else:
        receipt["rollback_cleanup"]["project_processes_after_exit"] = 1

    with pytest.raises(ValueError, match=match):
        validate_receipt(contract, receipt)


def test_contract_target_drift_fails_closed() -> None:
    contract = copy.deepcopy(_contract())
    contract["target"]["num_speculative_tokens"] = 3

    with pytest.raises(ValueError, match="target configuration"):
        validate_contract(contract)


def test_cli_require_authorized_rejects_checked_in_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_qwen35_correctness.py",
            str(CONTRACT_PATH),
            "--require-authorized",
        ],
    )

    assert main() == 2
    assert "FAIL_CLOSED" in capsys.readouterr().out
