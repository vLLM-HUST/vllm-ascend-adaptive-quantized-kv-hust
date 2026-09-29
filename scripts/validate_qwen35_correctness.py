#!/usr/bin/env python3
"""Validate the Qwen3.5 correctness contract and completed runtime receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "adaptive-quantized-kv-qwen35-correctness-contract/v1"
RECEIPT_SCHEMA = "adaptive-quantized-kv-qwen35-correctness-receipt/v1"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXTENSION_ID = "org.vllm-hust.ascend-adaptive-quantized-kv"
MANAGER_PHASES = ["discover", "check", "plan", "run"]
CASE_CONTRACTS = [
    {
        "id": "provider-disabled-baseline-equivalence",
        "provider": "disabled",
        "expected_path": "native",
        "exact_output_required": True,
    },
    {
        "id": "enabled-ineligible-native-fallback",
        "provider": "enabled_ineligible",
        "expected_path": "native_fallback",
        "exact_output_required": True,
    },
    {
        "id": "apc-continuing-prefill-eager",
        "provider": "enabled_eligible",
        "expected_path": "paged_int8_continuing_prefill",
        "exact_output_required": False,
    },
    {
        "id": "apc-continuing-prefill-full-and-piecewise-replay",
        "provider": "enabled_eligible",
        "expected_path": "paged_int8_capture_replay",
        "exact_output_required": False,
    },
    {
        "id": "apc-mtp2-sync",
        "provider": "enabled_eligible",
        "expected_path": "paged_int8_continuing_prefill",
        "exact_output_required": False,
    },
    {
        "id": "apc-mtp2-async",
        "provider": "enabled_eligible",
        "expected_path": "paged_int8_continuing_prefill",
        "exact_output_required": False,
    },
    {
        "id": "mixed-decode-continuing-prefill",
        "provider": "enabled_eligible",
        "expected_path": "mixed_native_and_paged_int8",
        "exact_output_required": False,
    },
    {
        "id": "unsupported-cleanup-restart",
        "provider": "enabled_unsupported",
        "expected_path": "fail_closed_then_native_restart",
        "exact_output_required": True,
    },
]
CASE_IDS = [case["id"] for case in CASE_CONTRACTS]
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_sha(value: object, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase digest")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def canonical_contract_sha256(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_target(target: dict[str, Any]) -> None:
    expected = {
        "model": "Qwen/Qwen3.5-35B-A3B",
        "model_revision": MODEL_REVISION,
        "weight_precision": "BF16",
        "tensor_parallel_size": 2,
        "automatic_prefix_caching": True,
        "num_speculative_tokens": 2,
        "async_scheduling": True,
        "compilation_mode": "FULL_AND_PIECEWISE",
        "accelerator": "Ascend 910B2",
    }
    if target != expected:
        raise ValueError("target configuration drifted")


def _validate_static_contract(contract: dict[str, Any]) -> None:
    _validate_target(_require_mapping(contract, "target"))
    if contract.get("manager_phases") != MANAGER_PHASES:
        raise ValueError("manager phase order drifted")

    if _require(contract, "matrix") != CASE_CONTRACTS:
        raise ValueError("correctness matrix contract drifted")

    workers = _require_mapping(contract, "worker_contract")
    if workers.get("required_ranks") != [0, 1]:
        raise ValueError("TP2 worker ranks drifted")
    if workers.get("full_attention_layers") != FULL_ATTENTION_LAYERS:
        raise ValueError("hybrid full-attention layer scope drifted")
    if workers.get("linear_attention_layers_rejected") != 30:
        raise ValueError("linear-attention rejection count drifted")

    policy = _require_mapping(contract, "resource_policy")
    expected_policy = {
        "exclusive_device_count": 2,
        "allow_co_tenancy": False,
        "max_session_minutes": 45,
        "release_on_any_failure": True,
    }
    if policy != expected_policy:
        raise ValueError("resource policy drifted")


def _validate_authorized_inputs(inputs: dict[str, Any]) -> None:
    for name in ("host_revision", "plugin_revision", "manager_revision"):
        _require_sha(inputs.get(name), name, GIT_SHA_RE)
    for name in ("model_weight_sha256", "tokenizer_sha256"):
        _require_sha(inputs.get(name), name, SHA256_RE)

    profile = _require_mapping(inputs, "profile")
    _require_sha(profile.get("logical_tensor_sha256"), "profile digest", SHA256_RE)
    shards = profile.get("tp2_shard_sha256")
    if not isinstance(shards, list) or len(shards) != 2:
        raise ValueError("profile must contain exactly two TP2 shard digests")
    for index, digest in enumerate(shards):
        _require_sha(digest, f"profile shard {index}", SHA256_RE)
    _require_text(profile.get("provenance_url"), "profile provenance URL")

    lease = _require_mapping(inputs, "resource_lease")
    for name in ("lease_id", "window_start", "window_end", "cleanup_owner"):
        _require_text(lease.get(name), f"resource lease {name}")

    oracle = _require_mapping(inputs, "numerical_oracle")
    for name in ("absolute_tolerance", "relative_tolerance"):
        value = oracle.get(name)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"numerical oracle {name} must be non-negative")
    _require_text(oracle.get("approval_source"), "numerical oracle approval source")


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unexpected correctness contract schema")
    _validate_static_contract(contract)

    authorized = contract.get("execution_authorized") is True
    expected_status = (
        "authorized_for_correctness" if authorized else "preregistered_blocked_on_h3"
    )
    if contract.get("status") != expected_status:
        raise ValueError("contract status and authorization disagree")
    if authorized:
        _validate_authorized_inputs(_require_mapping(contract, "runtime_inputs"))

    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": "PASS",
        "execution_authorized": authorized,
        "contract_sha256": canonical_contract_sha256(contract),
        "case_count": len(CASE_IDS),
    }


def _validate_pass_records(
    records: object,
    expected_ids: list[str],
    label: str,
) -> None:
    if not isinstance(records, list):
        raise ValueError(f"{label} records must be a list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} records must contain objects")
    if [record.get("id") for record in records] != expected_ids:
        raise ValueError(f"{label} record order or coverage drifted")
    for record in records:
        if record.get("status") != "PASS":
            raise ValueError(f"{label} {record.get('id')} did not pass")
        _require_sha(record.get("raw_receipt_sha256"), label, SHA256_RE)


def validate_receipt(
    contract: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    contract_result = validate_contract(contract)
    if contract_result["execution_authorized"] is not True:
        raise ValueError("correctness execution was not authorized")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unexpected correctness receipt schema")
    if receipt.get("status") != "PASS":
        raise ValueError("correctness receipt status is not PASS")
    if receipt.get("contract_sha256") != contract_result["contract_sha256"]:
        raise ValueError("correctness receipt is not bound to this contract")

    inputs = _require_mapping(contract, "runtime_inputs")
    identities = _require_mapping(receipt, "identities")
    expected_identities = {
        "host_revision": inputs["host_revision"],
        "plugin_revision": inputs["plugin_revision"],
        "manager_revision": inputs["manager_revision"],
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": inputs["model_weight_sha256"],
        "tokenizer_sha256": inputs["tokenizer_sha256"],
        "profile_logical_tensor_sha256": inputs["profile"]["logical_tensor_sha256"],
    }
    if identities != expected_identities:
        raise ValueError("runtime identities do not match the authorized contract")

    resource = _require_mapping(receipt, "resource")
    if resource.get("lease_id") != inputs["resource_lease"]["lease_id"]:
        raise ValueError("resource lease identity drifted")
    if resource.get("exclusive_device_count") != 2:
        raise ValueError("exactly two exclusive devices are required")
    if resource.get("co_tenant_processes") != 0:
        raise ValueError("shared-device execution is not accepted")
    if resource.get("exclusive") is not True:
        raise ValueError("resource lease was not exclusive")
    session_minutes = resource.get("session_minutes")
    if (
        not isinstance(session_minutes, (int, float))
        or not math.isfinite(session_minutes)
        or session_minutes <= 0
        or session_minutes > contract["resource_policy"]["max_session_minutes"]
    ):
        raise ValueError("resource session exceeded the authorized bound")

    manager = _require_mapping(receipt, "manager")
    if manager.get("extension_id") != EXTENSION_ID:
        raise ValueError("unexpected Manager extension identity")
    _validate_pass_records(manager.get("phases"), MANAGER_PHASES, "Manager")

    coverage = _require_mapping(receipt, "worker_coverage")
    ranks = coverage.get("ranks")
    if not isinstance(ranks, list) or [rank.get("rank") for rank in ranks] != [0, 1]:
        raise ValueError("TP2 worker coverage is incomplete")
    for rank, expected_digest in zip(
        ranks, inputs["profile"]["tp2_shard_sha256"], strict=True
    ):
        if rank.get("status") != "PASS":
            raise ValueError(f"worker rank {rank.get('rank')} did not pass")
        if rank.get("profile_shard_sha256") != expected_digest:
            raise ValueError("worker profile shard identity drifted")
        if rank.get("full_attention_layers") != FULL_ATTENTION_LAYERS:
            raise ValueError("worker full-attention coverage drifted")
    if coverage.get("linear_attention_layers_rejected") != 30:
        raise ValueError("linear-attention layers were not rejected")

    _validate_pass_records(receipt.get("cases"), CASE_IDS, "correctness case")
    oracle_contract = inputs["numerical_oracle"]
    for case, expected_case in zip(receipt["cases"], CASE_CONTRACTS, strict=True):
        if case.get("observed_path") != expected_case["expected_path"]:
            raise ValueError(f"dispatch path drifted for {case['id']}")
        oracle = _require_mapping(case, "numerical_oracle")
        if oracle.get("status") != "PASS":
            raise ValueError(f"numerical oracle failed for {case['id']}")
        if oracle.get("output_token_ids_match") is not True:
            raise ValueError(f"output tokens drifted for {case['id']}")
        absolute_error = oracle.get("max_absolute_error")
        relative_error = oracle.get("max_relative_error")
        for value, label in (
            (absolute_error, "absolute"),
            (relative_error, "relative"),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid {label} error for {case['id']}")
        if expected_case["exact_output_required"]:
            if absolute_error != 0 or relative_error != 0:
                raise ValueError(f"exact output equivalence failed for {case['id']}")
        elif (
            absolute_error > oracle_contract["absolute_tolerance"]
            or relative_error > oracle_contract["relative_tolerance"]
        ):
            raise ValueError(f"numerical tolerance exceeded for {case['id']}")

    graph = _require_mapping(receipt, "graph_lifecycle")
    for stage in ("capture", "first_replay", "subsequent_replay"):
        if graph.get(stage) != "PASS":
            raise ValueError(f"graph lifecycle {stage} did not pass")
    if graph.get("provider_identity_stable") is not True:
        raise ValueError("provider identity changed across graph replay")
    _require_sha(graph.get("raw_receipt_sha256"), "graph lifecycle", SHA256_RE)

    fallback = _require_mapping(receipt, "fallback")
    if (
        fallback.get("status") != "PASS"
        or fallback.get("native_path_restored") is not True
    ):
        raise ValueError("native fallback did not pass")
    _require_sha(fallback.get("raw_receipt_sha256"), "fallback", SHA256_RE)

    cleanup = _require_mapping(receipt, "rollback_cleanup")
    for stage in ("rollback", "uninstall", "service_exit", "device_release"):
        if cleanup.get(stage) != "PASS":
            raise ValueError(f"cleanup stage {stage} did not pass")
    if cleanup.get("project_processes_after_exit") != 0:
        raise ValueError("project processes remained after cleanup")
    if cleanup.get("device_memory_released") is not True:
        raise ValueError("device memory was not released")
    _require_sha(cleanup.get("raw_receipt_sha256"), "cleanup", SHA256_RE)

    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "correctness_complete": True,
        "contract_sha256": contract_result["contract_sha256"],
        "case_count": len(CASE_IDS),
        "performance_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--require-authorized", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = validate_contract(contract)
    if args.require_authorized and result["execution_authorized"] is not True:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "error": "correctness execution is not authorized",
                    "contract_sha256": result["contract_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if args.receipt is not None:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        result = validate_receipt(contract, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
