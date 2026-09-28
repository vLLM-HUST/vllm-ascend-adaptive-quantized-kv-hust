#!/usr/bin/env python3
"""Validate the frozen Qwen3.5 prefix-repetition pilot contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "adaptive-quantized-kv-qwen35-prefix-pilot/v1"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
BENCHMARK_COMMIT = "47c12e518a9188bb714bb6193d00260a7741841b"


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    if contract.get("status") != "preregistered_blocked_on_correctness":
        raise ValueError("pilot must remain blocked until correctness gates close")

    target = _require(contract, "target")
    expected_target = {
        "model": "Qwen/Qwen3.5-35B-A3B",
        "model_revision": MODEL_REVISION,
        "weight_precision": "BF16",
        "tensor_parallel_size": 2,
        "automatic_prefix_caching": True,
        "num_speculative_tokens": 2,
        "async_scheduling": True,
        "compilation_mode": "FULL_AND_PIECEWISE",
        "accelerator": "Ascend 910B2",
        "exclusive_accelerator_count": 2,
    }
    if target != expected_target:
        raise ValueError("target configuration drifted")

    benchmark = _require(contract, "benchmark")
    if benchmark.get("repository") != "vLLM-HUST/vllm-hust-benchmark":
        raise ValueError("unexpected benchmark repository")
    if benchmark.get("commit") != BENCHMARK_COMMIT:
        raise ValueError("benchmark commit drifted")
    if benchmark.get("scenario") != "prefix-repetition-online-2chip":
        raise ValueError("benchmark scenario drifted")
    expected_defaults = {
        "backend": "vllm",
        "endpoint": "/v1/completions",
        "dataset_name": "prefix_repetition",
        "num_prompts": 200,
        "input_len": 4096,
        "output_len": 256,
    }
    if benchmark.get("defaults") != expected_defaults:
        raise ValueError("benchmark defaults drifted")

    arms = _require(contract, "arms")
    if arms.get("baseline", {}).get("kv_cache_precision") != "BF16":
        raise ValueError("baseline must use BF16 KV")
    treatment = arms.get("treatment", {})
    if treatment.get("kv_cache_precision") != "INT8":
        raise ValueError("treatment must use INT8 KV")
    if treatment.get("profile") != "revision_bound_calibrated_zero_offset":
        raise ValueError("treatment profile is not fail closed")

    execution = _require(contract, "execution")
    if execution.get("order") != ["baseline", "treatment"]:
        raise ValueError("pilot order drifted")
    if execution.get("max_minutes_per_arm") != 30:
        raise ValueError("resource bound drifted")
    if execution.get("release_devices_after_each_arm") is not True:
        raise ValueError("device release gate is required")

    gates = _require(contract, "gates")
    primary = gates.get("primary_metric", {})
    if primary != {
        "name": "median_ttft_ms",
        "direction": "lower",
        "minimum_relative_improvement_percent": 5.0,
    }:
        raise ValueError("primary metric drifted")
    guardrails = gates.get("guardrails", {})
    if guardrails.get("successful_requests") != 200:
        raise ValueError("request completeness gate drifted")
    if guardrails.get("error_rate_percent_max") != 0.0:
        raise ValueError("error gate drifted")
    if guardrails.get("no_correctness_or_graph_failure") is not True:
        raise ValueError("correctness gate is required")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "execution_authorized": False,
        "model_revision": MODEL_REVISION,
        "benchmark_commit": BENCHMARK_COMMIT,
        "scenario": benchmark["scenario"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    print(json.dumps(validate_contract(contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
