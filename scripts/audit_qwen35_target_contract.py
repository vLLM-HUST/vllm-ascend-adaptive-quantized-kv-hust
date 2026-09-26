#!/usr/bin/env python3
"""Derive the pinned Qwen3.5-35B-A3B C8 profile shape contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vllm-ascend-qwen35-target-contract-audit/v1"
MODEL_REPOSITORY = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
CONFIG_SHA256 = "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944"
EXPECTED_LAYER_TYPES = [
    "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
    for layer in range(40)
]


class AuditError(RuntimeError):
    """Raised when the pinned config no longer satisfies the target contract."""


def _expect(config: dict[str, Any], key: str, expected: Any) -> None:
    actual = config.get(key)
    if actual != expected:
        raise AuditError(f"unexpected {key}: expected {expected!r}, got {actual!r}")


def analyze_config(config: dict[str, Any], *, tp_size: int = 2) -> dict[str, Any]:
    _expect(config, "architectures", ["Qwen3_5MoeForConditionalGeneration"])
    _expect(config, "model_type", "qwen3_5_moe")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise AuditError("config.text_config is not a JSON object")
    expected = {
        "model_type": "qwen3_5_moe_text",
        "dtype": "bfloat16",
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
    }
    for key, value in expected.items():
        _expect(text_config, key, value)
    _expect(text_config, "layer_types", EXPECTED_LAYER_TYPES)
    if tp_size <= 0:
        raise AuditError("tp_size must be positive")
    num_kv_heads = int(text_config["num_key_value_heads"])
    if num_kv_heads % tp_size:
        raise AuditError("num_key_value_heads is not divisible by tp_size")

    head_dim = int(text_config["head_dim"])
    full_attention_layers = [
        layer
        for layer, layer_type in enumerate(EXPECTED_LAYER_TYPES)
        if layer_type == "full_attention"
    ]
    linear_attention_layers = [
        layer
        for layer, layer_type in enumerate(EXPECTED_LAYER_TYPES)
        if layer_type == "linear_attention"
    ]
    global_channels = num_kv_heads * head_dim
    local_kv_heads = num_kv_heads // tp_size
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "pinned-public-config-contract-audit",
        "status": "PASS",
        "source": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "file": "config.json",
            "sha256": CONFIG_SHA256,
        },
        "model": {
            **expected,
            "full_attention_layers": full_attention_layers,
            "linear_attention_layers": linear_attention_layers,
        },
        "c8_profile_contract": {
            "covered_layers": full_attention_layers,
            "excluded_layers": linear_attention_layers,
            "profile_tensors_per_layer": 4,
            "profile_tensor_count": len(full_attention_layers) * 4,
            "global_channels_per_tensor": global_channels,
            "tp_size": tp_size,
            "tp_local_kv_heads": local_kv_heads,
            "tp_local_channels_per_tensor": local_kv_heads * head_dim,
            "offset_policy": "explicit-zero-only",
        },
        "runtime_compatible": None,
        "npu_started": False,
        "performance_claim": False,
        "limitations": [
            "This derives a shape and layer-coverage contract from config.json.",
            "It does not provide calibrated scale values or a C8 model artifact.",
            "It does not establish Host/provider activation or runtime correctness.",
        ],
    }


def audit_config(path: Path, *, tp_size: int = 2) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read pinned config: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CONFIG_SHA256:
        raise AuditError(
            f"config SHA256 mismatch: expected {CONFIG_SHA256}, got {digest}"
        )
    if not isinstance(config, dict):
        raise AuditError("config root is not a JSON object")
    return analyze_config(config, tp_size=tp_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_config(args.config, tp_size=args.tp_size)
    except (AuditError, KeyError, TypeError, ValueError) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "classification": "pinned-public-config-contract-audit",
            "status": "FAIL",
            "source": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "file": "config.json",
                "sha256": CONFIG_SHA256,
            },
            "npu_started": False,
            "performance_claim": False,
            "error": str(error),
        }
        exit_code = 2
    else:
        exit_code = 0
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
