#!/usr/bin/env python3
"""Audit a BF16-weight/C8-KV ModelSlim artifact without importing Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vllm-ascend-c8-profile-artifact-audit/v1"
PROFILE_CONTENT_DIGEST_ALGORITHM = "sha256-framed-logical-tensors-v1"
KV_NAME = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<kind>[kv])_proj\.kv_cache_(?P<field>scale|offset)$"
)
REQUIRED_FIELDS = {"k_scale", "k_offset", "v_scale", "v_offset"}


class AuditError(RuntimeError):
    """Raised when an artifact cannot satisfy the profile contract."""


def _reject_constant(value: str) -> None:
    raise AuditError(f"JSON contains non-finite value {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AuditError(f"cannot read JSON file {path.name}: {error}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise AuditError(f"cannot read JSON file {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object in {path.name}")
    return value


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    if nested is None:
        return config
    if not isinstance(nested, dict):
        raise AuditError("config.text_config is not a JSON object")
    return nested


def _attention_layers(
    config: dict[str, Any], num_hidden_layers: int
) -> tuple[list[int], list[str] | None]:
    raw_layer_types = config.get("layer_types")
    if raw_layer_types is None:
        return list(range(num_hidden_layers)), None
    if not isinstance(raw_layer_types, list) or not all(
        isinstance(value, str) for value in raw_layer_types
    ):
        raise AuditError("config.layer_types is not a string list")
    if len(raw_layer_types) != num_hidden_layers:
        raise AuditError("config.layer_types length does not match num_hidden_layers")
    supported = {"full_attention", "linear_attention"}
    unknown = sorted(set(raw_layer_types) - supported)
    if unknown:
        raise AuditError(f"unsupported hybrid layer types: {unknown}")
    attention_layers = [
        index
        for index, layer_type in enumerate(raw_layer_types)
        if layer_type == "full_attention"
    ]
    if not attention_layers:
        raise AuditError("hybrid model declares no full_attention layers")
    return attention_layers, raw_layer_types


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_path(model_dir: Path) -> Path:
    candidates = (
        model_dir / "quant_model_weights.safetensors.index.json",
        model_dir / "model.safetensors.index.json",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise AuditError(
            "expected exactly one supported safetensors index, found "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _shard_path(model_dir: Path, shard_name: str) -> Path:
    relative = Path(shard_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditError(f"unsafe shard path in weight_map: {shard_name!r}")
    root = model_dir.resolve()
    path = (model_dir / relative).resolve()
    if not path.is_relative_to(root):
        raise AuditError(f"shard escapes model directory: {shard_name!r}")
    if not path.is_file():
        raise AuditError(f"profile shard is missing: {shard_name!r}")
    return path


def _load_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise AuditError(f"truncated safetensors header in {path.name}")
        length = struct.unpack("<Q", raw_length)[0]
        if length > 128 * 1024 * 1024:
            raise AuditError(f"unreasonable safetensors header in {path.name}")
        raw_header = stream.read(length)
    try:
        header = json.loads(
            raw_header,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid safetensors header in {path.name}") from error
    if not isinstance(header, dict):
        raise AuditError(f"invalid safetensors header object in {path.name}")
    data_start = 8 + length
    _validate_header_ranges(path, data_start, header)
    return data_start, header


def _validate_header_ranges(
    path: Path,
    data_start: int,
    header: dict[str, Any],
) -> None:
    data_size = path.stat().st_size - data_start
    if data_size < 0:
        raise AuditError(f"safetensors header exceeds file size in {path.name}")
    ranges: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise AuditError(f"invalid tensor metadata for {name!r} in {path.name}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise AuditError(f"invalid tensor offsets for {name!r} in {path.name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise AuditError(
                f"tensor range is outside shard data for {name!r} in {path.name}"
            )
        ranges.append((start, end, name))
    previous_end = 0
    previous_name: str | None = None
    for start, end, name in sorted(ranges):
        if start < previous_end:
            raise AuditError(
                f"overlapping tensor ranges in {path.name}: "
                f"{previous_name!r} and {name!r}"
            )
        previous_end = end
        previous_name = name


def _decode_values(dtype: str, raw: bytes) -> list[float]:
    if dtype == "BF16":
        if len(raw) % 2:
            raise AuditError("BF16 tensor byte length is not even")
        values = struct.unpack(f"<{len(raw) // 2}H", raw)
        return [
            struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in values
        ]
    if dtype == "F32":
        if len(raw) % 4:
            raise AuditError("F32 tensor byte length is not divisible by four")
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    raise AuditError(f"unsupported profile tensor dtype: {dtype}")


def _read_tensor(
    path: Path,
    data_start: int,
    metadata: dict[str, Any],
) -> tuple[list[int], str, list[float], bytes]:
    try:
        shape = [int(value) for value in metadata["shape"]]
        dtype = str(metadata["dtype"])
        start, end = (int(value) for value in metadata["data_offsets"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(f"invalid tensor metadata in {path.name}") from error
    if start < 0 or end < start:
        raise AuditError(f"invalid tensor offsets in {path.name}")
    with path.open("rb") as stream:
        stream.seek(data_start + start)
        raw = stream.read(end - start)
    values = _decode_values(dtype, raw)
    if math.prod(shape) != len(values):
        raise AuditError(f"tensor shape/byte mismatch in {path.name}")
    return shape, dtype, values, raw


def _profile_digest_component(
    *,
    name: str,
    dtype: str,
    shape: list[int],
    raw: bytes,
) -> tuple[bytes, str]:
    """Frame one logical tensor for the canonical profile receipt stream."""

    descriptor = json.dumps(
        {"name": name, "dtype": dtype, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tensor_digest = hashlib.sha256(raw).hexdigest()
    component = b"".join(
        (
            len(descriptor).to_bytes(8, "big"),
            descriptor,
            len(raw).to_bytes(8, "big"),
            raw,
        )
    )
    return component, tensor_digest


def audit_profile(model_dir: Path, *, label: str, tp_size: int = 1) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    description_path = model_dir / "quant_model_description.json"
    index_path = _index_path(model_dir)
    config = _read_json(config_path)
    description = _read_json(description_path)
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AuditError("safetensors index has no weight_map object")
    if description.get("model_quant_type") != "FLOAT":
        raise AuditError("profile is not backed by FLOAT/BF16 model weights")
    if description.get("kv_cache_type") != "C8":
        raise AuditError("profile does not declare kv_cache_type=C8")

    model_config = _text_config(config)
    num_heads = int(model_config["num_attention_heads"])
    num_kv_heads = int(model_config["num_key_value_heads"])
    hidden_size = int(model_config["hidden_size"])
    num_hidden_layers = int(model_config["num_hidden_layers"])
    raw_head_size = model_config.get("head_dim")
    if raw_head_size is None:
        if hidden_size % num_heads:
            raise AuditError(
                "hidden_size is not divisible by num_attention_heads and head_dim "
                "is absent"
            )
        head_size = hidden_size // num_heads
    else:
        head_size = int(raw_head_size)
    if tp_size <= 0:
        raise AuditError("tp_size must be positive")
    if num_kv_heads % tp_size:
        raise AuditError("num_key_value_heads is not divisible by tp_size")
    expected_attention_layers, layer_types = _attention_layers(
        model_config, num_hidden_layers
    )
    expected_channels = num_kv_heads * head_size
    tp_local_channels = expected_channels // tp_size

    profile_names: dict[int, dict[str, str]] = {}
    for name, quant_type in description.items():
        match = KV_NAME.fullmatch(name)
        if match is None:
            continue
        if quant_type != "C8":
            raise AuditError(f"profile tensor is not declared C8: {name}")
        layer = int(match.group("layer"))
        field = f"{match.group('kind')}_{match.group('field')}"
        profile_names.setdefault(layer, {})[field] = name
    if not profile_names:
        raise AuditError("profile declares no dense-attention C8 layers")
    for layer, fields in profile_names.items():
        if set(fields) != REQUIRED_FIELDS:
            raise AuditError(
                f"layer {layer} has incomplete profile fields: {sorted(fields)}"
            )
    declared_profile_names = {
        name for fields in profile_names.values() for name in fields.values()
    }
    indexed_profile_names = {
        name for name in weight_map if KV_NAME.fullmatch(name) is not None
    }
    if indexed_profile_names != declared_profile_names:
        missing = sorted(declared_profile_names - indexed_profile_names)
        unexpected = sorted(indexed_profile_names - declared_profile_names)
        raise AuditError(
            "profile tensor names differ between description and weight_map: "
            f"missing={missing}, unexpected={unexpected}"
        )
    covered_layers = sorted(profile_names)
    if covered_layers != expected_attention_layers:
        missing = sorted(set(expected_attention_layers) - set(covered_layers))
        unexpected = sorted(set(covered_layers) - set(expected_attention_layers))
        raise AuditError(
            "profile coverage does not match dense-attention layers: "
            f"missing={missing}, unexpected={unexpected}"
        )

    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    profile_digest = hashlib.sha256()
    profile_digest.update(PROFILE_CONTENT_DIGEST_ALGORITHM.encode("ascii") + b"\0")
    for layer in sorted(profile_names):
        for field in sorted(REQUIRED_FIELDS):
            name = profile_names[layer][field]
            shard_name = weight_map.get(name)
            if not isinstance(shard_name, str):
                raise AuditError(f"profile tensor is absent from weight_map: {name}")
            shard = _shard_path(model_dir, shard_name)
            if shard_name not in headers:
                headers[shard_name] = _load_header(shard)
            data_start, header = headers[shard_name]
            metadata = header.get(name)
            if not isinstance(metadata, dict):
                raise AuditError(f"profile tensor is absent from shard header: {name}")
            shape, dtype, values, raw = _read_tensor(shard, data_start, metadata)
            if shape != [expected_channels]:
                raise AuditError(
                    f"profile tensor {name} has shape {shape}; "
                    f"expected [{expected_channels}]"
                )
            finite = all(math.isfinite(value) for value in values)
            if not finite:
                raise AuditError(f"profile tensor contains non-finite values: {name}")
            nonzero = sum(value != 0.0 for value in values)
            if field.endswith("offset") and nonzero:
                raise AuditError(
                    f"nonzero offset is unsupported by Host fallback: {name}"
                )
            if field.endswith("scale") and (
                nonzero != len(values) or min(values) <= 0.0
            ):
                raise AuditError(f"scale must be finite and strictly positive: {name}")
            digest_component, tensor_sha256 = _profile_digest_component(
                name=name,
                dtype=dtype,
                shape=shape,
                raw=raw,
            )
            profile_digest.update(digest_component)
            summaries.append(
                {
                    "layer": layer,
                    "field": field,
                    "dtype": dtype,
                    "shape": shape,
                    "minimum": min(values),
                    "maximum": max(values),
                    "nonzero": nonzero,
                    "elements": len(values),
                    "sha256": tensor_sha256,
                }
            )

    layers = covered_layers
    offsets = [item for item in summaries if item["field"].endswith("offset")]
    scales = [item for item in summaries if item["field"].endswith("scale")]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "read-only-local-development-profile-audit",
        "status": "PASS",
        "label": label,
        "runtime_compatible": None,
        "target_match": None,
        "npu_started": False,
        "performance_claim": False,
        "model": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "text_model_type": model_config.get("model_type"),
            "torch_dtype": model_config.get("torch_dtype", model_config.get("dtype")),
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_size": head_size,
            "layer_types": layer_types,
        },
        "profile": {
            "model_quant_type": description["model_quant_type"],
            "kv_cache_type": description["kv_cache_type"],
            "covered_layers": layers,
            "covered_layer_count": len(layers),
            "full_layer_coverage": layers == list(range(num_hidden_layers)),
            "expected_attention_layers": expected_attention_layers,
            "full_attention_layer_coverage": True,
            "channels_per_tensor": expected_channels,
            "tp_size": tp_size,
            "tp_local_channels_per_tensor": tp_local_channels,
            "offset_tensor_count": len(offsets),
            "offset_nonzero_elements": sum(item["nonzero"] for item in offsets),
            "scale_tensor_count": len(scales),
            "scale_minimum": min(item["minimum"] for item in scales),
            "scale_maximum": max(item["maximum"] for item in scales),
            "profile_content_digest_algorithm": PROFILE_CONTENT_DIGEST_ALGORITHM,
            "profile_content_sha256": profile_digest.hexdigest(),
            "tensor_digests": [
                {
                    "layer": item["layer"],
                    "field": item["field"],
                    "sha256": item["sha256"],
                }
                for item in summaries
            ],
        },
        "metadata_files": {
            config_path.name: _sha256(config_path),
            description_path.name: _sha256(description_path),
            index_path.name: _sha256(index_path),
        },
        "limitations": [
            "This validates profile structure and values, not calibration quality.",
            "The audit does not load model weights, import Torch, or run an NPU.",
            "A development fixture does not establish identity or correctness for "
            "the Qwen3.5-35B-A3B target.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_profile(args.model_dir, label=args.label, tp_size=args.tp_size)
    except (AuditError, OSError, KeyError, TypeError, ValueError) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "classification": "read-only-local-development-profile-audit",
            "status": "FAIL",
            "label": args.label,
            "runtime_compatible": None,
            "target_match": None,
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
