"""Fail-closed configuration contract for the C8 provider candidate."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "vllm-ascend-adaptive-quantized-kv-provider/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
HOST_LAYER_NAME_PATTERN = re.compile(
    r"model\.layers\.(?P<layer_id>[0-9]+)\.self_attn\.attn"
)

_REQUIRED_KEYS = {
    "schema_version",
    "expected_model",
    "expected_model_revision",
    "model_config_sha256",
    "profile_path",
    "profile_sha256",
    "calibration_provenance",
    "supported_soc",
    "full_attention_layer_ids",
    "global_channels_per_tensor",
    "tp_size",
    "tp_local_channels_per_tensor",
    "zero_offsets_attested",
    "kv_layout",
    "cache_write_owner",
}


class ProviderConfigError(ValueError):
    """Raised when activation cannot prove the pinned provider contract."""


def _reject_constant(value: str) -> None:
    raise ProviderConfigError(f"provider config contains non-finite value {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderConfigError(f"provider config contains duplicate key {key!r}")
        result[key] = value
    return result


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ProviderConfigError(f"{key} must be a non-empty string")
    return value


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if type(value) is not int or value <= 0:
        raise ProviderConfigError(f"{key} must be a positive integer")
    return value


def _require_sha256(payload: dict[str, Any], key: str) -> str:
    value = _require_string(payload, key)
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ProviderConfigError(f"{key} must be a lowercase SHA256 digest")
    return value


def _require_string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or not value:
        raise ProviderConfigError(f"{key} must be a non-empty JSON array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ProviderConfigError(f"{key} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ProviderConfigError(f"{key} must not contain duplicates")
    return result


def _require_layer_ids(payload: dict[str, Any]) -> tuple[int, ...]:
    value = payload["full_attention_layer_ids"]
    if not isinstance(value, list) or not value:
        raise ProviderConfigError(
            "full_attention_layer_ids must be a non-empty JSON array"
        )
    result = tuple(value)
    if any(type(layer_id) is not int or layer_id < 0 for layer_id in result):
        raise ProviderConfigError(
            "full_attention_layer_ids must contain non-negative integers"
        )
    if tuple(sorted(set(result))) != result:
        raise ProviderConfigError(
            "full_attention_layer_ids must be sorted and contain no duplicates"
        )
    return result


@dataclass(frozen=True, slots=True)
class ProviderRuntimeIdentity:
    """Host facts that must match the provider activation document."""

    model: str
    model_revision: str | None
    tensor_parallel_rank: int
    tensor_parallel_size: int
    layer_id: int
    num_kv_heads: int
    head_size: int
    kv_cache_dtype: str


def _host_attribute(config: object, name: str) -> Any:
    try:
        return getattr(config, name)
    except AttributeError as error:
        raise ProviderConfigError(
            f"Host provider config is missing required attribute {name!r}"
        ) from error


def _host_int(config: object, name: str) -> int:
    value = _host_attribute(config, name)
    if type(value) is not int:
        raise ProviderConfigError(f"Host provider config {name} must be an integer")
    return value


def runtime_identity_from_host_config(config: object) -> ProviderRuntimeIdentity:
    """Extract the fail-closed runtime identity carried by Host PR #35.

    The adapter deliberately uses structural attribute access instead of
    importing vLLM Ascend or Torch. The fixed target admits only canonical
    ``model.layers.N.self_attn.attn`` names; accepting suffix or substring
    matches would make layer admission ambiguous.
    """

    layer_name = _host_attribute(config, "layer_name")
    if not isinstance(layer_name, str):
        raise ProviderConfigError("Host provider config layer_name must be a string")
    match = HOST_LAYER_NAME_PATTERN.fullmatch(layer_name)
    if match is None:
        raise ProviderConfigError(
            "Host provider config layer_name must match "
            "'model.layers.N.self_attn.attn' exactly"
        )

    model = _host_attribute(config, "model")
    if not isinstance(model, str) or not model:
        raise ProviderConfigError("Host provider config model must be non-empty")
    model_revision = _host_attribute(config, "model_revision")
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision
    ):
        raise ProviderConfigError(
            "Host provider config model_revision must be non-empty or None"
        )

    return ProviderRuntimeIdentity(
        model=model,
        model_revision=model_revision,
        tensor_parallel_rank=_host_int(config, "tensor_parallel_rank"),
        tensor_parallel_size=_host_int(config, "tensor_parallel_size"),
        layer_id=int(match.group("layer_id")),
        num_kv_heads=_host_int(config, "num_kv_heads"),
        head_size=_host_int(config, "head_size"),
        kv_cache_dtype=str(_host_attribute(config, "kv_cache_dtype")),
    )


@dataclass(frozen=True, slots=True)
class ProviderActivationConfig:
    """Immutable, construction-time C8 provider expectations."""

    expected_model: str
    expected_model_revision: str
    model_config_sha256: str
    profile_path: str
    profile_sha256: str
    calibration_provenance: str
    supported_soc: tuple[str, ...]
    full_attention_layer_ids: tuple[int, ...]
    global_channels_per_tensor: int
    tp_size: int
    tp_local_channels_per_tensor: int
    zero_offsets_attested: bool
    kv_layout: str
    cache_write_owner: str

    @classmethod
    def from_json(cls, raw: str) -> ProviderActivationConfig:
        if not isinstance(raw, str) or not raw:
            raise ProviderConfigError("provider config must be non-empty JSON text")
        try:
            payload = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as error:
            raise ProviderConfigError(
                f"provider config is invalid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ProviderConfigError("provider config root must be a JSON object")
        keys = set(payload)
        if keys != _REQUIRED_KEYS:
            missing = sorted(_REQUIRED_KEYS - keys)
            unknown = sorted(keys - _REQUIRED_KEYS)
            raise ProviderConfigError(
                f"provider config keys differ: missing={missing}, unknown={unknown}"
            )
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ProviderConfigError(
                f"unsupported schema_version {payload['schema_version']!r}"
            )
        zero_offsets_attested = payload["zero_offsets_attested"]
        if zero_offsets_attested is not True:
            raise ProviderConfigError(
                "zero_offsets_attested must be true for the current Host fallback"
            )
        global_channels = _require_positive_int(payload, "global_channels_per_tensor")
        tp_size = _require_positive_int(payload, "tp_size")
        local_channels = _require_positive_int(payload, "tp_local_channels_per_tensor")
        if global_channels != tp_size * local_channels:
            raise ProviderConfigError(
                "global channels must equal tp_size * TP-local channels"
            )
        return cls(
            expected_model=_require_string(payload, "expected_model"),
            expected_model_revision=_require_string(payload, "expected_model_revision"),
            model_config_sha256=_require_sha256(payload, "model_config_sha256"),
            profile_path=_require_string(payload, "profile_path"),
            profile_sha256=_require_sha256(payload, "profile_sha256"),
            calibration_provenance=_require_string(payload, "calibration_provenance"),
            supported_soc=_require_string_tuple(payload, "supported_soc"),
            full_attention_layer_ids=_require_layer_ids(payload),
            global_channels_per_tensor=global_channels,
            tp_size=tp_size,
            tp_local_channels_per_tensor=local_channels,
            zero_offsets_attested=True,
            kv_layout=_require_string(payload, "kv_layout"),
            cache_write_owner=_require_string(payload, "cache_write_owner"),
        )

    def validate_runtime(self, runtime: ProviderRuntimeIdentity) -> None:
        """Reject a Host layer that does not match the activation contract."""

        if runtime.model != self.expected_model:
            raise ProviderConfigError(
                f"runtime model mismatch: expected {self.expected_model!r}, "
                f"got {runtime.model!r}"
            )
        if runtime.model_revision != self.expected_model_revision:
            raise ProviderConfigError(
                "runtime model revision does not match the activation contract"
            )
        if runtime.tensor_parallel_size != self.tp_size:
            raise ProviderConfigError(
                f"runtime TP size must be {self.tp_size}, "
                f"got {runtime.tensor_parallel_size}"
            )
        if not 0 <= runtime.tensor_parallel_rank < runtime.tensor_parallel_size:
            raise ProviderConfigError("runtime TP rank is outside the TP group")
        if runtime.layer_id not in self.full_attention_layer_ids:
            raise ProviderConfigError(
                f"layer {runtime.layer_id} is not an admitted full-attention layer"
            )
        if type(runtime.num_kv_heads) is not int or runtime.num_kv_heads <= 0:
            raise ProviderConfigError("runtime num_kv_heads must be positive")
        if type(runtime.head_size) is not int or runtime.head_size <= 0:
            raise ProviderConfigError("runtime head_size must be positive")
        observed_channels = runtime.num_kv_heads * runtime.head_size
        if observed_channels != self.tp_local_channels_per_tensor:
            raise ProviderConfigError(
                "runtime TP-local KV channels do not match the activation contract"
            )
        if runtime.kv_cache_dtype != "torch.int8":
            raise ProviderConfigError(
                "runtime KV cache dtype must be torch.int8, "
                f"got {runtime.kv_cache_dtype!r}"
            )

    def validate_host_config(self, config: object) -> ProviderRuntimeIdentity:
        """Validate Host PR #35 construction facts and return their identity."""

        runtime = runtime_identity_from_host_config(config)
        self.validate_runtime(runtime)
        return runtime


@dataclass(frozen=True, slots=True)
class ProviderConstructionContext:
    """Atomically parsed activation expectations and validated Host identity."""

    activation: ProviderActivationConfig
    runtime: ProviderRuntimeIdentity

    @classmethod
    def from_host_config(cls, config: object) -> ProviderConstructionContext:
        raw = _host_attribute(config, "provider_config_json")
        activation = ProviderActivationConfig.from_json(raw)
        runtime = activation.validate_host_config(config)
        return cls(activation=activation, runtime=runtime)
