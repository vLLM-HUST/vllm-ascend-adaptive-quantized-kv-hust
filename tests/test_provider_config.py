import json
from types import SimpleNamespace

import pytest

from vllm_ascend_adaptive_quantized_kv.provider_config import (
    SCHEMA_VERSION,
    ProviderActivationConfig,
    ProviderConfigError,
    ProviderConstructionContext,
    ProviderRuntimeIdentity,
    runtime_identity_from_host_config,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_model": "Qwen/Qwen3.5-35B-A3B",
        "expected_model_revision": ("59d61f3ce65a6d9863b86d2e96597125219dc754"),
        "model_config_sha256": (
            "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944"
        ),
        "profile_path": "/profiles/qwen35-c8.json",
        "profile_sha256": "a1" * 32,
        "calibration_provenance": "dataset=x; method=y; owner=z",
        "supported_soc": ["Ascend910B2"],
        "full_attention_layer_ids": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
        "global_channels_per_tensor": 512,
        "tp_size": 2,
        "tp_local_channels_per_tensor": 256,
        "zero_offsets_attested": True,
        "kv_layout": "paged-nz-int8",
        "cache_write_owner": "host-c8-kv-cache-writer",
    }


def _runtime(**overrides: object) -> ProviderRuntimeIdentity:
    values: dict[str, object] = {
        "model": "Qwen/Qwen3.5-35B-A3B",
        "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "tensor_parallel_rank": 0,
        "tensor_parallel_size": 2,
        "layer_id": 3,
        "num_kv_heads": 1,
        "head_size": 256,
        "kv_cache_dtype": "torch.int8",
    }
    values.update(overrides)
    return ProviderRuntimeIdentity(**values)  # type: ignore[arg-type]


def _host_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "layer_name": "model.layers.3.self_attn.attn",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "tensor_parallel_rank": 0,
        "tensor_parallel_size": 2,
        "num_kv_heads": 1,
        "head_size": 256,
        "kv_cache_dtype": "torch.int8",
        "provider_config_json": json.dumps(_payload()),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parse_and_validate_pinned_runtime_contract() -> None:
    config = ProviderActivationConfig.from_json(json.dumps(_payload()))
    config.validate_runtime(_runtime())

    assert config.full_attention_layer_ids == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
    assert config.global_channels_per_tensor == 512
    assert config.tp_local_channels_per_tensor == 256


def test_host_config_adapter_derives_and_validates_runtime_identity() -> None:
    config = ProviderActivationConfig.from_json(json.dumps(_payload()))
    host_config = _host_config(kv_cache_dtype="torch.int8")

    runtime = config.validate_host_config(host_config)

    assert runtime == _runtime()


def test_construction_context_atomically_parses_and_validates_host_config() -> None:
    context = ProviderConstructionContext.from_host_config(_host_config())

    assert context.activation.expected_model == "Qwen/Qwen3.5-35B-A3B"
    assert context.runtime == _runtime()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"provider_config_json": ""}, "non-empty JSON text"),
        ({"provider_config_json": "{}"}, "keys differ"),
        ({"model": "other"}, "model mismatch"),
        ({"kv_cache_dtype": "torch.bfloat16"}, "torch.int8"),
    ],
)
def test_construction_context_fails_before_returning_partial_state(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ProviderConfigError, match=message):
        ProviderConstructionContext.from_host_config(_host_config(**overrides))


@pytest.mark.parametrize(
    ("layer_name", "message"),
    [
        ("layers.3.self_attn.attn", "must match"),
        ("model.layers.3.self_attn", "must match"),
        ("prefix.model.layers.3.self_attn.attn", "must match"),
        ("model.layers.3.self_attn.attn.suffix", "must match"),
        ("model.layers.-1.self_attn.attn", "must match"),
    ],
)
def test_host_config_adapter_rejects_ambiguous_layer_name(
    layer_name: str, message: str
) -> None:
    with pytest.raises(ProviderConfigError, match=message):
        runtime_identity_from_host_config(_host_config(layer_name=layer_name))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": ""}, "model must be non-empty"),
        ({"model_revision": ""}, "model_revision"),
        ({"tensor_parallel_rank": True}, "must be an integer"),
        ({"tensor_parallel_size": "2"}, "must be an integer"),
        ({"num_kv_heads": 1.0}, "must be an integer"),
        ({"head_size": None}, "must be an integer"),
    ],
)
def test_host_config_adapter_rejects_invalid_identity_fields(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ProviderConfigError, match=message):
        runtime_identity_from_host_config(_host_config(**overrides))


def test_host_config_adapter_rejects_missing_attribute() -> None:
    host_config = _host_config()
    del host_config.num_kv_heads

    with pytest.raises(ProviderConfigError, match="missing required attribute"):
        runtime_identity_from_host_config(host_config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("zero_offsets_attested", False, "zero_offsets_attested"),
        ("tp_local_channels_per_tensor", 128, "global channels"),
        ("model_config_sha256", "not-a-digest", "model_config_sha256"),
        ("profile_sha256", "A1" * 32, "profile_sha256"),
        ("full_attention_layer_ids", [7, 3], "sorted"),
        ("full_attention_layer_ids", [3, 3], "duplicates"),
        ("supported_soc", [], "non-empty JSON array"),
    ],
)
def test_invalid_activation_contract_fails_closed(
    key: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[key] = value
    with pytest.raises(ProviderConfigError, match=message):
        ProviderActivationConfig.from_json(json.dumps(payload))


def test_unknown_missing_duplicate_and_non_finite_json_fail_closed() -> None:
    payload = _payload()
    payload["unknown"] = True
    with pytest.raises(ProviderConfigError, match="unknown"):
        ProviderActivationConfig.from_json(json.dumps(payload))

    payload = _payload()
    del payload["profile_path"]
    with pytest.raises(ProviderConfigError, match="profile_path"):
        ProviderActivationConfig.from_json(json.dumps(payload))

    with pytest.raises(ProviderConfigError, match="duplicate key"):
        ProviderActivationConfig.from_json(
            '{"schema_version":"x","schema_version":"y"}'
        )

    with pytest.raises(ProviderConfigError, match="non-finite"):
        ProviderActivationConfig.from_json('{"value":NaN}')


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "other"}, "model mismatch"),
        ({"model_revision": None}, "model revision"),
        ({"tensor_parallel_size": 1}, "TP size"),
        ({"tensor_parallel_rank": 2}, "TP rank"),
        ({"layer_id": 0}, "full-attention layer"),
        ({"num_kv_heads": 2}, "TP-local KV channels"),
        ({"head_size": 128}, "TP-local KV channels"),
        ({"kv_cache_dtype": "torch.bfloat16"}, "torch.int8"),
    ],
)
def test_runtime_mismatch_fails_closed(
    overrides: dict[str, object], message: str
) -> None:
    config = ProviderActivationConfig.from_json(json.dumps(_payload()))
    with pytest.raises(ProviderConfigError, match=message):
        config.validate_runtime(_runtime(**overrides))
