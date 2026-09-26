import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_c8_profile_artifact import AuditError, audit_profile


def _bf16(values: list[float]) -> bytes:
    words = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
    return struct.pack(f"<{len(words)}H", *words)


def _write_fixture(
    model_dir: Path,
    *,
    nonzero_offset: bool = False,
    hybrid: bool = False,
) -> None:
    text_config = {
        "architectures": ["FixtureForCausalLM"],
        "model_type": "fixture",
        "torch_dtype": "bfloat16",
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
    }
    profile_layers = range(2)
    if hybrid:
        text_config.update(
            {
                "head_dim": 2,
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention",
                ],
            }
        )
        config = {
            "architectures": ["FixtureForConditionalGeneration"],
            "model_type": "fixture_multimodal",
            "text_config": text_config,
        }
        profile_layers = (1, 3)
    else:
        config = text_config
    description = {
        "model_quant_type": "FLOAT",
        "kv_cache_type": "C8",
    }
    tensors: dict[str, bytes] = {}
    for layer in profile_layers:
        for kind in ("k", "v"):
            for field in ("scale", "offset"):
                name = f"model.layers.{layer}.self_attn.{kind}_proj.kv_cache_{field}"
                description[name] = "C8"
                value = 0.25 if field == "scale" else 0.0
                if nonzero_offset and layer == 0 and kind == "k" and field == "offset":
                    value = 1.0
                tensors[name] = _bf16([value] * 4)

    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for name in sorted(tensors):
        raw = tensors[name]
        start = len(data)
        data.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    shard_name = "model-00001-of-00001.safetensors"
    (model_dir / shard_name).write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + data
    )
    weight_map = {name: shard_name for name in tensors}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "quant_model_description.json").write_text(json.dumps(description))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_valid_float_weight_c8_profile_passes(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = audit_profile(tmp_path, label="fixture")

    assert result["status"] == "PASS"
    assert result["profile"]["covered_layers"] == [0, 1]
    assert result["profile"]["full_layer_coverage"] is True
    assert result["profile"]["channels_per_tensor"] == 4
    assert result["profile"]["tp_local_channels_per_tensor"] == 4
    assert result["profile"]["offset_nonzero_elements"] == 0
    assert result["profile"]["scale_minimum"] == pytest.approx(0.25)


def test_nonzero_offset_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path, nonzero_offset=True)

    with pytest.raises(AuditError, match="nonzero offset"):
        audit_profile(tmp_path, label="fixture")


def test_missing_profile_tensor_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    description_path = tmp_path / "quant_model_description.json"
    description = json.loads(description_path.read_text())
    del description["model.layers.1.self_attn.v_proj.kv_cache_offset"]
    description_path.write_text(json.dumps(description))

    with pytest.raises(AuditError, match="incomplete profile fields"):
        audit_profile(tmp_path, label="fixture")


def test_hybrid_profile_covers_only_full_attention_layers(tmp_path: Path) -> None:
    _write_fixture(tmp_path, hybrid=True)

    result = audit_profile(tmp_path, label="hybrid-fixture", tp_size=2)

    assert result["model"]["head_size"] == 2
    assert result["profile"]["covered_layers"] == [1, 3]
    assert result["profile"]["full_layer_coverage"] is False
    assert result["profile"]["full_attention_layer_coverage"] is True
    assert result["profile"]["channels_per_tensor"] == 4
    assert result["profile"]["tp_local_channels_per_tensor"] == 2


def test_missing_hybrid_attention_layer_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path, hybrid=True)
    description_path = tmp_path / "quant_model_description.json"
    description = json.loads(description_path.read_text())
    for name in list(description):
        if name.startswith("model.layers.3."):
            del description[name]
    description_path.write_text(json.dumps(description))

    with pytest.raises(AuditError, match=r"missing=\[3\]"):
        audit_profile(tmp_path, label="hybrid-fixture")


def test_profile_on_linear_attention_layer_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path, hybrid=True)
    description_path = tmp_path / "quant_model_description.json"
    description = json.loads(description_path.read_text())
    for kind in ("k", "v"):
        for field in ("scale", "offset"):
            source = f"model.layers.1.self_attn.{kind}_proj.kv_cache_{field}"
            target = f"model.layers.0.self_attn.{kind}_proj.kv_cache_{field}"
            description[target] = description[source]
    description_path.write_text(json.dumps(description))

    with pytest.raises(AuditError, match=r"unexpected=\[0\]"):
        audit_profile(tmp_path, label="hybrid-fixture")
