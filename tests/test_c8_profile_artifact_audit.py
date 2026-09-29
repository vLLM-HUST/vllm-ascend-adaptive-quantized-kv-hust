import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_c8_profile_artifact import (
    PROFILE_CONTENT_DIGEST_ALGORITHM,
    AuditError,
    audit_profile,
)


def _bf16(values: list[float]) -> bytes:
    words = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
    return struct.pack(f"<{len(words)}H", *words)


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
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
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def _rewrite_safetensors_header(path: Path, header: dict[str, object]) -> None:
    raw = path.read_bytes()
    old_length = struct.unpack("<Q", raw[:8])[0]
    data = raw[8 + old_length :]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def _write_fixture(
    model_dir: Path,
    *,
    nonzero_offset: bool = False,
    hybrid: bool = False,
    scale_value: float = 0.25,
    split_shards: bool = False,
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
                value = scale_value if field == "scale" else 0.0
                if nonzero_offset and layer == 0 and kind == "k" and field == "offset":
                    value = 1.0
                tensors[name] = _bf16([value] * 4)

    if split_shards:
        names = sorted(tensors)
        shard_names = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        groups = (
            {name: tensors[name] for name in names[::2]},
            {name: tensors[name] for name in names[1::2]},
        )
        weight_map = {}
        for shard_name, group in zip(shard_names, groups, strict=True):
            _write_safetensors(model_dir / shard_name, group)
            weight_map.update(dict.fromkeys(group, shard_name))
    else:
        shard_name = "model-00001-of-00001.safetensors"
        _write_safetensors(model_dir / shard_name, tensors)
        weight_map = dict.fromkeys(tensors, shard_name)
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
    assert (
        result["profile"]["profile_content_digest_algorithm"]
        == PROFILE_CONTENT_DIGEST_ALGORITHM
    )
    assert len(result["profile"]["profile_content_sha256"]) == 64
    assert len(result["profile"]["tensor_digests"]) == 8


def test_profile_digest_binds_tensor_values(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_fixture(first, scale_value=0.25)
    _write_fixture(second, scale_value=0.5)

    first_result = audit_profile(first, label="first")
    second_result = audit_profile(second, label="second")

    assert (
        first_result["profile"]["profile_content_sha256"]
        != second_result["profile"]["profile_content_sha256"]
    )


def test_profile_digest_is_independent_of_label(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    first = audit_profile(tmp_path, label="first")
    second = audit_profile(tmp_path, label="second")

    assert (
        first["profile"]["profile_content_sha256"]
        == second["profile"]["profile_content_sha256"]
    )


def test_profile_digest_is_independent_of_shard_packaging(tmp_path: Path) -> None:
    single = tmp_path / "single"
    split = tmp_path / "split"
    single.mkdir()
    split.mkdir()
    _write_fixture(single)
    _write_fixture(split, split_shards=True)

    single_result = audit_profile(single, label="single")
    split_result = audit_profile(split, label="split")

    assert (
        single_result["profile"]["profile_content_sha256"]
        == split_result["profile"]["profile_content_sha256"]
    )
    assert single_result["metadata_files"] != split_result["metadata_files"]


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


def test_weight_map_profile_names_must_match_description(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_name = next(iter(index["weight_map"].values()))
    index["weight_map"]["model.layers.99.self_attn.k_proj.kv_cache_scale"] = shard_name
    index_path.write_text(json.dumps(index))

    with pytest.raises(AuditError, match=r"unexpected=.*layers\.99"):
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
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in list(index["weight_map"]):
        if name.startswith("model.layers.3."):
            del index["weight_map"][name]
    index_path.write_text(json.dumps(index))

    with pytest.raises(AuditError, match=r"missing=\[3\]"):
        audit_profile(tmp_path, label="hybrid-fixture")


def test_profile_on_linear_attention_layer_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path, hybrid=True)
    description_path = tmp_path / "quant_model_description.json"
    description = json.loads(description_path.read_text())
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for kind in ("k", "v"):
        for field in ("scale", "offset"):
            source = f"model.layers.1.self_attn.{kind}_proj.kv_cache_{field}"
            target = f"model.layers.0.self_attn.{kind}_proj.kv_cache_{field}"
            description[target] = description[source]
            index["weight_map"][target] = index["weight_map"][source]
    description_path.write_text(json.dumps(description))
    index_path.write_text(json.dumps(index))

    with pytest.raises(AuditError, match=r"unexpected=\[0\]"):
        audit_profile(tmp_path, label="hybrid-fixture")


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    description_path = tmp_path / "quant_model_description.json"
    raw = description_path.read_text()
    description_path.write_text(
        raw.replace(
            '"model_quant_type": "FLOAT"',
            '"model_quant_type": "FLOAT", "model_quant_type": "FLOAT"',
            1,
        )
    )

    with pytest.raises(AuditError, match="duplicate key 'model_quant_type'"):
        audit_profile(tmp_path, label="fixture")


def test_non_finite_json_value_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["unexpected_non_finite"] = float("nan")
    config_path.write_text(json.dumps(config))

    with pytest.raises(AuditError, match="non-finite value NaN"):
        audit_profile(tmp_path, label="fixture")


def test_weight_map_cannot_escape_model_directory(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    tensor_name = next(iter(index["weight_map"]))
    index["weight_map"][tensor_name] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))

    with pytest.raises(AuditError, match="unsafe shard path"):
        audit_profile(tmp_path, label="fixture")


def test_duplicate_safetensors_header_key_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    raw = shard.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header_raw = raw[8 : 8 + header_length]
    header = json.loads(header_raw)
    tensor_name = next(name for name in header if name != "__metadata__")
    duplicate = json.dumps(header[tensor_name], separators=(",", ":"))
    text = header_raw.decode().rstrip()
    rewritten = f'{text[:-1]},"{tensor_name}":{duplicate}}}'.encode()
    padding = (-len(rewritten)) % 8
    rewritten += b" " * padding
    data = raw[8 + header_length :]
    shard.write_bytes(struct.pack("<Q", len(rewritten)) + rewritten + data)

    with pytest.raises(AuditError, match=f"duplicate key {tensor_name!r}"):
        audit_profile(tmp_path, label="fixture")


def test_overlapping_safetensors_ranges_fail_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    raw = shard.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    tensor_names = [name for name in header if name != "__metadata__"]
    header[tensor_names[1]]["data_offsets"] = header[tensor_names[0]]["data_offsets"]
    _rewrite_safetensors_header(shard, header)

    with pytest.raises(AuditError, match="overlapping tensor ranges"):
        audit_profile(tmp_path, label="fixture")


def test_out_of_bounds_safetensors_range_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    raw = shard.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    tensor_name = next(name for name in header if name != "__metadata__")
    header[tensor_name]["data_offsets"] = [0, len(raw) * 2]
    _rewrite_safetensors_header(shard, header)

    with pytest.raises(AuditError, match="outside shard data"):
        audit_profile(tmp_path, label="fixture")
