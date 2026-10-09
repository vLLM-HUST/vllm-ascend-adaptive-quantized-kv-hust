import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_qwen35_target_contract import (
    EXPECTED_LAYER_TYPES,
    AuditError,
    analyze_config,
)


def _config() -> dict:
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
            "layer_types": EXPECTED_LAYER_TYPES,
        },
    }


def test_exact_target_contract_derives_tp2_profile_shape() -> None:
    result = analyze_config(_config(), tp_size=2)

    assert result["model"]["full_attention_layers"] == [
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        31,
        35,
        39,
    ]
    assert result["c8_profile_contract"]["profile_tensor_count"] == 40
    assert result["c8_profile_contract"]["global_channels_per_tensor"] == 512
    assert result["c8_profile_contract"]["tp_local_kv_heads"] == 1
    assert result["c8_profile_contract"]["tp_local_channels_per_tensor"] == 256


def test_layer_type_drift_fails_closed() -> None:
    config = copy.deepcopy(_config())
    config["text_config"]["layer_types"][3] = "linear_attention"

    with pytest.raises(AuditError, match="layer_types"):
        analyze_config(config)


def test_non_divisible_tp_fails_closed() -> None:
    with pytest.raises(AuditError, match="not divisible"):
        analyze_config(_config(), tp_size=3)
