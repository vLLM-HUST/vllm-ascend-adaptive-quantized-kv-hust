from types import SimpleNamespace

import pytest

from vllm_ascend_adaptive_quantized_kv.provider_config import (
    runtime_identity_from_host_config,
)

torch = pytest.importorskip("torch")


def test_host_torch_int8_dtype_normalizes_without_host_imports() -> None:
    host_config = SimpleNamespace(
        layer_name="model.layers.3.self_attn.attn",
        model="Qwen/Qwen3.5-35B-A3B",
        model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        num_kv_heads=1,
        head_size=256,
        kv_cache_dtype=torch.int8,
    )

    runtime = runtime_identity_from_host_config(host_config)

    assert runtime.kv_cache_dtype == "torch.int8"
