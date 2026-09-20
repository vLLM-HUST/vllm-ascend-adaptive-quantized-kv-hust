import pytest

torch = pytest.importorskip("torch")

from vllm_ascend_adaptive_quantized_kv import snapshot_reference  # noqa: E402
from vllm_ascend_adaptive_quantized_kv.reference import (  # noqa: E402
    paged_int8_attention_reference,
)


def make_snapshot():
    rng = torch.Generator().manual_seed(51)
    shape = (4, 2, 2, 32, 32)
    key_nz = torch.randint(-4, 5, shape, dtype=torch.int8, generator=rng)
    value_nz = torch.randint(-4, 5, shape, dtype=torch.int8, generator=rng)
    scale = torch.rand(1, 2, 64, generator=rng) + 0.1
    query = torch.randn(15, 4, 64, generator=rng)
    query[:2].fill_(float("nan"))  # decode prefix is outside the reference scope
    query[14:].fill_(float("nan"))  # graph padding must not become an active query
    return (
        {
            "query": query,
            "key_cache": key_nz.view(4, 32, 2, 64),
            "value_cache": value_nz.view(4, 32, 2, 64),
            "storage_layout": "NZ_PACKED_4D",
            "block_table": torch.tensor([[-1, 999], [2, 0], [1, 3]]),
            "actual_seq_lengths_q": [2, 9, 14],
            "kv_lengths": [50, 48, 39],
            "num_decodes": 1,
            "num_decode_tokens": 2,
            "num_prefills": 2,
            "key_scale": scale,
            "value_scale": scale * 2,
            "key_offset": torch.zeros_like(scale),
            "value_offset": torch.zeros_like(scale),
            "causal": True,
            "attention_type": "decoder",
            "sliding_window": None,
            "query_tile_size": 4,
        },
        key_nz,
        value_nz,
    )


def test_cumulative_metadata_and_decode_prefix_conversion(monkeypatch):
    snapshot, key_nz, value_nz = make_snapshot()
    snapshot["query"].requires_grad_(True)
    original = snapshot["query"].clone()
    seen = {}

    def inspect(query, key, value, **kwargs):
        assert key.data_ptr() == snapshot["key_cache"].data_ptr()
        assert value.data_ptr() == snapshot["value_cache"].data_ptr()
        assert query.data_ptr() == snapshot["query"][2:].data_ptr()
        seen.update(kwargs)
        return paged_int8_attention_reference(query, key, value, **kwargs)

    monkeypatch.setattr(snapshot_reference, "paged_int8_attention_reference", inspect)
    actual = snapshot_reference.continuing_prefill_snapshot_reference(**snapshot)
    expected = paged_int8_attention_reference(
        snapshot["query"][2:14],
        key_nz,
        value_nz,
        block_table=snapshot["block_table"][1:],
        kv_lengths=[48, 39],
        query_lengths=[7, 5],
        key_scale=snapshot["key_scale"][0],
        value_scale=snapshot["value_scale"][0],
    )
    assert seen["query_lengths"] == [7, 5]  # not cumulative [7, 12]
    assert actual.shape == (12, 4, 64)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(snapshot["query"], original, equal_nan=True)


def test_prefill_only_snapshot():
    snapshot, _, _ = make_snapshot()
    snapshot.update(
        query=snapshot["query"][2:14],
        block_table=snapshot["block_table"][1:],
        actual_seq_lengths_q=(7, 12),
        kv_lengths=(48, 39),
        num_decodes=0,
        num_decode_tokens=0,
    )
    assert (
        snapshot_reference.continuing_prefill_snapshot_reference(**snapshot).shape[0]
        == 12
    )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("storage_layout", "ND", "storage declaration"),
        ("causal", False, "causal decoder"),
        ("attention_type", "encoder_decoder", "causal decoder"),
        ("sliding_window", 16, "causal decoder"),
        ("actual_seq_lengths_q", [2, 2, 14], "increase"),
        ("actual_seq_lengths_q", [2, 9, 16], "fit query"),
        ("actual_seq_lengths_q", [0, 9, 14], "positive Python"),
        ("actual_seq_lengths_q", [2, 9], "row counts"),
        ("kv_lengths", [], "nonempty Python"),
        ("num_decodes", -1, "nonnegative Python"),
        ("num_decodes", True, "nonnegative Python"),
        ("num_decode_tokens", 1, "cumulative query split"),
        ("num_prefills", 3, "partition"),
        ("num_prefills", 0, "partition"),
        ("kv_lengths", [50, 7, 39], "continuing-only"),
        ("kv_lengths", [50, 48, 3], "continuing-only"),
    ],
)
def test_rejects_ambiguous_or_unsupported_metadata(field, value, message):
    snapshot, _, _ = make_snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError, match=message):
        snapshot_reference.continuing_prefill_snapshot_reference(**snapshot)


@pytest.mark.parametrize(
    "field,transform,message",
    [
        ("key_cache", lambda t: t.float(), "contiguous INT8"),
        ("key_cache", lambda t: torch.stack((t, t), dim=-1)[..., 0], "contiguous INT8"),
        ("key_cache", lambda t: t[:0], "nonempty"),
        ("value_cache", lambda t: t[:1], "shapes must match"),
        ("query", lambda t: t.to("meta"), "CPU snapshots"),
        ("query", lambda t: None, "Torch tensors"),
        ("query", lambda t: t.flatten(), "packed TND"),
        ("block_table", lambda t: t[:1], "metadata rows"),
        ("block_table", lambda t: t.fill_(-1), "page IDs"),
        ("key_scale", lambda t: t[0], "TP-local"),
        ("value_scale", lambda t: t.fill_(float("nan")), "finite and positive"),
        ("key_offset", lambda t: t.fill_(1), "zero-offset"),
        ("value_offset", lambda t: t.fill_(float("nan")), "zero-offset"),
    ],
)
def test_rejects_invalid_snapshot_storage(field, transform, message):
    snapshot, _, _ = make_snapshot()
    snapshot[field] = transform(snapshot[field])
    with pytest.raises(ValueError, match=message):
        snapshot_reference.continuing_prefill_snapshot_reference(**snapshot)
