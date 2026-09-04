"""Offline conversion of declared host-shaped CPU snapshots, never live hooks."""

from __future__ import annotations

import torch

from .reference import paged_int8_attention_reference


def continuing_prefill_snapshot_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    storage_layout: str,
    block_table: torch.Tensor,
    actual_seq_lengths_q: list[int] | tuple[int, ...],
    kv_lengths: list[int] | tuple[int, ...],
    num_decodes: int,
    num_decode_tokens: int,
    num_prefills: int,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    key_offset: torch.Tensor,
    value_offset: torch.Tensor,
    causal: bool,
    attention_type: str,
    sliding_window: int | None,
    query_tile_size: int = 64,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Return only continuing-prefill rows; do not execute or write decode rows.

    `NZ_PACKED_4D` declares NZ physical byte order wrapped as [P, B, Hkv, D].
    It is NOT ordinary ND data of the same shape, nor an assumption that an
    arbitrary NPU `.cpu()` copy preserves NZ format. The caller must construct
    a CPU snapshot with this explicit representation; no NPU exporter exists.
    Q lengths are cumulative end offsets WITHOUT a leading zero, covering a
    decode prefix followed by continuing-prefill requests. Graph-padding Q rows
    beyond the final offset are ignored. Scales/offsets are TP-local [1,Hkv,D].
    New-only/mixed-new prefill, nonzero offsets, windows and noncausal or cross
    attention are rejected. This is not a graph-safe runtime adapter.
    """
    tensors = (
        query,
        key_cache,
        value_cache,
        block_table,
        key_scale,
        value_scale,
        key_offset,
        value_offset,
    )
    if any(not isinstance(item, torch.Tensor) for item in tensors):
        raise ValueError("snapshot inputs must be Torch tensors")
    if any(item.device.type != "cpu" for item in tensors):
        raise ValueError("only explicit CPU snapshots are accepted")
    if storage_layout != "NZ_PACKED_4D":
        raise ValueError("explicit NZ_PACKED_4D storage declaration required")
    if causal is not True or attention_type != "decoder" or sliding_window is not None:
        raise ValueError("only causal decoder attention without a window is supported")
    if key_cache.ndim != 4 or min(key_cache.shape) <= 0:
        raise ValueError("packed cache must have nonempty [P,B,Hkv,D] dimensions")
    if value_cache.shape != key_cache.shape:
        raise ValueError("key/value packed cache shapes must match")
    if any(
        cache.dtype != torch.int8 or not cache.is_contiguous()
        for cache in (key_cache, value_cache)
    ):
        raise ValueError(
            "packed caches must be contiguous INT8; implicit copies forbidden"
        )
    pages, block_size, kv_heads, dim = key_cache.shape
    if block_size % 32 or dim % 32:
        raise ValueError(
            "host NZ snapshot requires block size and head dim divisible by 32"
        )
    if query.ndim != 3:
        raise ValueError("query must be packed TND")
    for lengths in (actual_seq_lengths_q, kv_lengths):
        if not isinstance(lengths, (list, tuple)) or not lengths:
            raise ValueError("metadata lengths must be nonempty Python lists or tuples")
        if any(type(length) is not int or length <= 0 for length in lengths):
            raise ValueError("metadata lengths must be positive Python integers")
    if len(actual_seq_lengths_q) != len(kv_lengths):
        raise ValueError("query/KV metadata row counts must agree")
    if block_table.ndim != 2 or block_table.shape[0] != len(kv_lengths):
        raise ValueError("block table must align with all metadata rows")
    if any(
        type(count) is not int or count < 0
        for count in (num_decodes, num_decode_tokens, num_prefills)
    ):
        raise ValueError("decode/prefill counts must be nonnegative Python integers")
    if num_prefills <= 0 or num_decodes + num_prefills != len(kv_lengths):
        raise ValueError("decode/prefill request counts must partition metadata rows")
    previous = (0, *actual_seq_lengths_q[:-1])
    lengths = [
        end - start for start, end in zip(previous, actual_seq_lengths_q, strict=True)
    ]
    if min(lengths) <= 0 or actual_seq_lengths_q[-1] > query.shape[0]:
        raise ValueError("cumulative query ends must increase and fit query rows")
    expected_decode_tokens = actual_seq_lengths_q[num_decodes - 1] if num_decodes else 0
    if num_decode_tokens != expected_decode_tokens:
        raise ValueError("decode token count disagrees with cumulative query split")
    prefill_lengths = lengths[num_decodes:]
    prefill_kv = kv_lengths[num_decodes:]
    if any(kv <= q for kv, q in zip(prefill_kv, prefill_lengths, strict=True)):
        raise ValueError(
            "snapshot requires continuing-only prefill; new/mixed-new unsupported"
        )
    for tensor in (key_scale, value_scale, key_offset, value_offset):
        if tensor.shape != (1, kv_heads, dim) or not tensor.is_floating_point():
            raise ValueError("TP-local scales/offsets must be floating [1,Hkv,D]")
    if any(not (offset == 0).all() for offset in (key_offset, value_offset)):
        raise ValueError("only symmetric zero-offset NZ snapshots are supported")

    # Reinterpret declared NZ bytes without materializing a whole-cache copy.
    nz_shape = (pages, kv_heads, dim // 32, block_size, 32)
    return paged_int8_attention_reference(
        query[num_decode_tokens : actual_seq_lengths_q[-1]],
        key_cache.view(nz_shape),
        value_cache.view(nz_shape),
        block_table=block_table[num_decodes:],
        kv_lengths=prefill_kv,
        query_lengths=prefill_lengths,
        key_scale=key_scale[0],
        value_scale=value_scale[0],
        query_tile_size=query_tile_size,
        softmax_scale=softmax_scale,
    )
