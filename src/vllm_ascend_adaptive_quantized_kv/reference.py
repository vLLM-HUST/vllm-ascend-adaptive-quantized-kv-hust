"""CPU-only blockwise attention reference, adapted from advisor PR #19.

See docs/PR19_TAKEOVER_CONTROL_20260904.md for provenance and boundaries.
This module is explicitly imported for component testing, never by discovery
or the manifest. It is not an Ascend kernel or a runtime integration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _validate_inputs(
    query: torch.Tensor,
    key_nz: torch.Tensor,
    value_nz: torch.Tensor,
    block_table: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    kv_lengths: Sequence[int],
    query_lengths: Sequence[int],
    query_tile_size: int,
) -> list[list[int]]:
    tensors = (query, key_nz, value_nz, block_table, key_scale, value_scale)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("all tensor inputs must be Torch tensors")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("reference accepts CPU tensors only; no device execution")
    float_types = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if query.ndim != 3 or min(query.shape) <= 0 or query.dtype not in float_types:
        raise ValueError("query must be a nonempty floating TND tensor")
    if key_nz.ndim != 5 or min(key_nz.shape) <= 0 or key_nz.shape[-1] != 32:
        raise ValueError("key cache must be 5D NZ [pages, kv_heads, D/32, B, 32]")
    if value_nz.shape != key_nz.shape:
        raise ValueError("key/value cache shapes must match")
    if key_nz.dtype != torch.int8 or value_nz.dtype != torch.int8:
        raise ValueError("key/value caches must both be INT8")
    num_pages, kv_heads, dim_chunks, block_size, _ = key_nz.shape
    if query.shape[2] != dim_chunks * 32 or query.shape[1] % kv_heads:
        raise ValueError("query head dimension or GQA head ratio is incompatible")
    if block_table.ndim != 2 or block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block table must be a 2D integer tensor")
    if not isinstance(kv_lengths, (list, tuple)) or not isinstance(
        query_lengths, (list, tuple)
    ):
        raise ValueError("lengths must be Python lists or tuples")
    if (
        not kv_lengths
        or len(kv_lengths) != len(query_lengths)
        or block_table.shape[0] != len(kv_lengths)
    ):
        raise ValueError("sequence metadata and block-table rows must align")
    if any(type(length) is not int or length <= 0 for length in kv_lengths):
        raise ValueError("KV lengths must be positive Python integers")
    if any(type(length) is not int or length <= 0 for length in query_lengths):
        raise ValueError("query lengths must be positive Python integers")
    if sum(query_lengths) != query.shape[0]:
        raise ValueError("query lengths must cover exactly the packed query")
    if type(query_tile_size) is not int or query_tile_size <= 0:
        raise ValueError("query tile size must be a positive Python integer")
    for scale in (key_scale, value_scale):
        if scale.shape != (kv_heads, query.shape[2]) or scale.dtype not in float_types:
            raise ValueError("scales must be floating [kv_heads, head_dim] tensors")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("scales must be finite and positive")
    if not torch.isfinite(query).all():
        raise ValueError("query must be finite")

    page_ids: list[list[int]] = []
    for row, (kv_len, query_len) in enumerate(
        zip(kv_lengths, query_lengths, strict=True)
    ):
        if query_len > kv_len:
            raise ValueError("query length cannot exceed KV length")
        page_count = (kv_len + block_size - 1) // block_size
        if page_count > block_table.shape[1]:
            raise ValueError("block table does not cover the sequence")
        used = block_table[row, :page_count].tolist()
        if any(page < 0 or page >= num_pages for page in used):
            raise ValueError("used physical page IDs must be in cache bounds")
        page_ids.append(used)
    return page_ids


def _dequantize_page(
    cache: torch.Tensor,
    page_id: int,
    valid_tokens: int,
    scale: torch.Tensor,
) -> torch.Tensor:
    # NZ [H, D/32, B, 32] -> logical [valid_B, H, D], one physical page only.
    page = cache[page_id, :, :, :valid_tokens, :].permute(2, 0, 1, 3)
    return (
        page.reshape(valid_tokens, scale.shape[0], scale.shape[1]).to(scale.dtype)
        * scale
    )


@torch.no_grad()
@torch.autocast(device_type="cpu", enabled=False)
def paged_int8_attention_reference(
    query: torch.Tensor,
    key_nz: torch.Tensor,
    value_nz: torch.Tensor,
    *,
    block_table: torch.Tensor,
    kv_lengths: Sequence[int],
    query_lengths: Sequence[int],
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    query_tile_size: int = 64,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Compute causal suffix attention without a full-cache floating K/V copy.

    Lengths are per-request, not cumulative. Q is packed TND; K/V are symmetric
    INT8 NZ pages with a 32-element inner lane. Only used block IDs are checked;
    padding IDs are ignored. KV heads are shared within each GQA group.
    Arithmetic is FP64 for FP64 queries, otherwise FP32, then cast to Q dtype.
    This mathematical oracle does not promise bitwise BF16 FIA equivalence.
    """
    page_ids = _validate_inputs(
        query,
        key_nz,
        value_nz,
        block_table,
        key_scale,
        value_scale,
        kv_lengths,
        query_lengths,
        query_tile_size,
    )
    if softmax_scale is None:
        softmax_scale = query.shape[2] ** -0.5
    if (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, (int, float))
        or not math.isfinite(softmax_scale)
        or softmax_scale <= 0
    ):
        raise ValueError("softmax scale must be finite and positive")

    _, kv_heads, _, block_size, _ = key_nz.shape
    _, num_heads, head_dim = query.shape
    groups = num_heads // kv_heads
    math_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    k_scale, v_scale = key_scale.to(math_dtype), value_scale.to(math_dtype)
    output = torch.empty_like(query)
    offset = 0
    for row, (kv_len, query_len) in enumerate(
        zip(kv_lengths, query_lengths, strict=True)
    ):
        for start in range(0, query_len, query_tile_size):
            end = min(start + query_tile_size, query_len)
            tile_q = query[offset + start : offset + end].to(math_dtype)
            tile_q = tile_q.reshape(end - start, kv_heads, groups, head_dim)
            query_positions = torch.arange(
                kv_len - query_len + start,
                kv_len - query_len + end,
                device=query.device,
            )
            running_max = torch.full(
                (kv_heads, groups, end - start),
                -torch.inf,
                dtype=math_dtype,
                device=query.device,
            )
            running_sum = torch.zeros_like(running_max)
            accumulator = torch.zeros(
                (*running_max.shape, head_dim), dtype=math_dtype, device=query.device
            )
            for logical_page, physical_page in enumerate(page_ids[row]):
                page_start = logical_page * block_size
                valid_tokens = min(block_size, kv_len - page_start)
                block_k = _dequantize_page(key_nz, physical_page, valid_tokens, k_scale)
                block_v = _dequantize_page(
                    value_nz, physical_page, valid_tokens, v_scale
                )
                scores = torch.einsum("qhgd,bhd->hgqb", tile_q, block_k) * softmax_scale
                key_positions = torch.arange(
                    page_start, page_start + valid_tokens, device=query.device
                )
                causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                scores = scores.masked_fill(~causal, -torch.inf)

                # Online softmax rescales previous mass when a block raises the max.
                next_max = torch.maximum(running_max, scores.amax(dim=-1))
                old_weight = torch.exp(running_max - next_max)
                weights = torch.exp(scores - next_max.unsqueeze(-1))
                running_sum = running_sum * old_weight + weights.sum(dim=-1)
                accumulator = accumulator * old_weight.unsqueeze(-1) + torch.einsum(
                    "hgqb,bhd->hgqd", weights, block_v
                )
                running_max = next_max
            tile_output = (accumulator / running_sum.unsqueeze(-1)).permute(2, 0, 1, 3)
            output[offset + start : offset + end] = tile_output.reshape(
                end - start, num_heads, head_dim
            ).to(query.dtype)
        offset += query_len
    if not torch.isfinite(output).all():
        raise ValueError("nonfinite reference output; numerical contract failed")
    return output
