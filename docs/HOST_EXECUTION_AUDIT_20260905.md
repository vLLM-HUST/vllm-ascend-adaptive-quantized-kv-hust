# Continuing-Prefill Host Execution Audit

## Classification and exact inputs

This is a source audit and CPU snapshot bridge, not live runtime evidence.

| Input | Revision | Role |
| --- | --- | --- |
| `vLLM-HUST/vllm-hust` | `a4d6aa022fb1885a25a802a6e29372c81eac6c9f` | candidate core source |
| `vLLM-HUST/vllm-ascend-hust` | `2c8c722107a54127999a64c4eb0ec86139df8c26` | candidate Ascend source |
| this extension | `ab071bb3e54d4cea9c36e14dae31b0f1cc4966f2` | S2 base before this audit |

The candidate hosts advanced after the active parent v5 gate was frozen. They
are audited rather than silently substituted for its exact host pins.

## Source findings

1. Core `vllm/v1/attention/backends/registry.py` supports registering
   `AttentionBackendEnum.CUSTOM`. Core selector code nevertheless delegates to
   `current_platform.get_attn_backend_cls(...)`; registration does not force a
   platform to return `CUSTOM`.
2. Ascend `platform.py:get_attn_backend_cls` accepts `selected_backend` but its
   dense-attention path returns fixed Ascend class paths from `backend_map`.
   Thus a plugin registration alone is not an execution route on this host.
3. Ascend `kv_c8.py:AscendC8KVCacheAttentionMethod.create_weights` assigns
   `layer.impl.__class__ = AscendC8AttentionBackendImpl`. A whole-backend plugin
   override would not be a stable C8 ownership boundary even if selected.
4. In `AscendC8AttentionBackendImpl.forward`, graph capture dispatches to
   `full_graph_fia` before the `ChunkedPrefill` state branch. Non-capture
   chunked prefill dispatches to `_forward_c8_chunked_prefill`. A replacement at
   only one branch would create capture/replay semantic drift.
5. The continuing-prefill fallback remains present in both chunked-prefill and
   general prefill paths: paged INT8 cache is gathered/dequantized, then TND FIA
   receives dense K/V with `block_table=None`. Decode remains a distinct native
   BNSD paged-INT8 path and is outside this replacement scope.

## Completed local bridge

`snapshot_reference.continuing_prefill_snapshot_reference` maps a declared CPU
snapshot of the observed host data contract to the S2 mathematical reference:

- explicit NZ physical bytes wrapped as contiguous `[P,B,Hkv,D]` are viewed as
  `[P,Hkv,D/32,B,32]` without a whole-cache copy;
- cumulative `actual_seq_lengths_q` are converted to per-request lengths;
- the decode prefix and graph-padding query rows are excluded;
- only prefill block-table/KV-length rows are forwarded;
- TP-local `[1,Hkv,D]` scales are reduced to the component shape;
- unsupported or ambiguous semantics fail closed.

This bridge deliberately accepts CPU snapshots, not host metadata objects. It
does not import vLLM/Ascend, export NPU NZ tensors, dispatch a kernel or modify
host output. The explicit storage label prevents ordinary ND `[P,B,H,D]` data
from being mistaken for NZ merely because the dimensions agree.

## Minimal owner-delivered execution boundary

A usable host boundary must be reachable for C8 after cache write and before
both graph-capture and non-capture attention dispatch. It must preserve native
decode and new-only prefill behavior while allowing a continuing-only provider
to receive:

- active TND Q and cumulative/per-request query boundaries;
- paged INT8 NZ K/V, block table, block size and valid KV lengths;
- TP-local K/V scales and offsets with an explicit quantization convention;
- head configuration, causal/window/attention type and stable graph shapes;
- host-owned output/workspace plus capture/replay lifecycle identity.

The host must define provider discovery, eligibility and fallback; disabled
behavior must be baseline-equivalent. It must also define whether mixed
new/continuing batches are partitioned by the host or rejected by the provider.
The extension must not replace implementation classes, import private graph
registries or monkey-patch `forward`.

No contract identifier is claimed here. Issue #1 should resolve whether an
existing interface satisfies this boundary or which owner-delivered route will
provide it. After that decision, the next implementation is a minimal adapter
to the owner API followed by CPU contract tests, NPU correctness, graph
capture/replay, cleanup, and only then matched performance measurement.
