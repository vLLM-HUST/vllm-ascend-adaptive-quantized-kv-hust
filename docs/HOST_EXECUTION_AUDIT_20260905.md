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
6. The graph-capture C8 branch is not evidence of native continuing prefill. It
   views paged NZ K/V, but reshapes Q with `unsqueeze(2)`, selects BNSD, clears
   the attention mask and uses `sparse_mode=0`. This is a decode-shaped call;
   source inspection alone does not prove whether multi-token continuing
   prefill can reach it or whether it would be numerically valid.
7. Replay update code repeats the BNSD C8 contract. The C8 decode, chunked
   prefill and general prefill methods call `torch_npu` FIA directly, while
   `DeviceOperator` is selected from a fixed hardware-family map. Therefore the
   existing device adaptor is not a sufficient extension provider boundary.

## Reproducible source receipt

Run the audit without importing either host or starting a device runtime:

```bash
python scripts/audit_host_execution.py \
  --core-repo /path/to/vllm-hust \
  --ascend-repo /path/to/vllm-ascend-hust \
  --output /path/to/new-receipt.json
```

The retained S3b revisions above remain the defaults. To audit newer fetched
commits without changing that baseline, pass both exact 40-character SHAs:

```bash
python scripts/audit_host_execution.py \
  --core-repo /path/to/vllm-hust \
  --ascend-repo /path/to/vllm-ascend-hust \
  --core-revision <full-core-sha> \
  --ascend-revision <full-ascend-sha> \
  --output /path/to/additive-refresh-receipt.json
```

The output is created exclusively so an earlier receipt cannot be overwritten.
It records exact revisions, SHA256 for every inspected source blob, AST-derived
symbol locations, source findings and explicit limits. If a required definition
or source structure changes, the audit returns `INCONCLUSIVE` with exit code 2.
The scan of two host test files reports only direct target-symbol references;
it cannot establish that indirect coverage is absent.

The retained run is
[`docs/evidence/HOST_EXECUTION_SOURCE_AUDIT_20260905.json`](evidence/HOST_EXECUTION_SOURCE_AUDIT_20260905.json),
SHA256 `4e7ce1e659b3990a8ca58228f3314dc1c8a7ebe1ca46b29847a6f9446010ac86`.

## Latest-main refresh

On 2026-09-05 the same audit was run against fetched core
`88cca78bbc92e9113067cb62252c5d8ae2bbdd06` and Ascend
`d0433ba3aeb3b6643787177d7b1fefe4c742ef6e`. Compared with the retained S3b
inputs, core changed only scheduler/preemption files and Ascend changed only
MoE offload metadata. None of the eight inspected execution or test files
changed; their blob hashes are identical to S3b. The AST audit independently
reconfirmed all seven findings, including dense continuing-prefill
materialization and decode-shaped C8 capture/replay.

The additive receipt is
[`docs/evidence/HOST_EXECUTION_SOURCE_AUDIT_MAIN_REFRESH_20260905.json`](evidence/HOST_EXECUTION_SOURCE_AUDIT_MAIN_REFRESH_20260905.json),
SHA256 `9332fd8ede32c52a40f0f2cd1f35a02e445a3eca6a7f2f8934ac1062af0ac3cd`.
This refresh does not move the parent gate, activate the extension or establish
runtime compatibility. It confirms that the owner decision described below is
still required at these newer repository heads.

At the audited Ascend revision, `test_kv_c8.py` directly checks one
`_dequant_paged_kv_to_dense` round trip. Neither scanned file directly
references `_forward_c8_chunked_prefill` or `full_graph_fia`. The remaining
test gap is therefore end-to-end dispatch and graph behavior, rather than the
basic materializer conversion itself.

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

The smallest host-owner implementation map is:

1. Add one public, host-owned C8 execution-provider lookup after cache write and
   before capture/non-capture dispatch. The host keeps eligibility, output and
   fallback ownership.
2. Define continuing-only metadata explicitly: active TND Q, query boundaries,
   paged NZ K/V, block table, valid KV lengths, scales and quantization mode.
   Preserve current decode and all-new prefill behavior. Mixed batches must have
   a declared host partition or a provider rejection path.
3. Carry provider identity and stable shape data into graph capture records, and
   invoke the matching provider update during replay. A forward-only hook cannot
   preserve capture semantics.
4. Add host tests for non-capture output, capture/first replay/subsequent replay,
   fallback equivalence and mixed-batch handling before enabling the extension.

No contract identifier is claimed here. Issue #1 should resolve whether an
existing interface satisfies this boundary or which owner-delivered route will
provide it. After that decision, the next implementation is a minimal adapter
to the owner API followed by CPU contract tests, NPU correctness, graph
capture/replay, cleanup, and only then matched performance measurement.

## Retained operator-feasibility boundary

The parent research carrier already tested two narrower native-operator
candidates on its pinned 910B2 stack. Raw 5D NZ paged INT8 K/V failed the
multi-token dimension gate; an NZ-to-4D-ND repack then failed paged prompt
antiquantization validation. A later read-only V2 audit also found that the
documented special case did not directly cover the observed Qwen3-14B head
pair or query lengths. Exact commits, summary hashes, raw-receipt limitations
and admissible conclusions are indexed in
[`PAGED_INT8_OPERATOR_EVIDENCE.md`](PAGED_INT8_OPERATOR_EVIDENCE.md).

Those findings rule out repeating the same calls on the same stack, not every
future CANN or custom-kernel route. They also do not remove the host-interface
requirement above: even a capable operator still needs host-owned eligibility,
fallback, capture and replay integration.
