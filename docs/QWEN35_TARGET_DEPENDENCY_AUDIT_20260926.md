# Qwen3.5 Target Dependency Audit, 2026-09-26

## Scope and revisions

This is a source-only H1 audit. It does not import the Ascend runtime, inspect
an NPU, start a service, or establish correctness or performance.

- host base: `vLLM-HUST/vllm-ascend-hust@fbe4911bb54ce493b3fcbbf6238b032b9dc07ec6`;
- candidate host contract: Draft PR #35 at
  `11382832d9b88e6a7bdf7a20f4d47c7361fd4c8e`, rebased onto that host base;
- required target: `Qwen3.5-35B-A3B` BF16, TP2, APC, MTP2, async scheduling, and
  `FULL_AND_PIECEWISE`;
- plugin state: `import_only`, no active provider.

The question is not whether each feature appears somewhere in the repository.
The acceptance question is whether one pinned C8 model artifact and one exact
host revision support the complete combination without changing its semantics.

## Dependency matrix

| Dependency | Source evidence | Test evidence | Classification | Required closure |
| --- | --- | --- | --- | --- |
| Exact model family | `patch_qwen3_5.py` patches Qwen3.5 decoder and MTP behavior; hybrid-cache code names Qwen3.5 explicitly | Exact BF16 `Qwen/Qwen3.5-35B-A3B` appears in four-card and speculative-decode tests | Source-backed, partly test-backed | Pin the exact BF16 model revision/artifact |
| TP2 | Exact config has 2 KV heads and explicit `head_dim=256`; checkpoint profile tensors therefore have 512 global channels and Host `_prepare_c8_scales()` shards them to 256 channels per TP2 rank | Exact-model TP2 appears in `test_qwen3_5_35b_a3b_w8a8.py`; another exact-model test uses DP2/TP2; the profile auditor has synthetic TP2 hybrid coverage | Source-derived and CPU-contract-tested | Confirm the real target artifact has 512-channel tensors and each rank receives its expected one-head, 256-channel shard |
| C8 KV on the exact target | PR #35 is wired only through `AscendC8AttentionBackendImpl`; that implementation is selected by checkpoint quantization metadata with `kv_cache_type=C8` and loaded scale/offset weights. Ascend disables runtime `calculate_kv_scales` | Provider unit tests use synthetic C8 tensors; the exact public BF16 model manifest has no ModelSlim description, while the separate TP2 test uses W8A8 weights | Blocking reachability mismatch | Provide a revision-bound BF16 KV calibration profile or Host-owned scale source; do not substitute the W8A8/C8 checkpoint or default `1/0` parameters |
| C8 offset semantics | Cache write computes `round(x / scale + offset)`; the dense fallback reconstructs `q * scale` without subtracting offset | Existing CPU round-trip coverage uses zero offsets | Blocking for asymmetric profiles | Require explicit zero offsets for the correctness candidate or fix and validate every native/fallback dequantization path |
| BF16+C8 profile precedent | A local Qwen3-14B fixture retains BF16/FLOAT weights plus per-channel C8 KV metadata for all 40 layers; 80 scale tensors are positive and all 80 offset tensors are zero | Standard-library artifact audit passes without loading model weights or an NPU | Format precedent only | Reproduce the profile for the exact Qwen3.5 revision with calibration provenance; do not reuse Qwen3-14B values |
| Hybrid attention/Mamba model | Pinned config has 40 layers: full attention only at `3,7,11,15,19,23,27,31,35,39`; the other 30 are linear attention. Qwen3.5 patch and hybrid-cache code separate these paths | The artifact auditor rejects missing full-attention coverage and any profile on a linear-attention layer; no real C8-provider test yet covers recurrent-state and attention-cache interaction | Scope fixed; runtime interaction unknown | Require exactly 40 profile tensors on the ten full-attention layers, then prove hybrid cache bookkeeping and MTP state remain unchanged |
| APC | Provider eligibility recognizes cached multi-token prefill by `query_length > 1` and `kv_length > query_length`; hybrid config has explicit prefix-cache alignment behavior | APC tests exist for other Qwen3.5 variants/platforms; no exact-target C8-provider APC test | Source-backed intent, combination untested | Define the exact two-request prefix-hit fixture and prove provider dispatch only on the second continuing-prefill request |
| MTP2 | Qwen3.5 MTP patch and speculative configuration are present | Exact target appears in MTP-related tests, but retained examples use MTP3 or disable prefix caching; no MTP2+C8 continuing-prefill test | Combination unknown | Freeze `num_speculative_tokens=2` and verify request metadata, accepted-token rollback, cache writes and provider output ownership |
| Async scheduling | Scheduler/model-runner code contains Qwen3.5 and async handling | Async tests exist, but no exact-target APC+C8-provider+MTP2 combination | Combination unknown | Prove decode/prefill row ordering, cumulative Q lengths and KV lengths remain valid after async compaction/reuse |
| `FULL_AND_PIECEWISE` | Hybrid/Mamba config enables this graph mode by default; PR #35 offers the provider before `full_graph_fia` while capturing and retains workspace tensors | Provider unit tests cover a mocked capture branch; exact target tests use either unspecified/default graph mode or `FULL_DECODE_ONLY` | Source-backed intent, real graph unverified | Confirm capture partition, replay lifetime, shape identity and disabled-path equivalence on the exact target |
| Mixed decode/prefill batch | PR #35 slices prefill rows after `num_decodes` and leaves decode on native paged C8 | CPU-only tests cover classification plus aligned slicing of query/output views, cumulative Q/KV lengths and block tables; no exact-target runtime test | Unit-test-backed only | Confirm scheduler ordering guarantee under MTP2+async and test mixed output assembly |
| Provider implementation/kernel | PR #35 defines only discovery, request/result objects and host dispatch | No project provider or NPU kernel exists in the plugin | Required implementation | Add a lazy, fail-closed provider and prove manager enablement reaches it before active publication |
| Host acceptance | PR #35 is open Draft | No host-owner review, accepted API revision or merge receipt | Blocking governance gate | Record architecture disposition, delivery owner and exact accepted revision in Issue #1/PR #35 |
| Host CI | Ruff, format and focused syntax checks pass on the current PR head | Test-only PR #37 passes pre-commit/mypy after declaring the local `V41CacheLayer.kv_cache`; CPU UT then fails during collection because the CI image cannot import `triton.runtime.jit` | Type fix prepared; baseline environment gate open | Land PR #37, repair or identify the baseline Triton image issue, then rerun PR #35 candidate unit tests |

## What PR #35 can already establish

At source level, the candidate removes the need for a plugin monkey patch and
places a default-off provider boundary after C8 cache preparation. It carries
paged NZ INT8 K/V, block tables, TP-local scales and offsets, cumulative
lengths, output ownership, masks and capture state. It distinguishes cached
multi-token prefill from decode and all-new prefill, and it makes decline versus
post-acceptance failure explicit.

Those properties are sufficient to request host review. They are not
sufficient to claim that the complete target matrix works. In particular, the
candidate does not identify a C8 Qwen3.5 artifact, implement the provider, or
prove hybrid-cache, MTP2, async and graph replay interactions.

## Minimum accepted host response

H2 can close only when the host owner or assigned maintainer records all of:

1. accept/revise/replace disposition for the provider architecture;
2. final host delivery owner and exact accepted host revision;
3. exact BF16 `Qwen3.5-35B-A3B` artifact identity, supported hardware, and the
   reviewed route that creates its INT8 KV scales/offsets and paged cache;
   the current candidate must use zero offsets unless Host dequantization is
   revised and validated;
4. whether full-attention-only provider scope is correct for the hybrid model;
5. required APC, MTP2, async and `FULL_AND_PIECEWISE` semantics;
6. capture/replay workspace and failure/fallback requirements;
7. the CI route that actually runs the candidate unit tests.

## Prepared validation matrix, not authorized for execution

After H2 and plugin adapter review, correctness must be established in this
order on an assigned idle device:

1. provider disabled: exact baseline equivalence;
2. provider enabled but ineligible: exact native fallback equivalence;
3. APC continuing prefill without MTP/async, eager first;
4. the same case under `FULL_AND_PIECEWISE` capture and repeated replay;
5. MTP2 added while scheduling remains synchronous;
6. async added last;
7. mixed decode plus continuing-prefill batch;
8. cleanup/restart and an unsupported-case fail-closed receipt.

Every stage requires pinned model/source/config identities and dense-baseline
output agreement. Timing starts only after the complete correctness sequence
passes. No part of this matrix should be executed before H3 pins an enableable
Host/provider/plugin candidate and its fail-closed activation contract.
