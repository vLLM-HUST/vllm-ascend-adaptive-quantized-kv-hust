# BF16-to-C8 Profile Delivery Decision, 2026-09-26

## Decision

The public `Qwen/Qwen3.5-35B-A3B` BF16 checkpoint cannot enter the current C8
continuing-prefill provider through plugin activation alone. The project must
choose one of two profile-delivery routes before a usable correctness
candidate can be built.

For the shortest correctness baseline, use a ModelSlim-compatible BF16+C8
sidecar/checkpoint artifact. For the eventual product-quality plugin route,
add a Host-owned external C8 profile source. The two routes may share the same
calibrated values and attestation schema, but they have different Host
ownership and implementation cost.

## Source-backed ordering constraint

Host startup currently runs:

1. `maybe_auto_detect_quantization(vllm_config)`;
2. `_fix_incompatible_config(vllm_config)`;
3. `init_ascend_config(vllm_config)`.

Therefore quantization detection and `vllm_config.quant_config` construction
happen before `AscendConfig` materializes the provider factory/config fields.
Later, only `AscendC8KVCacheAttentionMethod` changes the attention
implementation to `AscendC8AttentionBackendImpl`, allocates INT8 cache scale
and offset parameters, and configures the continuing-prefill provider.

This rules out a plugin-only solution: the provider factory is downstream of
the decision that creates the C8 cache and tensors it consumes.

## Route A: ModelSlim-compatible BF16+C8 artifact

Package the unchanged BF16 model weights with:

- `quant_model_description.json` declaring `model_quant_type=FLOAT` and
  `kv_cache_type=C8`;
- exactly 40 scale/offset tensors for full-attention layers
  `3,7,11,15,19,23,27,31,35,39`;
- 512 global channels per tensor, sharded to 256 channels per TP2 rank;
- finite positive scales and explicit zero offsets;
- pinned model/config/profile digests and calibration provenance.

The operator supplies this artifact or wrapper model path. Existing ModelSlim
auto-detection, parameter creation, checkpoint loading and TP sharding remain
the source of truth. Extension Manager still uses its normal vLLM Provider;
the published manifest must not hard-code the deployment path.

### Cost and risk

- Host production changes: low, assuming the current ModelSlim path accepts
  the exact hybrid model and the profile tensors load under the expected names.
- Artifact tooling and validation: medium; the wrapper/index must be
  reproducible and must not copy or mutate the 14 BF16 weight shards silently.
- Runtime risk: medium; hybrid cache, MTP2, async and graph replay still need
  the complete correctness matrix.
- Main dependency: calibrated Qwen3.5 values do not yet exist in this project.

This route is the fastest way to establish the requested engineering baseline,
but the deployment artifact, not the plugin, owns C8 selection.

## Route B: Host-owned external C8 profile source

Add a generic Host profile contract that is visible before quantization
detection. A reviewed implementation must:

1. parse and validate the Host-owned profile selector from raw
   `vllm_config.additional_config` before `init_ascend_config()`;
2. reconstruct `vllm_config.quant_config` as a FLOAT-weight/C8-KV config
   without requiring a ModelSlim description inside the public model repo;
3. select C8 only for the ten full-attention layers;
4. load revision-bound K/V scales and zero offsets before cache use;
5. preserve existing TP sharding, cache-write ownership and weight loading;
6. compare the Host-loaded profile with the provider's immutable attestation;
7. reject model/revision/hash/layer/shape/offset mismatches before graph
   capture.

The provider JSON added at Host head `084f70f5` remains useful for construction
attestation, but it cannot be the earlier Host quantization selector because it
is plugin-owned and is materialized too late in the current startup order.

### Cost and risk

- Host production changes: high; quantization detection, configuration,
  ModelSlim C8 metadata creation and scale loading all change.
- Host tests: high; disabled-path equivalence, explicit-user-quantization
  precedence, hybrid-layer filtering, TP2 sharding and failure atomicity need
  unit coverage before runtime testing.
- Plugin changes: medium after Host review; profile attestation, provider
  construction and strict eligibility can reuse the schema already in PR #4.
- Runtime risk: high until APC/MTP2/async/FULL_AND_PIECEWISE replay is proven.

This is the cleaner long-term plugin route because the public BF16 model stays
unchanged and the Host explicitly owns cache quantization.

## Rejected routes

- Activating only the current plugin: cannot create C8 cache state upstream of
  provider construction.
- Treating default scale/offset `1/0` as a profile: constructible but not
  calibrated correctness evidence.
- Reading NPU offset values in `is_eligible()`: violates the side-effect-free
  request gate and graph-capture contract.
- Online scale calculation: disabled in the pinned Ascend source and would
  introduce a different calibration/runtime contract.
- Substituting the W8A8/C8 model: does not satisfy the fixed BF16 target.

## Work that can proceed before the decision

- keep the strict provider activation schema and source audits green;
- define the reproducible profile manifest and calibration provenance without
  inventing scale values;
- prepare Manager merge/conflict tests and the correctness matrix;
- keep Host PR #35 default-off and plugin publication `import_only`.

No service, NPU run or performance claim is justified until one delivery route
has an actual calibrated profile and passes the off-device construction gates.
