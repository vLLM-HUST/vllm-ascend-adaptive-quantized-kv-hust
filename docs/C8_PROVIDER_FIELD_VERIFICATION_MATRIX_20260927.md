# C8 Provider Field Verification Matrix, 2026-09-27

## Purpose

This matrix separates four claims that must not be conflated:

1. the activation document is syntactically valid;
2. a configured expectation matches Host facts in a CPU-only contract test;
3. an external artifact has been read and cryptographically bound;
4. the fixed target has passed real-device correctness.

The current plugin proves only the first two classes. The standalone profile
auditor can produce evidence for part of the third class, but no provider
factory currently loads that artifact or compares it with the activation
document. Nothing in this matrix is NPU correctness or performance evidence.

## Evidence levels

| Level | Meaning |
| --- | --- |
| S | Schema-only: type, shape, enum-like policy, or cross-field consistency |
| H | Host-bound CPU contract: a fact supplied by Host is compared fail-closed |
| A | Artifact-bound: bytes or metadata are independently read and verified |
| N | Real-device correctness: the pinned target has passed the registered NPU gate |

`H`, `A`, and `N` are cumulative claims only when the same execution path binds
them together. A standalone `A` receipt does not raise an activation field to
`A` until the provider checks that receipt or recomputes the same fact.

## Activation fields

| Field | Current evidence | What is actually proved | Missing closure |
| --- | --- | --- | --- |
| `schema_version` | S | Exact `vllm-ascend-adaptive-quantized-kv-provider/v1` match | Pin the same parser revision in the enabled Manager artifact |
| `expected_model` | H (CPU) | Host-supplied model string is compared exactly | Exercise the comparison while constructing the real target |
| `expected_model_revision` | H (CPU) | Host-supplied revision is compared exactly, including rejection of `None` | Prove the runtime resolves the pinned revision rather than a mutable default |
| `model_config_sha256` | S | Lowercase SHA256 syntax only | Read the resolved `config.json`, hash its exact bytes, and compare before provider activation |
| `profile_path` | S | Non-empty, NUL-free string only | Choose the delivery route, resolve a deployment-owned path safely, and load it before graph capture |
| `profile_sha256` | S; standalone A available | Digest syntax is checked; `audit_c8_profile_artifact.py` can independently emit the logical profile digest | Provider must recompute/consume the versioned logical digest and compare it before activation |
| `calibration_provenance` | S | Non-empty string only | Define a machine-checkable provenance receipt and bind it to the exact profile and model revision |
| `supported_soc` | S | Non-empty, unique strings only | Obtain the runtime SoC identity from an authoritative Host/runtime source and reject non-members |
| `full_attention_layer_ids` | H (CPU) | Sorted unique IDs are required; the Host adapter accepts only canonical `model.layers.N.self_attn.attn` names and the derived ID must be admitted | Provider factory must call the adapter; the real target must prove only the ten full-attention layers activate |
| `global_channels_per_tensor` | S | Positive value and `global = tp_size * local` | Bind global shape to the loaded profile and pinned model config |
| `tp_size` | H (CPU) | Host TP world size and rank bounds are compared | Exercise both ranks on the fixed TP2 target |
| `tp_local_channels_per_tensor` | H (CPU) | `num_kv_heads * head_size` is compared with the configured local count | Bind each loaded scale/offset tensor to this local shape on both ranks |
| `zero_offsets_attested` | S; standalone A available | Config must be `true`; the standalone artifact audit rejects nonzero offset values | Provider must load the same audited profile and re-establish zero offsets before activation |
| `kv_layout` | S | Non-empty string only | Replace free text with a pinned identifier and compare it with the Host request/cache layout before accepting a request |
| `cache_write_owner` | S | Non-empty string only | Define a pinned owner identifier from the Host path and prove one writer owns cache mutation for the accepted request |

## Host facts already carried

Host PR #35 constructs the provider before warmup/graph capture and supplies
`layer_name`, model/revision, TP rank/size, heads, KV heads, head size, attention
scale, KV cache dtype, and canonical provider JSON. Its request object supplies
paged INT8 K/V views, block table, antiquant scales/offsets, output, sequence
lengths, block size, sparse mode, and capture state. CPU tests establish the
carrier and fail-closed provider protocol; they do not establish that a usable
provider exists.

The current plugin has no provider factory. `ProviderActivationConfig` can
compare a `ProviderRuntimeIdentity`, and `runtime_identity_from_host_config`
can now derive that identity from Host's construction carrier without a
Host/Torch import. The adapter is not yet called by an actual provider factory.
Therefore the current honest status remains `import_only` and
`HOST_CONFIG_AND_PLUGIN_SCHEMA_READY_PROFILE_AND_PROVIDER_STILL_MISSING`.

## Ordered closure

1. Select the BF16-to-C8 delivery route: ModelSlim-compatible sidecar first, or
   a new Host-owned external profile source.
2. Implement provider construction that converts Host facts into
   `ProviderRuntimeIdentity`, derives the admitted layer ID, and validates it.
3. Load the exact profile before graph capture; compare the versioned logical
   digest, exact tensor names/shapes, zero offsets, model config digest, and
   provenance.
4. Add authoritative runtime checks for SoC, KV layout, and cache-write
   ownership. Do not silently weaken these fields to comments or labels.
5. Run Manager disabled/conflict/enabled tests. Only then leave `import_only`.
6. Run the preregistered real-device correctness matrix. Performance remains
   blocked until correctness, graph replay, cleanup, and resource gates pass.
