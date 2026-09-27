# C8 Provider Configuration Contract, 2026-09-26

## Scope

`provider_config.py` defines the plugin-owned, construction-time activation
document carried by Host PR #35. It is deliberately independent of vLLM,
vLLM Ascend, Torch and NPU runtime imports.

The parser rejects unknown, missing or duplicate keys, non-finite JSON values,
invalid digests, inconsistent TP channel counts and any offset policy other
than explicit zero-only. Runtime validation then compares actual Host facts
with the configured model, revision, TP group, full-attention layer set,
TP-local channel count and INT8 cache dtype.

`runtime_identity_from_host_config` is the dependency-free adapter for Host PR
#35's construction carrier. It accepts only the fixed target's canonical
`model.layers.N.self_attn.attn` layer name, requires all identity attributes,
normalizes the Torch dtype through its stable string representation, and then
feeds the existing fail-closed runtime comparison. It does not construct an
attention provider or import Host/Torch.

## Required document

The schema is
`vllm-ascend-adaptive-quantized-kv-provider/v1` and requires:

- expected model and revision;
- model-config SHA256 and the logical profile-content SHA256 digest;
- deployment-owned profile path and calibration provenance;
- supported SoC list;
- admitted full-attention layer IDs;
- global and TP-local channel counts plus exact TP size;
- explicit zero-offset attestation;
- paged KV layout and cache-write owner identifiers.

For the fixed target, the admitted layers are
`3,7,11,15,19,23,27,31,35,39`, global channels are 512, and TP2-local
channels are 256.

`profile_sha256` is not the digest of a deployment directory, archive or full
model shard. It is the deterministic `profile_content_sha256` emitted by
`scripts/audit_c8_profile_artifact.py`: each admitted logical tensor contributes
its canonical name/dtype/shape descriptor, byte length and exact payload bytes
in sorted layer/field order under the explicit
`sha256-framed-logical-tensors-v1` domain. The receipt also exposes the
algorithm identifier and each tensor digest. This binds activation to the
reviewed calibration values while remaining stable if the same tensors are
repacked into different safetensors shards. It does not establish calibration
quality; provenance and correctness gates remain separate requirements.

The artifact audit also rejects duplicate/non-finite JSON, ambiguous
safetensors headers, overlapping/out-of-bounds tensor ranges, and any absolute
or parent-traversing shard path before reading tensor bytes. These checks
protect the receipt boundary only; they do not make an untrusted model artifact
safe to execute.

## Boundary

This commit provides schema parsing and runtime-identity validation only. It
does not provide calibrated scale values, load a profile, construct a Host
provider, activate the extension, start an NPU service or establish runtime
correctness/performance. The per-field evidence and missing closure are
enumerated in
[`C8_PROVIDER_FIELD_VERIFICATION_MATRIX_20260927.md`](C8_PROVIDER_FIELD_VERIFICATION_MATRIX_20260927.md).
The manifest remains `import_only` until those gates close.
