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

## Required document

The schema is
`vllm-ascend-adaptive-quantized-kv-provider/v1` and requires:

- expected model and revision;
- model-config and profile SHA256 digests;
- deployment-owned profile path and calibration provenance;
- supported SoC list;
- admitted full-attention layer IDs;
- global and TP-local channel counts plus exact TP size;
- explicit zero-offset attestation;
- paged KV layout and cache-write owner identifiers.

For the fixed target, the admitted layers are
`3,7,11,15,19,23,27,31,35,39`, global channels are 512, and TP2-local
channels are 256.

## Boundary

This commit provides schema parsing and runtime-identity validation only. It
does not provide calibrated scale values, load a profile, construct a Host
provider, activate the extension, start an NPU service or establish runtime
correctness/performance. The manifest remains `import_only` until those gates
close.
