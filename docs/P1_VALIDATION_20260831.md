# P1 Static Bundle Validation

> Historical evidence: this record validates the superseded Bundle v1 prototype.
> Current packaging uses `vllm-hust-ext` and Manifest 0.2 experimental; the old
> command names below are retained only to preserve the original evidence.
> Current Manager releases reject enablement while the implementation status is
> `import_only`; this supersedes the historical enablement observation below.

## Scope

This receipt validates packaging and lifecycle-manager discovery only. It does
not validate a vLLM runtime, an Ascend operator, model correctness, NPU use, or
performance.

## Source

- Bundle version: `0.1.0.dev0`
- Bundle ID: `org.vllm-hust.ascend-adaptive-quantized-kv`
- Lifecycle manager: `vllmhust` at
  `bcb74a9fb5e65979a61e169913c1bc7647fbd70a`
- Validation Python: CPython `3.12.13`
- Validation environment:
  `/Users/anon/Projects/.venvs/aqkv-bundle-p1-20260831`
- Isolated lifecycle config:
  `/Users/anon/Projects/vllm-ascend-adaptive-quantized-kv-hust/artifacts/p1-validation/config.json`

## Results

| Gate | Result |
| --- | --- |
| Unit tests | PASS, 9 tests |
| Ruff lint | PASS |
| Ruff format check | PASS |
| Isolated sdist build | PASS |
| Isolated wheel build | PASS |
| Repeated artifact digest | PASS, unchanged |
| Wheel manifest count | PASS, exactly one |
| `vllmhust plugin list` | PASS |
| `vllmhust plugin inspect` | PASS |
| `vllmhust plugin validate` | PASS |
| Explicit enablement | PASS |
| Runtime activation remains absent | PASS, no `VLLM_PLUGINS` emitted |
| PyPI name lookup | Unregistered as of 2026-08-31; reservation not guaranteed |

## Artifacts

- wheel:
  `vllm_ascend_adaptive_quantized_kv_hust-0.1.0.dev0-py3-none-any.whl`
- wheel SHA256:
  `59705e20ad543dac2376ed3612f899de4773e3821890b6c76e3cb3bb05b34211`
- sdist:
  `vllm_ascend_adaptive_quantized_kv_hust-0.1.0.dev0.tar.gz`

Generated package files under `dist/` remain ignored. A release rebuild must
record fresh wheel and sdist digests from the committed source and must not
reuse these local development artifacts. The source document does not embed
its own sdist digest because doing so would be self-referential.

## Lifecycle observation

After explicit enablement in an isolated config, `vllmhust plugin env` emitted:

```json
{
  "VLLMHUST_ENABLED_BUNDLES": "org.vllm-hust.ascend-adaptive-quantized-kv"
}
```

The absence of `VLLM_PLUGINS` is required in P1. It proves that package
enablement cannot accidentally load project behavior before the host contract
is accepted.

The first enablement command was detected using the default user config rather
than the intended isolated config. It was immediately disabled and verified as
empty. The reported result above was then reproduced with the explicit
`VLLMHUST_CONFIG` path. No vLLM process, model service, or NPU task was started.

## Next gate

P2 requires repository-owner review of the generic observer and ACL graph
lifecycle contracts in `HOST_CONTRACT_PROPOSAL.md`. No runtime activation or
NPU run starts before that review.
