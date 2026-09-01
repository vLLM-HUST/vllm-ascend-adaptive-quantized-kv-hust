# vLLM Ascend Adaptive Quantized KV

English | [Simplified Chinese](README.zh.md)

Independent vLLM-HUST extension Bundle for the Adaptive Quantized KV research
project on Ascend.

Repository:
[`vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust`](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust)

## Current scope

This repository is at package stage P1. It provides:

- a static Bundle v1 manifest discoverable by `vllmhust` without importing
  implementation modules;
- fail-closed host-contract compatibility receipts;
- typed validation of the retained ACL graph reset and recapture receipts;
- the proposed generic host-contract boundary for continuing-prefill
  observation and ACL graph lifecycle control.

It does not yet activate runtime behavior, patch vLLM or vLLM Ascend, launch a
model service, or make a performance claim. The current host baselines do not
yet expose the reviewed contracts required by this Bundle. Installing the
package is therefore inert.

## Delivery policy

This organization plugin is the project implementation carrier. The project
does not submit optimization commits or PRs directly to `vllm-hust` or
`vllm-ascend-hust`. Missing generic host interfaces are reviewed in this
repository and must be delivered by the host owner or another explicitly
approved route. Reproducible workload contracts remain in the research carrier
with its fixed `intellistream/llm-serving-workloads` gitlink.

## Identity

| Layer | Identifier |
| --- | --- |
| Distribution | `vllm-ascend-adaptive-quantized-kv-hust` |
| Python package | `vllm_ascend_adaptive_quantized_kv` |
| Bundle ID | `org.vllm-hust.ascend-adaptive-quantized-kv` |
| Component | `continuing-prefill-profiler` |
| Proposed contract | `vllm.ascend.continuing-prefill.observer.v1` |

## Validation

```bash
uv venv --python 3.12
uv pip install -e ".[test]"
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m build
.venv/bin/python scripts/validate_distribution.py --dist dist
```

Install the resulting wheel together with `vllmhust` in a clean environment,
then run:

```bash
vllmhust plugin list
vllmhust plugin inspect org.vllm-hust.ascend-adaptive-quantized-kv
vllmhust plugin validate org.vllm-hust.ascend-adaptive-quantized-kv
```

Validation proves package discovery and metadata only. It is not a runtime or
NPU result.

The repository CI runs the tests on Python 3.10, 3.12, and 3.14. Python 3.12
also builds the wheel and sdist, validates the inert Bundle boundary, and
retains the validated distributions for 14 days. CI does not publish to PyPI.

## Research evidence

Experiment contracts, raw evidence, runners, and claims remain in
[`intellistream/ascend-adaptive-quantized-kv`](https://github.com/intellistream/ascend-adaptive-quantized-kv).
They are not copied into the runtime package.

## Security and release

- Installation never enables the Bundle.
- Runtime activation will require an explicit `vllmhust plugin enable` after
  the host contract is accepted.
- PyPI publication should use Trusted Publishing.
- Passwords, 2FA codes, recovery codes, and long-lived API tokens must not be
  stored in this repository or CI configuration.
