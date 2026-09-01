# Legacy Contract Evidence

## Owner direction

The 2026-09-01 owner reply on plugin Issue #1 maps the reconciled Ascend
profiling drafts to this existing plugin carrier rather than to another plugin:

- [legacy PR #271](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/271);
- [legacy PR #279](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/279).

These links retain source-level evidence for the proposed continuing-prefill
observer and ACL graph-lifecycle contracts. They do not make either contract an
available host API.

## Exact records

| Record | Frozen head | State | Contract evidence | Admissible conclusion |
| --- | --- | --- | --- | --- |
| Legacy PR #271 | `3689bbc11b419ef60817b80f24a33268f0a4f24a` | Closed Draft | continuing-prefill scope placement; opt-in graph reset, recapture, and seal receipts | The source boundaries were reviewable, but r5 failed output stability, the 5% perturbation gate, and replay identity. |
| Legacy PR #279 | `360056b2741fca6e612640c15313eec04084eec8` | Open Draft | descriptor-bound ACL graph capture/replay markers and lifecycle tests | A bounded scalar probe established feasibility only; no accepted 14B mapping, runtime, or performance result exists. |

PR #271 maps to both proposed contracts because it contains the
continuing-prefill observation sites and the dev-only graph lifecycle helper.
PR #279 retains those sites and adds the graph-safe identity candidate needed
to reason about capture/replay ownership.

## Evidence boundary

- The legacy repository remains the source location; old implementation files
  are not copied into this plugin.
- The records are contract-design boundary and negative-boundary evidence, not
  runtime dependencies, host baselines, merge dependencies, or delivery
  carriers.
- Closed or Draft state is preserved. Neither record is represented as merged,
  accepted, or available through a documented host API.
- No latency, throughput, correctness, kernel, or service-level claim may be
  derived from these records.
- A later change to PR #279 must be recorded as a new reviewed exact revision;
  this table must not follow a moving branch implicitly.
- Runtime activation remains blocked on the owner decision in plugin Issue #1
  and on host-owner delivery of any required interface.

Raw experiment receipts and claim adjudication remain in the parent research
carrier. This document stores only the immutable source references and their
contract interpretation.
