# Paged INT8 Continuing-Prefill Operator Evidence

## Purpose and classification

This document maps retained operator-feasibility evidence from the parent
research carrier into the PR #19 component takeover. It is an immutable
evidence index, not a copied experiment, a runtime dependency, or an NPU
acceptance result.

The parent summaries record two real Ascend 910B2 pilot executions. Both are
classified `PILOT_DIAGNOSTIC`: they stopped at operator capability checks
before the preregistered numerical and timing matrix. The raw remote receipts
are not stored in Git; the committed summaries retain their remote paths and
SHA256 values. Consequently, this repository may cite the failures and their
boundaries, but cannot independently reconstruct the raw operator logs from
the Git history alone.

## Frozen records

The M0 probe records and executable were committed together in
[`intellistream/ascend-adaptive-quantized-kv@69137e4e6bd08333707bb9c74241b2501c885eb0`](https://github.com/intellistream/ascend-adaptive-quantized-kv/commit/69137e4e6bd08333707bb9c74241b2501c885eb0).
They are byte-identical at the current retained parent head
`58f93560ee738ebdc548612a1965210d567a6b39`.

| Record | Committed-file SHA256 | Classification | Retained raw receipt |
| --- | --- | --- | --- |
| `artifacts/m0/paged-int8-prefill-probe-contract-20260818.json` | `5dedf67b28e68741f36f5a0e3124c34e702cc78599fcced6b32bef0a04fc6e76` | preregistered revision 2 contract | n/a |
| `artifacts/m0/paged-int8-prefill-nz-pilot-20260818.json` | `97d11c02fb7d6294422903a4461ed6069781d646d1a671394be3e40782af20d1` | pilot diagnostic | remote SHA256 `f2809988027d13b2428d7a742ae05c3829825a9145e6bcf9f4a7c74ca2dad9c9` |
| `artifacts/m0/paged-int8-prefill-nd-pilot-20260818.json` | `9a4ff50b15b671b7038186bbb7fa1bda8cd5752077aba864ae7b12780671f3a9` | pilot diagnostic | remote SHA256 `8bd782e0f127262ed130db86306ee51a479d85164630903fa871e87eb6f05a22` |
| `scripts/probe_m0_paged_int8_prefill.py` | `7ba1bab9a54649e258a0e407024536b0ce9092cdd04723edd4198b9d506086dd` | retained probe implementation | n/a |

The later exact-stack compatibility record was committed in parent revision
[`c3d4c667b74479678e81a86598b3bd864b3d6955`](https://github.com/intellistream/ascend-adaptive-quantized-kv/commit/c3d4c667b74479678e81a86598b3bd864b3d6955):

| Record | Committed-file SHA256 | Classification |
| --- | --- | --- |
| `artifacts/m1/v2-compatibility-audit-20260826.json` | `2c5db68afe94a5e67ad242af991a4f8519fe159b0dc26af65f6d470f6bf0cdd0` | read-only source, docstring, model-config and archived-dispatch audit; no execution |

That audit binds its source facts to parent
`8be03304c1657de2f5eb75de6b859d899e92bef5` and Ascend host
`e2e1c96e3b2f040aaba06b8ba333070d19be4738`.

## Reproducible integrity audit

`scripts/audit_parent_operator_evidence.py` reads these files directly from
the two exact parent commits with `git show`. It rejects non-exact revisions,
hash drift, schema/status changes, changed pilot outcomes, changed raw-receipt
hashes, or a V2 record that no longer declares its no-execution boundary. It
does not fetch branches, import the parent runtime, or run an operator.

```bash
python scripts/audit_parent_operator_evidence.py \
  --research-repo /path/to/ascend-adaptive-quantized-kv \
  --output /path/to/new-exclusive-receipt.json
```

The output is created exclusively and an inconclusive audit exits with status
2. The retained result is
[`evidence/PARENT_OPERATOR_EVIDENCE_AUDIT_20260905.json`](evidence/PARENT_OPERATOR_EVIDENCE_AUDIT_20260905.json),
SHA256 `e17e3a97e796bcaf578f447af15649b4777987b4dac89952f7374fa20dc94f0b`.
It reports `SUPPORTED_BY_PINNED_SUMMARIES`, not runtime acceptance.

## Admissible findings

1. On the pinned 2026-08-18 stack (`torch`/`torch_npu` 2.10.0 and
   `vllm_ascend` 0.21.0rc1), the raw 5D NZ paged INT8 candidate was rejected
   for B1/Q128 because page attention required 3D or 4D K/V when query length
   exceeded 16.
2. A timed NZ-to-4D-ND INT8 repack passed that dimension gate, but the paged
   prompt call rejected the combined per-channel antiquantization parameters
   with error 561002 (`antiquant scale is nullptr`).
3. Both pilots failed before numerical comparison or timing. They provide no
   latency, throughput, quality, HBM, graph-capture, or service evidence.
4. The later V2 audit found no direct documented coverage for the observed
   Qwen3-14B continuing-prefill workload: the `(40,8)` query/KV head pair was
   outside the listed special-case pairs, and all 3,840 archived fallback
   dispatches had query lengths 128-8,320 rather than 1-16. That result is a
   compatibility audit, not an operator execution failure.

These records reject repeating the same two native-call candidates on the
same pinned stack without a changed operator contract. They do not establish
that Ascend hardware, a later CANN release, or a custom kernel cannot implement
paged INT8 continuing-prefill attention.

## Takeover decision

- Do not replace the current host fallback with a simple V1-to-V2 call or the
  already rejected 5D-NZ/4D-ND candidates.
- Keep the PR #19 CPU implementation as a mathematical and input-validation
  reference only. It is not evidence that the same algorithm is available as
  an Ascend operator.
- A fused dense materializer remains an intermediate route: it could remove
  separate gather/layout/cast/scale operations but still emits full dense BF16
  K/V. The parent estimate is 7-12 effective working days, plus 3-5 days if a
  compiler change is required.
- Native paged INT8 attention remains the full route. The parent estimate is
  20-35 effective working days for the bounded Qwen3-14B contract and does not
  include broader generalization.
- Neither route may start from this plugin until the host owner identifies or
  delivers the public C8 execution boundary described in
  `HOST_EXECUTION_AUDIT_20260905.md`. Operator feasibility alone does not solve
  host dispatch ownership, mixed-batch behavior, or graph capture/replay
  workspace lifetime.

The next admissible runtime step is therefore still an owner-reviewed host
interface followed by component correctness and graph-lifecycle validation.
A later operator probe must use a materially changed stack or contract, retain
raw receipts in the approved evidence carrier, and pass correctness gates
before any performance claim.
