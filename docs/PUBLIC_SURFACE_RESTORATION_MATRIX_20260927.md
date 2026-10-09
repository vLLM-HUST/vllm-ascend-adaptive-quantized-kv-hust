# Public Surface Restoration Matrix, 2026-09-27

## Purpose

Plugin Issue #1 records that website PR
[`vLLM-HUST/vllm-hust-website#303`](https://github.com/vLLM-HUST/vllm-hust-website/pull/303)
merged as `67555a509024564ddc23ad05fab31e825f3d5154` and removed
`Adaptive Quantized KV` from the public MOD Workshop while retaining historical
registry and evidence records. This matrix maps each owner requirement to
evidence, remaining work and the responsible owner. It is an execution
checklist, not evidence that any incomplete gate passed.

Current public state: **delisted**. Current manifest state: **`import_only`**.

## Restoration requirements

| Gate | Required evidence | Current evidence | Status | Next action and owner |
| --- | --- | --- | --- | --- |
| R1 profile delivery | Revision-bound calibrated Qwen3.5-35B-A3B KV profile; auditable method, dataset, model/config/profile digests and explicit zero-offset policy | Strict profile schema, logical tensor digest and artifact auditor exist; no calibrated target values exist | Blocked | Host owner selects ModelSlim-compatible sidecar or Host-owned external profile source; assigned calibration owner produces the profile and provenance |
| R2 Host reachability | Accepted C8 selection/loading and provider carrier on an exact Host revision; only ten full-attention layers enter C8; unsupported inputs fail closed | Host Draft PR #35 carries the provider contract/config; source audit proves the public BF16 target does not reach C8 from plugin activation alone. PR #37 head `3fb17f2a9` passes pre-commit/mypy and the complete no-device CPU job with 6,890 passed and 49 skipped; the required `ready-all` source-change gate is queued on the self-hosted test runner and the PR is not landed | Blocked | Complete the `ready-all` gate and land PR #37, validate PR #35 CPU tests, then obtain Host-owner architecture acceptance and exact delivery revision |
| R3 provider construction | Provider loads and revalidates the exact profile before graph capture; model, revision, config, profile, TP, layer, layout, SoC and cache-write ownership match | Atomic off-device `ProviderConstructionContext` validates canonical config plus Host identity; profile loading and execution are absent | In progress | Implement only after R1/R2 choose the actual data source and accepted Host interface |
| R4 Manager lifecycle | On exact Host/plugin heads, real `discover`, `check`, `plan` and `run` reach the intended provider; conflicts and unsupported states fail closed | Static Manager source audit and inert distribution checks pass; no real activation run exists | Blocked | Pin accepted R1-R3 heads, build an internal correctness candidate, then retain Manager command/output receipts |
| R5 NPU serving correctness | Worker coverage, TP2, APC, MTP2, async, hybrid attention/Mamba cache, `FULL_AND_PIECEWISE`, numerical oracle, graph capture/replay and fallback all pass | CPU reference and contract tests only | Blocked | Execute the preregistered correctness matrix after R4; stop on any provenance, correctness, graph or fallback failure |
| R6 rollback and cleanup | Disable/uninstall/rollback restores the native path; service exit releases processes and NPU memory; no project task remains | Source-level fallback/ownership contracts only | Blocked | Capture process, memory, uninstall, rollback and exit receipts in the same bounded R5 run |
| R7 quality and performance | Each conclusion links raw artifacts, exact revisions, commands, workload and checksums; activation success is not treated as a result | No authorized target result | Blocked | After R5/R6, run preregistered matched baselines; report failures and null results as retained evidence |
| R8 restoration packet | Issue #1 links final PRs/heads/CI and all R1-R7 evidence; owner approves restoration | This matrix and existing Draft evidence only | Blocked | Request `public_surface` restoration only after every preceding row is complete |

## Current executable order

1. Finish PR #37's Host-main CPU test gate on current `main`.
2. Rebase PR #35 only after the baseline fix is available; run its CPU contract
   tests without folding the baseline repair into provider code.
3. Obtain the Host owner's R1 profile-delivery and R2 architecture decisions.
4. Implement R3 against the accepted revisions and calibrated artifact, keeping
   the public package `import_only`.
5. Produce an internal, fail-closed correctness candidate and run R4.
6. Run one bounded R5/R6 correctness session with retained raw receipts.
7. Run R7 only after correctness and cleanup pass.
8. Assemble R8 in Issue #1 and request owner review. Do not edit the website or
   claim restored availability before approval.

## Claim boundary

Schema tests, source audits, CPU references, inert wheel validation, Manager
dry runs and successful imports are not substitutes for R4-R7. No row may be
marked complete from a projected result or evidence produced for a different
model, revision, TP degree, cache mode or graph mode.
