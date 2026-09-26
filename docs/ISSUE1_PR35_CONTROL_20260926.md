# Issue #1 / Host PR #35 Control, 2026-09-26

## Authority and precedence

This document records the latest owner direction in plugin Issue #1 and
supersedes older execution plans where they conflict. It does not rewrite or
invalidate historical evidence.

The CPU reference and the existing evidence package are accepted as delivered.
The next host validation target is fixed to:

- model: `Qwen3.5-35B-A3B` BF16;
- tensor parallelism: `TP2`;
- required retained behavior: APC, MTP2, async scheduling, and
  `FULL_AND_PIECEWISE` graph mode.

The released/default plugin remains `import_only` and fail closed until the
Host/provider revision and manager-activation contract pass off-device review.
An opt-in active candidate is then required for H4 correctness. No model
service, operator retry, performance test, or performance claim is allowed
before its stage gate.

The owner follow-up on 2026-09-26 asks the existing PRs to produce an actually
usable Host/provider version that the extension manager can enable, then prove
the complete target's correctness. Performance validation remains owner-gated
after that evidence. The current `import_only` state is still correct while the
provider and activation contract are incomplete; it is now a staging state,
not a direction to stop implementation.

## Ownership boundary

| Area | Owner | Current boundary |
| --- | --- | --- |
| Final host route and go/no-go | Shuhao / host owner | Selects or rejects the host integration route and exact accepted host revision |
| Host interface delivery | Shuhao / assigned host maintainer | Remains host-owned; it is not transferred to the plugin project |
| Candidate host contract | XilingGao, through Draft PR #35 | Review input only: minimal, generic, default-off implementation allowed by the 2026-09-20 correction |
| Plugin contracts and adapter | XilingGao | May implement the reviewed candidate off-device, but must prove target reachability before activation |
| CPU reference, tests, and evidence | XilingGao | Delivered; may be maintained without implying NPU support |
| Correctness-only activation | XilingGao plus host review | Allowed only at H3 for the pinned candidate; remains opt-in and fail closed |
| Performance evidence | Shuhao / owner-gated delivery | Blocked until the complete H4 correctness matrix passes |

Draft PR #35 therefore does not contradict host ownership. It is a candidate
implementation submitted for CODEOWNERS/maintainer review. Acceptance, adoption,
assignment, merge, and the final delivery revision still belong to the host
owner and maintainers.

## What Draft PR #35 resolves as a candidate

Draft PR
[`vLLM-HUST/vllm-ascend-hust#35`](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/35)
is currently pinned here at
`084f70f50dfcdf2daf66b3a31813bc982c2d1d09`, rebased without patch changes
onto `vllm-ascend-hust@fbe4911bb54ce493b3fcbbf6238b032b9dc07ec6`. Its
candidate source addresses:

1. Default-off `module:factory` provider discovery instead of a project-specific
   import or monkey patch.
2. A typed request boundary carrying causal multi-token TND query/output,
   paged 5D NZ INT8 K/V, block table, cumulative query/KV lengths, TP-local
   scales and offsets, GQA/head/block/layout metadata, masks, and graph/capture
   context.
3. Host-side identification of cached multi-token prefill; all-new prefill and
   decode remain on native paths.
4. Mixed-batch separation where continuing-prefill rows may reach the provider
   while decode rows remain native.
5. Explicit pre-acceptance fallback through `is_eligible=False`; after provider
   acceptance, provider errors propagate fail closed.
6. Output and workspace lifetime retention needed by capture-sensitive code.
7. A default-disabled path intended to preserve current host behavior when the
   provider is not configured.
8. A CPU-only mixed-batch contract test now locks the decode-row removal and
   corresponding query/output, cumulative-length, KV-length, and block-table
   slicing assumptions.
9. A finite JSON provider-config channel now combines operator-owned settings
   with actual model/revision, TP rank/size, layer shape and dtype in one
   immutable factory input before graph capture.

These are source-level properties of the candidate, not an accepted public API
or a real-device result.

## What still requires host-owner confirmation

Before plugin integration or NPU work, the host owner must confirm:

1. Whether PR #35's provider architecture is accepted, should be revised, or
   should be replaced by another host-owned interface.
2. The assigned host maintainer, review route, and exact accepted/merged host
   commit that plugin work may depend on.
3. Whether the request and result semantics cover `Qwen3.5-35B-A3B` with TP2,
   APC, MTP2, async scheduling, mixed batches, and
   `FULL_AND_PIECEWISE` capture/replay.
4. Capture ordering, replay stability, workspace lifetime, fallback behavior,
   failure behavior, and disabled-path equivalence.
5. Whether the separate observer and ACL graph lifecycle contracts are needed,
   and their accepted identifiers or replacements.
6. The provider/kernel delivery route. PR #35 does not contain a project
   provider, a native paged-INT8 continuing-prefill operator, or activation.
7. How to resolve the current CPU-UT CI environment failure before treating
   the PR as validated. Test-only PR #37 closes the unchanged `V41CacheLayer`
   mypy declaration and passes pre-commit. Its CPU UT then reaches collection
   but fails because the CI image cannot import `triton.runtime.jit`; this is
   still upstream of provider tests and must not be reported as their result.
8. How the standard BF16 target reaches C8 storage and the provider. Current
   source configures PR #35 only through checkpoint metadata declaring
   `kv_cache_type=C8`; plugin enablement alone cannot make the BF16 path C8.
   See [`BF16_C8_REACHABILITY_AUDIT_20260926.md`](BF16_C8_REACHABILITY_AUDIT_20260926.md).
9. Which revision-bound calibration profile or Host-owned scale source supplies
   the exact BF16 target's K/V scales. Runtime `calculate_kv_scales` is disabled
   on Ascend, and default `1/0` parameters are not correctness evidence.
10. Whether the correctness candidate accepts only zero offsets. The current
    cache writer adds offsets while the dense fallback does not subtract them;
    nonzero offsets must fail closed until the Host paths share one formula.
11. Whether the supplied profile matches the pinned hybrid model exactly: C8
    metadata only for full-attention layers
    `3,7,11,15,19,23,27,31,35,39`, 512 global channels per tensor, and 256
    channels per TP2 rank. Linear-attention layers must not be admitted to the
    provider.

## Allowed work before confirmation

- Keep this ownership/dependency map current in Issue #1 and PR #35.
- Audit the exact host source for target-model and required-mode dependencies.
- Add fail-closed, CPU-only fixtures and contract tests that do not claim host
  acceptance or runtime reachability.
- Implement a lazy, default-off provider candidate and its manager-activation
  contract off-device, while retaining `import_only` until target reachability
  and activation tests are complete.
- Use the Extension Manager's built-in vLLM Provider. Static activation may fix
  the provider factory, while deployment profile settings remain
  operator-owned `--additional-config`; do not create a private Manager
  Provider to bypass the Host contract.
- Define and test the plugin-owned provider JSON schema now that the Host
  candidate supplies canonical construction-time configuration and runtime
  identity. Profile loading, digest checks and zero-offset validation remain
  provider-construction work, not per-request eligibility. The source-backed
  status is recorded in
  [`C8_MANAGER_PROVIDER_ACTIVATION_AUDIT_20260926.md`](C8_MANAGER_PROVIDER_ACTIVATION_AUDIT_20260926.md).
- Preserve the delivered CPU reference and historical evidence.
- Prepare a bounded real-device correctness matrix without executing it.

## Prohibited work before confirmation

- Do not change `import_only`, publish an active wheel, or run a service before
  the H3 provider/manager contract is complete and pinned for correctness.
- Do not start a model service, reserve an NPU, run an operator probe, or repeat
  previously failed native operator calls.
- Do not publish performance, memory, graph, or quality claims.
- Do not treat a Draft PR, dry run, projected result, or source audit as host
  acceptance or real-device evidence.
- Do not make project code responsible for private host graph-pool state.

## Ordered stages

| Stage | Work | Exit condition |
| --- | --- | --- |
| H0 | Publish this PR #35-to-owner-decision mapping in Issue #1 | Issue reply links the exact PR head and separates resolved candidate work from pending host decisions |
| H1 | Source-only target/mode dependency audit | Every Qwen3.5/TP2/APC/MTP2/async/FULL_AND_PIECEWISE dependency is mapped to source, test, unknown, or blocker in [`QWEN35_TARGET_DEPENDENCY_AUDIT_20260926.md`](QWEN35_TARGET_DEPENDENCY_AUDIT_20260926.md) |
| H2 | Host CI and BF16-to-C8 reachability | PR #37 is landed and the Host CPU-UT environment can collect tests; a revision-bound scale profile or accepted Host scale source covers exactly the ten full-attention layers with 512 global/256 TP2-local channels, zero offsets and calibration provenance, allowing the exact BF16 target to construct the reviewed paged-INT8 request without substituting a different model |
| H3 | Host/provider and plugin-manager candidate | Exact Host/Manager/provider revisions are pinned; the Host configuration carrier is present at `084f70f50dfcdf2daf66b3a31813bc982c2d1d09`; the plugin provider validates model/config/profile identity and zero-offset policy; Manager dry-run merges the fixed factory with operator-owned settings; enablement reaches the provider; disabled, conflicting and unsupported paths fail closed |
| H4 | Real-device correctness | Pinned BF16 model/source/config passes APC/MTP2/async/hybrid-cache/output/graph capture-replay/cleanup gates |
| H5 | Performance validation | Only after H4; preregistered matched runs may measure TTFT/TPOT, throughput, memory, and conversion cost |

## Stop conditions

Stop immediately on ambiguous ownership, unpinned source or model identity,
unsupported target semantics, disabled-path drift, graph capture/replay drift,
incorrect output, incomplete cleanup, shared-device conflict, timeout, or a
request to infer performance from non-device evidence. Record the receipt and
return to the responsible owner; do not silently fall back to a different
target or route.
