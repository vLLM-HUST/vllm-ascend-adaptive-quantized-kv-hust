# PR #19 Student-Owned Component Takeover

## Frozen source and ownership

- Advisor source: `intellistream/ascend-adaptive-quantized-kv#19`, closed without
  merge on 2026-09-04. Head: `470f1151b8ec6acd31a4b63f8de3cd882602f826`.
- Source base: `8be03304c1657de2f5eb75de6b859d899e92bef5`.
- Embedded runtime patch base: `e2e1c96e3b2f040aaba06b8ba333070d19be4738`;
  patch commit: `1d61d8ae4c69a1679ee609168db541ef80004ed3`.
- Compressed patch SHA256:
  `3db28f84f8c94ee81ccb1761f38245cae15d8cb507bb64ecb2c561e53bc78d50`.
- Current extension base: `ddd306fce8d885b9b9cfeb8c947ed576c5269e66`.
- Student owner: XilingGao. Branch: `feature/pr19-component-takeover`.

The owner requested a student-authored implementation carrier rather than a
merge of the advisor's mixed runtime branch. This is an attributed adaptation,
not a claim that the original algorithm or prototype was authored here. The
useful starting idea is block-local INT8 dequantization with online softmax.
This work does not reopen PR #19 or modify a core host repository.

## Scope and stages

| Stage | Work | Exit criterion | Status |
| --- | --- | --- | --- |
| S0 | Audit exact PR and reproduce admission/correctness gaps off-device | Record findings and disposition of every changed area | Complete |
| S1 | Implement a standalone CPU tensor reference in this extension | Dense-reference agreement for grouped heads, variable lengths, non-unit scales, page order, causal masks and partial blocks; invalid inputs rejected | Complete; validation below |
| S2 | Verify packaging and submit a student-authored Draft PR | Existing inert-package tests plus component tests, isolated build and CI; exact source attribution | Complete: Draft #3; four CI jobs pass |
| S3a | Recheck host selection and bridge explicit CPU snapshots to the reference | Pinned source audit; cumulative-length/decode-prefix/NZ conversion tests | Complete: 122 total tests pass |
| S3b | Freeze a reproducible read-only audit of C8 dispatch, capture/replay and dense fallback | Structured receipt records exact host revisions and hashes; source drift fails closed | Complete: 127 total tests pass |
| S3c | Re-audit newer host heads without moving the retained S3b baseline | Exact-revision parameters; additive receipt; changed-path and AST checks agree on the execution boundary | Complete: execution sources unchanged |
| S3 | Integrate with a reviewed host dispatch/operator interface | Actual public host API and graph-safe layout/lifecycle contract are available; new bounded execution plan reviewed | Blocked |
| S4 | Run matched NPU component/service validation | Correctness, graph replay, provenance and resource gates pass before timing claims or matrix expansion | Blocked |

The initial takeover executed S0-S2. S3a was added on 2026-09-05 as off-device
integration preparation. The existing manifest stays `import_only`; there is
no registration, environment activation, monkey patch, service, NPU request or
change to the parent v5 gate. A CPU component pass cannot advance D1 or D2.

2026-09-05 continuation: S3a is now in scope, following the owner request to
continue local integration preparation. S3a is not activation or host surgery.
The candidate source observations are Ascend
`2c8c722107a54127999a64c4eb0ec86139df8c26` and core
`a4d6aa022fb1885a25a802a6e29372c81eac6c9f`; these do not replace active pins.
Before live S3, resolve platform backend selection, C8 implementation ownership
and the separate graph-capture dispatch branch. Offline snapshot conversion
will reject unsupported semantics rather than invent a host hook.

The completed source audit and owner-boundary request are in
`docs/HOST_EXECUTION_AUDIT_20260905.md`. The bridge is
`snapshot_reference.continuing_prefill_snapshot_reference`; it is intentionally
not imported by normal discovery. The three explicit owner outcomes are in
`docs/HOST_CONTRACT_PROPOSAL.md`; no new host contract name is claimed.

S3b converts the manual host reading into a standard-library audit. It reads
only the two exact candidate revisions, hashes every inspected blob, parses
definitions with Python AST, and records a structured receipt. A changed or
ambiguous source shape produces `INCONCLUSIVE` and a nonzero exit. The receipt
does not claim runtime reachability, correctness or performance.

The retained S3b receipt is
`docs/evidence/HOST_EXECUTION_SOURCE_AUDIT_20260905.json`, SHA256
`4e7ce1e659b3990a8ca58228f3314dc1c8a7ebe1ca46b29847a6f9446010ac86`.
It records seven findings supported by the pinned source and the direct-symbol
coverage found in the two scanned host test files. Local validation is 127
tests plus Ruff and format checks.

S3c keeps those defaults and that receipt immutable while allowing a caller to
name newer full 40-character commit SHAs explicitly. Branch names, tags,
abbreviated SHAs and uppercase object names are rejected. Each refresh writes a
new receipt and reruns the same structural checks; unrelated repository changes
therefore do not silently move the evidence baseline or create a false host-API
claim.

The 2026-09-05 refresh audited core
`88cca78bbc92e9113067cb62252c5d8ae2bbdd06` and Ascend
`d0433ba3aeb3b6643787177d7b1fefe4c742ef6e`. The intervening core change is
limited to scheduler/preemption files, and the Ascend change is limited to MoE
offload metadata. All eight inspected execution/test blobs have the same
SHA256 values as S3b, and all seven structural findings reproduce. The additive
receipt is
`docs/evidence/HOST_EXECUTION_SOURCE_AUDIT_MAIN_REFRESH_20260905.json`, SHA256
`9332fd8ede32c52a40f0f2cd1f35a02e445a3eca6a7f2f8934ac1062af0ac3cd`.
No public C8 execution boundary appeared, so S3 remains blocked. Local
validation after S3c is 134 tests plus Ruff, format, diff and isolated
distribution checks; the built wheel remains inert.

## Reproduced audit and disposition

1. **NZ layout:** the old helper reads `key_cache.shape[1]` as block size, but
   its new positive test passes 5D NZ storage `[pages, kv_heads, D/32, B, 32]`.
   Axis 1 is the KV-head count, not B. The replacement takes explicit 5D NZ
   inputs and validates all dimensions; raw host 4D storage needs a reviewed
   adapter, not a guessed reshape.
2. **Page safety:** old indexing does not reject negative physical page IDs,
   which PyTorch treats as indexing from the end. Validate every used ID before
   reading KV; ignore unused block-table padding rather than dereferencing it.
3. **Temporary storage:** the old candidate expands K/V with `repeat_interleave`
   and processes all query tokens at once. The CPU reference will retain grouped
   KV heads and tile queries. This is a source-level working-set property, not
   an allocator, graph-pool or HBM measurement.
4. **Capacity provenance:** the old runner assigns
   `scheduler_visible_kv_cache_tokens` from the same parsed server-log value as
   reported capacity. It is not an independent scheduler observation. Do not
   adopt that field as cross-layer proof or change the invalid M0 result.
5. **D1 admission:** the old finalizer checks summary flags, coordinates,
   dispatch counts and capacity but does not independently require raw outputs,
   graph replay, source/model/workload identity or release evidence. Its positive
   test creates only synthetic summaries. Do not reuse this as a real-online
   finalizer.
6. **Host integration:** the old environment switch/core patch conflicts with
   the organization extension route. An observer is not an execution dispatch
   API. Reference correctness does not establish an available integration point.
7. **CLI regression:** the old runner defaults the mechanism arm to `c8` when
   only the original variant argument is supplied, but its new case statement
   accepts `c8-fallback`/`c8-native`, not `c8`. The extracted admission case exits
   2 with `unknown mechanism arm: c8`. Do not adopt the changed runner wholesale.

The old runner/merger/tests remain review input in the closed PR; they are not
copied into current execution paths. Neither the full-cache-free algorithm nor
its Torch implementation is a demonstrated fused/CANN/NPU-native kernel.

### Offline reproduction, 2026-09-04

`scripts/reproduce_pr19_gaps.py` reads only the frozen revisions above, checks
the compressed patch digest, extracts the original functions via AST, and
executes the CPU attention method and summary finalizer in isolation. It does
not execute the shell runner or import the old runtime. Run from this plugin:

```bash
.venv/bin/python scripts/reproduce_pr19_gaps.py \
  --research-repo /path/to/ascend-adaptive-quantized-kv \
  --legacy-backend-repo /path/to/vllm-ascend-hust-legacy \
  --output /path/to/local-audit.json
```

Observed with Python 3.12 / Torch 2.8.0 on CPU:

- The original positive-test dimensions (P=2, Hkv=2, D=32, B=32, KV=48, Q=7)
  raise `block table does not cover the continuing-prefill sequence`. This is
  the explicit 5D fixture/layout mismatch, not proof that the old 4D production
  wrapper always fails.
- With the old 4D wrapper, physical page `-1` silently selects the final page.
- Three synthetic summary files alone produce a claimed `PASS_D1` and
  `real-online`. This demonstrates insufficient independent verification by
  the finalizer, not that the full runner produced invalid live data. The audit
  itself is classified `synthetic-cpu-audit-only`, with `d1_accepted=false`.
- Both capacity fields refer to the same server-log expression; the extracted
  default C8 arm case exits 2. No service, allocation or model was started.

Changed-area disposition: archived runtime patch -> attributed CPU reference
and regression tests here; old contract/D0 JSON -> historical readiness only;
runner/merger/orchestration tests -> audit input only, not execution admission.

## Component contract

CPU tensors only: floating TND Q; symmetric INT8 NZ K/V with lane size 32;
per-KV-head/channel finite positive K/V scales; integer block table; per-request
KV and query lengths. Q is the suffix of each causal sequence. GQA is supported
when query heads are divisible by KV heads. No sliding window, cross attention,
offset quantization, graph capture, device dispatch or KV lifecycle mutation.

Query tiling bounds score/accumulator dimensions; each dequantization sees only
one used cache page, with valid-tail slicing. Shared pages between requests are
read-only. Dense attention exists solely in tests as an independent oracle.
Tests must check both numerical correctness and invalid inputs, plus that normal
package discovery does not import Torch or activate the new module.

## Next decision after S2

Report findings, actual tests and limitations on the student Draft PR and
existing plugin Issue #1. Propose no new contract name as an implemented API.
Request an existing or owner-delivered dispatch/operator integration point;
do not equate approval of an observer with approval of attention replacement.

## Validation handoff

- Python 3.12 / Torch 2.8.0: CPU component and inert-package tests are required
  together; the final count is recorded in the Draft PR/CI result.
- Clean environment with no Torch: 9 inert-package tests pass, the optional
  component module is skipped, and a separate isolated import of the installed
  wheel confirms that Torch/reference/runtime modules are not loaded.
- The dedicated CI component job explicitly imports CPU Torch before pytest;
  the no-Torch job's skip is never presented as component acceptance.
- Ruff, format, diff checks and an isolated sdist-to-wheel build pass. Default
  runtime dependencies remain empty; only the `component` extra pins Torch.
- No NPU/SSH/service, host patch, PyPI release, graph acceptance, quality result
  or speedup claim is included. The parent v5 gate remains unchanged.

### S2 receipt, 2026-09-05

- Student implementation: `3fdbf47fee98a0d338c8b6bc2448e58a076819e0`.
- [Owner-authored Draft PR #3](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/pull/3)
  is open for review; the archived research PR #19 stays closed and unmerged.
- [CI run 33895674890](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/actions/runs/33895674890)
  passed CPU component (Python 3.12) and inert-package jobs on Python
  3.10/3.12/3.14. Local full suite: **91 passed**, including 82 component cases.
- The installed-wheel import check passed in an isolated no-Torch environment.
  Local wheel SHA256:
  `df90bf0ea8caff2fc6c4fed4b06c6811e950925c4417edd6efaaf798f6cab840`.
- The S2 exit criterion is a tested Draft, not merged or active runtime code.
  S3/S4 remain blocked: no NPU result, graph approval or D1 acceptance follows
  from this receipt.
