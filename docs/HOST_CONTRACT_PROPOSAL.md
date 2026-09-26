# Generic Host Contract Proposal

## Current status, 2026-09-26

The latest ownership and execution control is
[`ISSUE1_PR35_CONTROL_20260926.md`](ISSUE1_PR35_CONTROL_20260926.md).
Host Draft PR
[`vLLM-HUST/vllm-ascend-hust#35`](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/35)
at `27032f3e611847784652503ade4270a1930f2bc4` is the permitted minimal,
generic, default-off provider-contract candidate. It implements part of the
attention-execution decision below for review; it is not an accepted host API,
does not provide the project provider/kernel, and does not transfer delivery or
go/no-go ownership away from the host owner.

The CPU reference and evidence package are delivered. Any next real-device
validation must target `Qwen3.5-35B-A3B`, TP2, with APC, MTP2, async scheduling,
and `FULL_AND_PIECEWISE` preserved. The plugin remains `import_only` pending
host review and real-device correctness.

## Missing seam

The archived project branch reached into private Ascend backend state to list
`ACLGraphWrapper` instances, release concrete graph objects, replace the shared
graph pool, and seal recapture. Project-specific profiling labels were also
inserted directly in the attention and graph hot paths.

Those mechanisms cannot be shipped as an independent plugin against the reset
repositories. Importing private registries or monkey-patching hot-path methods
would make compatibility implicit and would allow a failed plugin load to be
mistaken for a successful service start.

## Proposed contracts

### Continuing-prefill observer

Contract ID: `vllm.ascend.continuing-prefill.observer.v1`

The host emits stable path events with shape/layout/precision metadata and a
host-generated graph identity. The event sink is absent by default and must
add no labels, synchronization, allocation, or Python callbacks when disabled.
Project-specific marker names and receipt formatting stay in this plugin.

### ACL graph lifecycle controller

Contract ID: `vllm.ascend.acl-graph.lifecycle.v1`

The host provides idle-only operations to:

1. snapshot wrapper, entry, and pool identities;
2. reset all owned graph entries and return a structured result;
3. bind a fresh pool through a host-owned operation;
4. seal exactly one requested recapture;
5. prove capture state after every transition.

The host owns synchronization, graph object release, shared workspace cleanup,
and model-runner bookkeeping. The plugin must never access private wrapper
registries or mutate a platform global directly.

### Separate attention-execution decision

The two contracts above support measurement and graph lifecycle; neither lets
the extension replace continuing-prefill attention. The PR #19 takeover needs
a separate host decision, but this proposal does not invent a contract ID for
an interface that the host does not advertise.

The candidate-source audit is pinned to
`vLLM-HUST/vllm-hust@a4d6aa022fb1885a25a802a6e29372c81eac6c9f` and
`vLLM-HUST/vllm-ascend-hust@2c8c722107a54127999a64c4eb0ec86139df8c26`.
Those revisions are audit inputs, not active runtime pins. On that source,
core `CUSTOM` backend registration does not force Ascend platform selection,
C8 setup replaces the attention implementation class, and graph capture
dispatches through `full_graph_fia` before the normal `ChunkedPrefill` branch.
Hooking only the visible non-capture fallback would therefore create a
capture/replay behavior split.

The host owner should select one of these outcomes:

1. Identify an existing public interface and its exact host revision. The
   interface must be reachable for C8 in both capture/replay and non-capture
   execution and define input, output, fallback, and workspace lifetime.
2. Deliver a generic host-owned provider seam at the C8 execution boundary.
   The host owns discovery, eligibility, mixed-batch policy, native fallback,
   graph-stable output/workspace, and disabled-path equivalence. The extension
   owns only its provider implementation and project receipts.
3. Confirm that no supported extension execution route will be provided. In
   that case the PR #19 takeover stops at the CPU reference and source audit;
   it must not be activated or described as an NPU implementation.

Required provider inputs are causal multi-token TND Q, paged INT8 NZ K/V,
block table, valid query/KV lengths, TP-local per-channel scales, head/layout
metadata, and graph-stable shape identity. Native decode and all-new prefill
must retain host behavior. Unsupported or mixed cases must follow a
host-defined fallback or fail closed; the extension may not infer that policy.

## Rejected alternatives

- Copy the old backend branch into the reset core: carries project policy and
  stale assumptions across a repository reset.
- Register a general plugin that monkey-patches the backend: undocumented,
  import-order dependent, and not fail-closed when the vLLM loader catches an
  exception.
- Treat profiler output as the control interface: observations do not provide
  safe lifecycle ownership.
- Activate P1 with a placeholder entry point: could start a service without the
  required behavior and create false runtime evidence.

## Legacy evidence mapping

Owner direction maps reconciled Ascend PR #271 at `3689bbc` and open Draft
#279 at `360056b` to this plugin as design evidence for the observer and graph
lifecycle contracts. Their exact roles and negative-result boundaries are
recorded in `LEGACY_CONTRACT_EVIDENCE.md`. They are not host APIs, runtime
dependencies, or permission to restore a direct core delivery route.

## Acceptance

The plugin may add runtime activation only after repository owners approve the
contract identifiers or their replacements, the pinned host source advertises
them, and disabled-path tests prove baseline equivalence.

The PR #19 attention component additionally requires resolution of the
separate execution decision above. Approval of only the observer and lifecycle
contracts does not authorize attention replacement, an NPU run, or a
performance claim.

The original 2026-09-01 policy treated this proposal only as a host dependency
request. The 2026-09-20 correction permits the student to submit a minimal,
generic, default-off candidate host contract for review, which is now PR #35.
That permission changes the authorship route, not ownership: the host owner
still decides whether to adopt it, who delivers the final interface, which
revision is accepted, and when real-device work may start. All project-specific
Adaptive Quantized KV behavior, receipts, and policy remain in this plugin.
