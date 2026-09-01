# Generic Host Contract Proposal

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

Under the 2026-09-01 organization delivery policy, this proposal is a host
dependency request rather than permission for this project to submit an
optimization PR directly to `vllm-hust` or `vllm-ascend-hust`. If the contracts
are required, the host owner must provide them or explicitly name another
accepted delivery route. All Adaptive Quantized KV behavior, receipts, policy,
and visible implementation progress remain in this organization plugin.
