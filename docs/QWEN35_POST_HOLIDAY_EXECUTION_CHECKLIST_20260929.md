# Qwen3.5 Post-Holiday Execution Checklist

Date: 2026-09-29

## Objective

Run exactly one bounded, preregistered `Qwen3.5-35B-A3B` correctness session
and, only after it passes, one baseline/treatment prefix-repetition pair. This
checklist does not authorize execution by itself.

## Frozen target

| Input | Frozen value |
| --- | --- |
| Model | `Qwen/Qwen3.5-35B-A3B` |
| Model revision | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| Weight baseline | BF16 |
| Treatment KV | calibrated zero-offset C8/INT8 |
| Parallelism | TP2 on two exclusive 910B2 devices |
| Runtime modes | APC, MTP2, async scheduling, `FULL_AND_PIECEWISE` |
| Benchmark | `vLLM-HUST/vllm-hust-benchmark@47c12e518a9188bb714bb6193d00260a7741841b` |
| Scenario | `prefix-repetition-online-2chip` |
| Shape | 200 requests, 4096 input tokens, up to 256 output tokens |

PR and branch heads recorded before the holiday are planning inputs, not
accepted runtime revisions. Replace them with exact merged or owner-accepted
SHAs in the execution receipt.

## E0: owner input gate

Do not download weights, start a service or reserve a device until one public
owner record supplies all of the following:

- accepted Host carrier and exact revision, including the disposition of Host
  PR #35 and baseline PR #37;
- accepted plugin and Manager revisions;
- ModelSlim-compatible BF16+C8 sidecar/checkpoint or a Host-owned external
  profile source;
- calibration owner, method, dataset, model/config/profile digests, zero-offset
  policy and TP2 shard digests;
- exact model and tokenizer location and digests;
- two exclusive 910B2 device IDs, lease window and cleanup owner;
- approved numerical oracle tolerances.

Record the owner URL and timestamp. A Slack summary without an attributable
decision URL is not sufficient provenance.

**Stop if:** any value is missing, points to the W8A8 fixture, uses placeholder
scales/offsets, or disagrees with the frozen model revision.

## E1: restore and identity gate

1. Clone the public Host, plugin, Manager and benchmark repositories.
2. Check out the exact accepted revisions in detached state.
3. Restore only project-owned, non-reproducible evidence needed for comparison.
4. Rebuild the environment from manifests; do not restore old virtualenv,
   conda, kernel-cache or model-cache directories.
5. Verify model, tokenizer, profile, TP2 shards and benchmark scenario digests.
6. Run all no-device tests and distribution validation before requesting NPU
   resources.

**Stop if:** a Git revision, artifact digest, schema, supported-layer set or
profile shard differs from the owner record.

## E2: resource admission gate

1. Confirm there is no unfinished project session or service.
2. Observe both assigned devices before launch and record raw device/process
   output.
3. Reject co-tenancy; never stop or signal an unknown process.
4. Set a bounded correctness timeout and identify the cleanup owner.

**Stop if:** either assigned device is occupied, the lease is ambiguous, or
cleanup responsibility is not assigned. Release the lease without waiting on
the device.

## E3: Manager and NPU correctness

On the exact E0/E1 identities, retain raw evidence for:

1. Manager `discover`, `check`, `plan` and `run`;
2. both TP2 workers, profile shards and all ten full-attention layers;
3. eight frozen correctness cases covering decode, continuing prefill, mixed
   batch, APC, MTP2 rollback, async scheduling and hybrid-cache rejection;
4. graph capture, first replay and subsequent replay;
5. native fallback;
6. rollback, uninstall, process exit and device-memory release.

Validate the 26-artifact bundle from PR #5. Preserve service stdout/stderr,
commands, environment and raw Manager/worker/case/lifecycle receipts.

**Stop if:** provenance, numerical output, worker coverage, dispatch path,
capture/replay, fallback or cleanup fails. Do not proceed to timing and do not
describe activation as a performance result.

## E4: one matched pilot pair

Use a fresh service per arm and the same exact model, tokenizer, request set,
request order, software revisions, device pair and runtime modes.

1. Run baseline first: plugin disabled, native BF16 KV.
2. Stop the service and retain all nine baseline artifacts.
3. Verify zero project processes and released device memory.
4. Run treatment: reviewed plugin/provider with the calibrated zero-offset C8
   profile.
5. Stop the service and retain all nine treatment artifacts.
6. Verify cleanup again and release the two-device lease.

Each arm is bounded to 30 minutes. Do not repeat a failed pair to search for a
favorable result without a new preregistered contract.

## E5: immutable analysis and report

Run:

```bash
python scripts/validate_qwen35_prefix_pilot_result.py \
  RESULT.json \
  --pilot-contract \
    docs/evidence/QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json \
  --root RESULT_BUNDLE_ROOT
```

The validator must return one of:

- `BOUNDED_POSITIVE`: 200/200 requests, matching outputs, no safety/cleanup
  failure, median TTFT improvement at least 5%, throughput regression at most
  3%;
- `BOUNDED_NEGATIVE`: all evidence is valid but a frozen performance threshold
  is missed.

Publish a sanitized evidence index with exact revisions, commands, workload,
metric values, file sizes and SHA-256. Provide the raw bundle to the teacher or
assigned engineer through the approved project channel; do not publish secrets,
tokens, private paths or credentials. Report negative and failed results without
changing thresholds.

One positive pair is a project-delivery pilot only. It does not authorize a
cross-model performance claim, quality claim, PyPI release or restoration of
the public website entry.

## Completion record

The post-holiday run is complete only when all boxes can be backed by retained
artifacts:

- [ ] owner input record and exact accepted revisions;
- [ ] model, tokenizer, profile and request digests;
- [ ] exclusive TP2 lease and before/after resource receipts;
- [ ] complete 26-artifact correctness bundle;
- [ ] complete 18-artifact baseline/treatment result bundle;
- [ ] mechanically validated verdict;
- [ ] raw bundle delivered to the responsible teacher/engineer;
- [ ] sanitized public Issue/PR update with URLs and hashes.
