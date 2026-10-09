# Qwen3.5 Prefix-Repetition Pilot, 2026-09-28

## Why this group

The project-group delivery notice requires at least one `Qwen3.5-35B` test
group with a positive optimization result. The selected group is the official
`prefix-repetition-online-2chip` scenario from
`vLLM-HUST/vllm-hust-benchmark@47c12e5`.

This is the narrowest relevant group because it simultaneously exercises:

- the fixed two-device `TP2` target;
- repeated 4096-token prefixes, which can create APC hits and continuing
  prefill instead of measuring an unrelated all-new prefill path;
- online TTFT, the metric most directly affected by avoiding full dense KV
  materialization;
- the project's required `Qwen3.5-35B-A3B` serving combination.

Random-online and ordinary ShareGPT are not the first choice: neither guarantees
the repeated-prefix condition needed to exercise the optimization. AgentX is a
separate optional 256k benchmark and does not replace this delivery group.

## Frozen comparison

Both arms use the same exact model revision, benchmark commit and scenario,
request order, 200 requests, 4096 input tokens, 256 output tokens, TP2 device
pair, APC, MTP2, async scheduling, and `FULL_AND_PIECEWISE` graph mode.

- Baseline: plugin disabled, native BF16 KV path.
- Treatment: plugin enabled, calibrated zero-offset INT8 KV profile and the
  reviewed paged-INT8 continuing-prefill provider.

The machine-readable contract is
[`evidence/QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json`](evidence/QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json).

## Admission and result rule

Do not start the model service while the plugin is `import_only`, the BF16-to-C8
profile is absent, Host/provider delivery is unaccepted, or the complete
correctness matrix has not passed. This prevents using NPU time for a treatment
that cannot be reached or interpreted.

For the first bounded pair, a positive project-delivery result requires:

1. 200/200 successful requests and no correctness, graph or cleanup failure;
2. treatment median TTFT at least 5% lower than baseline;
3. treatment throughput no more than 3% below baseline;
4. matching output-token accounting and retained raw outputs;
5. exact Host, plugin, model, profile, benchmark, hardware and command receipts.

The pair is capped at 30 minutes per arm and releases the two leased devices
after each arm. A positive pair is a pilot result only; broader claims require
replication and a capacity curve.

## Current status

Preregistered, not executable yet. The benchmark group and result rule are
closed. Execution remains blocked by the existing correctness gates: accepted
Host reachability, a calibrated Qwen3.5 C8 profile, an activatable Manager/plugin
candidate, and real-device correctness through graph replay and rollback.
