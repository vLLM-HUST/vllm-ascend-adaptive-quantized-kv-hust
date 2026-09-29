# Qwen3.5 Prefix Pilot Result Validation

Date: 2026-09-29

## Purpose

This validator closes the evidence boundary for the single preregistered
`Qwen3.5-35B-A3B` baseline/treatment pair. It does not authorize execution and
does not turn one passing pair into a general performance claim.

The frozen pilot contract remains
[`evidence/QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json`](evidence/QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json).
The completed result must bind its canonical SHA-256 and a completed correctness
evidence-bundle SHA-256.

## Evidence retained per arm

Baseline and treatment each retain nine distinct, non-empty files:

1. raw benchmark result;
2. ordered request-level records;
3. resolved command;
4. sanitized runtime environment identity;
5. service stdout;
6. service stderr;
7. device-memory telemetry;
8. correctness, graph and fallback safety counts;
9. process and device cleanup receipt.

Every file has a relative POSIX path, byte count and SHA-256 in the result
document. Validation rejects missing files, path traversal, symlinks, duplicate
paths, hard-link reuse, size drift and digest drift.

The environment file binds both arms to the same exact Host, plugin, Manager,
benchmark, model, tokenizer, profile, request set, request order, resource lease
and two device IDs. The arm order is always baseline then treatment, with a
fresh service and device release after each arm.

## Metric semantics

The primary metric is the raw benchmark field `median_ttft_ms`. It is not the
public leaderboard's normalized `ttft_ms`, which may be derived from a mean
TTFT field. Substituting the normalized field changes the preregistered metric
and therefore fails closed.

The standard aggregate benchmark result alone is insufficient for this
project. The result bundle separately retains:

- 200 ordered request records with input/output token counts, TTFT, TPOT and an
  output-token-ID digest;
- measured memory samples for both leased devices and peaks recomputed from
  those samples;
- safety and cleanup receipts.

The validator cross-checks aggregate token totals against request records and
requires baseline/treatment output-token counts and token-ID digests to match
for every request.

## Verdict

After all identity, correctness and cleanup checks pass, the validator computes:

- TTFT improvement as
  `(baseline_median - treatment_median) / baseline_median * 100`;
- throughput regression as
  `(baseline_throughput - treatment_throughput) / baseline_throughput * 100`.

The only accepted verdicts are:

- `BOUNDED_POSITIVE`: TTFT improves by at least 5% and throughput regression is
  no more than 3%;
- `BOUNDED_NEGATIVE`: the evidence is valid but one or both performance gates
  are missed.

The declared verdict must equal the computed verdict. Correctness, graph,
fallback, provenance, identity or cleanup failures are rejected rather than
relabelled as a bounded negative result. Both verdicts keep general performance,
quality and public-surface claims disabled.

## Validation command

```bash
python scripts/validate_qwen35_prefix_pilot_result.py \
  RESULT.json \
  --pilot-contract \
    docs/evidence/QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json \
  --root RESULT_BUNDLE_ROOT
```

The validator is an offline evidence check. It does not connect to a server,
start a model service, reserve an accelerator or publish raw logs.
