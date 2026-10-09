# Qwen3.5 Manager/NPU Correctness Runbook

Date: 2026-09-29

This runbook maps the P1 correctness contract to the public Extension Manager
CLI at `vLLM-HUST/extension-manager@cf1ea71e3e2cb81ab06267ef05eddb3e580ea20b`.
It is a prepared procedure, not current execution authorization.

## Mandatory preconditions

Do not configure, enable or run the extension until all conditions hold:

1. the correctness contract contains accepted 40-character Host, plugin and
   Manager revisions;
2. the exact BF16 model/tokenizer and calibrated zero-offset C8 profile/TP2
   shard digests are present;
3. the numerical oracle thresholds cite an owner-approved source;
4. an exclusive two-device lease and cleanup owner are recorded;
5. the Host owner has accepted a reachable C8 selection/loading and provider
   revision;
6. the checked-in `import_only` manifest has been replaced only in an internal,
   reviewed correctness candidate.

The first command in the execution environment must be:

```bash
python scripts/validate_qwen35_correctness.py \
  /path/to/authorized-correctness-contract.json \
  --require-authorized
```

Exit code `2` or `execution_authorized=false` stops the session before Manager
state changes, model loading or device use. The checked-in template intentionally
returns exit code `2` with this flag.

## Isolated Manager state

Use a run-specific configuration root so rollback never modifies another
operator's saved intent:

```bash
set -euo pipefail
export EXTENSION_ID=org.vllm-hust.ascend-adaptive-quantized-kv
export RUN_ROOT=/path/to/new-exclusive-run-directory
export XDG_CONFIG_HOME="$RUN_ROOT/xdg-config"
export RECEIPTS="$RUN_ROOT/manager-receipts"
mkdir -p "$RECEIPTS"
```

The run directory must be newly created, access-controlled and bound to the
authorized contract digest. Do not reuse or overwrite earlier receipts.

## Manager phases

The public Manager has no standalone command named `discover`; discovery is the
entry-point scan performed by `extension list`, `validate`, `check` and `plan`.
Record the lifecycle as follows:

```bash
vllm-hust-ext extension list --json \
  | tee "$RECEIPTS/01-discover.json"
vllm-hust-ext extension validate "$EXTENSION_ID" \
  | tee "$RECEIPTS/02-validate.json"
vllm-hust-ext extension check "$EXTENSION_ID" \
  | tee "$RECEIPTS/03-check-before-enable.json"
```

After confirming that the discovered distribution and manifest digests equal
the contract, configure and enable the reviewed internal candidate:

```bash
vllm-hust-ext extension configure "$EXTENSION_ID" \
  --file /path/to/revision-bound-provider-config.json
vllm-hust-ext extension enable "$EXTENSION_ID"
vllm-hust-ext extension check "$EXTENSION_ID" \
  | tee "$RECEIPTS/04-check-enabled.json"
vllm-hust-ext extension plan "$EXTENSION_ID" \
  | tee "$RECEIPTS/05-plan.json"
```

`enable` failing on `import_only`, an identity mismatch, an activation conflict
or an unsupported Host is a stop result, not permission to edit around the
Manager.

## Dry run and launch

The exact `vllm serve` argv must be generated from the accepted Host revision and
stored as a hashed artifact. It must encode the frozen BF16 target, TP2, APC,
MTP2, async scheduling and `FULL_AND_PIECEWISE`; it must not substitute the
existing W8A8 fixture. Before the real launch, run:

```bash
vllm-hust-ext run --dry-run -- <reviewed-vllm-serve-argv> \
  | tee "$RECEIPTS/06-run-dry-run.json"
```

Compare the emitted environment and merged `--additional-config` against the
contract. Only an exact match may proceed to:

```bash
vllm-hust-ext run -- <reviewed-vllm-serve-argv> \
  >"$RECEIPTS/07-service.stdout.log" \
  2>"$RECEIPTS/07-service.stderr.log"
```

The bounded executor, not an interactive shell left running, owns the 45-minute
timeout, failure trap and process group. It executes the eight cases in the
contract order and stops on the first provenance, numerical, graph, fallback or
resource failure.

## Rollback and cleanup

Whether the matrix passes or fails, the executor must stop the owned process
group, then record Manager rollback without touching unknown processes:

```bash
vllm-hust-ext extension disable "$EXTENSION_ID"
vllm-hust-ext extension forget "$EXTENSION_ID"
vllm-hust-ext extension status "$EXTENSION_ID" \
  | tee "$RECEIPTS/08-status-after-forget.json"
```

Package uninstall is operator-owned and separate from Manager `forget`; perform
it only for the run-specific environment. Record zero remaining project
processes and released device memory. Never stop or signal an unknown process.

Finally hash every raw receipt without rewriting it and assemble the single
correctness receipt consumed by:

```bash
python scripts/validate_qwen35_correctness.py \
  /path/to/authorized-correctness-contract.json \
  --receipt /path/to/completed-correctness-receipt.json
```

A PASS closes correctness only. The validator returns
`performance_authorized=false`; benchmark timing remains a separate gated run.
