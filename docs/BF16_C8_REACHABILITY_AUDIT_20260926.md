# BF16 Target to C8 Provider Reachability Audit, 2026-09-26

## Decision being checked

The owner follow-up fixes the correctness target to `Qwen3.5-35B-A3B` BF16,
TP2, APC, MTP2, async scheduling, and `FULL_AND_PIECEWISE`. It also asks for a
Host/provider version that the extension manager can enable before performance
testing begins.

This audit asks a narrower question: can that BF16 target reach Host Draft PR
#35's C8 continuing-prefill provider merely by enabling the plugin?

## Source result

No. The current source has a reachability gap before provider execution:

1. `AscendModelSlimConfig._add_kvcache_quant_metadata()` enables dense C8 KV
   only when a model quantization description declares `kv_cache_type=C8`.
2. It records C8 layers only when the checkpoint metadata contains
   `k_proj.kv_cache_scale` entries.
3. `AscendC8KVCacheAttentionMethod.create_weights()` changes the layer
   implementation to `AscendC8AttentionBackendImpl` and creates the K/V cache
   scale and offset parameters.
4. Only `AscendC8KVCacheAttentionMethod.process_weights_after_loading()` calls
   `configure_c8_continuing_prefill_provider()`.
5. Draft PR #35 therefore provides a provider boundary inside an already-C8
   checkpoint path. It does not turn an ordinary BF16 model into an INT8-KV
   model and does not define how BF16-target K/V scales and offsets are
   produced, sharded, stored, or validated.

The existing exact-model BF16 test uses `Qwen/Qwen3.5-35B-A3B` without an
Ascend quantization description. The exact-model TP2 quantized test instead
uses `Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp`; it is not the requested BF16 target
and does not cover APC, MTP2, async scheduling, and `FULL_AND_PIECEWISE`
together.

## Consequence

Changing the plugin manifest from `import_only` to active before closing this
gap would prove only that the manager can import the package. It would not
prove that the required BF16 target dispatches to the provider. Such a build
must not be reported as a usable Adaptive Quantized KV implementation.

The likely intended treatment is BF16 model weights with an INT8 KV cache. If
so, the usable Host/provider version also needs a reviewed policy and data
contract for:

- how K/V quantization scales and offsets are obtained;
- whether they are static per-channel, calibrated, or dynamic;
- how TP2 shards own those parameters;
- where BF16 K/V is quantized before the paged cache write;
- how the paged INT8 layout and metadata survive APC, MTP2, async compaction,
  hybrid attention/Mamba cache bookkeeping, and graph capture/replay;
- how unsupported layers or shapes fail closed without silently changing the
  target.

## Routes and disposition

| Route | What it proves | Disposition |
| --- | --- | --- |
| Use a checkpoint with `kv_cache_type=C8` | PR #35 provider reachability for an already-quantized model | Useful development fixture, but not a substitute for the specified BF16 target |
| BF16 weights plus a reviewed runtime INT8-KV path | The requested target can actually reach paged-INT8 attention | Required product route; Host/provider contract must include quantization metadata and cache-write ownership |
| Activate only the current observer/CPU reference | Extension-manager import and offline reference availability | Insufficient; no runtime provider reachability |

## Ordered closure

1. Fix or land the unrelated Host-main mypy test-double declaration so PR #35
   can run its CPU unit tests.
2. Record whether “BF16” means BF16 model weights plus INT8 KV treatment. Do
   not silently replace it with the W8A8/C8 checkpoint.
3. Extend or revise the Host/provider contract so the exact BF16 target can
   create, retain, and consume the required paged INT8 KV metadata.
4. Implement the plugin provider with lazy runtime imports, strict eligibility,
   and fail-closed result semantics while the manifest remains `import_only`.
5. Add an active manifest revision only when manager enablement reaches that
   provider in a source/contract test.
6. Run the fixed correctness matrix on an assigned device. Performance remains
   out of scope until the owner accepts the complete correctness evidence.

This is source evidence only. No device, service, operator, correctness, or
performance result follows from it.
