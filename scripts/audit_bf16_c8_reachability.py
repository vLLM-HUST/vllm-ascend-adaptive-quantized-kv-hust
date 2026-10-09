#!/usr/bin/env python3
"""Audit whether the pinned BF16 target can reach the Host C8 provider."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASCEND_REPOSITORY = "vLLM-HUST/vllm-ascend-hust"
DEFAULT_ASCEND_REVISION = "084f70f50dfcdf2daf66b3a31813bc982c2d1d09"
MODEL_REPOSITORY = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
SCHEMA_VERSION = "vllm-ascend-bf16-c8-reachability-audit/v1"

MODELSLIM_CONFIG = "vllm_ascend/quantization/configs/modelslim_config.py"
C8_METHOD = "vllm_ascend/quantization/methods/kv_cache/kv_c8.py"
ATTENTION = "vllm_ascend/attention/attention_v1.py"
PLATFORM = "vllm_ascend/platform.py"

BF16_MODEL_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    *(f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)),
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


class AuditInconclusive(RuntimeError):
    """Raised when a pinned source no longer has the expected structure."""


@dataclass(frozen=True, slots=True)
class SourceUnit:
    path: str
    text: str
    sha256: str


def require_exact_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AuditInconclusive(
            f"revision must be a full lowercase 40-character commit SHA: {revision!r}"
        )
    return revision


def canonical_repository_origin(origin: str) -> str:
    value = origin.strip().rstrip("/")
    patterns = (
        r"https://github\.com/(?P<repository>[^?#]+)",
        r"ssh://git@github\.com/(?P<repository>[^?#]+)",
        r"git@github\.com:(?P<repository>[^?#]+)",
    )
    repository = None
    for pattern in patterns:
        match = re.fullmatch(pattern, value)
        if match is not None:
            repository = match.group("repository").removesuffix(".git")
            break
    if repository is None:
        raise AuditInconclusive(f"unsupported GitHub origin: {origin!r}")
    if repository.lower() != ASCEND_REPOSITORY.lower():
        raise AuditInconclusive(
            "unexpected repository origin: "
            f"expected {ASCEND_REPOSITORY}, got {repository}"
        )
    return ASCEND_REPOSITORY


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def load_source(repo: Path, revision: str, path: str) -> SourceUnit:
    require_exact_revision(revision)
    resolved = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.decode().strip() != revision:
        raise AuditInconclusive(f"revision did not resolve exactly: {revision}")
    raw = _git(repo, "show", f"{revision}:{path}")
    return SourceUnit(
        path=path, text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest()
    )


def find_qualified_def(source: str, qualified_name: str) -> ast.AST:
    body: list[ast.stmt] = ast.parse(source).body
    node: ast.AST | None = None
    for part in qualified_name.split("."):
        matches = [
            item
            for item in body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == part
        ]
        if len(matches) != 1:
            raise AuditInconclusive(
                f"expected exactly one definition for {qualified_name!r}"
            )
        node = matches[0]
        body = node.body
    assert node is not None
    return node


def source_ref(unit: SourceUnit, symbol: str, node: ast.AST) -> dict[str, Any]:
    return {
        "path": unit.path,
        "symbol": symbol,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "blob_sha256": unit.sha256,
    }


def _unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def _require_text(node: ast.AST, *needles: str) -> None:
    rendered = _unparse(node)
    missing = [needle for needle in needles if needle not in rendered]
    if missing:
        raise AuditInconclusive(f"expected source fragments are absent: {missing}")


def analyze(units: dict[str, SourceUnit]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    modelslim = units[MODELSLIM_CONFIG]
    maybe_update = find_qualified_def(
        modelslim.text, "AscendModelSlimConfig.maybe_update_config"
    )
    _require_text(
        maybe_update,
        "if self.quant_description:\n        return",
        "get_model_file(model_name, MODELSLIM_CONFIG_FILENAME",
        "ModelSlim Quantization Config Not Found",
    )
    add_metadata = find_qualified_def(
        modelslim.text, "AscendModelSlimConfig._add_kvcache_quant_metadata"
    )
    _require_text(
        add_metadata,
        "self.enable_c8_quant = kv_quant_type == 'C8'",
        "if 'k_proj.kv_cache_scale' in key",
        "self.c8_quant_layers.append(int(_id))",
    )
    findings.append(
        {
            "id": "checkpoint-metadata-gate",
            "status": "confirmed-in-pinned-source",
            "summary": (
                "Dense C8 is selected from ModelSlim metadata loaded from the model "
                "directory; layers are admitted only when kv_cache_type is C8 and a "
                "k_proj.kv_cache_scale key identifies the layer."
            ),
            "evidence": [
                source_ref(
                    modelslim,
                    "AscendModelSlimConfig.maybe_update_config",
                    maybe_update,
                ),
                source_ref(
                    modelslim,
                    "AscendModelSlimConfig._add_kvcache_quant_metadata",
                    add_metadata,
                ),
            ],
        }
    )

    c8_method = units[C8_METHOD]
    create_weights = find_qualified_def(
        c8_method.text, "AscendC8KVCacheAttentionMethod.create_weights"
    )
    _require_text(
        create_weights,
        "layer.impl.__class__ = AscendC8AttentionBackendImpl",
        "layer.k_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype)",
        "layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype)",
        "layer.v_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype)",
        "layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype)",
    )
    process_weights = find_qualified_def(
        c8_method.text,
        "AscendC8KVCacheAttentionMethod.process_weights_after_loading",
    )
    _require_text(
        process_weights,
        "layer.impl.configure_c8_continuing_prefill_provider",
    )
    findings.append(
        {
            "id": "provider-is-downstream-of-c8-weight-setup",
            "status": "confirmed-in-pinned-source",
            "summary": (
                "The provider is configured only after ModelSlim has selected the C8 "
                "attention method and completed C8 weight setup. The new provider JSON "
                "carrier does not select this method. Missing checkpoint scales leave "
                "one/zero initialization; provider activation does not calibrate or "
                "validate those values."
            ),
            "evidence": [
                source_ref(
                    c8_method,
                    "AscendC8KVCacheAttentionMethod.create_weights",
                    create_weights,
                ),
                source_ref(
                    c8_method,
                    "AscendC8KVCacheAttentionMethod.process_weights_after_loading",
                    process_weights,
                ),
            ],
        }
    )

    platform = units[PLATFORM]
    fix_config = find_qualified_def(platform.text, "_fix_incompatible_config")
    _require_text(
        fix_config,
        "getattr(vllm_config.cache_config, 'calculate_kv_scales', False)",
        "vllm_config.cache_config.calculate_kv_scales = False",
    )
    findings.append(
        {
            "id": "runtime-scale-calculation-disabled",
            "status": "confirmed-in-pinned-source",
            "summary": (
                "Ascend explicitly resets calculate_kv_scales to false, so the "
                "standard BF16 target cannot close the scale contract through the "
                "upstream runtime option."
            ),
            "evidence": [source_ref(platform, "_fix_incompatible_config", fix_config)],
        }
    )

    attention = units[ATTENTION]
    quantize = find_qualified_def(
        attention.text, "AscendC8AttentionBackendImpl._quantize_kv_to_int8"
    )
    dequantize = find_qualified_def(
        attention.text, "AscendC8AttentionBackendImpl._dequant_paged_kv_to_dense"
    )
    _require_text(
        quantize,
        "actual_key * layer._c8_k_inv_scale + layer._c8_k_offset",
        "actual_value * layer._c8_v_inv_scale + layer._c8_v_offset",
    )
    dequantized = _unparse(dequantize)
    if "layer._c8_k_scale" not in dequantized or "layer._c8_v_scale" not in dequantized:
        raise AuditInconclusive("dense fallback scale application changed")
    if "_c8_k_offset" in dequantized or "_c8_v_offset" in dequantized:
        raise AuditInconclusive("dense fallback now consumes offsets")
    findings.append(
        {
            "id": "offset-contract-is-symmetric-only",
            "status": "confirmed-in-pinned-source",
            "summary": (
                "Cache write adds configured offsets, while the dense fallback "
                "dequantizes with scale only. Correctness therefore requires zero "
                "offsets until both native and fallback paths share an asymmetric "
                "dequantization contract."
            ),
            "evidence": [
                source_ref(
                    attention,
                    "AscendC8AttentionBackendImpl._quantize_kv_to_int8",
                    quantize,
                ),
                source_ref(
                    attention,
                    "AscendC8AttentionBackendImpl._dequant_paged_kv_to_dense",
                    dequantize,
                ),
            ],
        }
    )
    return findings


def audit(
    ascend_repo: Path, *, ascend_revision: str = DEFAULT_ASCEND_REVISION
) -> dict[str, Any]:
    revision = require_exact_revision(ascend_revision)
    origin = _git(ascend_repo, "remote", "get-url", "origin").decode()
    canonical_repository_origin(origin)
    paths = (MODELSLIM_CONFIG, C8_METHOD, ATTENTION, PLATFORM)
    units = {path: load_source(ascend_repo, revision, path) for path in paths}
    findings = analyze(units)
    model_files = sorted(BF16_MODEL_FILES)
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "read-only-pinned-source-and-model-manifest-audit",
        "source_audit_status": "BLOCKED_BEFORE_PROVIDER_REACHABILITY",
        "runtime_compatible": False,
        "npu_started": False,
        "performance_claim": False,
        "host": {
            "repository": ASCEND_REPOSITORY,
            "revision": revision,
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "files_source": (
                "https://huggingface.co/api/models/"
                f"{MODEL_REPOSITORY}/tree/{MODEL_REVISION}"
            ),
            "files": model_files,
            "files_sha256": hashlib.sha256(
                ("\n".join(model_files) + "\n").encode()
            ).hexdigest(),
            "has_modelslim_description": "quant_model_description.json" in model_files,
        },
        "sources": [
            {"path": unit.path, "blob_sha256": unit.sha256} for unit in units.values()
        ],
        "findings": findings,
        "required_closure": [
            "Provide a model-revision-bound KV calibration profile with per-layer "
            "K/V scales and explicitly zero offsets, or implement and validate an "
            "equivalent Host-owned scale source.",
            "Define a reviewed BF16-to-C8 entry path that selects C8 only for the ten "
            "full-attention layers, loads the profile before cache allocation/use, "
            "and preserves TP2 sharding and cache-write ownership.",
            "Keep the extension import_only until manager enablement reaches a "
            "provider with the pinned Host and scale profile and fails closed on "
            "identity or shape mismatch.",
        ],
        "limitations": [
            "The model file list is a retained manifest observation for the exact "
            "public model revision; the audit does not download model tensors.",
            "No Host module was imported and no runtime, service, or NPU path ran.",
            "The audit proves a reachability blocker, not numerical quality or "
            "performance.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-repo", type=Path, required=True)
    parser.add_argument("--ascend-revision", default=DEFAULT_ASCEND_REVISION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.ascend_repo, ascend_revision=args.ascend_revision)
    except (
        AuditInconclusive,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "classification": "read-only-pinned-source-and-model-manifest-audit",
            "source_audit_status": "INCONCLUSIVE",
            "runtime_compatible": None,
            "npu_started": False,
            "performance_claim": False,
            "requested_revision": args.ascend_revision,
            "error": str(error),
        }
        exit_code = 2
    else:
        exit_code = 0

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
