#!/usr/bin/env python3
"""Audit pinned host sources for the C8 continuing-prefill execution boundary."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CORE_REVISION = "a4d6aa022fb1885a25a802a6e29372c81eac6c9f"
DEFAULT_ASCEND_REVISION = "2c8c722107a54127999a64c4eb0ec86139df8c26"
SCHEMA_VERSION = "vllm-ascend-c8-host-source-audit/v1"
CORE_REPOSITORY = "vLLM-HUST/vllm-hust"
ASCEND_REPOSITORY = "vLLM-HUST/vllm-ascend-hust"

CORE_REGISTRY = "vllm/v1/attention/backends/registry.py"
CORE_SELECTOR = "vllm/v1/attention/selector.py"
ASCEND_PLATFORM = "vllm_ascend/platform.py"
ASCEND_C8_METHOD = "vllm_ascend/quantization/methods/kv_cache/kv_c8.py"
ASCEND_ATTENTION = "vllm_ascend/attention/attention_v1.py"
ASCEND_DEVICE_OP = "vllm_ascend/device/device_op.py"
ASCEND_ATTENTION_TEST = "tests/ut/attention/test_attention_v1.py"
ASCEND_C8_TEST = "tests/ut/quantization/methods/test_kv_c8.py"


class AuditInconclusive(RuntimeError):
    """Raised when a pinned source no longer has the expected structure."""


def require_exact_revision(revision: str) -> str:
    """Reject branches, tags and abbreviated object names."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AuditInconclusive(
            f"revision must be a full lowercase 40-character commit SHA: {revision!r}"
        )
    return revision


@dataclass(frozen=True, slots=True)
class SourceUnit:
    repository: str
    revision: str
    path: str
    text: str
    sha256: str


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        [
            "git",
            "-c",
            "http.proxy=",
            "-c",
            "https.proxy=",
            "-C",
            str(repo),
            *args,
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def load_source(
    repo: Path,
    repository: str,
    revision: str,
    path: str,
) -> SourceUnit:
    require_exact_revision(revision)
    resolved = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.decode().strip() != revision:
        raise AuditInconclusive(f"revision did not resolve exactly: {revision}")
    raw = _git(repo, "show", f"{revision}:{path}")
    return SourceUnit(
        repository=repository,
        revision=revision,
        path=path,
        text=raw.decode("utf-8"),
        sha256=hashlib.sha256(raw).hexdigest(),
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


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def loaded_name_count(node: ast.AST, name: str) -> int:
    return sum(
        isinstance(item, ast.Name)
        and isinstance(item.ctx, ast.Load)
        and item.id == name
        for item in ast.walk(node)
    )


def contains_dotted_name(node: ast.AST, name: str) -> bool:
    return any(dotted_name(item) == name for item in ast.walk(node))


def calls_with_suffix(node: ast.AST, suffix: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (dotted_name(item.func) or "").endswith(suffix)
    ]


def literal_keyword(call: ast.Call, name: str) -> Any:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise AuditInconclusive(f"expected one {name!r} keyword")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise AuditInconclusive(f"{name!r} keyword is not literal") from error


def has_class_rewrite(node: ast.AST) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, (ast.Assign, ast.AnnAssign)):
            continue
        targets = item.targets if isinstance(item, ast.Assign) else [item.target]
        value = item.value
        if any(dotted_name(target) == "layer.impl.__class__" for target in targets):
            return dotted_name(value) == "AscendC8AttentionBackendImpl"
    return False


def literal_assignments(node: ast.AST) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or len(item.targets) != 1:
            continue
        name = dotted_name(item.targets[0])
        if name is None:
            continue
        try:
            values[name] = ast.literal_eval(item.value)
        except (ValueError, TypeError):
            continue
    return values


def find_ifs(
    node: ast.AST,
    required_name: str,
    required_call: str | None = None,
) -> list[ast.If]:
    matches = []
    for item in ast.walk(node):
        if not isinstance(item, ast.If):
            continue
        if not contains_dotted_name(item.test, required_name):
            continue
        if required_call is not None:
            body = ast.Module(body=item.body, type_ignores=[])
            if not calls_with_suffix(body, required_call):
                continue
        matches.append(item)
    return matches


def find_if(
    node: ast.AST,
    required_name: str,
    required_call: str | None = None,
) -> ast.If:
    matches = find_ifs(node, required_name, required_call)
    if len(matches) != 1:
        raise AuditInconclusive(
            f"expected one branch containing {required_name!r} and {required_call!r}"
        )
    return matches[0]


def source_ref(unit: SourceUnit, symbol: str, node: ast.AST) -> dict[str, Any]:
    return {
        "path": unit.path,
        "symbol": symbol,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "blob_sha256": unit.sha256,
    }


def confirmed(identifier: str, summary: str, *evidence: dict[str, Any]) -> dict:
    return {
        "id": identifier,
        "status": "confirmed-in-pinned-source",
        "summary": summary,
        "evidence": list(evidence),
    }


def analyze(units: dict[str, SourceUnit]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    registry = units[CORE_REGISTRY]
    registry_enum = find_qualified_def(registry.text, "AttentionBackendEnum")
    custom_members = [
        item
        for item in registry_enum.body  # type: ignore[attr-defined]
        if isinstance(item, ast.Assign)
        and any(dotted_name(target) == "CUSTOM" for target in item.targets)
        and isinstance(item.value, ast.Constant)
        and item.value.value is None
    ]
    register = find_qualified_def(registry.text, "register_backend")
    if len(custom_members) != 1 or not contains_dotted_name(
        register, "_ATTN_OVERRIDES"
    ):
        raise AuditInconclusive("core CUSTOM registration structure changed")

    selector = units[CORE_SELECTOR]
    cached_selector = find_qualified_def(selector.text, "_cached_get_attn_backend")
    if not calls_with_suffix(cached_selector, "current_platform.get_attn_backend_cls"):
        raise AuditInconclusive("core selector no longer delegates to the platform")

    platform = units[ASCEND_PLATFORM]
    platform_selector = find_qualified_def(
        platform.text, "NPUPlatform.get_attn_backend_cls"
    )
    assignments = literal_assignments(platform_selector)
    if loaded_name_count(platform_selector, "selected_backend") != 0:
        raise AuditInconclusive("Ascend platform now consumes selected_backend")
    backend_map = assignments.get("backend_map")
    if (
        not isinstance(backend_map, dict)
        or backend_map.get((False, False, False))
        != "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    ):
        raise AuditInconclusive("Ascend dense-attention backend map changed")
    findings.append(
        confirmed(
            "backend-selection",
            "Core CUSTOM registration still delegates selection to Ascend, whose "
            "candidate selector does not consume selected_backend and returns "
            "fixed maps.",
            source_ref(registry, "register_backend", register),
            source_ref(selector, "_cached_get_attn_backend", cached_selector),
            source_ref(
                platform,
                "NPUPlatform.get_attn_backend_cls",
                platform_selector,
            ),
        )
    )

    c8_method = units[ASCEND_C8_METHOD]
    create_weights = find_qualified_def(
        c8_method.text, "AscendC8KVCacheAttentionMethod.create_weights"
    )
    if not has_class_rewrite(create_weights):
        raise AuditInconclusive("C8 implementation class replacement was not found")
    findings.append(
        confirmed(
            "c8-implementation-ownership",
            "C8 weight setup replaces layer.impl.__class__ with the host C8 subclass.",
            source_ref(
                c8_method,
                "AscendC8KVCacheAttentionMethod.create_weights",
                create_weights,
            ),
        )
    )

    attention = units[ASCEND_ATTENTION]
    forward = find_qualified_def(attention.text, "AscendC8AttentionBackendImpl.forward")
    capture_branches = find_ifs(forward, "_EXTRA_CTX.capturing", "full_graph_fia")
    if len(capture_branches) != 2:
        raise AuditInconclusive("expected two C8 capture dispatch branches")
    chunked_branch = find_if(
        forward,
        "AscendAttentionState.ChunkedPrefill",
        "_forward_c8_chunked_prefill",
    )
    if capture_branches[0].lineno >= chunked_branch.lineno:
        raise AuditInconclusive("capture no longer precedes chunked-prefill dispatch")
    findings.append(
        confirmed(
            "capture-dispatch-order",
            "C8 capture returns through full_graph_fia before the "
            "ChunkedPrefill branch.",
            source_ref(attention, "AscendC8AttentionBackendImpl.forward", forward),
        )
    )

    full_graph = find_qualified_def(
        attention.text, "AscendAttentionBackendImpl.full_graph_fia"
    )
    c8_capture = find_if(full_graph, "self.enable_c8_quant", "self._nz_5d_view")
    capture_values = literal_assignments(c8_capture)
    if capture_values.get("input_layout") != "BNSD":
        raise AuditInconclusive("C8 capture layout is no longer BNSD")
    if capture_values.get("attn_mask", object()) is not None:
        raise AuditInconclusive("C8 capture mask is no longer literal None")
    if capture_values.get("sparse_mode") != 0:
        raise AuditInconclusive("C8 capture sparse mode is no longer zero")
    if not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "unsqueeze"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 2
        for call in ast.walk(c8_capture)
        if isinstance(call, ast.Call)
    ):
        raise AuditInconclusive("C8 capture query.unsqueeze(2) was not found")
    findings.append(
        confirmed(
            "capture-is-decode-shaped",
            "The C8 graph branch uses paged NZ K/V but reshapes Q with "
            "unsqueeze(2), selects BNSD, clears the mask and sets sparse_mode=0. "
            "This is decode-shaped and does not establish native multi-token "
            "continuing-prefill support.",
            source_ref(
                attention, "AscendAttentionBackendImpl.full_graph_fia", full_graph
            ),
        )
    )

    update_graph = find_qualified_def(
        attention.text, "AscendAttentionBackendImpl.update_graph_params"
    )
    replay_branch = find_if(update_graph, "c8_k_aq_scale")
    if not calls_with_suffix(update_graph, "npu_fused_infer_attention_score.out"):
        raise AuditInconclusive("FIA replay update call was not found")
    replay_values = literal_assignments(replay_branch)
    if (
        replay_values.get("input_layout") != "BNSD"
        or replay_values.get("sparse_mode") != 0
    ):
        raise AuditInconclusive("C8 replay update is no longer BNSD sparse_mode=0")
    findings.append(
        confirmed(
            "capture-replay-coupling",
            "Graph replay updates repeat the C8 BNSD/sparse_mode=0 contract, so a "
            "host provider boundary must cover capture records and replay updates.",
            source_ref(
                attention,
                "AscendAttentionBackendImpl.update_graph_params",
                update_graph,
            ),
        )
    )

    materializer = find_qualified_def(
        attention.text, "AscendC8AttentionBackendImpl._dequant_paged_kv_to_dense"
    )
    chunked = find_qualified_def(
        attention.text, "AscendC8AttentionBackendImpl._forward_c8_chunked_prefill"
    )
    if not calls_with_suffix(chunked, "_dequant_paged_kv_to_dense"):
        raise AuditInconclusive(
            "chunked prefill no longer calls the dense materializer"
        )
    tnd_dense_calls = [
        call
        for call in calls_with_suffix(chunked, "npu_fused_infer_attention_score")
        if any(keyword.arg == "input_layout" for keyword in call.keywords)
        and literal_keyword(call, "input_layout") == "TND"
        and literal_keyword(call, "block_table") is None
    ]
    if len(tnd_dense_calls) != 1:
        raise AuditInconclusive("expected one dense TND continuing-prefill call")
    if not contains_dotted_name(materializer, "block_table.reshape") or not any(
        isinstance(item, ast.Subscript)
        and dotted_name(item.value) in {"key_nz", "value_nz"}
        for item in ast.walk(materializer)
    ):
        raise AuditInconclusive("paged-KV gather structure changed")
    findings.append(
        confirmed(
            "dense-materialization",
            "Continuing prefill gathers paged INT8 blocks, materializes dense K/V, "
            "and calls TND FIA with block_table=None.",
            source_ref(
                attention,
                "AscendC8AttentionBackendImpl._dequant_paged_kv_to_dense",
                materializer,
            ),
            source_ref(
                attention,
                "AscendC8AttentionBackendImpl._forward_c8_chunked_prefill",
                chunked,
            ),
        )
    )

    direct_methods = {}
    for name in (
        "AscendC8AttentionBackendImpl._forward_c8_decode",
        "AscendC8AttentionBackendImpl._forward_c8_chunked_prefill",
        "AscendC8AttentionBackendImpl._forward_c8_fused_infer_attention",
    ):
        method = find_qualified_def(attention.text, name)
        if not calls_with_suffix(method, "torch_npu.npu_fused_infer_attention_score"):
            raise AuditInconclusive(f"{name} no longer calls torch_npu FIA directly")
        if contains_dotted_name(method, "DeviceOperator"):
            raise AuditInconclusive(f"{name} now uses DeviceOperator")
        direct_methods[name] = method

    device_op = units[ASCEND_DEVICE_OP]
    get_adaptor = find_qualified_def(device_op.text, "get_device_adaptor")
    if not contains_dotted_name(get_adaptor, "adaptor_by_family"):
        raise AuditInconclusive("device adaptor map was not found")
    findings.append(
        confirmed(
            "device-adaptor-is-not-c8-provider",
            "The audited C8 methods call torch_npu FIA directly. DeviceOperator is "
            "selected from a hardware-family map and is not their provider boundary.",
            *(
                source_ref(attention, name, method)
                for name, method in direct_methods.items()
            ),
            source_ref(device_op, "get_device_adaptor", get_adaptor),
        )
    )

    return findings


def audit(
    core_repo: Path,
    ascend_repo: Path,
    *,
    core_revision: str = DEFAULT_CORE_REVISION,
    ascend_revision: str = DEFAULT_ASCEND_REVISION,
) -> dict[str, Any]:
    core_revision = require_exact_revision(core_revision)
    ascend_revision = require_exact_revision(ascend_revision)
    specs = (
        (core_repo, CORE_REPOSITORY, core_revision, CORE_REGISTRY),
        (core_repo, CORE_REPOSITORY, core_revision, CORE_SELECTOR),
        (ascend_repo, ASCEND_REPOSITORY, ascend_revision, ASCEND_PLATFORM),
        (ascend_repo, ASCEND_REPOSITORY, ascend_revision, ASCEND_C8_METHOD),
        (ascend_repo, ASCEND_REPOSITORY, ascend_revision, ASCEND_ATTENTION),
        (ascend_repo, ASCEND_REPOSITORY, ascend_revision, ASCEND_DEVICE_OP),
        (
            ascend_repo,
            ASCEND_REPOSITORY,
            ascend_revision,
            ASCEND_ATTENTION_TEST,
        ),
        (ascend_repo, ASCEND_REPOSITORY, ascend_revision, ASCEND_C8_TEST),
    )
    units = {
        path: load_source(repo, repository, revision, path)
        for repo, repository, revision, path in specs
    }
    findings = analyze(units)
    target_symbols = (
        "_forward_c8_chunked_prefill",
        "_dequant_paged_kv_to_dense",
        "full_graph_fia",
    )
    direct_test_references = {
        path: {symbol: unit.text.count(symbol) for symbol in target_symbols}
        for path, unit in units.items()
        if path in {ASCEND_ATTENTION_TEST, ASCEND_C8_TEST}
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "read-only-pinned-source-audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_audit_status": "SUPPORTED_BY_PINNED_SOURCE",
        "runtime_compatible": None,
        "npu_started": False,
        "performance_claim": False,
        "sources": [
            {
                "repository": unit.repository,
                "revision": unit.revision,
                "path": unit.path,
                "blob_sha256": unit.sha256,
            }
            for unit in units.values()
        ],
        "findings": findings,
        "direct_target_symbol_references_in_scanned_tests": direct_test_references,
        "limitations": [
            "No host module was imported and no runtime or NPU path was executed.",
            "Branch reachability and numerical behavior remain unproven.",
            "The two scanned test files cannot prove the absence of indirect coverage.",
            "Candidate source revisions do not replace the active parent gate pins.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-repo", type=Path, required=True)
    parser.add_argument("--ascend-repo", type=Path, required=True)
    parser.add_argument(
        "--core-revision",
        default=DEFAULT_CORE_REVISION,
        help="full core commit SHA (defaults to the retained S3b candidate)",
    )
    parser.add_argument(
        "--ascend-revision",
        default=DEFAULT_ASCEND_REVISION,
        help="full Ascend commit SHA (defaults to the retained S3b candidate)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(
            args.core_repo,
            args.ascend_repo,
            core_revision=args.core_revision,
            ascend_revision=args.ascend_revision,
        )
    except (
        AuditInconclusive,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "classification": "read-only-pinned-source-audit",
            "source_audit_status": "INCONCLUSIVE",
            "runtime_compatible": None,
            "npu_started": False,
            "performance_claim": False,
            "requested_revisions": {
                CORE_REPOSITORY: args.core_revision,
                ASCEND_REPOSITORY: args.ascend_revision,
            },
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
