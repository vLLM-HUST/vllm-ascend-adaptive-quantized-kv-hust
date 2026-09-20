#!/usr/bin/env python3
"""CPU audit of pinned PR #19 functions, never its shell runner or NPU code."""

from __future__ import annotations

import argparse
import ast
import contextlib
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

PR_HEAD = "470f1151b8ec6acd31a4b63f8de3cd882602f826"
BACKEND_BASE = "e2e1c96e3b2f040aaba06b8ba333070d19be4738"
PATCH_SHA256 = "3db28f84f8c94ee81ccb1761f38245cae15d8cb507bb64ecb2c561e53bc78d50"
PATCH_PATH = (
    "artifacts/m0/native-prefill-runtime-patch/"
    "0001-feat-attention-add-bounded-C8-prefill-path.patch.gz"
)


def git_show(repo: Path, revision: str, name: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{name}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def extract_function(source: str, name: str):
    nodes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1:
        raise ValueError(f"expected exactly one {name}")
    node = nodes[0]
    node.decorator_list = []
    namespace = {
        "torch": torch,
        "AttentionLayer": object,
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<pinned-function>", "exec"),
        namespace,
    )
    return namespace[name]


def reproduce(research_repo: Path, backend_repo: Path) -> dict:
    compressed = git_show(research_repo, PR_HEAD, PATCH_PATH)
    if hashlib.sha256(compressed).hexdigest() != PATCH_SHA256:
        raise ValueError("archived patch digest mismatch")
    diff = gzip.decompress(compressed).decode()
    section = diff.split("diff --git a/vllm_ascend/attention/attention_v1.py", 1)[1]
    additions = "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    start = additions.index("    def _blockwise_paged_int8_prefill(")
    terminal = "        return torch.cat(outputs, dim=0)"
    end = additions.index(terminal, start) + len(terminal)
    method = extract_function(
        textwrap.dedent(additions[start:end]), "_blockwise_paged_int8_prefill"
    )
    backend = git_show(
        backend_repo, BACKEND_BASE, "vllm_ascend/attention/attention_v1.py"
    ).decode()
    view = extract_function(backend, "_nz_5d_view")
    impl = SimpleNamespace(num_heads=4, num_kv_heads=2, head_size=32, scale=32**-0.5)
    impl._nz_5d_view = lambda cache, block_size: view(impl, cache, block_size)
    impl.key_cache = torch.zeros(2, 2, 1, 32, 32, dtype=torch.int8)
    impl.value_cache = torch.zeros_like(impl.key_cache)
    layer = SimpleNamespace(
        _c8_k_scale=torch.ones(1, 2, 32), _c8_v_scale=torch.ones(1, 2, 32)
    )
    try:
        method(impl, torch.zeros(7, 4, 32), torch.tensor([[0, 1]]), [48], [7], layer)
    except ValueError as error:
        raw_5d_error = str(error)
    else:
        raise AssertionError("expected pinned positive-test shape to fail")
    assert "block table does not cover" in raw_5d_error

    # Old 4D physical wrapper isolates negative-ID handling from the 5D bug.
    impl.key_cache = torch.zeros(2, 32, 2, 32, dtype=torch.int8)
    impl.value_cache = torch.ones_like(impl.key_cache)
    impl.value_cache[1].fill_(7)
    negative = method(
        impl, torch.zeros(1, 4, 32), torch.tensor([[-1]]), [1], [1], layer
    )
    assert torch.equal(negative, torch.full_like(negative, 7))

    live = git_show(
        research_repo, PR_HEAD, "scripts/run_m0_native_prefill_live_cell.sh"
    ).decode()
    finalizer = live.split("<<'PY' > \"$SESSION/live-cell-summary.json\"\n", 1)[
        1
    ].split("\nPY\n", 1)[0]
    with tempfile.TemporaryDirectory(prefix="pr19-cpu-audit-") as temporary:
        root = Path(temporary)
        for arm in ("bf16", "c8-fallback", "c8-native"):
            counts = {}
            if arm == "c8-fallback":
                counts["chunked_prefill_fia_tnd_dequantized_cache"] = 1
            if arm == "c8-native":
                counts["chunked_prefill_blockwise_paged_int8_resident"] = 1
            summary = {
                "overall_pass": True,
                "kv_cache_tokens": 100 if arm == "bf16" else 200,
                "cells": [
                    {
                        "cell": {
                            "context_tokens": 4096,
                            "num_prefixes": 12,
                            "round": 1,
                            "seed": 19,
                        },
                        "attention_dispatch_counts": counts,
                    }
                ],
            }
            (root / arm).mkdir()
            (root / arm / "summary.json").write_text(json.dumps(summary))
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["audit", str(root), "r1-c4096-p12"]),
            contextlib.redirect_stdout(output),
        ):
            exec(compile(finalizer, "<pinned-finalizer>", "exec"), {})
        claimed = json.loads(output.getvalue())
        assert claimed["decision"] == "PASS_D1"
        assert claimed["evidence_level"] == "real-online"

    runner = git_show(
        research_repo, PR_HEAD, "scripts/run_m0_clean_cache_matrix.sh"
    ).decode()
    assert '"reported_kv_cache_tokens": kv_token_observations[0]' in runner
    assert '"scheduler_visible_kv_cache_tokens": kv_token_observations[0]' in runner
    arm_case = runner.split('case "$MECHANISM_ARM" in\n', 1)[1].split("\nesac", 1)[0]
    arm_probe = subprocess.run(
        [
            "bash",
            "-c",
            "VARIANT=c8\nMECHANISM_ARM=${M0_MECHANISM_ARM:-$VARIANT}\n"
            + 'case "$MECHANISM_ARM" in\n'
            + arm_case
            + "\nesac",
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert arm_probe.returncode == 2
    return {
        "evidence_level": "synthetic-cpu-audit-only",
        "npu_started": False,
        "d1_accepted": False,
        "pr_head": PR_HEAD,
        "backend_base": BACKEND_BASE,
        "patch_sha256": PATCH_SHA256,
        "torch_version": torch.__version__,
        "raw_5d_test_error": raw_5d_error,
        "negative_page_reads_last_page": True,
        "summary_only_finalizer_claim": {
            "decision": claimed["decision"],
            "evidence_level": claimed["evidence_level"],
            "input_files": "three synthetic summary.json files only",
            "accepted_as_evidence": False,
        },
        "capacity_fields_share_same_log_value": True,
        "default_c8_arm": {
            "exit_code": arm_probe.returncode,
            "stderr": arm_probe.stderr.strip(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-repo", type=Path, required=True)
    parser.add_argument("--legacy-backend-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = reproduce(args.research_repo, args.legacy_backend_repo)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
