#!/usr/bin/env python3
"""Audit exact local MMLU prompt tokens without sending model requests."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from scripts.prepare_pujiang_mmlu_source import (
        _canonical,
        _checked,
        _require,
        _sha,
        load_scorer,
        prepare_source,
    )
except ModuleNotFoundError:
    from prepare_pujiang_mmlu_source import (
        _canonical,
        _checked,
        _require,
        _sha,
        load_scorer,
        prepare_source,
    )

MODEL_REPOSITORY = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
TOKENIZER_FILES = {
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
}


def tokenize_tasks(
    source: dict[str, Any], scorer: ModuleType, tokenizer: Any
) -> tuple[list[dict], dict]:
    """Audit supplied tasks; prepare_tokens checks source/tokenizer provenance."""
    tasks = copy.deepcopy(source["tasks"])
    context = source["transport_intent"]["context_tokens"]
    _require(type(context) is int and context > 0, "Invalid context budget")
    _require(bool(tasks), "No source tasks")
    # Some tokenizers copy the vocabulary when queried with added tokens.
    vocabulary_size = len(tokenizer)
    _require(vocabulary_size > 0, "Empty tokenizer vocabulary")
    for task in tasks:
        request = task["candidate_request"]
        _require(_sha(_canonical(request)) == task["request_sha256"], "Request changed")
        _require(
            request["chat_template_kwargs"] == {"enable_thinking": False},
            "Only the pinned non-thinking prompt mode is supported",
        )
        budget = request["max_tokens"]
        _require(type(budget) is int and budget > 0, "Invalid generation budget")
        messages = request["messages"]
        _require(
            len(messages) == 1 and messages[0]["role"] == "user",
            "Expected the source formatter's single user message",
        )
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = scorer.encode_prompt(tokenizer, messages[0]["content"])
        direct = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        _require(
            isinstance(ids, list)
            and len(ids) >= 16
            and all(
                type(identity) is int and 0 <= identity < vocabulary_size
                for identity in ids
            ),
            "Invalid actual prompt token IDs",
        )
        _require(ids == direct, "Shared encoder and direct chat tokenization disagree")
        _require(
            _sha(messages[0]["content"].encode()) == task["prompt_text_sha256"],
            "Source prompt changed",
        )
        task.update(
            prompt_token_ids=ids,
            prompt_tokens=len(ids),
            prompt_token_ids_sha256=_sha(_canonical(ids)),
            rendered_prompt_sha256=_sha(rendered.encode()),
            context_headroom_tokens=context - len(ids) - budget,
            within_context_budget=len(ids) + budget <= context,
        )
    lengths = [task["prompt_tokens"] for task in tasks]
    exceeded = [task["case_id"] for task in tasks if not task["within_context_budget"]]
    return tasks, {
        "context_tokens": context,
        "prompt_tokens_min": min(lengths),
        "prompt_tokens_max": max(lengths),
        "context_exceeded_case_ids": exceeded,
        "all_tasks_retained": True,
        "truncation_applied": False,
        "context_budget_passed": not exceeded,
    }


def prepare_tokens(
    assets: Path,
    reference: Path,
    selection: Path,
    benchmark_checkout: Path,
    tokenizer_directory: Path,
) -> dict:
    source = prepare_source(assets, reference, selection, benchmark_checkout)
    scorer = load_scorer(benchmark_checkout)
    verified = {
        name: _checked(tokenizer_directory / name, digest)
        for name, digest in TOKENIZER_FILES.items()
    }
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ValueError("Local token audit requires Transformers") from error
    # Load only verified bytes, not extra config/code or changing source files.
    with tempfile.TemporaryDirectory(prefix="aqkv-pinned-tokenizer-") as temporary:
        directory = Path(temporary)
        for name, raw in verified.items():
            destination = directory / name
            destination.write_bytes(raw)
            destination.chmod(0o600)
        tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        _require(
            tokenizer.chat_template == verified["chat_template.jinja"].decode(),
            "Loaded chat template differs from the pinned file",
        )
        tasks, summary = tokenize_tasks(source, scorer, tokenizer)
        tokenizer_class = type(tokenizer).__name__
    report = copy.deepcopy(source)
    report.update(
        schema_version="aqkv-mmlu-token-plan/v1",
        status="local_tokens_verified_candidate_plan_execution_blocked"
        if summary["context_budget_passed"]
        else "context_budget_failed_execution_blocked",
        source_tasks_sha256=source["tasks_sha256"],
        tasks=tasks,
        tasks_sha256=_sha(_canonical(tasks)),
        tokenizer_verified=True,
        serving_tokenizer_verified=False,
        model_weights_verified=False,
        tokenizer={
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "files_sha256": TOKENIZER_FILES.copy(),
            "class": tokenizer_class,
            "library_versions": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "tokenizers", "jinja2")
            },
            "scope": "local-text-only-not-serving-runtime",
        },
        context_audit=summary,
        pending_gates=[
            "source rights and candidate selection/prompt/generation contract review",
            "serving model/tokenizer/template/library identity and token parity",
            "accepted Host/Manager activation and C8 profile correctness",
            "attributable B0/B1 service identity and bounded transport receipts",
            "device lease, cleanup, paired raw outputs and target-path observations",
        ],
    )
    return report


def scoring_bundle(plan: dict) -> dict:
    """Export scorer inputs; provenance is established by rebuilding prepare_tokens."""
    try:
        from scripts.validate_pujiang_mmlu_pair import (
            SCORING_BUNDLE_SCHEMA,
            SOURCE_SCHEMA,
        )
    except ModuleNotFoundError:
        from validate_pujiang_mmlu_pair import SCORING_BUNDLE_SCHEMA, SOURCE_SCHEMA
    _require(
        plan.get("schema_version") == "aqkv-mmlu-token-plan/v1"
        and plan.get("tokenizer_verified") is True
        and plan["context_audit"]["context_budget_passed"] is True,
        "Scorer export requires a context-valid local token plan",
    )
    tasks = plan["tasks"]
    _require(
        bool(tasks)
        and len(tasks) == plan["task_count"]
        and _sha(_canonical(tasks)) == plan["tasks_sha256"],
        "Token plan task count/hash changed",
    )
    cases = []
    generation = None
    for task in tasks:
        request = task["candidate_request"]
        settings = {
            key: value
            for key, value in request.items()
            if key not in ("model", "messages")
        }
        if generation is None:
            generation = settings
        _require(
            _canonical(settings) == _canonical(generation)
            and request["model"] == plan["selection"]["model_alias"]
            and _sha(_canonical(request)) == task["request_sha256"]
            and task["within_context_budget"] is True
            and task["prompt_tokens"] == len(task["prompt_token_ids"])
            and _sha(_canonical(task["prompt_token_ids"]))
            == task["prompt_token_ids_sha256"],
            "Task request/tokens differ from the source token plan",
        )
        cases.append(
            {
                **{
                    name: copy.deepcopy(task[name])
                    for name in (
                        "case_id",
                        "question_id",
                        "category",
                        "source_row_sha256",
                        "validation_example_ids",
                        "answer",
                        "option_count",
                        "prompt_tokens",
                        "prompt_text_sha256",
                        "rendered_prompt_sha256",
                        "candidate_request",
                        "request_sha256",
                    )
                },
                "prompt": copy.deepcopy(task["prompt_token_ids"]),
                "prompt_sha256": task["prompt_token_ids_sha256"],
                "max_tokens": request["max_tokens"],
                "ignore_eos": request["ignore_eos"],
            }
        )
    binding = {
        name: copy.deepcopy(plan[name])
        for name in (
            "source_data_sha256",
            "reference_revision",
            "reference_sha256",
            "instruction_sha256",
            "selection_sha256",
            "selection",
            "coverage",
            "source_tasks_sha256",
            "tasks_sha256",
            "tokenizer",
            "context_audit",
        )
    }
    binding["cases_sha256"] = _sha(_canonical(cases))
    return {
        "schema_version": SCORING_BUNDLE_SCHEMA,
        "status": "source_bound_scorer_inputs_only_execution_blocked",
        "cases": cases,
        "envelope_contract": {
            "schema_version": SOURCE_SCHEMA,
            "scope": "verified-local-source-selection",
            "dataset": plan["dataset"],
            "dataset_revision": plan["dataset_revision"],
            "benchmark_revision": plan["benchmark_revision"],
            "scorer_source_sha256": plan["scorer_source_sha256"],
            "task_count": len(cases),
            "generation": copy.deepcopy(generation),
            "source_binding": binding,
        },
        "required_outcome_hashes": [
            "prompt_sha256",
            "request_sha256",
            "rendered_prompt_sha256",
        ],
        "serving_tokenizer_verified": False,
        "transport_verified": False,
        "model_requests_sent": 0,
        "claim_boundary": copy.deepcopy(plan["claim_boundary"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "assets",
        "reference",
        "selection",
        "benchmark-checkout",
        "tokenizer-directory",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scoring-output", type=Path)
    args = parser.parse_args()
    try:
        _require(not args.output.exists(), "Output already exists")
        if args.scoring_output is not None:
            _require(
                args.scoring_output.resolve() != args.output.resolve()
                and not args.scoring_output.exists(),
                "Scoring output must be a distinct new file",
            )
        report = prepare_tokens(
            args.assets,
            args.reference,
            args.selection,
            args.benchmark_checkout,
            args.tokenizer_directory,
        )
        bundle = (
            scoring_bundle(report)
            if args.scoring_output is not None
            and report["context_audit"]["context_budget_passed"]
            else None
        )
        with args.output.open("x", encoding="utf-8") as output:
            args.output.chmod(0o600)
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if bundle is not None:
            with args.scoring_output.open("x", encoding="utf-8") as output:
                args.scoring_output.chmod(0o600)
                output.write(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(
        json.dumps(
            {"status": report["status"], "context_audit": report["context_audit"]}
        )
    )
    return 0 if report["context_audit"]["context_budget_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
