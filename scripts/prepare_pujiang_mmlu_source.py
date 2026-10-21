#!/usr/bin/env python3
"""Prepare pinned MMLU-Pro source prompts, not a model run or accepted contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from scripts.validate_pujiang_mmlu_pair import (
        BENCHMARK_REVISION,
        DATASET_REVISION,
        SCORER_SHA256,
        load_scorer,
    )
    from scripts.validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json
except ModuleNotFoundError:
    from validate_pujiang_mmlu_pair import (
        BENCHMARK_REVISION,
        DATASET_REVISION,
        SCORER_SHA256,
        load_scorer,
    )
    from validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json

SELECTION_SCHEMA = "aqkv-mmlu-source-selection/v1"
PLAN_SCHEMA = "aqkv-mmlu-source-plan/v1"
SOURCE_COUNTS = {"test": 12032, "validation": 70}
GENERATION = {
    "temperature": 0,
    "seed": 17,
    "max_tokens": 2048,
    "ignore_eos": False,
    "n": 1,
    "stop": None,
    "chat_template_kwargs": {"enable_thinking": False},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _checked(path: Path, digest: str) -> bytes:
    raw = path.read_bytes()
    _require(_sha(raw) == digest, f"Pinned source checksum mismatch: {path.name}")
    return raw


def validate_selection(selection: object) -> dict[str, Any]:
    _require(isinstance(selection, dict), "Selection must be an object")
    _require(selection.get("schema_version") == SELECTION_SCHEMA, "Selection schema")
    _require(
        selection.get("dataset_revision") == DATASET_REVISION
        and selection.get("split") == "test",
        "Only the pinned MMLU-Pro test split is supported",
    )
    _require(
        selection.get("review_status") == "candidate-unreviewed",
        "Source preparation cannot declare contract acceptance",
    )
    _require(
        isinstance(selection.get("selection_reason"), str)
        and bool(selection["selection_reason"].strip()),
        "Record the prospective selection reason",
    )
    _require(
        isinstance(selection.get("model_alias"), str)
        and bool(selection["model_alias"].strip()),
        "A proposed served-model alias is required, not model provenance",
    )
    ids = selection.get("question_ids")
    _require(
        isinstance(ids, list)
        and 0 < len(ids) <= SOURCE_COUNTS["test"]
        and all(type(identity) is int and identity >= 0 for identity in ids)
        and len(set(ids)) == len(ids),
        "Need ordered unique integer test question IDs",
    )
    return selection


def build_tasks(
    tables: dict[str, list[dict]],
    instruction: str,
    selection: object,
    scorer: ModuleType,
) -> list[dict]:
    """Construct messages from supplied rows; only prepare_source verifies assets."""
    choice = validate_selection(selection)
    _require(instruction.count("{$}") == 1, "Expected one subject placeholder")
    _require(set(tables) == {"test", "validation"}, "Need both official splits")
    lookup = {}
    for split, rows in tables.items():
        _require(isinstance(rows, list) and bool(rows), f"Empty {split} split")
        seen = set()
        for row in rows:
            _require(isinstance(row, dict), "Source row must be an object")
            identity = row.get("question_id")
            _require(
                type(identity) is int and identity >= 0 and identity not in seen,
                f"Duplicate or invalid {split} question ID",
            )
            seen.add(identity)
            for field in ("category", "question", "answer", "cot_content"):
                _require(isinstance(row.get(field), str), f"Invalid source {field}")
            _require(bool(row["category"].strip()), "Empty source category")
            options = row.get("options")
            _require(
                isinstance(options, list)
                and 2 <= len([x for x in options if x != "N/A"]) <= 10
                and all(isinstance(x, str) for x in options)
                and type(row.get("answer_index")) is int,
                "Invalid source options/index",
            )
            # Reuse the checksum-pinned answer/filter consistency checks.
            scorer.format_example(row, answered=(split == "validation"))
            if split == "test":
                lookup[identity] = row
    categories = {row["category"] for row in tables["test"]}
    _require(
        {row["category"] for row in tables["validation"]} == categories,
        "Validation/test category coverage mismatch",
    )
    prefixes = {}
    example_ids = {}
    for category in sorted(categories):
        examples = [x for x in tables["validation"] if x["category"] == category]
        _require(
            len(examples) == 5, "Require all five source-order examples per subject"
        )
        prefixes[category] = (
            instruction.replace("{$}", category)
            + "\n"
            + "".join(scorer.format_example(row, answered=True) for row in examples)
        )
        example_ids[category] = [row["question_id"] for row in examples]
    _require(set(choice["question_ids"]) <= set(lookup), "Unknown test question ID")
    tasks = []
    for identity in choice["question_ids"]:
        row = lookup[identity]
        content = prefixes[row["category"]] + scorer.format_example(row, answered=False)
        request = {
            "model": choice["model_alias"],
            "messages": [{"role": "user", "content": content}],
            **copy.deepcopy(GENERATION),
            "stream": False,
        }
        tasks.append(
            {
                "case_id": f"mmlu-pro-{identity:05d}",
                "question_id": identity,
                "category": row["category"],
                "source_row_sha256": _sha(_canonical(row)),
                "validation_example_ids": example_ids[row["category"]],
                "answer": row["answer"],
                "option_count": len([x for x in row["options"] if x != "N/A"]),
                "prompt_text_sha256": _sha(content.encode()),
                "candidate_request": request,
                "request_sha256": _sha(_canonical(request)),
                "prompt_token_ids": None,
                "prompt_tokens": None,
            }
        )
    return tasks


def prepare_source(
    assets: Path, reference: Path, selection_path: Path, benchmark_checkout: Path
) -> dict[str, Any]:
    scorer = load_scorer(benchmark_checkout)
    source_bytes = {
        split: _checked(assets / f"{split}-00000-of-00001.parquet", digest)
        for split, digest in scorer.DATA_HASHES.items()
    }
    _checked(reference / "evaluate_from_local.py", scorer.REFERENCE_HASH)
    instruction_raw = _checked(
        reference / "cot_prompt_lib/initial_prompt.txt", scorer.INITIAL_PROMPT_HASH
    )
    selection_raw = selection_path.read_bytes()
    selection = validate_selection(parse_json(selection_raw))
    # No pickle, dataset loader, tokenizer, model, or reference runner is executed.
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ValueError("Offline source preparation requires pyarrow") from error
    tables = {
        split: pq.read_table(io.BytesIO(raw)).to_pylist()
        for split, raw in source_bytes.items()
    }
    _require(
        {split: len(rows) for split, rows in tables.items()} == SOURCE_COUNTS,
        "Pinned dataset cardinality mismatch",
    )
    tasks = build_tasks(tables, instruction_raw.decode(), selection, scorer)
    subjects = sorted({row["category"] for row in tables["test"]})
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "source_verified_candidate_plan_execution_blocked",
        "dataset": "mmlu-pro",
        "dataset_revision": DATASET_REVISION,
        "benchmark_revision": BENCHMARK_REVISION,
        "scorer_source_sha256": SCORER_SHA256,
        "split": "test",
        "source_counts": SOURCE_COUNTS.copy(),
        "source_data_sha256": dict(scorer.DATA_HASHES),
        "reference_revision": scorer.REFERENCE_REVISION,
        "reference_sha256": scorer.REFERENCE_HASH,
        "instruction_sha256": scorer.INITIAL_PROMPT_HASH,
        "selection_sha256": _sha(selection_raw),
        "selection": selection,
        "coverage": "full-test-source"
        if len(tasks) == SOURCE_COUNTS["test"]
        else "explicit-named-subset",
        "subjects": subjects,
        "task_count": len(tasks),
        "tasks_sha256": _sha(_canonical(tasks)),
        "tasks": tasks,
        "transport_intent": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "concurrency": 1,
            "retries": 0,
            "context_tokens": 32768,
            "truncation": "reject-after-exact-tokenization",
        },
        "tokenizer_verified": False,
        "transport_verified": False,
        "runtime_activation_verified": False,
        "model_requests_sent": 0,
        "task_quality": None,
        "pending_gates": [
            "source rights and candidate selection/prompt/generation contract review",
            "exact model/tokenizer revision, chat template and token/context audit",
            "accepted Host/Manager activation and C8 profile correctness",
            "attributable B0/B1 service identity and bounded transport receipts",
            "device lease, cleanup, paired raw outputs and target-path observations",
        ],
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "reference", "selection", "benchmark-checkout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = prepare_source(
            args.assets, args.reference, args.selection, args.benchmark_checkout
        )
        # Raw questions and answers stay outside the public source tree.
        with args.output.open("x", encoding="utf-8") as output:
            args.output.chmod(0o600)
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(json.dumps({"status": report["status"], "task_count": report["task_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
