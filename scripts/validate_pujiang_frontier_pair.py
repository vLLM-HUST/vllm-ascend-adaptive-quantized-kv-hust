#!/usr/bin/env python3
"""Validate stored FrontierScience pairs offline; never authorize real scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from scripts.pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from scripts.validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json
except ModuleNotFoundError:
    from pujiang_shared_source import BENCHMARK_REVISION, load_verified_module
    from validate_pujiang_readiness import CLAIM_BOUNDARY, parse_json

DATASET_REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"
SOURCE_FILES = {
    "inputs": (
        "src/vllm_hust_benchmark/frontierscience_inputs.py",
        "3dcc94f282f055776f129688469320978c1dae7474284ce41e5e7e100519c8ff",
    ),
    "scorer": (
        "src/vllm_hust_benchmark/frontierscience_pyramidkv.py",
        "1e3b6a4e4eade0144f2372c6ed16292ae143bb4243bb672e1a374baa26560560",
    ),
}
SCHEMA = "aqkv-frontier-offline-envelope/v1"
TRACK_COUNTS = {"olympiad": 100, "research": 60}
EVIDENCE_KINDS = ("development-fixture", "unverified-stored-outputs")
REQUEST_FIELDS = ("prompt_sha256", "prompt_tokens", "generation_sha256")
STATUSES = ("graded", "solver_error", "judge_error", "solved-ungraded")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def obj(value: object, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value  # type: ignore[return-value]


def text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_scorer(checkout: Path) -> ModuleType:
    return load_verified_module(
        checkout,
        SOURCE_FILES,
        "scorer",
        ("inputs", "vllm_hust_benchmark.frontierscience_inputs"),
    )


def validate_inputs(
    envelope: object, inventory: object, arms: dict[str, object]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    record = obj(envelope, "envelope")
    require(record.get("schema_version") == SCHEMA, "Envelope schema mismatch")
    require(record.get("evidence_kind") in EVIDENCE_KINDS, "Offline evidence only")
    require(record.get("scope") == "supplied-manifest-only", "No official score scope")
    require(
        record.get("dataset") == "frontierscience"
        and record.get("dataset_revision") == DATASET_REVISION
        and record.get("benchmark_revision") == BENCHMARK_REVISION
        and record.get("source_sha256")
        == {name: value[1] for name, value in SOURCE_FILES.items()},
        "Dataset/scorer provenance mismatch",
    )
    generation = obj(record.get("generation"), "generation")
    require(
        set(generation) == {"max_tokens", "temperature", "seed", "trials", "retries"}
        and type(generation.get("max_tokens")) is int
        and generation["max_tokens"] > 0
        and type(generation.get("temperature")) in (int, float)
        and math.isfinite(generation["temperature"])
        and generation["temperature"] >= 0
        and type(generation.get("seed")) is int
        and type(generation.get("trials")) is int
        and generation["trials"] == 1
        and type(generation.get("retries")) is int
        and generation["retries"] == 0,
        "Need an explicit single-attempt offline generation budget",
    )
    require(isinstance(inventory, list) and bool(inventory), "Need task inventory")
    tasks = [obj(row, "task") for row in inventory]
    require(
        type(record.get("task_count")) is int and record["task_count"] == len(tasks),
        "Task count differs from supplied manifest",
    )
    identities = set()
    counts = dict.fromkeys(TRACK_COUNTS, 0)
    for task in tasks:
        track = task.get("track")
        require(track in tuple(TRACK_COUNTS), "Unknown track")
        index = task.get("source_row")
        require(
            type(index) is int and 0 <= index < TRACK_COUNTS[track],
            "Source row index outside the frozen track",
        )
        identity = task.get("task_id")
        require(
            identity == f"{track}/test/{index:03d}" and identity not in identities,
            "Task identity must be unique track/test/source-row, not group ID",
        )
        identities.add(identity)
        counts[track] += 1
        require(
            digest(task.get("source_row_sha256"))
            and text(task.get("problem"))
            and text(task.get("reference")),
            "Need source row digest and separate problem/reference text",
        )
        prompt = task.get("prompt")
        require(
            isinstance(prompt, list)
            and bool(prompt)
            and all(type(token) is int and token >= 0 for token in prompt),
            "Prompt must contain actual nonnegative token IDs",
        )
        require(
            type(task.get("prompt_tokens")) is int
            and task["prompt_tokens"] == len(prompt)
            and task.get("prompt_sha256") == sha(canonical(prompt))
            and task.get("generation_sha256") == sha(canonical(generation)),
            "Task prompt or generation digest mismatch",
        )
    require(
        all(counts.values()), "Both tracks must be present; never pool their scores"
    )
    require(set(arms) == {"B0", "B1"}, "Require exactly B0 and B1")
    lookup = {task["task_id"]: task for task in tasks}
    checked = {}
    for arm, value in arms.items():
        bundle = obj(value, arm)
        obj(bundle.get("collection"), f"{arm}: collection")
        outcomes = bundle.get("outcomes")
        require(isinstance(outcomes, list), f"{arm}: outcomes must be a list")
        rows = [obj(row, f"{arm}: outcome") for row in outcomes]
        require(len(rows) == len(tasks), f"{arm}: missing or duplicate outcomes")
        seen = set()
        for row in rows:
            identity = row.get("task_id")
            require(
                isinstance(identity, str)
                and identity in lookup
                and identity not in seen,
                f"{arm}: unknown or duplicate task ID",
            )
            seen.add(identity)
            task = lookup[identity]
            require(row.get("track") == task["track"], f"{arm}: track mismatch")
            for field in REQUEST_FIELDS:
                require(
                    type(row.get(field)) is type(task[field])
                    and row[field] == task[field],
                    f"{arm}: unmatched request field {field}",
                )
            require(row.get("status") in STATUSES, f"{arm}: invalid outcome status")
            if row["status"] == "solver_error":
                require(text(row.get("error")), f"{arm}: retain solver failure")
                require(row.get("judge") is None, f"{arm}: cannot grade solver failure")
            else:
                require(
                    isinstance(row.get("attempt"), str), f"{arm}: raw attempt needed"
                )
                require(
                    row.get("finish_reason") in ("stop", "length"),
                    f"{arm}: retain solver stop/length reason",
                )
                if row["status"] == "judge_error":
                    require(text(row.get("error")), f"{arm}: retain judge failure")
        checked[arm] = {"collection": bundle["collection"], "outcomes": rows}
    return record, tasks, checked


def judge_binding(
    record: dict[str, Any], spec: object, calibration: object, scorer: ModuleType
) -> str:
    if spec is None:
        require(calibration is None, "Judge and calibration must be supplied together")
        return "missing"
    judge = obj(spec, "judge contract")
    receipt = obj(calibration, "calibration")
    prompts = {
        key: sha(value.encode()) for key, value in scorer.JUDGE_INSTRUCTIONS.items()
    }
    require(
        judge.get("scope") == "independently-authored-candidate"
        and judge.get("scorer_source_sha256") == SOURCE_FILES["scorer"][1]
        and judge.get("instructions_sha256") == prompts
        and text(judge.get("model_revision"))
        and text(judge.get("runtime_identity")),
        "Judge model/runtime, candidate scope or prompt binding mismatch",
    )
    budget = obj(judge.get("generation"), "judge generation")
    require(
        set(budget) == {"max_tokens", "temperature"}
        and type(budget.get("max_tokens")) is int
        and budget["max_tokens"] > 0
        and type(budget.get("temperature")) in (int, float)
        and math.isfinite(budget["temperature"])
        and budget["temperature"] >= 0,
        "Judge generation budget mismatch",
    )
    require(
        judge.get("calibration_sha256") == record["files_sha256"]["calibration"]
        and receipt.get("judge_model_revision") == judge["model_revision"]
        and receipt.get("judge_runtime_identity") == judge["runtime_identity"]
        and receipt.get("instructions_sha256") == prompts
        and isinstance(receipt.get("tracks"), list)
        and all(isinstance(track, str) for track in receipt["tracks"])
        and sorted(receipt["tracks"]) == sorted(TRACK_COUNTS)
        and type(receipt.get("passed")) is bool,
        "Calibration identity, prompts or two-track coverage mismatch",
    )
    expected_kind = (
        "development-fixture"
        if record["evidence_kind"] == "development-fixture"
        else "unverified-external-receipt"
    )
    require(
        receipt.get("evidence_kind") == expected_kind, "Calibration evidence mismatch"
    )
    if not receipt["passed"]:
        return "not_passed"
    return (
        "fixture_only" if expected_kind == "development-fixture" else "bound_unverified"
    )


def normalize_outcomes(
    bundle: dict[str, Any],
    tasks: list[dict[str, Any]],
    record: dict[str, Any],
    spec: object,
    scorer: ModuleType,
) -> list[dict[str, Any]]:
    collection = bundle["collection"]
    files = record["files_sha256"]
    require(
        collection.get("judge_contract_sha256") == files["judge_contract"]
        and collection.get("calibration_sha256") == files["calibration"]
        and collection.get("frozen_before_solver_requests") is (spec is not None),
        "Collection differs from the judge/calibration declared before solver requests",
    )
    lookup = {task["task_id"]: task for task in tasks}
    normalized = []
    for row in bundle["outcomes"]:
        result = {
            "task_id": row["task_id"],
            "track": row["track"],
            "status": row["status"],
        }
        if row["status"] == "solved-ungraded":
            result["status"] = "judge_error"
        elif row["status"] == "graded":
            if spec is None:
                # No judge/calibration means no score, even if cached grades exist.
                result["status"] = "judge_error"
            else:
                task = lookup[row["task_id"]]
                judge = obj(row.get("judge"), "raw judge record")
                require(
                    judge.get("runtime_identity") == spec["runtime_identity"]
                    and judge.get("judge_contract_sha256") == files["judge_contract"]
                    and judge.get("calibration_sha256") == files["calibration"],
                    "Per-attempt judge identity mismatch",
                )
                require(
                    canonical(judge.get("request"))
                    == canonical(
                        {
                            "model": spec["model_revision"],
                            "messages": scorer.judge_messages(
                                row["track"],
                                task["problem"],
                                task["reference"],
                                row["attempt"],
                            ),
                            **spec["generation"],
                        }
                    ),
                    "Judge request differs from the matched problem/reference/attempt",
                )
                response = obj(judge.get("response"), "raw judge response")
                require(
                    response.get("model") == spec["model_revision"],
                    "Judge returned another model",
                )
                try:
                    choices = response["choices"]
                    require(
                        isinstance(choices, list) and len(choices) == 1,
                        "Need one judge choice",
                    )
                    choice = obj(choices[0], "judge choice")
                    require(
                        choice.get("finish_reason") == "stop", "Judge output truncated"
                    )
                    answer = obj(choice.get("message"), "judge message")["content"]
                    scorer.parse_verdict(row["track"], answer)
                    result["judge_response"] = answer
                except (ValueError, TypeError, KeyError) as error:
                    result["status"] = "judge_error"
                    result["judge_parse_error"] = str(error)
        normalized.append(result)
    return normalized


def validate_pair(
    envelope_path: Path,
    inventory_path: Path,
    b0_path: Path,
    b1_path: Path,
    benchmark_checkout: Path,
    judge_path: Path | None = None,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    envelope_raw = envelope_path.read_bytes()
    record = obj(parse_json(envelope_raw), "envelope")
    files = obj(record.get("files_sha256"), "files_sha256")
    paths = {
        "inventory": inventory_path,
        "B0": b0_path,
        "B1": b1_path,
        "judge_contract": judge_path,
        "calibration": calibration_path,
    }
    require(
        set(files) == set(paths), "Require exact input checksums, including judge slots"
    )
    require(
        (judge_path is None) == (calibration_path is None),
        "Supply judge and calibration together",
    )
    parsed = {}
    for name, path in paths.items():
        if path is None:
            require(
                files[name] is None, f"{name}: absent artifact must have null digest"
            )
            parsed[name] = None
        else:
            raw = path.read_bytes()
            require(
                digest(files[name]) and sha(raw) == files[name],
                f"{name}: artifact checksum mismatch",
            )
            parsed[name] = parse_json(raw)
    record, tasks, arms = validate_inputs(
        record, parsed["inventory"], {arm: parsed[arm] for arm in ("B0", "B1")}
    )
    scorer = load_scorer(benchmark_checkout)
    binding = judge_binding(
        record, parsed["judge_contract"], parsed["calibration"], scorer
    )
    normalized = {
        arm: normalize_outcomes(bundle, tasks, record, parsed["judge_contract"], scorer)
        for arm, bundle in arms.items()
    }
    counters = {}
    for arm, rows in normalized.items():
        counters[arm] = {}
        for track in TRACK_COUNTS:
            subset = [row for row in rows if row["track"] == track]
            counters[arm][track] = {
                "tasks": len(subset),
                "solver_errors": sum(row["status"] == "solver_error" for row in subset),
                "ungraded": sum(row["status"] == "judge_error" for row in subset),
                "length_limited": sum(
                    row.get("finish_reason") == "length" and row["track"] == track
                    for row in arms[arm]["outcomes"]
                ),
            }
    return {
        "status": "offline_validated_runtime_unverified",
        "evidence_kind": record["evidence_kind"],
        "benchmark_revision": BENCHMARK_REVISION,
        "envelope_sha256": sha(envelope_raw),
        "artifact_sha256": dict(files),
        "scope": "supplied-manifest-only; not official or AQKV task scores",
        "source_membership_verified": False,
        "judge_binding": binding,
        "semantic_calibration_verified": False,
        "pre_generation_binding_verified": False,
        "track_counts": counters,
        "normalized_outcomes": normalized,
        "fixture_scores": {
            arm: scorer.aggregate(tasks, rows) for arm, rows in normalized.items()
        }
        if binding == "fixture_only"
        else None,
        "task_quality": None,
        "aqkv_path_observations": None,
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("envelope", "inventory", "b0", "b1", "benchmark-checkout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("judge-contract", "calibration"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        report = validate_pair(
            args.envelope,
            args.inventory,
            args.b0,
            args.b1,
            args.benchmark_checkout,
            args.judge_contract,
            args.calibration,
        )
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
