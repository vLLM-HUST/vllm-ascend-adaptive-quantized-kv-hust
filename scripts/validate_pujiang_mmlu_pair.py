#!/usr/bin/env python3
"""Rescore stored MMLU-Pro pairs offline; never authorize AQKV or task claims."""

from __future__ import annotations

import argparse
import hashlib
import json
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

SCORER_PATH = "src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py"
SCORER_SHA256 = "c2237c69f848e2358abe71c68f54143b5c090a81bd4f7915088b19b8ae157801"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SCHEMA = "aqkv-mmlu-offline-envelope/v1"
SOURCE_SCHEMA = "aqkv-mmlu-source-bound-envelope/v2"
SCORING_BUNDLE_SCHEMA = "aqkv-mmlu-scoring-bundle/v1"
EVIDENCE_KINDS = {"development-fixture", "unverified-stored-outputs"}
REQUEST_FIELDS = ("prompt_sha256", "prompt_tokens", "max_tokens", "ignore_eos")
SOURCE_REQUEST_FIELDS = (*REQUEST_FIELDS, "request_sha256", "rendered_prompt_sha256")
SOURCE_INPUTS = ("assets", "reference", "selection", "tokenizer-directory")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _object(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value  # type: ignore[return-value]


def load_scorer(checkout: Path) -> ModuleType:
    return load_verified_module(
        checkout, {"scorer": (SCORER_PATH, SCORER_SHA256)}, "scorer"
    )


def validate_inputs(
    envelope: object, cases: object, arms: dict[str, object]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    record = _object(envelope, "envelope")
    schema = record.get("schema_version")
    _require(schema in (SCHEMA, SOURCE_SCHEMA), "Envelope schema mismatch")
    _require(
        isinstance(record.get("evidence_kind"), str)
        and record["evidence_kind"] in EVIDENCE_KINDS,
        "Offline envelopes cannot declare measured or accepted evidence",
    )
    _require(
        record.get("scope")
        == (
            "verified-local-source-selection"
            if schema == SOURCE_SCHEMA
            else "supplied-manifest-only"
        ),
        "Scope does not match the envelope's source verification mode",
    )
    _require(
        record.get("dataset") == "mmlu-pro"
        and record.get("dataset_revision") == DATASET_REVISION
        and record.get("benchmark_revision") == BENCHMARK_REVISION
        and record.get("scorer_source_sha256") == SCORER_SHA256,
        "Dataset/scorer provenance mismatch",
    )
    generation = _object(record.get("generation"), "generation")
    _require(
        type(generation.get("max_tokens")) is int
        and generation["max_tokens"] > 0
        and type(generation.get("ignore_eos")) is bool,
        "Generation needs a positive token budget and boolean ignore_eos",
    )
    _require(isinstance(cases, list) and bool(cases), "Need a nonempty task manifest")
    _require(
        type(record.get("task_count")) is int and record["task_count"] == len(cases),
        "Task count differs from supplied manifest",
    )
    manifest = [_object(row, "case") for row in cases]
    identities = set()
    for row in manifest:
        identity = row.get("case_id")
        _require(
            isinstance(identity, str) and bool(identity) and identity not in identities,
            "Task IDs must be nonempty and unique",
        )
        identities.add(identity)
        prompt = row.get("prompt")
        _require(
            isinstance(prompt, list)
            and bool(prompt)
            and all(type(token) is int and token >= 0 for token in prompt),
            "Prompt must contain actual nonnegative token IDs",
        )
        _require(
            type(row.get("prompt_tokens")) is int
            and row["prompt_tokens"] == len(prompt)
            and row.get("prompt_sha256")
            == _sha(json.dumps(prompt, separators=(",", ":")).encode()),
            "Prompt count/hash differs from actual token IDs",
        )
        _require(
            type(row.get("option_count")) is int
            and 2 <= row["option_count"] <= 10
            and row.get("answer") in tuple("ABCDEFGHIJ"[: row["option_count"]])
            and isinstance(row.get("category"), str)
            and bool(row["category"]),
            "Invalid option count, answer, or category",
        )
        _require(
            type(row.get("max_tokens")) is int
            and row["max_tokens"] == generation["max_tokens"]
            and type(row.get("ignore_eos")) is bool
            and row["ignore_eos"] is generation["ignore_eos"],
            "Case generation differs from envelope",
        )
        if schema == SOURCE_SCHEMA:
            _require(
                all(_digest(row.get(field)) for field in SOURCE_REQUEST_FIELDS[4:]),
                "Source-bound cases need full request and rendered prompt hashes",
            )
    _require(set(arms) == {"B0", "B1"}, "Require exactly B0 and B1")
    lookup = {row["case_id"]: row for row in manifest}
    validated = {}
    for arm, value in arms.items():
        _require(isinstance(value, list), f"{arm}: outcomes must be a list")
        rows = [_object(row, f"{arm}: outcome") for row in value]
        _require(len(rows) == len(manifest), f"{arm}: missing or duplicate outcomes")
        seen = set()
        for row in rows:
            identity = row.get("case_id")
            _require(
                isinstance(identity, str)
                and identity in lookup
                and identity not in seen,
                f"{arm}: unknown or duplicate task ID",
            )
            seen.add(identity)
            case = lookup[identity]
            fields = (
                SOURCE_REQUEST_FIELDS if schema == SOURCE_SCHEMA else REQUEST_FIELDS
            )
            for field in fields:
                _require(
                    type(row.get(field)) is type(case[field])
                    and row[field] == case[field],
                    f"{arm}: unmatched request field {field}",
                )
            _require(
                type(row.get("success")) is bool, f"{arm}: success must be boolean"
            )
            _require(
                not row["success"] or isinstance(row.get("output"), str),
                f"{arm}: successful request needs raw output text",
            )
            _require(
                row.get("finish_reason") is None
                or isinstance(row["finish_reason"], str),
                f"{arm}: finish_reason must be text or null",
            )
        validated[arm] = rows
    return record, manifest, validated


def verify_source_bundle(
    record: dict[str, Any],
    bundle: object,
    benchmark_checkout: Path,
    source_inputs: dict[str, Path] | None,
) -> tuple[list[dict], dict]:
    _require(
        source_inputs is not None and set(source_inputs) == set(SOURCE_INPUTS),
        "Source-bound scoring requires all official assets, "
        "selection and tokenizer inputs",
    )
    # Rebuild from pinned bytes; stored flags and self-consistent hashes are not proof.
    try:
        from scripts.tokenize_pujiang_mmlu_source import prepare_tokens, scoring_bundle
    except ModuleNotFoundError:
        from tokenize_pujiang_mmlu_source import prepare_tokens, scoring_bundle
    plan = prepare_tokens(
        source_inputs["assets"],
        source_inputs["reference"],
        source_inputs["selection"],
        benchmark_checkout,
        source_inputs["tokenizer-directory"],
    )
    expected = scoring_bundle(plan)
    _require(
        json.dumps(bundle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        == json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False),
        "Stored scoring bundle differs from rebuilt official source/token plan",
    )
    contract = expected["envelope_contract"]
    for field, value in contract.items():
        _require(
            json.dumps(record.get(field), sort_keys=True, allow_nan=False)
            == json.dumps(value, sort_keys=True, allow_nan=False),
            f"Source-bound envelope mismatch: {field}",
        )
    return expected["cases"], contract["source_binding"]


def score_pair(
    envelope_path: Path,
    cases_path: Path,
    b0_path: Path,
    b1_path: Path,
    benchmark_checkout: Path,
    *,
    source_inputs: dict[str, Path] | None = None,
    stored_transport_root: Path | None = None,
    stored_token_parity_root: Path | None = None,
) -> dict[str, Any]:
    envelope_raw = envelope_path.read_bytes()
    record = _object(parse_json(envelope_raw), "envelope")
    files = _object(record.get("files_sha256"), "files_sha256")
    _require(set(files) == {"cases", "B0", "B1"}, "Require cases/B0/B1 checksums")
    paths = {"cases": cases_path, "B0": b0_path, "B1": b1_path}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    for name in paths:
        _require(
            _digest(files[name]) and _sha(raw[name]) == files[name],
            f"{name}: stored artifact checksum mismatch",
        )
    manifest = parse_json(raw["cases"])
    source_binding = None
    if record.get("schema_version") == SOURCE_SCHEMA:
        manifest, source_binding = verify_source_bundle(
            record, manifest, benchmark_checkout, source_inputs
        )
    else:
        _require(
            source_inputs is None, "Legacy envelopes cannot claim source verification"
        )
    arm_values = {arm: parse_json(raw[arm]) for arm in ("B0", "B1")}
    stored_transport = {}
    if stored_transport_root is not None:
        _require(
            source_binding is not None,
            "Stored transport requires verified source-bound scoring",
        )
        try:
            from scripts.pujiang_mmlu_transport import normalize_stored_arm
        except ModuleNotFoundError:
            from pujiang_mmlu_transport import normalize_stored_arm
        for arm in ("B0", "B1"):
            arm_values[arm], stored_transport[arm] = normalize_stored_arm(
                arm_values[arm],
                manifest,
                stored_transport_root / arm,
                arm,
                _sha(raw["cases"]),
                record["evidence_kind"],
            )
    record, cases, arms = validate_inputs(
        record,
        manifest,
        arm_values,
    )
    stored_token_parity = {}
    _require(
        stored_token_parity_root is not None
        or "stored_token_parity_sha256" not in record,
        "Envelope binds token parity; provide --stored-token-parity-root",
    )
    if stored_token_parity_root is not None:
        _require(
            source_binding is not None, "Token parity requires source-bound scoring"
        )
        parity_files = _object(
            record.get("stored_token_parity_sha256"), "token parity hashes"
        )
        _require(
            set(parity_files) == {"B0", "B1"}, "Require both token parity checksums"
        )
        try:
            from scripts.pujiang_mmlu_token_parity import audit_stored_token_parity
            from scripts.pujiang_mmlu_transport import read_artifact
        except ModuleNotFoundError:
            from pujiang_mmlu_token_parity import audit_stored_token_parity
            from pujiang_mmlu_transport import read_artifact
        for arm in ("B0", "B1"):
            root = stored_token_parity_root / arm
            path = root / "receipt.json"
            parity_raw = read_artifact(
                root,
                {
                    "path": "receipt.json",
                    "bytes": path.stat().st_size,
                    "sha256": parity_files[arm],
                },
            )
            stored_token_parity[arm] = audit_stored_token_parity(
                parse_json(parity_raw),
                cases,
                root,
                arm,
                _sha(raw["cases"]),
                record["evidence_kind"],
                source_binding["tokenizer"],
                source_binding["context_audit"]["context_tokens"],
            )
    parity_blocked = any(
        not audit["all_token_ids_match"] for audit in stored_token_parity.values()
    )
    scores = None
    if not parity_blocked:
        scorer = load_scorer(benchmark_checkout)
        scores = {}
        for arm, rows in arms.items():
            score = scorer.score_results(rows, cases)
            # Upstream eligibility is PyramidKV's 4096-token rule, not AQKV's.
            score.pop("eligible")
            score.pop("ineligible")
            for outcome in score["outcomes"]:
                outcome.pop("eligible")
            scores[arm] = score
    if parity_blocked:
        status = "stored_token_parity_mismatch_scoring_blocked"
    elif source_binding is not None:
        status = "offline_scored_source_bound_runtime_unverified"
    else:
        status = "offline_scored_runtime_unverified"
    report = {
        "status": status,
        "evidence_kind": record["evidence_kind"],
        "scope": (
            "verified-local-source-selection; stored outputs, "
            "not official/runtime scores"
        )
        if source_binding is not None
        else "supplied-manifest-only; not official scores or runtime verification",
        "source_binding_verified": source_binding is not None,
        "source_binding": source_binding,
        "serving_tokenizer_verified": False,
        "transport_verified": False,
        "benchmark_revision": BENCHMARK_REVISION,
        "scorer_source_sha256": SCORER_SHA256,
        "envelope_sha256": _sha(envelope_raw),
        "artifact_sha256": dict(files),
        "offline_quality": scores,
        "aqkv_path_observations": None,
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }
    if stored_transport:
        report.update(
            stored_transport=stored_transport,
            normalized_stored_outcomes=arms,
        )
    if stored_token_parity:
        report["stored_token_parity"] = stored_token_parity
        report["stored_token_parity_sha256"] = dict(parity_files)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("envelope", "cases", "b0", "b1", "benchmark-checkout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in SOURCE_INPUTS:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--stored-transport-root", type=Path)
    parser.add_argument("--stored-token-parity-root", type=Path)
    args = parser.parse_args()
    try:
        supplied = {
            name: getattr(args, name.replace("-", "_"))
            for name in SOURCE_INPUTS
            if getattr(args, name.replace("-", "_")) is not None
        }
        _require(
            not supplied or set(supplied) == set(SOURCE_INPUTS),
            "Provide all four source verification inputs together",
        )
        report = score_pair(
            args.envelope,
            args.cases,
            args.b0,
            args.b1,
            args.benchmark_checkout,
            source_inputs=supplied or None,
            stored_transport_root=args.stored_transport_root,
            stored_token_parity_root=args.stored_token_parity_root,
        )
        with args.output.open("x", encoding="utf-8") as output:
            args.output.chmod(0o600)
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return (
        1 if report["status"] == "stored_token_parity_mismatch_scoring_blocked" else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
