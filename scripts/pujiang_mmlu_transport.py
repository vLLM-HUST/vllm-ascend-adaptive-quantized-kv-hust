"""Translate stored non-streaming receipts; this module never sends requests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.validate_pujiang_mmlu_pair import (
        SOURCE_REQUEST_FIELDS,
        _digest,
        _object,
        _require,
        _sha,
    )
    from scripts.validate_pujiang_readiness import parse_json
except ModuleNotFoundError:
    from validate_pujiang_mmlu_pair import (
        SOURCE_REQUEST_FIELDS,
        _digest,
        _object,
        _require,
        _sha,
    )
    from validate_pujiang_readiness import parse_json

SCHEMA = "aqkv-mmlu-stored-transport/v1"
MAX_BODY_BYTES = 1_048_576
FAILURE_KINDS = {"timeout", "connection-error", "not-attempted", "response-too-large"}


def read_artifact(root: Path, descriptor: object) -> bytes:
    record = _object(descriptor, "stored body")
    _require(set(record) == {"path", "sha256", "bytes"}, "Invalid body descriptor")
    _require(
        isinstance(record["path"], str)
        and bool(record["path"])
        and _digest(record["sha256"])
        and type(record["bytes"]) is int
        and 0 <= record["bytes"] <= MAX_BODY_BYTES,
        "Stored body identity/size is invalid",
    )
    relative = Path(record["path"])
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts != (),
        "Stored body path must stay inside its arm directory",
    )
    directory = root.resolve(strict=True)
    _require(directory.is_dir() and not root.is_symlink(), "Invalid arm directory")
    candidate = directory
    for part in relative.parts:
        candidate /= part
        _require(not candidate.is_symlink(), "Stored body symlinks are not allowed")
    destination = candidate.resolve(strict=True)
    _require(
        directory in destination.parents and destination.is_file(),
        "Stored body path escapes its arm directory",
    )
    with destination.open("rb") as handle:
        raw = handle.read(MAX_BODY_BYTES + 1)
    _require(
        len(raw) == record["bytes"] and _sha(raw) == record["sha256"],
        "Stored body size/checksum mismatch",
    )
    return raw


def response_outcome(case: dict, raw: bytes, status: int) -> dict:
    """Malformed/failed responses remain failed tasks, not discarded records."""
    result = {"success": False, "output": "", "finish_reason": None}

    def failure(kind: str) -> dict:
        return {**result, "error_kind": kind}

    if status != 200:
        return failure("http-error")
    try:
        response = parse_json(raw)
    except (ValueError, UnicodeError):
        return failure("invalid-json-response")
    if not isinstance(response, dict):
        return failure("invalid-response")
    if response.get("error") is not None:
        return failure("api-error")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return failure("invalid-choices")
    choice = choices[0]
    if (
        not isinstance(choice, dict)
        or type(choice.get("index")) is not int
        or choice["index"] != 0
        or choice.get("finish_reason") not in ("stop", "length")
    ):
        return failure("invalid-choice")
    result["finish_reason"] = choice["finish_reason"]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return failure("invalid-message")
    if message.get("refusal"):
        return failure("refusal")
    if message.get("reasoning_content") not in (None, ""):
        return failure("unexpected-thinking-output")
    text = message.get("content")
    if not isinstance(text, str):
        return failure("invalid-content")
    usage = response.get("usage")
    if not isinstance(usage, dict) or not all(
        type(usage.get(name)) is int and usage[name] >= 0
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return failure("invalid-usage")
    result["recorded_usage"] = {
        name: usage[name]
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    result["recorded_prompt_count_matches_local"] = (
        usage["prompt_tokens"] == case["prompt_tokens"]
    )
    if not result["recorded_prompt_count_matches_local"]:
        return failure("prompt-count-mismatch")
    if (
        usage["completion_tokens"] > case["max_tokens"]
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
        or (bool(text) and usage["completion_tokens"] == 0)
    ):
        return failure("generation-usage-mismatch")
    return {**result, "success": True, "output": text, "error_kind": None}


def normalize_stored_arm(
    receipt: object,
    cases: list[dict[str, Any]],
    root: Path,
    arm: str,
    bundle_sha256: str,
    evidence_kind: str,
) -> tuple[list[dict], dict]:
    record = _object(receipt, "transport receipt")
    _require(
        set(record)
        == {
            "schema_version",
            "arm",
            "evidence_kind",
            "scorer_bundle_sha256",
            "request_method",
            "request_path",
            "requests",
        }
        and record["schema_version"] == SCHEMA
        and record["arm"] == arm
        and record["evidence_kind"] == evidence_kind
        and evidence_kind in ("development-fixture", "unverified-stored-outputs")
        and record["scorer_bundle_sha256"] == bundle_sha256
        and record["request_method"] == "POST"
        and record["request_path"] == "/v1/chat/completions",
        "Stored transport provenance/method/arm mismatch",
    )
    rows = record["requests"]
    _require(isinstance(rows, list) and len(rows) == len(cases), "Missing stored tasks")
    expected = {case["case_id"]: case for case in cases}
    _require(len(expected) == len(cases), "Duplicate source case IDs")
    outcomes = {}
    identities = []
    for value in rows:
        row = _object(value, "stored request")
        _require(
            set(row) == {"case_id", "request", "attempts", "result"}
            and isinstance(row["case_id"], str)
            and row["case_id"] in expected
            and row["case_id"] not in outcomes,
            "Unknown or duplicate stored task",
        )
        case = expected[row["case_id"]]
        request_raw = read_artifact(root, row["request"])
        request = parse_json(request_raw)
        _require(
            json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
            == json.dumps(
                case["candidate_request"],
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "Stored request body differs from the source-bound candidate request",
        )
        response = _object(row["result"], "stored result")
        kind = response.get("kind")
        _require(
            isinstance(kind, str)
            and kind in ("http", *FAILURE_KINDS)
            and type(row["attempts"]) is int
            and row["attempts"] == (0 if kind == "not-attempted" else 1),
            "Stored attempts/kind mismatch; retries are unsupported",
        )
        identity = {
            "case_id": case["case_id"],
            "request_record_sha256": _sha(request_raw),
            "attempts": row["attempts"],
            "kind": kind,
        }
        if kind == "http":
            _require(
                set(response) == {"kind", "status", "body"}
                and type(response["status"]) is int
                and 200 <= response["status"] <= 599,
                "Invalid stored HTTP result",
            )
            body = read_artifact(root, response["body"])
            outcome = response_outcome(case, body, response["status"])
            identity.update(http_status=response["status"], response_sha256=_sha(body))
        else:
            _require(set(response) == {"kind"}, "Invalid stored failure result")
            outcome = {
                "success": False,
                "output": "",
                "finish_reason": None,
                "error_kind": kind,
            }
        identities.append(identity)
        outcomes[case["case_id"]] = {
            "case_id": case["case_id"],
            **{field: case[field] for field in SOURCE_REQUEST_FIELDS},
            **outcome,
        }
    ordered = [outcomes[case["case_id"]] for case in cases]
    return ordered, {
        "scope": "stored-records-only-not-network-or-serving-verification",
        "case_count": len(cases),
        "failure_counts": dict(
            Counter(row["error_kind"] for row in ordered if not row["success"])
        ),
        "artifacts": identities,
        "serving_tokenizer_verified": False,
        "transport_verified": False,
        "execution_authorized": False,
    }
