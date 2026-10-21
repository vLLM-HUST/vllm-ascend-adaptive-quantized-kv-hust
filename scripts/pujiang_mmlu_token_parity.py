"""Compare stored /tokenize bytes offline; never verify a live server's origin."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.pujiang_mmlu_transport import FAILURE_KINDS, read_artifact
    from scripts.validate_pujiang_mmlu_pair import _digest, _object, _require, _sha
    from scripts.validate_pujiang_readiness import parse_json
except ModuleNotFoundError:
    from pujiang_mmlu_transport import FAILURE_KINDS, read_artifact
    from validate_pujiang_mmlu_pair import _digest, _object, _require, _sha
    from validate_pujiang_readiness import parse_json

SCHEMA = "aqkv-mmlu-stored-token-parity/v1"
IDENTITY_SCHEMA = "aqkv-mmlu-tokenizer-declaration/v1"
TOKENIZE_REFERENCE = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
TOKENIZE_SOURCE_FILES = {
    "protocol.py": "0bb4ddf2c0b1412e88c9d853e2ef7163f62665df0b8bbef86df5bdbdd9a76e44",
    "serving.py": "fb9fd00cc71df5c052f420a959e39434ce5fd3c84556275eb9dc25d43ff59750",
    "api_router.py": "dac2a4d0cd6076d50b568f3bae65dea0c6a65547fda073217a7dab9a1e2d8ccc",
}
TOKENIZER_FIELDS = (
    "repository",
    "revision",
    "files_sha256",
    "class",
    "library_versions",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def tokenize_request(case: dict) -> dict:
    """Project the pinned text-only chat contract without generating tokens."""
    request = _object(case.get("candidate_request"), "candidate request")
    _require(
        _sha(_canonical(request)) == case.get("request_sha256")
        and request.get("chat_template_kwargs") == {"enable_thinking": False}
        and request.get("stream") is False
        and isinstance(request.get("model"), str)
        and bool(request["model"].strip())
        and isinstance(request.get("messages"), list)
        and len(request["messages"]) == 1,
        "Tokenize projection requires the unchanged non-thinking candidate request",
    )
    message = _object(request["messages"][0], "candidate message")
    _require(
        set(message) == {"role", "content"}
        and message["role"] == "user"
        and isinstance(message["content"], str),
        "Only the source formatter's text-only user message is supported",
    )
    _require(
        set(request)
        <= {
            "model",
            "messages",
            "temperature",
            "seed",
            "max_tokens",
            "ignore_eos",
            "n",
            "stop",
            "chat_template_kwargs",
            "stream",
        },
        "Unsupported token-affecting candidate request field",
    )
    return {
        "model": request["model"],
        "messages": copy.deepcopy(request["messages"]),
        "add_generation_prompt": True,
        "continue_final_message": False,
        "add_special_tokens": False,
        "chat_template": None,
        "chat_template_kwargs": copy.deepcopy(request["chat_template_kwargs"]),
        "return_token_strs": False,
        "tools": None,
        "media_io_kwargs": None,
        "mm_processor_kwargs": None,
    }


def _identity(
    value: object, tokenizer: dict, model_alias: str, context_tokens: int
) -> None:
    declaration = _object(value, "tokenizer declaration")
    _require(
        set(declaration)
        == {
            "schema_version",
            "tokenizer",
            "served_model_name",
            "context_tokens",
            "server_configuration_sha256",
            "release_manifest_sha256",
        }
        and declaration["schema_version"] == IDENTITY_SCHEMA
        and declaration["served_model_name"] == model_alias
        and type(declaration["context_tokens"]) is int
        and declaration["context_tokens"] == context_tokens
        and _digest(declaration["server_configuration_sha256"])
        and _digest(declaration["release_manifest_sha256"]),
        "Stored tokenizer declaration identity/context mismatch",
    )
    expected = {field: tokenizer[field] for field in TOKENIZER_FIELDS}
    _require(
        _canonical(declaration["tokenizer"]) == _canonical(expected),
        "Declared tokenizer files/template/class/libraries differ from local source",
    )


def _compare_response(case: dict, raw: bytes, status: int, context_tokens: int) -> dict:
    def failure(kind: str, **details) -> dict:
        return {"match": False, "error_kind": kind, **details}

    if status != 200:
        return failure("http-error")
    try:
        response = _object(parse_json(raw), "tokenize response")
    except (ValueError, UnicodeError):
        return failure("invalid-json-response")
    if set(response) != {"count", "max_model_len", "tokens", "token_strs"}:
        return failure("invalid-tokenize-response")
    ids = response["tokens"]
    if not (
        isinstance(ids, list)
        and all(type(token) is int and 0 <= token < 2**63 for token in ids)
        and type(response["count"]) is int
        and response["count"] == len(ids)
        and response["token_strs"] is None
    ):
        return failure("invalid-token-ids-or-count")
    if (
        type(response["max_model_len"]) is not int
        or response["max_model_len"] != context_tokens
    ):
        return failure("context-mismatch")
    expected = case["prompt"]
    identity = {
        "stored_token_count": len(ids),
        "stored_token_ids_sha256": _sha(_canonical(ids)),
    }
    if ids != expected:
        index = next(
            (
                i
                for i, (left, right) in enumerate(zip(ids, expected, strict=False))
                if left != right
            ),
            min(len(ids), len(expected)),
        )
        return failure("token-ids-mismatch", first_mismatch_index=index, **identity)
    return {"match": True, "error_kind": None, **identity}


def audit_stored_token_parity(
    receipt: object,
    cases: list[dict],
    root: Path,
    arm: str,
    bundle_sha256: str,
    evidence_kind: str,
    tokenizer: dict,
    context_tokens: int,
) -> dict:
    """Call with rebuilt source cases; matching files do not establish origin."""
    record = _object(receipt, "token parity receipt")
    _require(
        set(record)
        == {
            "schema_version",
            "arm",
            "evidence_kind",
            "scorer_bundle_sha256",
            "tokenize_reference_revision",
            "request_method",
            "request_path",
            "tokenizer_declaration",
            "requests",
        }
        and record["schema_version"] == SCHEMA
        and arm in ("B0", "B1")
        and record["arm"] == arm
        and record["evidence_kind"] == evidence_kind
        and evidence_kind in ("development-fixture", "unverified-stored-outputs")
        and _digest(bundle_sha256)
        and record["scorer_bundle_sha256"] == bundle_sha256
        and record["tokenize_reference_revision"] == TOKENIZE_REFERENCE
        and record["request_method"] == "POST"
        and record["request_path"] == "/tokenize",
        "Stored token parity provenance/method/arm mismatch",
    )
    _require(bool(cases), "Need source-bound token cases")
    _require(
        type(context_tokens) is int and context_tokens > 0, "Invalid context budget"
    )
    expected = {case["case_id"]: case for case in cases}
    _require(len(expected) == len(cases), "Duplicate source token cases")
    for case in cases:
        ids = case.get("prompt")
        _require(
            isinstance(ids, list)
            and bool(ids)
            and all(type(token) is int and 0 <= token < 2**63 for token in ids)
            and type(case.get("prompt_tokens")) is int
            and case["prompt_tokens"] == len(ids)
            and case.get("prompt_sha256") == _sha(_canonical(ids)),
            "Invalid source-bound token IDs/count/hash",
        )
        _require(
            case["candidate_request"]["model"]
            == cases[0]["candidate_request"]["model"],
            "Token cases must use one declared model alias",
        )
    identity_raw = read_artifact(root, record["tokenizer_declaration"])
    _identity(
        parse_json(identity_raw),
        tokenizer,
        cases[0]["candidate_request"]["model"],
        context_tokens,
    )
    rows = record["requests"]
    _require(
        isinstance(rows, list) and len(rows) == len(cases), "Missing token parity tasks"
    )
    outcomes = {}
    for value in rows:
        row = _object(value, "token parity request")
        _require(
            set(row)
            == {"case_id", "completion_request_sha256", "request", "attempts", "result"}
            and isinstance(row["case_id"], str)
            and row["case_id"] in expected
            and row["case_id"] not in outcomes,
            "Unknown or duplicate token parity task",
        )
        case = expected[row["case_id"]]
        request_raw = read_artifact(root, row["request"])
        _require(
            row["completion_request_sha256"] == case["request_sha256"]
            and _canonical(parse_json(request_raw))
            == _canonical(tokenize_request(case)),
            "Stored tokenize request differs from the full source-bound projection",
        )
        response = _object(row["result"], "token parity result")
        kind = response.get("kind")
        _require(
            isinstance(kind, str)
            and kind in ("http", *FAILURE_KINDS)
            and type(row["attempts"]) is int
            and row["attempts"] == (0 if kind == "not-attempted" else 1),
            "Stored token parity attempts/kind mismatch",
        )
        details = {"request_record_sha256": _sha(request_raw)}
        if kind == "http":
            _require(
                set(response) == {"kind", "status", "body"}
                and type(response["status"]) is int
                and 200 <= response["status"] <= 599,
                "Invalid stored tokenize HTTP result",
            )
            body = read_artifact(root, response["body"])
            outcome = _compare_response(case, body, response["status"], context_tokens)
            details.update(response_sha256=_sha(body), http_status=response["status"])
        else:
            _require(set(response) == {"kind"}, "Invalid token parity failure result")
            outcome = {"match": False, "error_kind": kind}
        outcomes[case["case_id"]] = {"case_id": case["case_id"], **outcome, **details}
    ordered = [outcomes[case["case_id"]] for case in cases]
    matched = all(row["match"] for row in ordered)
    return {
        "status": "stored_token_ids_match_origin_unverified"
        if matched
        else "stored_token_parity_mismatch_scoring_blocked",
        "scope": "stored-tokenize-content-only-not-authenticated-serving",
        "case_count": len(cases),
        "all_cases_retained": True,
        "all_token_ids_match": matched,
        "tokenizer_declaration_sha256": _sha(identity_raw),
        "declared_tokenizer_matches_local": True,
        "tokenizer_declaration_origin_verified": False,
        "release_signature_verified": False,
        "failure_counts": dict(
            Counter(row["error_kind"] for row in ordered if not row["match"])
        ),
        "outcomes": ordered,
        "serving_tokenizer_verified": False,
        "transport_verified": False,
        "execution_authorized": False,
    }
