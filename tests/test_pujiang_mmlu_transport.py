import copy

import pytest

from scripts import pujiang_mmlu_transport as m
from scripts import validate_pujiang_mmlu_pair as pair
from scripts.prepare_pujiang_mmlu_source import GENERATION, _canonical


@pytest.fixture
def cases():
    result = []
    for index in range(2):
        request = {
            "model": "candidate-alias",
            "messages": [{"role": "user", "content": f"synthetic-{index}"}],
            **copy.deepcopy(GENERATION),
            "stream": False,
        }
        ids = list(range(16 + index))
        result.append(
            {
                "case_id": f"synthetic-{index}",
                "candidate_request": request,
                "prompt_sha256": m._sha(_canonical(ids)),
                "prompt_tokens": len(ids),
                "max_tokens": request["max_tokens"],
                "ignore_eos": False,
                "request_sha256": m._sha(_canonical(request)),
                "rendered_prompt_sha256": "1" * 64,
            }
        )
    return result


def body(case):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer is B"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": case["prompt_tokens"],
            "completion_tokens": 2,
            "total_tokens": case["prompt_tokens"] + 2,
        },
    }


def store(root, name, raw):
    destination = root / name
    destination.write_bytes(raw)
    return {"path": name, "sha256": m._sha(raw), "bytes": len(raw)}


def receipt(root, cases, arm="B0"):
    requests = []
    for index, case in enumerate(cases):
        requests.append(
            {
                "case_id": case["case_id"],
                "request": store(
                    root, f"request-{index}.json", _canonical(case["candidate_request"])
                ),
                "attempts": 1,
                "result": {
                    "kind": "http",
                    "status": 200,
                    "body": store(
                        root, f"response-{index}.json", _canonical(body(case))
                    ),
                },
            }
        )
    return {
        "schema_version": m.SCHEMA,
        "arm": arm,
        "evidence_kind": "development-fixture",
        "scorer_bundle_sha256": "2" * 64,
        "request_method": "POST",
        "request_path": "/v1/chat/completions",
        "requests": requests,
    }


def normalize(value, cases, root):
    return m.normalize_stored_arm(
        value, cases, root, "B0", "2" * 64, "development-fixture"
    )


def test_success_keeps_exact_content_usage_and_no_parity_claim(cases):
    original = body(cases[0])
    original["choices"][0]["message"]["content"] = "  answer is B\n"
    result = m.response_outcome(cases[0], _canonical(original), 200)
    assert result["success"] is True
    assert result["output"] == "  answer is B\n"
    assert result["recorded_usage"] == original["usage"]
    assert result["recorded_prompt_count_matches_local"] is True
    assert "serving_tokenizer_verified" not in result


@pytest.mark.parametrize("raw", [b"not json", b"\xff", b'{"choices":[],"choices":[]}'])
def test_malformed_raw_response_is_a_retained_failure(cases, raw):
    result = m.response_outcome(cases[0], raw, 200)
    assert result["success"] is False
    assert result["error_kind"] == "invalid-json-response"
    assert result["output"] == ""


@pytest.mark.parametrize(
    "mutation",
    [
        "array",
        "api-error",
        "no-choices",
        "multiple-choices",
        "index-bool",
        "tool-call",
        "wrong-role",
        "missing-content",
        "refusal",
        "reasoning",
        "no-usage",
        "usage-bool",
        "negative-usage",
        "prompt-count",
        "too-long",
        "total-mismatch",
        "zero-output-count",
    ],
)
def test_invalid_response_contract_does_not_become_a_scored_success(cases, mutation):
    raw = body(cases[0])
    choice = raw["choices"][0]
    if mutation == "array":
        raw = []
    elif mutation == "api-error":
        raw["error"] = {"message": "synthetic server error"}
    elif mutation == "no-choices":
        raw.pop("choices")
    elif mutation == "multiple-choices":
        raw["choices"].append(copy.deepcopy(choice))
    elif mutation == "index-bool":
        choice["index"] = False
    elif mutation == "tool-call":
        choice["finish_reason"] = "tool_calls"
    elif mutation == "wrong-role":
        choice["message"]["role"] = "user"
    elif mutation == "missing-content":
        choice["message"].pop("content")
    elif mutation == "refusal":
        choice["message"]["refusal"] = "synthetic refusal"
    elif mutation == "reasoning":
        choice["message"]["reasoning_content"] = "unexpected hidden answer"
    elif mutation == "no-usage":
        raw.pop("usage")
    elif mutation == "usage-bool":
        raw["usage"]["completion_tokens"] = True
    elif mutation == "negative-usage":
        raw["usage"]["completion_tokens"] = -1
    elif mutation == "prompt-count":
        raw["usage"]["prompt_tokens"] += 1
    elif mutation == "too-long":
        raw["usage"]["completion_tokens"] = cases[0]["max_tokens"] + 1
        raw["usage"]["total_tokens"] = (
            raw["usage"]["completion_tokens"] + cases[0]["prompt_tokens"]
        )
    elif mutation == "total-mismatch":
        raw["usage"]["total_tokens"] += 1
    else:
        raw["usage"]["completion_tokens"] = 0
        raw["usage"]["total_tokens"] = cases[0]["prompt_tokens"]
    result = m.response_outcome(cases[0], _canonical(raw), 200)
    assert result["success"] is False
    assert result["output"] == ""


def test_length_limited_and_empty_content_are_not_discarded(cases):
    raw = body(cases[0])
    raw["choices"][0]["finish_reason"] = "length"
    result = m.response_outcome(cases[0], _canonical(raw), 200)
    assert result["success"] is True
    assert result["finish_reason"] == "length"
    raw["choices"][0]["message"]["content"] = ""
    raw["usage"].update(completion_tokens=0, total_tokens=cases[0]["prompt_tokens"])
    assert m.response_outcome(cases[0], _canonical(raw), 200)["output"] == ""


def test_http_error_cannot_extract_answer_from_error_body(cases):
    result = m.response_outcome(cases[0], b"answer is B", 429)
    assert result["success"] is False
    assert result["error_kind"] == "http-error"
    assert result["output"] == ""


def test_complete_receipts_are_ordered_and_do_not_authorize_transport(tmp_path, cases):
    value = receipt(tmp_path, cases)
    value["requests"].reverse()
    before = copy.deepcopy(value)
    rows, summary = normalize(value, cases, tmp_path)
    assert value == before
    assert [row["case_id"] for row in rows] == [case["case_id"] for case in cases]
    assert all(row["success"] for row in rows)
    assert summary["case_count"] == 2
    assert summary["failure_counts"] == {}
    assert not summary["serving_tokenizer_verified"]
    assert not summary["transport_verified"]
    assert not summary["execution_authorized"]


@pytest.mark.parametrize("kind", ["timeout", "connection-error", "not-attempted"])
def test_failed_and_unattempted_cases_remain_in_denominator(tmp_path, cases, kind):
    value = receipt(tmp_path, cases)
    row = value["requests"][0]
    row["result"] = {"kind": kind}
    row["attempts"] = 0 if kind == "not-attempted" else 1
    rows, summary = normalize(value, cases, tmp_path)
    assert len(rows) == 2
    assert rows[0]["success"] is False
    assert summary["failure_counts"] == {kind: 1}


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "arm",
        "bundle",
        "evidence",
        "method",
        "path",
        "claim",
        "missing",
        "duplicate",
        "unknown",
        "retry",
        "attempt-bool",
        "status-bool",
        "request",
        "request-hash",
        "response-hash",
        "response-size",
        "failure-extra",
    ],
)
def test_provenance_or_artifact_corruption_stops_conversion(tmp_path, cases, mutation):
    value = receipt(tmp_path, cases)
    row = value["requests"][0]
    if mutation in {"schema", "arm", "bundle", "evidence", "method", "path"}:
        key = {
            "schema": "schema_version",
            "arm": "arm",
            "bundle": "scorer_bundle_sha256",
            "evidence": "evidence_kind",
            "method": "request_method",
            "path": "request_path",
        }[mutation]
        value[key] = "changed"
    elif mutation == "claim":
        value["transport_verified"] = True
    elif mutation == "missing":
        value["requests"].pop()
    elif mutation == "duplicate":
        value["requests"][1] = copy.deepcopy(row)
    elif mutation == "unknown":
        row["case_id"] = "unknown"
    elif mutation == "retry":
        row["attempts"] = 2
    elif mutation == "attempt-bool":
        row["attempts"] = True
    elif mutation == "status-bool":
        row["result"]["status"] = True
    elif mutation == "request":
        changed = copy.deepcopy(cases[0]["candidate_request"])
        changed["seed"] = 99
        row["request"] = store(tmp_path, "request-0.json", _canonical(changed))
    elif mutation == "request-hash":
        row["request"]["sha256"] = "0" * 64
    elif mutation == "response-hash":
        row["result"]["body"]["sha256"] = "0" * 64
    elif mutation == "response-size":
        row["result"]["body"]["bytes"] += 1
    else:
        row["result"] = {"kind": "timeout", "output": "answer is B"}
    with pytest.raises(ValueError):
        normalize(value, cases, tmp_path)


@pytest.mark.parametrize("path", ["../outside.json", "/absolute.json", "."])
def test_artifact_paths_cannot_escape_arm_root(tmp_path, path):
    descriptor = {"path": path, "sha256": "0" * 64, "bytes": 0}
    with pytest.raises(ValueError):
        m.read_artifact(tmp_path, descriptor)


def test_symlink_artifacts_are_rejected_even_with_valid_digest(tmp_path):
    actual = tmp_path / "actual.json"
    actual.write_bytes(b"{}")
    (tmp_path / "link.json").symlink_to(actual)
    with pytest.raises(ValueError, match="symlinks"):
        m.read_artifact(
            tmp_path, {"path": "link.json", "sha256": m._sha(b"{}"), "bytes": 2}
        )


def test_body_read_is_bounded_including_false_size_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MAX_BODY_BYTES", 8)
    descriptor = store(tmp_path, "large.json", b"x" * 9)
    with pytest.raises(ValueError, match="identity/size"):
        m.read_artifact(tmp_path, descriptor)
    descriptor["bytes"] = 8
    with pytest.raises(ValueError, match="size/checksum"):
        m.read_artifact(tmp_path, descriptor)


def test_legacy_scoring_cannot_normalize_stored_receipts(tmp_path):
    case = {
        "case_id": "test",
        "category": "synthetic",
        "prompt": [0],
        "prompt_sha256": m._sha(b"[0]"),
        "prompt_tokens": 1,
        "max_tokens": 32,
        "ignore_eos": False,
        "option_count": 2,
        "answer": "B",
    }
    paths = {}
    for name, value in {"cases": [case], "B0": {}, "B1": {}}.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(_canonical(value))
    record = {
        "schema_version": pair.SCHEMA,
        "evidence_kind": "development-fixture",
        "scope": "supplied-manifest-only",
        "dataset": "mmlu-pro",
        "dataset_revision": pair.DATASET_REVISION,
        "benchmark_revision": pair.BENCHMARK_REVISION,
        "scorer_source_sha256": pair.SCORER_SHA256,
        "task_count": 1,
        "generation": {"max_tokens": 32, "ignore_eos": False},
        "files_sha256": {
            name: m._sha(path.read_bytes()) for name, path in paths.items()
        },
    }
    envelope = tmp_path / "envelope.json"
    envelope.write_bytes(_canonical(record))
    with pytest.raises(ValueError, match="requires verified source-bound"):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            stored_transport_root=tmp_path,
        )
