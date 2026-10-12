import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_pujiang_mmlu_pair as m


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def payload():
    cases = []
    arms = {"B0": [], "B1": []}
    for index in range(4):
        prompt = list(range(16 + index))
        case = {
            "case_id": f"synthetic-{index}",
            "category": "synthetic-unit-test",
            "prompt": prompt,
            "prompt_sha256": digest(json.dumps(prompt, separators=(",", ":")).encode()),
            "prompt_tokens": len(prompt),
            "max_tokens": 32,
            "ignore_eos": False,
            "option_count": 2,
            "answer": "B",
        }
        cases.append(case)
        for rows in arms.values():
            rows.append(
                {
                    "case_id": case["case_id"],
                    **{field: case[field] for field in m.REQUEST_FIELDS},
                    "success": index != 3,
                    "output": ["answer is B", "answer is J", "no choice", "B"][index],
                    "finish_reason": "length" if index == 0 else None,
                }
            )
    envelope = {
        "schema_version": m.SCHEMA,
        "evidence_kind": "development-fixture",
        "scope": "supplied-manifest-only",
        "dataset": "mmlu-pro",
        "dataset_revision": m.DATASET_REVISION,
        "benchmark_revision": m.BENCHMARK_REVISION,
        "scorer_source_sha256": m.SCORER_SHA256,
        "task_count": len(cases),
        "generation": {"max_tokens": 32, "ignore_eos": False},
    }
    return envelope, cases, arms


def write_inputs(tmp_path, payload):
    envelope, cases, arms = copy.deepcopy(payload)
    paths = {}
    for name, value in {"cases": cases, **arms}.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    envelope["files_sha256"] = {
        name: digest(path.read_bytes()) for name, path in paths.items()
    }
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    return envelope_path, paths


def test_valid_fixture_has_complete_pair_not_runtime_evidence(payload):
    envelope, cases, arms = payload
    assert m.validate_inputs(envelope, cases, arms) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "measured-result/v1"),
        ("evidence_kind", "runtime-measured"),
        ("evidence_kind", []),
        ("scope", "full-dataset-official"),
        ("dataset", "mmlu"),
        ("dataset_revision", "0" * 40),
        ("benchmark_revision", "main"),
        ("scorer_source_sha256", "0" * 64),
        ("task_count", True),
        ("task_count", 5),
    ],
)
def test_provenance_and_scope_cannot_be_promoted(payload, field, value):
    envelope, cases, arms = payload
    envelope[field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, cases, arms)


@pytest.mark.parametrize("arm", ["B0", "B1"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_all_task_outcomes_are_required(payload, arm, mutation):
    envelope, cases, arms = payload
    if mutation == "missing":
        arms[arm].pop()
    elif mutation == "duplicate":
        arms[arm][-1] = arms[arm][0]
    else:
        arms[arm][-1]["case_id"] = "not-in-manifest"
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, cases, arms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_sha256", "0" * 64),
        ("prompt_tokens", 100),
        ("max_tokens", 64),
        ("ignore_eos", 0),
        ("success", 1),
        ("output", None),
        ("finish_reason", {}),
    ],
)
def test_request_mismatch_or_truthy_flags_fail_closed(payload, field, value):
    envelope, cases, arms = payload
    arms["B1"][0][field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, cases, arms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "synthetic-1"),
        ("case_id", None),
        ("prompt", [True] * 16),
        ("prompt", [-1] * 16),
        ("prompt", {"input_ids": [1] * 16}),
        ("prompt_tokens", True),
        ("prompt_sha256", "0" * 64),
        ("option_count", True),
        ("option_count", 11),
        ("answer", "J"),
        ("category", None),
        ("max_tokens", True),
        ("ignore_eos", 0),
    ],
)
def test_manifest_must_bind_actual_tokens_and_answer_range(payload, field, value):
    envelope, cases, arms = payload
    cases[0][field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, cases, arms)


@pytest.mark.parametrize("name", ["cases", "B0", "B1"])
def test_artifact_tampering_is_rejected_before_scorer_import(tmp_path, payload, name):
    envelope, paths = write_inputs(tmp_path, payload)
    paths[name].write_bytes(paths[name].read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        m.score_pair(envelope, paths["cases"], paths["B0"], paths["B1"], tmp_path)


def test_invalid_checkout_is_not_an_ambient_dependency(tmp_path):
    with pytest.raises(ValueError, match="exact pinned repository"):
        m.load_scorer(tmp_path)


def test_changed_scorer_is_rejected_before_execution(tmp_path, monkeypatch):
    source = tmp_path / m.SCORER_PATH
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('unverified code executed')\n")

    def git_identity(command, **kwargs):
        identity = (
            str(tmp_path) if command[-1] == "--show-toplevel" else m.BENCHMARK_REVISION
        )
        return SimpleNamespace(returncode=0, stdout=identity)

    monkeypatch.setattr(m.subprocess, "run", git_identity)
    with pytest.raises(ValueError, match="source checksum"):
        m.load_scorer(tmp_path)


@pytest.mark.parametrize("name", ["cases", "B0", "B1"])
def test_duplicate_json_keys_fail_even_with_matching_checksum(tmp_path, payload, name):
    envelope, paths = write_inputs(tmp_path, payload)
    paths[name].write_text('[{"case_id":"first","case_id":"second"}]')
    record = json.loads(envelope.read_text())
    record["files_sha256"][name] = digest(paths[name].read_bytes())
    envelope.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        m.score_pair(envelope, paths["cases"], paths["B0"], paths["B1"], tmp_path)


def test_pyramidkv_eligibility_is_removed_not_relabelled(
    tmp_path, payload, monkeypatch
):
    envelope, paths = write_inputs(tmp_path, payload)

    def scorer(rows, cases):
        return {
            "count": len(rows),
            "eligible": {"count": 0},
            "ineligible": {"count": len(rows)},
            "outcomes": [
                {"case_id": row["case_id"], "eligible": False} for row in rows
            ],
        }

    monkeypatch.setattr(
        m, "load_scorer", lambda checkout: SimpleNamespace(score_results=scorer)
    )
    report = m.score_pair(envelope, paths["cases"], paths["B0"], paths["B1"], tmp_path)
    assert report["claim_boundary"] == m.CLAIM_BOUNDARY
    assert report["aqkv_path_observations"] is None
    for score in report["offline_quality"].values():
        assert "eligible" not in score and "ineligible" not in score
        assert all("eligible" not in row for row in score["outcomes"])


@pytest.fixture
def real_scorer_checkout():
    value = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not value:
        pytest.skip("Pinned benchmark checkout is required for upstream integration")
    return Path(value)


def test_real_pinned_scorer_keeps_failures_and_truncation(
    tmp_path, payload, real_scorer_checkout
):
    envelope, paths = write_inputs(tmp_path, payload)
    report = m.score_pair(
        envelope, paths["cases"], paths["B0"], paths["B1"], real_scorer_checkout
    )
    for score in report["offline_quality"].values():
        assert score["count"] == 4
        assert score["correct"] == 1
        assert score["accuracy"] == 0.25
        assert score["invalid_answers"] == 3
        assert score["transport_failures"] == 1
        assert score["length_limited"] == 1
    assert report["evidence_kind"] == "development-fixture"
    assert all(value is False for value in report["claim_boundary"].values())


def test_above_pyramidkv_threshold_is_still_not_aqkv_eligibility(
    tmp_path, payload, real_scorer_checkout
):
    _, cases, arms = payload
    cases[0]["prompt"] = list(range(4097))
    cases[0]["prompt_tokens"] = 4097
    cases[0]["prompt_sha256"] = digest(
        json.dumps(cases[0]["prompt"], separators=(",", ":")).encode()
    )
    for rows in arms.values():
        rows[0].update({field: cases[0][field] for field in m.REQUEST_FIELDS})
    envelope, paths = write_inputs(tmp_path, payload)
    report = m.score_pair(
        envelope, paths["cases"], paths["B0"], paths["B1"], real_scorer_checkout
    )
    assert report["aqkv_path_observations"] is None
    assert all("eligible" not in score for score in report["offline_quality"].values())


def test_command_preserves_existing_output(
    tmp_path, payload, real_scorer_checkout, monkeypatch, capsys
):
    envelope, paths = write_inputs(tmp_path, payload)
    output = tmp_path / "report.json"
    argv = [
        "validate_pujiang_mmlu_pair.py",
        "--envelope",
        str(envelope),
        "--cases",
        str(paths["cases"]),
        "--b0",
        str(paths["B0"]),
        "--b1",
        str(paths["B1"]),
        "--benchmark-checkout",
        str(real_scorer_checkout),
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert m.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "offline_scored_runtime_unverified"
    )
    original = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o600
    assert m.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert output.read_bytes() == original
