import copy
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import validate_pujiang_frontier_pair as m


@pytest.fixture
def stub_scorer():
    def messages(track, problem, reference, attempt):
        return [
            {
                "role": "user",
                "content": json.dumps([track, problem, reference, attempt]),
            }
        ]

    def parser(track, value):
        if value not in ("VERDICT: CORRECT", "VERDICT: 8"):
            raise ValueError("Synthetic parser rejection")

    return SimpleNamespace(
        JUDGE_INSTRUCTIONS={
            track: f"unit-test-only-{track}" for track in m.TRACK_COUNTS
        },
        judge_messages=messages,
        parse_verdict=parser,
        aggregate=lambda inventory, outcomes: {"stub": "not a score"},
    )


def make_payload(scorer):
    generation = {
        "max_tokens": 32,
        "temperature": 0,
        "seed": 17,
        "trials": 1,
        "retries": 0,
    }
    tasks = []
    outcomes = []
    spec = {
        "scope": "independently-authored-candidate",
        "scorer_source_sha256": m.SOURCE_FILES["scorer"][1],
        "model_revision": "synthetic-judge-version",
        "runtime_identity": "synthetic-judge-runtime",
        "instructions_sha256": {
            key: m.sha(value.encode())
            for key, value in scorer.JUDGE_INSTRUCTIONS.items()
        },
        "generation": {"max_tokens": 32, "temperature": 0},
    }
    calibration = {
        "judge_model_revision": spec["model_revision"],
        "judge_runtime_identity": spec["runtime_identity"],
        "instructions_sha256": spec["instructions_sha256"],
        "passed": True,
        "tracks": list(m.TRACK_COUNTS),
        "evidence_kind": "development-fixture",
    }
    for track, index in (("olympiad", 0), ("research", 0), ("research", 1)):
        prompt = [1, 2, 3 + index]
        task = {
            "task_id": f"{track}/test/{index:03d}",
            "track": track,
            "source_row": index,
            "task_group_id": "deliberately-duplicated-group-id",
            "source_row_sha256": m.sha(f"synthetic-{track}-{index}".encode()),
            "problem": "synthetic science problem",
            "reference": "synthetic answer or rubric; never model input",
            "prompt": prompt,
            "prompt_tokens": len(prompt),
            "prompt_sha256": m.sha(m.canonical(prompt)),
            "generation_sha256": m.sha(m.canonical(generation)),
        }
        tasks.append(task)
        outcomes.append(
            {
                "task_id": task["task_id"],
                "track": track,
                **{field: task[field] for field in m.REQUEST_FIELDS},
                "status": "solver_error" if index == 1 else "graded",
                "error": "synthetic solver failure" if index == 1 else None,
                "attempt": "synthetic attempt",
                "finish_reason": None if index == 1 else "length",
            }
        )
        if index != 1:
            outcomes[-1]["judge"] = {
                "runtime_identity": spec["runtime_identity"],
                "request": {
                    "model": spec["model_revision"],
                    "messages": scorer.judge_messages(
                        track,
                        task["problem"],
                        task["reference"],
                        outcomes[-1]["attempt"],
                    ),
                    **spec["generation"],
                },
                "response": {
                    "model": spec["model_revision"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "VERDICT: CORRECT"
                                if track == "olympiad"
                                else "VERDICT: 8"
                            },
                        }
                    ],
                },
            }
    envelope = {
        "schema_version": m.SCHEMA,
        "evidence_kind": "development-fixture",
        "scope": "supplied-manifest-only",
        "dataset": "frontierscience",
        "dataset_revision": m.DATASET_REVISION,
        "benchmark_revision": m.BENCHMARK_REVISION,
        "source_sha256": {name: value[1] for name, value in m.SOURCE_FILES.items()},
        "task_count": len(tasks),
        "generation": generation,
    }
    arms = {
        arm: {"collection": {}, "outcomes": copy.deepcopy(outcomes)}
        for arm in ("B0", "B1")
    }
    return envelope, tasks, arms, spec, calibration


@pytest.fixture
def payload(stub_scorer):
    return make_payload(stub_scorer)


def write_inputs(tmp_path, payload, *, judge=True):
    envelope, tasks, arms, spec, calibration = copy.deepcopy(payload)
    paths = {}
    if judge:
        for name, value in (("calibration", calibration), ("judge_contract", spec)):
            if name == "judge_contract":
                value["calibration_sha256"] = m.sha(paths["calibration"].read_bytes())
            paths[name] = tmp_path / (name + ".json")
            paths[name].write_bytes(m.canonical(value))
    else:
        paths.update(judge_contract=None, calibration=None)
    digests = {
        name: m.sha(path.read_bytes()) if path else None for name, path in paths.items()
    }
    for bundle in arms.values():
        bundle["collection"] = {
            "judge_contract_sha256": digests["judge_contract"],
            "calibration_sha256": digests["calibration"],
            "frozen_before_solver_requests": judge,
        }
        for row in bundle["outcomes"]:
            if row.get("judge"):
                row["judge"].update(
                    {
                        key: bundle["collection"][key]
                        for key in ("judge_contract_sha256", "calibration_sha256")
                    }
                )
    for name, value in {"inventory": tasks, **arms}.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(m.canonical(value))
    envelope["files_sha256"] = {
        name: m.sha(path.read_bytes()) if path else None for name, path in paths.items()
    }
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_bytes(m.canonical(envelope))
    return envelope_path, paths


def validate_written(envelope, paths, checkout):
    return m.validate_pair(
        envelope,
        paths["inventory"],
        paths["B0"],
        paths["B1"],
        checkout,
        paths["judge_contract"],
        paths["calibration"],
    )


def change_artifact(envelope_path, paths, name, mutate):
    value = m.parse_json(paths[name].read_bytes())
    mutate(value)
    paths[name].write_bytes(m.canonical(value))
    envelope = m.parse_json(envelope_path.read_bytes())
    envelope["files_sha256"][name] = m.sha(paths[name].read_bytes())
    envelope_path.write_bytes(m.canonical(envelope))


def test_track_row_identity_never_deduplicates_group_id(payload):
    envelope, tasks, arms, _, _ = payload
    _, checked, _ = m.validate_inputs(envelope, tasks, arms)
    assert len(checked) == 3
    assert len({row["task_group_id"] for row in checked}) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "runtime-measured/v1"),
        ("evidence_kind", "official-score"),
        ("evidence_kind", []),
        ("scope", "full-dataset-official"),
        ("dataset", "hle"),
        ("dataset_revision", "0" * 40),
        ("benchmark_revision", "main"),
        ("source_sha256", {}),
        ("task_count", True),
        ("task_count", 160),
    ],
)
def test_envelope_cannot_claim_runtime_or_official_scope(payload, field, value):
    envelope, tasks, arms, _, _ = payload
    envelope[field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, tasks, arms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", True),
        ("max_tokens", 0),
        ("temperature", True),
        ("temperature", float("inf")),
        ("seed", True),
        ("trials", True),
        ("trials", 2),
        ("retries", True),
        ("retries", 1),
    ],
)
def test_generation_budget_is_explicit_and_single_attempt(payload, field, value):
    envelope, tasks, arms, _, _ = payload
    envelope["generation"][field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, tasks, arms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("track", "science"),
        ("track", []),
        ("source_row", True),
        ("source_row", 100),
        ("task_id", "group-id"),
        ("source_row_sha256", "bad"),
        ("problem", ""),
        ("reference", None),
        ("prompt", [True]),
        ("prompt", [-1]),
        ("prompt_tokens", True),
        ("prompt_sha256", "0" * 64),
        ("generation_sha256", "0" * 64),
    ],
)
def test_inventory_requires_matched_source_row_and_real_tokens(payload, field, value):
    envelope, tasks, arms, _, _ = payload
    tasks[0][field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, tasks, arms)


@pytest.mark.parametrize("arm", ["B0", "B1"])
@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "unknown", "track", "request"]
)
def test_pairs_include_every_task_and_error(payload, arm, mutation):
    envelope, tasks, arms, _, _ = payload
    rows = arms[arm]["outcomes"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "unknown":
        rows[-1]["task_id"] = "research/test/059"
    elif mutation == "track":
        rows[-1]["track"] = "olympiad"
    else:
        rows[-1]["generation_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, tasks, arms)


@pytest.mark.parametrize(
    "name", ["inventory", "B0", "B1", "judge_contract", "calibration"]
)
def test_tampered_files_rejected_before_scorer_execution(tmp_path, payload, name):
    envelope, paths = write_inputs(tmp_path, payload)
    paths[name].write_bytes(paths[name].read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_written(envelope, paths, tmp_path)


def test_missing_judge_suppresses_even_cached_grades(
    tmp_path, payload, stub_scorer, monkeypatch
):
    envelope, paths = write_inputs(tmp_path, payload, judge=False)
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    report = validate_written(envelope, paths, tmp_path)
    assert report["judge_binding"] == "missing"
    assert report["fixture_scores"] is None and report["task_quality"] is None
    assert report["track_counts"]["B0"]["research"]["tasks"] == 2
    assert report["track_counts"]["B0"]["research"]["solver_errors"] == 1
    assert report["track_counts"]["B0"]["research"]["ungraded"] == 1
    assert report["aqkv_path_observations"] is None
    assert all(value is False for value in report["claim_boundary"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "runtime",
        "prompts",
        "tracks",
        "duplicate-track",
        "bad-track-type",
        "passed-type",
        "evidence",
    ],
)
def test_calibration_cannot_change_identity_or_use_parser_tests_as_real_evidence(
    payload, stub_scorer, mutation
):
    envelope, _, _, spec, calibration = payload
    envelope["files_sha256"] = {"calibration": "c" * 64}
    spec["calibration_sha256"] = "c" * 64
    if mutation == "model":
        calibration["judge_model_revision"] = "another-judge"
    elif mutation == "runtime":
        calibration["judge_runtime_identity"] = "another-runtime"
    elif mutation == "prompts":
        calibration["instructions_sha256"] = {}
    elif mutation == "tracks":
        calibration["tracks"] = ["research"]
    elif mutation == "duplicate-track":
        calibration["tracks"].append("research")
    elif mutation == "bad-track-type":
        calibration["tracks"] = [{}, "research"]
    elif mutation == "passed-type":
        calibration["passed"] = 1
    else:
        calibration["evidence_kind"] = "runtime-verified"
    with pytest.raises(ValueError):
        m.judge_binding(envelope, spec, calibration, stub_scorer)


def test_failed_calibration_does_not_produce_fixture_scores(
    tmp_path, payload, stub_scorer, monkeypatch
):
    payload[-1]["passed"] = False
    envelope, paths = write_inputs(tmp_path, payload)
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    report = validate_written(envelope, paths, tmp_path)
    assert report["judge_binding"] == "not_passed"
    assert report["fixture_scores"] is None


def test_unverified_external_receipt_never_authorizes_or_generates_task_scores(
    tmp_path, payload, stub_scorer, monkeypatch
):
    payload[0]["evidence_kind"] = "unverified-stored-outputs"
    payload[-1]["evidence_kind"] = "unverified-external-receipt"
    envelope, paths = write_inputs(tmp_path, payload)
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    report = validate_written(envelope, paths, tmp_path)
    assert report["judge_binding"] == "bound_unverified"
    assert report["fixture_scores"] is None and report["task_quality"] is None
    assert report["semantic_calibration_verified"] is False
    assert report["pre_generation_binding_verified"] is False


@pytest.mark.parametrize(
    "field",
    ["judge_contract_sha256", "calibration_sha256", "frozen_before_solver_requests"],
)
def test_posthoc_or_different_collection_binding_rejected(
    tmp_path, payload, stub_scorer, monkeypatch, field
):
    envelope, paths = write_inputs(tmp_path, payload)
    change_artifact(
        envelope,
        paths,
        "B1",
        lambda bundle: bundle["collection"].__setitem__(
            field, False if field.startswith("frozen") else "0" * 64
        ),
    )
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    with pytest.raises(ValueError, match="before solver"):
        validate_written(envelope, paths, tmp_path)


@pytest.mark.parametrize(
    "mutation", ["runtime", "response-model", "attempt", "reference", "budget"]
)
def test_raw_judge_record_bound_to_attempt_reference_and_budget(
    tmp_path, payload, stub_scorer, monkeypatch, mutation
):
    envelope, paths = write_inputs(tmp_path, payload)

    def mutate(bundle):
        row = bundle["outcomes"][0]
        if mutation == "runtime":
            row["judge"]["runtime_identity"] = "another-runtime"
        elif mutation == "response-model":
            row["judge"]["response"]["model"] = "another-model"
        elif mutation == "attempt":
            row["attempt"] = "posthoc replacement"
        elif mutation == "reference":
            row["judge"]["request"]["messages"] = []
        else:
            row["judge"]["request"]["max_tokens"] += 1

    change_artifact(envelope, paths, "B1", mutate)
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    with pytest.raises(ValueError):
        validate_written(envelope, paths, tmp_path)


@pytest.mark.parametrize("mutation", ["verdict", "truncated", "choices", "content"])
def test_bad_or_truncated_judge_response_is_retained_ungraded(
    tmp_path, payload, stub_scorer, monkeypatch, mutation
):
    envelope, paths = write_inputs(tmp_path, payload)

    def mutate(bundle):
        response = bundle["outcomes"][1]["judge"]["response"]
        if mutation == "verdict":
            response["choices"][0]["message"]["content"] = "VERDICT: NaN"
        elif mutation == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        elif mutation == "choices":
            response["choices"] = []
        else:
            response["choices"][0]["message"]["content"] = None

    change_artifact(envelope, paths, "B1", mutate)
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    report = validate_written(envelope, paths, tmp_path)
    assert report["normalized_outcomes"]["B1"][1]["status"] == "judge_error"
    assert report["track_counts"]["B1"]["research"]["ungraded"] == 1


def test_duplicate_json_keys_rejected_despite_matching_checksum(tmp_path, payload):
    envelope, paths = write_inputs(tmp_path, payload)
    paths["B1"].write_bytes(b'{"outcomes":[],"outcomes":[]}')
    record = json.loads(envelope.read_text())
    record["files_sha256"]["B1"] = m.sha(paths["B1"].read_bytes())
    envelope.write_bytes(m.canonical(record))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_written(envelope, paths, tmp_path)


def test_raw_judge_budget_rejects_bool_even_when_equal_to_one(
    tmp_path, payload, stub_scorer, monkeypatch
):
    payload[3]["generation"]["max_tokens"] = 1
    for bundle in payload[2].values():
        for row in bundle["outcomes"]:
            if row.get("judge"):
                row["judge"]["request"]["max_tokens"] = 1
    payload[2]["B1"]["outcomes"][0]["judge"]["request"]["max_tokens"] = True
    envelope, paths = write_inputs(tmp_path, payload)
    monkeypatch.setattr(m, "load_scorer", lambda checkout: stub_scorer)
    with pytest.raises(ValueError, match="Judge request differs"):
        validate_written(envelope, paths, tmp_path)


@pytest.mark.parametrize("name", ["inputs", "scorer"])
def test_all_dependency_sources_checked_before_any_execution(
    tmp_path, monkeypatch, name
):
    for kind, (relative, _) in m.SOURCE_FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("raise AssertionError('unverified code executed')\n")
        if kind != name:
            monkeypatch.setitem(
                m.SOURCE_FILES, kind, (relative, m.sha(target.read_bytes()))
            )
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=str(tmp_path)
            if cmd[-1] == "--show-toplevel"
            else m.BENCHMARK_REVISION,
        ),
    )
    with pytest.raises(ValueError, match="source checksum mismatch"):
        m.load_scorer(tmp_path)


@pytest.fixture
def real_scorer_checkout():
    value = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not value:
        pytest.skip("Pinned benchmark checkout required for upstream integration")
    return Path(value)


def test_real_loader_restores_ambient_modules(real_scorer_checkout, monkeypatch):
    names = ("vllm_hust_benchmark", "vllm_hust_benchmark.frontierscience_inputs")
    original = {name: ModuleType(name) for name in names}
    for name, module in original.items():
        monkeypatch.setitem(sys.modules, name, module)
    scorer = m.load_scorer(real_scorer_checkout)
    assert scorer.REVISION == m.DATASET_REVISION
    assert scorer.parse_verdict("research", "VERDICT: 7")["correct"] is True
    assert all(sys.modules[name] is module for name, module in original.items())


def test_real_aggregate_keeps_errors_denominators_and_tracks(
    tmp_path, real_scorer_checkout
):
    scorer = m.load_scorer(real_scorer_checkout)
    payload = make_payload(scorer)
    envelope, paths = write_inputs(tmp_path, payload)
    report = validate_written(envelope, paths, real_scorer_checkout)
    for scores in report["fixture_scores"].values():
        assert scores["olympiad"]["accuracy"] == 1
        assert scores["research"]["tasks"] == 2
        assert scores["research"]["accuracy"] == 0.5
        assert scores["research"]["mean_points"] == 4
        assert scores["research"]["solver_errors"] == 1
    assert report["task_quality"] is None
    assert all(value is False for value in report["claim_boundary"].values())


def test_real_aggregate_does_not_drop_malformed_grade_or_truncation(
    tmp_path, real_scorer_checkout
):
    payload = make_payload(m.load_scorer(real_scorer_checkout))
    payload[2]["B1"]["outcomes"][1]["judge"]["response"]["choices"][0]["message"][
        "content"
    ] = "VERDICT: 8\nVERDICT: 10"
    payload[2]["B1"]["outcomes"][1]["correct"] = True
    envelope, paths = write_inputs(tmp_path, payload)
    report = validate_written(envelope, paths, real_scorer_checkout)
    scores = report["fixture_scores"]["B1"]["research"]
    assert scores["accuracy"] is None and scores["mean_points"] is None
    assert scores["tasks"] == 2 and scores["ungraded"] == 1
    assert report["track_counts"]["B1"]["research"]["length_limited"] == 1


def test_real_command_refuses_to_overwrite_existing_report(
    tmp_path, real_scorer_checkout, monkeypatch, capsys
):
    envelope, paths = write_inputs(
        tmp_path, make_payload(m.load_scorer(real_scorer_checkout))
    )
    output = tmp_path / "report.json"
    argv = [
        "validate_pujiang_frontier_pair.py",
        "--envelope",
        str(envelope),
        "--inventory",
        str(paths["inventory"]),
        "--b0",
        str(paths["B0"]),
        "--b1",
        str(paths["B1"]),
        "--judge-contract",
        str(paths["judge_contract"]),
        "--calibration",
        str(paths["calibration"]),
        "--benchmark-checkout",
        str(real_scorer_checkout),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "offline_validated_runtime_unverified"
    )
    previous = output.read_bytes()
    assert m.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert output.read_bytes() == previous
