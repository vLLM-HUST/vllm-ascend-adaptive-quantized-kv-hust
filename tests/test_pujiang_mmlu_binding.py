import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tokenize_pujiang_mmlu_source as tokens
from scripts import validate_pujiang_mmlu_pair as pair
from scripts.prepare_pujiang_mmlu_source import GENERATION, SELECTION_SCHEMA
from scripts.validate_pujiang_readiness import CLAIM_BOUNDARY


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def plan():
    tasks = []
    for identity in (70, 71):
        request = {
            "model": "candidate-alias",
            "messages": [{"role": "user", "content": f"synthetic question {identity}"}],
            **copy.deepcopy(GENERATION),
            "stream": False,
        }
        ids = list(range(identity, identity + 16))
        tasks.append(
            {
                "case_id": f"mmlu-pro-{identity:05d}",
                "question_id": identity,
                "category": "synthetic-test",
                "source_row_sha256": "1" * 64,
                "validation_example_ids": [0, 1, 2, 3, 4],
                "answer": "B",
                "option_count": 2,
                "prompt_text_sha256": sha(request["messages"][0]["content"].encode()),
                "candidate_request": request,
                "request_sha256": sha(tokens._canonical(request)),
                "prompt_token_ids": ids,
                "prompt_tokens": len(ids),
                "prompt_token_ids_sha256": sha(tokens._canonical(ids)),
                "rendered_prompt_sha256": "2" * 64,
                "within_context_budget": True,
            }
        )
    return {
        "schema_version": "aqkv-mmlu-token-plan/v1",
        "dataset": "mmlu-pro",
        "dataset_revision": pair.DATASET_REVISION,
        "benchmark_revision": pair.BENCHMARK_REVISION,
        "scorer_source_sha256": pair.SCORER_SHA256,
        "source_data_sha256": {"test": "3" * 64, "validation": "4" * 64},
        "reference_revision": "5" * 40,
        "reference_sha256": "6" * 64,
        "instruction_sha256": "7" * 64,
        "selection_sha256": "8" * 64,
        "selection": {"model_alias": "candidate-alias", "question_ids": [70, 71]},
        "coverage": "explicit-named-subset",
        "source_tasks_sha256": "9" * 64,
        "tasks_sha256": sha(tokens._canonical(tasks)),
        "tasks": tasks,
        "task_count": len(tasks),
        "tokenizer_verified": True,
        "tokenizer": {
            "repository": tokens.MODEL_REPOSITORY,
            "revision": tokens.MODEL_REVISION,
        },
        "context_audit": {"context_budget_passed": True, "context_tokens": 32768},
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }


def write_pair(directory, bundle):
    cases = bundle["cases"]
    rows = [
        {
            "case_id": case["case_id"],
            **{field: case[field] for field in pair.SOURCE_REQUEST_FIELDS},
            "success": False,
            "finish_reason": None,
        }
        for case in cases
    ]
    envelope = {
        **copy.deepcopy(bundle["envelope_contract"]),
        "evidence_kind": "development-fixture",
    }
    paths = {}
    for name, value in {"cases": bundle, "B0": rows, "B1": rows}.items():
        paths[name] = directory / (name + ".json")
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    envelope["files_sha256"] = {
        name: sha(destination.read_bytes()) for name, destination in paths.items()
    }
    envelope_path = directory / "envelope.json"
    envelope_path.write_text(json.dumps(envelope))
    return envelope_path, paths


@pytest.fixture
def source_inputs(tmp_path):
    return {name: tmp_path / name for name in pair.SOURCE_INPUTS}


@pytest.fixture
def prepared(plan, monkeypatch):
    calls = []

    def rebuild(*args):
        calls.append(args)
        return copy.deepcopy(plan)

    monkeypatch.setattr(tokens, "prepare_tokens", rebuild)
    return tokens.scoring_bundle(plan), calls


def test_export_preserves_source_order_full_requests_and_claim_boundaries(plan):
    original = copy.deepcopy(plan)
    bundle = tokens.scoring_bundle(plan)
    assert plan == original
    assert [case["question_id"] for case in bundle["cases"]] == [70, 71]
    assert bundle["envelope_contract"]["generation"] == {**GENERATION, "stream": False}
    for case, task in zip(bundle["cases"], plan["tasks"], strict=True):
        assert case["candidate_request"] == task["candidate_request"]
        assert case["prompt"] == task["prompt_token_ids"]
        assert case["prompt_sha256"] == sha(tokens._canonical(case["prompt"]))
        assert case["answer"] == task["answer"]
    assert not any(bundle["claim_boundary"].values())
    assert bundle["model_requests_sent"] == 0
    assert not bundle["transport_verified"]
    assert not bundle["serving_tokenizer_verified"]


@pytest.mark.parametrize(
    "mutation", ["schema", "tokenizer", "overflow", "count", "hash"]
)
def test_export_rejects_incomplete_token_plan(plan, mutation):
    if mutation == "schema":
        plan["schema_version"] = "aqkv-mmlu-source-plan/v1"
    elif mutation == "tokenizer":
        plan["tokenizer_verified"] = False
    elif mutation == "overflow":
        plan["context_audit"]["context_budget_passed"] = False
    elif mutation == "count":
        plan["task_count"] -= 1
    else:
        plan["tasks_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        tokens.scoring_bundle(plan)


@pytest.mark.parametrize("mutation", ["settings", "model", "tokens", "overflow"])
def test_export_rejects_self_rehashed_task_inconsistency(plan, mutation):
    task = plan["tasks"][1]
    if mutation == "settings":
        task["candidate_request"]["temperature"] = 1
    elif mutation == "model":
        task["candidate_request"]["model"] = "other-model"
    elif mutation == "tokens":
        task["prompt_token_ids"][0] = 0
    else:
        task["within_context_budget"] = False
    task["request_sha256"] = sha(tokens._canonical(task["candidate_request"]))
    plan["tasks_sha256"] = sha(tokens._canonical(plan["tasks"]))
    with pytest.raises(ValueError, match="Task request/tokens differ"):
        tokens.scoring_bundle(plan)


def test_source_bound_mode_cannot_use_only_stored_flags(tmp_path, prepared):
    bundle, calls = prepared
    envelope, paths = write_pair(tmp_path, bundle)
    with pytest.raises(ValueError, match="requires all official assets"):
        pair.score_pair(envelope, paths["cases"], paths["B0"], paths["B1"], tmp_path)
    assert not calls


@pytest.mark.parametrize(
    "mutation", ["answer", "tokens", "request", "id", "order", "subset", "template"]
)
def test_self_consistent_stored_bundle_tampering_is_rebuilt_not_trusted(
    tmp_path, prepared, source_inputs, mutation
):
    bundle, calls = prepared
    if mutation == "answer":
        bundle["cases"][0]["answer"] = "A"
    elif mutation == "tokens":
        bundle["cases"][0]["prompt"][0] = 0
    elif mutation == "request":
        bundle["cases"][0]["candidate_request"]["temperature"] = 1
    elif mutation == "id":
        bundle["cases"][0]["question_id"] = 72
    elif mutation == "order":
        bundle["cases"].reverse()
    elif mutation == "subset":
        bundle["cases"].pop()
    else:
        bundle["envelope_contract"]["source_binding"]["tokenizer"]["revision"] = (
            "0" * 40
        )
    bundle["envelope_contract"]["source_binding"]["cases_sha256"] = sha(
        tokens._canonical(bundle["cases"])
    )
    envelope, paths = write_pair(tmp_path, bundle)
    with pytest.raises(ValueError, match="differs from rebuilt official source"):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            source_inputs=source_inputs,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field", ["temperature", "seed", "stop", "chat_template_kwargs", "stream"]
)
def test_envelope_must_bind_all_generation_settings(
    tmp_path, prepared, source_inputs, field
):
    bundle, _ = prepared
    envelope, paths = write_pair(tmp_path, bundle)
    record = json.loads(envelope.read_text())
    record["generation"][field] = "changed"
    envelope.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Source-bound envelope mismatch: generation"):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            source_inputs=source_inputs,
        )


@pytest.mark.parametrize("field", ["request_sha256", "rendered_prompt_sha256"])
def test_arm_cannot_change_source_bound_request_hash(
    tmp_path, prepared, source_inputs, field
):
    bundle, _ = prepared
    envelope, paths = write_pair(tmp_path, bundle)
    rows = json.loads(paths["B1"].read_text())
    rows[0][field] = "0" * 64
    paths["B1"].write_text(json.dumps(rows))
    record = json.loads(envelope.read_text())
    record["files_sha256"]["B1"] = sha(paths["B1"].read_bytes())
    envelope.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=f"unmatched request field {field}"):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            source_inputs=source_inputs,
        )


def test_source_bound_report_does_not_promote_transport_or_runtime(
    tmp_path, prepared, source_inputs, monkeypatch
):
    bundle, calls = prepared
    envelope, paths = write_pair(tmp_path, bundle)
    monkeypatch.setattr(
        pair,
        "load_scorer",
        lambda _: SimpleNamespace(
            score_results=lambda rows, cases: {
                "count": len(cases),
                "eligible": {},
                "ineligible": {},
                "outcomes": [{"eligible": False} for row in rows],
            }
        ),
    )
    report = pair.score_pair(
        envelope,
        paths["cases"],
        paths["B0"],
        paths["B1"],
        tmp_path,
        source_inputs=source_inputs,
    )
    assert len(calls) == 1
    assert report["status"] == "offline_scored_source_bound_runtime_unverified"
    assert report["source_binding_verified"] is True
    assert report["source_binding"] == bundle["envelope_contract"]["source_binding"]
    assert not report["serving_tokenizer_verified"]
    assert not report["transport_verified"]
    assert not any(report["claim_boundary"].values())
    assert report["aqkv_path_observations"] is None


@pytest.mark.parametrize("existing", ["plan", "bundle", "same-file"])
def test_token_export_preserves_existing_files(
    tmp_path, prepared, monkeypatch, existing
):
    _, calls = prepared
    output = tmp_path / "plan.json"
    bundle_path = tmp_path / "bundle.json"
    if existing == "same-file":
        bundle_path = output
    else:
        (output if existing == "plan" else bundle_path).write_text("keep this")
    monkeypatch.setattr(
        "sys.argv",
        [
            "tokenize_pujiang_mmlu_source.py",
            *[
                arg
                for name in (
                    "assets",
                    "reference",
                    "selection",
                    "benchmark-checkout",
                    "tokenizer-directory",
                )
                for arg in ("--" + name, str(tmp_path))
            ],
            "--output",
            str(output),
            "--scoring-output",
            str(bundle_path),
        ],
    )
    assert tokens.main() == 1
    assert not calls
    if existing != "same-file":
        assert (
            output if existing == "plan" else bundle_path
        ).read_text() == "keep this"


def test_token_export_writes_private_plan_and_bundle(tmp_path, prepared, monkeypatch):
    _, calls = prepared
    output = tmp_path / "plan.json"
    bundle_path = tmp_path / "bundle.json"
    # The CLI status comes from prepare_tokens, not the pure fixture builder.
    real_prepare = tokens.prepare_tokens

    def rebuild(*args):
        plan = real_prepare(*args)
        plan["status"] = "local_tokens_verified_candidate_plan_execution_blocked"
        return plan

    monkeypatch.setattr(tokens, "prepare_tokens", rebuild)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tokenize_pujiang_mmlu_source.py",
            *[
                arg
                for name in (
                    "assets",
                    "reference",
                    "selection",
                    "benchmark-checkout",
                    "tokenizer-directory",
                )
                for arg in ("--" + name, str(tmp_path))
            ],
            "--output",
            str(output),
            "--scoring-output",
            str(bundle_path),
        ],
    )
    assert tokens.main() == 0
    assert len(calls) == 1
    assert output.stat().st_mode & 0o777 == 0o600
    assert bundle_path.stat().st_mode & 0o777 == 0o600
    assert (
        json.loads(bundle_path.read_text())["schema_version"]
        == pair.SCORING_BUNDLE_SCHEMA
    )


def test_overflow_cli_retains_plan_but_exports_no_scoring_bundle(
    tmp_path, plan, monkeypatch
):
    plan["status"] = "context_budget_failed_execution_blocked"
    plan["context_audit"]["context_budget_passed"] = False
    monkeypatch.setattr(tokens, "prepare_tokens", lambda *_: plan)
    output, bundle_path = tmp_path / "plan.json", tmp_path / "bundle.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tokenize_pujiang_mmlu_source.py",
            *[
                argument
                for name in (*pair.SOURCE_INPUTS, "benchmark-checkout")
                for argument in ("--" + name, str(tmp_path))
            ],
            "--output",
            str(output),
            "--scoring-output",
            str(bundle_path),
        ],
    )
    assert tokens.main() == 1
    assert len(json.loads(output.read_text())["tasks"]) == 2
    assert not bundle_path.exists()


def test_partial_source_cli_fails_before_reading_any_artifact(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_pujiang_mmlu_pair.py",
            *[
                argument
                for name in ("envelope", "cases", "b0", "b1", "benchmark-checkout")
                for argument in ("--" + name, str(tmp_path / "absent"))
            ],
            "--assets",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert pair.main() == 1
    assert not output.exists()


def test_legacy_envelope_cannot_silently_ignore_source_inputs(
    tmp_path, prepared, source_inputs
):
    bundle, calls = prepared
    envelope, paths = write_pair(tmp_path, bundle)
    record = json.loads(envelope.read_text())
    record["schema_version"] = pair.SCHEMA
    record["scope"] = "supplied-manifest-only"
    envelope.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Legacy envelopes cannot claim"):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            source_inputs=source_inputs,
        )
    assert not calls


def test_real_source_to_token_to_shared_scorer_keeps_failed_cases(tmp_path):
    names = {
        "assets": "AQKV_TEST_MMLU_ASSETS",
        "reference": "AQKV_TEST_MMLU_REFERENCE",
        "tokenizer-directory": "AQKV_TEST_MMLU_TOKENIZER",
    }
    if not all(
        os.environ.get(name)
        for name in (*names.values(), "AQKV_TEST_BENCHMARK_CHECKOUT")
    ):
        pytest.skip("Opt in with official assets, tokenizer and pinned benchmark")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": SELECTION_SCHEMA,
                "dataset_revision": pair.DATASET_REVISION,
                "split": "test",
                "review_status": "candidate-unreviewed",
                "selection_reason": "Four IDs for development-only failure scoring",
                "model_alias": "candidate-alias",
                "question_ids": [70, 71, 72, 73],
            }
        )
    )
    source_inputs = {key: Path(os.environ[name]) for key, name in names.items()}
    source_inputs["selection"] = selection
    checkout = Path(os.environ["AQKV_TEST_BENCHMARK_CHECKOUT"])
    plan = tokens.prepare_tokens(
        source_inputs["assets"],
        source_inputs["reference"],
        selection,
        checkout,
        source_inputs["tokenizer-directory"],
    )
    envelope, paths = write_pair(tmp_path, tokens.scoring_bundle(plan))
    report = pair.score_pair(
        envelope,
        paths["cases"],
        paths["B0"],
        paths["B1"],
        checkout,
        source_inputs=source_inputs,
    )
    for score in report["offline_quality"].values():
        assert score["count"] == 4
        assert score["transport_failures"] == 4
        assert score["accuracy"] == 0
    assert report["source_binding_verified"] is True
    assert report["evidence_kind"] == "development-fixture"
    assert report["aqkv_path_observations"] is None
    assert not any(report["claim_boundary"].values())


def test_stored_receipt_cli_connects_to_shared_scorer_without_runtime_claims(
    tmp_path, prepared, source_inputs, monkeypatch
):
    checkout = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not checkout:
        pytest.skip("Opt in with the pinned benchmark for shared scorer integration")
    from scripts import pujiang_mmlu_transport as transport

    bundle, _ = prepared
    envelope, paths = write_pair(tmp_path, bundle)
    for arm in ("B0", "B1"):
        directory = tmp_path / arm
        directory.mkdir()
        requests = []

        def artifact(name, value, root=directory):
            raw = tokens._canonical(value)
            (root / name).write_bytes(raw)
            return {"path": name, "sha256": sha(raw), "bytes": len(raw)}

        for index, case in enumerate(bundle["cases"]):
            result = {"kind": "timeout"}
            if index == 0:
                response = {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer is B"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": case["prompt_tokens"],
                        "completion_tokens": 2,
                        "total_tokens": case["prompt_tokens"] + 2,
                    },
                }
                result = {
                    "kind": "http",
                    "status": 200,
                    "body": artifact(f"response-{index}.json", response),
                }
            requests.append(
                {
                    "case_id": case["case_id"],
                    "request": artifact(
                        f"request-{index}.json", case["candidate_request"]
                    ),
                    "attempts": 1,
                    "result": result,
                }
            )
        paths[arm].write_bytes(
            tokens._canonical(
                {
                    "schema_version": transport.SCHEMA,
                    "arm": arm,
                    "evidence_kind": "development-fixture",
                    "scorer_bundle_sha256": sha(paths["cases"].read_bytes()),
                    "request_method": "POST",
                    "request_path": "/v1/chat/completions",
                    "requests": requests,
                }
            )
        )
    record = json.loads(envelope.read_text())
    record["files_sha256"] = {
        name: sha(path.read_bytes()) for name, path in paths.items()
    }
    envelope.write_text(json.dumps(record))
    output = tmp_path / "normalized-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
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
            checkout,
            "--stored-transport-root",
            str(tmp_path),
            *[
                value
                for name, destination in source_inputs.items()
                for value in ("--" + name, str(destination))
            ],
            "--output",
            str(output),
        ],
    )
    assert pair.main() == 0
    report = json.loads(output.read_text())
    assert output.stat().st_mode & 0o777 == 0o600
    for arm in ("B0", "B1"):
        score = report["offline_quality"][arm]
        assert score["count"] == 2
        assert score["correct"] == 1
        assert score["transport_failures"] == 1
        assert score["length_limited"] == 1
        assert report["stored_transport"][arm]["failure_counts"] == {"timeout": 1}
        assert not report["stored_transport"][arm]["transport_verified"]
    assert not report["serving_tokenizer_verified"]
    assert not report["transport_verified"]
    assert not any(report["claim_boundary"].values())
    before = output.read_bytes()
    assert pair.main() == 1
    assert output.read_bytes() == before
