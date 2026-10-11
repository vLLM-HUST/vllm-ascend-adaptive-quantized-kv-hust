import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import prepare_pujiang_mmlu_source as m


@pytest.fixture
def inputs():
    def row(identity, category):
        return {
            "question_id": identity,
            "category": category,
            "question": f"synthetic question {identity}",
            "options": ["choice A", "choice B"],
            "answer_index": 1,
            "answer": "B",
            "cot_content": "synthetic validation-only explanation",
        }

    tables = {
        "test": [row(20, "subject-b"), row(21, "subject-a")],
        "validation": [
            row(i, subject) for subject in ("subject-a", "subject-b") for i in range(5)
        ],
    }
    for i, example in enumerate(tables["validation"]):
        example["question_id"] = i
    selection = {
        "schema_version": m.SELECTION_SCHEMA,
        "dataset_revision": m.DATASET_REVISION,
        "split": "test",
        "review_status": "candidate-unreviewed",
        "selection_reason": "Explicit synthetic ordered IDs; no outcome selection",
        "model_alias": "candidate-served-model",
        "question_ids": [21, 20],
    }
    formatter = Mock(
        side_effect=lambda row, *, answered: (
            f"{row['question']} {'DEMO' if answered else 'TARGET'}\n"
        )
    )
    return (
        tables,
        "synthetic instruction about {$}",
        selection,
        SimpleNamespace(format_example=formatter),
    )


def test_ordered_ids_source_rows_examples_and_budget(inputs):
    tables, instruction, selection, scorer = inputs
    tasks = m.build_tasks(*inputs)
    assert [x["question_id"] for x in tasks] == [21, 20]
    assert tasks[0]["validation_example_ids"] == list(range(5))
    assert tasks[1]["validation_example_ids"] == list(range(5, 10))
    for task in tasks:
        request = task["candidate_request"]
        assert request["messages"][0]["content"].count("DEMO") == 5
        assert request["messages"][0]["content"].count("TARGET") == 1
        assert request["max_tokens"] == 2048
        assert request["ignore_eos"] is False
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert task["prompt_token_ids"] is None and task["prompt_tokens"] is None
        assert (
            task["request_sha256"] == hashlib.sha256(m._canonical(request)).hexdigest()
        )
        row = next(x for x in tables["test"] if x["question_id"] == task["question_id"])
        assert (
            task["source_row_sha256"] == hashlib.sha256(m._canonical(row)).hexdigest()
        )
    assert all(
        call.kwargs["answered"] is False
        for call in scorer.format_example.call_args_list
        if call.args[0] in tables["test"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "accepted/v1"),
        ("dataset_revision", "main"),
        ("split", "validation"),
        ("review_status", "accepted"),
        ("selection_reason", " "),
        ("model_alias", ""),
        ("question_ids", []),
        ("question_ids", [True]),
        ("question_ids", [-1]),
        ("question_ids", [20, 20]),
        ("question_ids", ["20"]),
        ("question_ids", [999]),
    ],
)
def test_invalid_selection_is_rejected(inputs, field, value):
    inputs[2][field] = value
    with pytest.raises(ValueError):
        m.build_tasks(*inputs)


@pytest.mark.parametrize("split", ["test", "validation"])
@pytest.mark.parametrize(
    "mutation", ["duplicate", "category", "question", "answer_index", "options"]
)
def test_source_identity_and_field_errors(inputs, split, mutation):
    rows = inputs[0][split]
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0][mutation] = {
            "category": "",
            "question": None,
            "answer_index": True,
            "options": ["only-one-option"],
        }[mutation]
    with pytest.raises(ValueError):
        m.build_tasks(*inputs)


@pytest.mark.parametrize("change", ["four", "six", "wrong-category", "placeholder"])
def test_five_shot_and_instruction_are_fixed(inputs, change):
    tables, instruction, selection, scorer = inputs
    if change == "four":
        tables["validation"].pop()
    elif change == "six":
        extra = copy.deepcopy(tables["validation"][-1])
        extra["question_id"] = 99
        tables["validation"].append(extra)
    elif change == "wrong-category":
        tables["validation"][-1]["category"] = "unknown"
    else:
        instruction = "no subject placeholder"
    with pytest.raises(ValueError):
        m.build_tasks(tables, instruction, selection, scorer)


def test_no_threshold_selection_or_target_cot_leak(inputs):
    before = copy.deepcopy(inputs[:3])
    inputs[0]["test"][0]["cot_content"] = "SECRET-TARGET-RATIONALE"
    tasks = m.build_tasks(*inputs)
    assert len(tasks) == len(before[2]["question_ids"])
    assert "SECRET-TARGET-RATIONALE" not in json.dumps(tasks)
    assert "eligible" not in json.dumps(tasks)


def test_request_mutation_does_not_change_other_tasks_or_defaults(inputs):
    tasks = m.build_tasks(*inputs)
    tasks[0]["candidate_request"]["chat_template_kwargs"]["enable_thinking"] = True
    assert (
        tasks[1]["candidate_request"]["chat_template_kwargs"]["enable_thinking"]
        is False
    )
    assert m.GENERATION["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("exists", [False, True])
def test_cli_never_overwrites_and_sets_private_file_permissions(
    monkeypatch, tmp_path, exists
):
    output = tmp_path / "plan.json"
    if exists:
        output.write_bytes(b"original receipt")
    monkeypatch.setattr(
        m, "prepare_source", lambda *_: {"status": "candidate", "task_count": 2}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--assets",
            str(tmp_path),
            "--reference",
            str(tmp_path),
            "--selection",
            str(tmp_path),
            "--benchmark-checkout",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert m.main() == int(exists)
    if exists:
        assert output.read_bytes() == b"original receipt"
    else:
        assert output.stat().st_mode & 0o777 == 0o600
        assert json.loads(output.read_bytes())["task_count"] == 2


def test_checksum_checked_before_data_loading(tmp_path, monkeypatch):
    source = tmp_path / "test-00000-of-00001.parquet"
    source.write_bytes(b"not parquet")
    monkeypatch.setattr(
        m, "load_scorer", lambda _: SimpleNamespace(DATA_HASHES={"test": "0" * 64})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        m.prepare_source(tmp_path, tmp_path, tmp_path, tmp_path)


@pytest.fixture
def official_assets():
    checkout = os.getenv("AQKV_TEST_BENCHMARK_CHECKOUT")
    assets = os.getenv("AQKV_TEST_MMLU_ASSETS")
    reference = os.getenv("AQKV_TEST_MMLU_REFERENCE")
    if not all((checkout, assets, reference)):
        pytest.skip("Opt-in pinned source assets/checkout not supplied")
    return Path(assets), Path(reference), Path(checkout)


def test_real_pinned_assets_prepare_without_runtime(official_assets, tmp_path):
    assets, reference, checkout = official_assets
    selection = {
        "schema_version": m.SELECTION_SCHEMA,
        "dataset_revision": m.DATASET_REVISION,
        "split": "test",
        "review_status": "candidate-unreviewed",
        "selection_reason": "Integration check, not an official evaluation",
        "model_alias": "unverified-candidate-alias",
        "question_ids": [70, 71],
    }
    selected = tmp_path / "selection.json"
    selected.write_text(json.dumps(selection))
    report = m.prepare_source(assets, reference, selected, checkout)
    assert report["source_counts"] == {"test": 12032, "validation": 70}
    assert len(report["subjects"]) == 14
    assert report["task_count"] == 2
    assert report["coverage"] == "explicit-named-subset"
    assert report["model_requests_sent"] == 0
    assert report["task_quality"] is None
    assert not any(report["claim_boundary"].values())
    assert report["tokenizer_verified"] is False
    assert report["runtime_activation_verified"] is False
    assert report["transport_verified"] is False
    for task in report["tasks"]:
        assert (
            "Answer: Let's think step by step."
            in (task["candidate_request"]["messages"][0]["content"])
        )
        assert len(task["validation_example_ids"]) == 5


def test_validation_ids_cannot_masquerade_as_test_tasks(official_assets, tmp_path):
    assets, reference, checkout = official_assets
    selected = tmp_path / "selection.json"
    selected.write_text(
        json.dumps(
            {
                "schema_version": m.SELECTION_SCHEMA,
                "dataset_revision": m.DATASET_REVISION,
                "split": "test",
                "review_status": "candidate-unreviewed",
                "selection_reason": "Negative split-identity integration check",
                "model_alias": "unverified-candidate-alias",
                "question_ids": [0],
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown test question ID"):
        m.prepare_source(assets, reference, selected, checkout)


@pytest.mark.parametrize(
    "relative", ["evaluate_from_local.py", "cot_prompt_lib/initial_prompt.txt"]
)
def test_real_reference_tampering_is_rejected(official_assets, tmp_path, relative):
    assets, reference, checkout = official_assets
    for name in ("evaluate_from_local.py", "cot_prompt_lib/initial_prompt.txt"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((reference / name).read_bytes())
    (tmp_path / relative).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        m.prepare_source(assets, tmp_path, tmp_path / "not-used", checkout)
