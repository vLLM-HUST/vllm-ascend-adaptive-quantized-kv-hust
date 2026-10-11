import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import tokenize_pujiang_mmlu_source as m
from scripts.validate_pujiang_readiness import CLAIM_BOUNDARY


@pytest.fixture
def inputs():
    request = {
        "model": "candidate-alias",
        "messages": [{"role": "user", "content": "synthetic prompt"}],
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    tasks = [
        {
            "case_id": identity,
            "candidate_request": copy.deepcopy(request),
            "request_sha256": m._sha(m._canonical(request)),
            "prompt_text_sha256": m._sha(b"synthetic prompt"),
        }
        for identity in ("task-a", "task-b")
    ]
    source = {
        "tasks": tasks,
        "tasks_sha256": m._sha(m._canonical(tasks)),
        "transport_intent": {"context_tokens": 64},
        "transport_verified": False,
        "runtime_activation_verified": False,
        "model_requests_sent": 0,
        "task_quality": None,
        "claim_boundary": CLAIM_BOUNDARY.copy(),
    }
    ids = list(range(32))
    scorer = SimpleNamespace(encode_prompt=Mock(return_value=ids))
    tokenizer = Mock()
    tokenizer.__len__ = Mock(return_value=64)
    tokenizer.chat_template = "synthetic template"
    tokenizer.apply_chat_template.side_effect = lambda messages, **kw: (
        ids if kw["tokenize"] else "synthetic rendered prompt"
    )
    return source, scorer, tokenizer


def test_exact_budget_token_binding_and_source_unchanged(inputs):
    source, scorer, tokenizer = inputs
    before = copy.deepcopy(source)
    tasks, summary = m.tokenize_tasks(*inputs)
    assert source == before
    assert summary["context_budget_passed"] is True
    assert summary["truncation_applied"] is False
    assert [x["case_id"] for x in tasks] == ["task-a", "task-b"]
    for task in tasks:
        assert task["context_headroom_tokens"] == 0
        assert task["prompt_tokens"] == 32
        assert task["prompt_token_ids"] == list(range(32))
        assert task["prompt_token_ids_sha256"] == m._sha(m._canonical(list(range(32))))
    assert scorer.encode_prompt.call_count == 2
    tokenizer.__len__.assert_called_once_with()
    assert all(
        call.kwargs
        == {
            "tokenize": bool(index % 2),
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        for index, call in enumerate(tokenizer.apply_chat_template.call_args_list)
    )


def test_overflow_is_retained_not_truncated_or_removed(inputs):
    inputs[0]["transport_intent"]["context_tokens"] = 63
    tasks, summary = m.tokenize_tasks(*inputs)
    assert summary["context_exceeded_case_ids"] == ["task-a", "task-b"]
    assert summary["context_budget_passed"] is False
    assert len(tasks) == 2
    assert all(task["context_headroom_tokens"] == -1 for task in tasks)


@pytest.mark.parametrize("value", [True, 0, -1, "64"])
def test_invalid_context(inputs, value):
    inputs[0]["transport_intent"]["context_tokens"] = value
    with pytest.raises(ValueError, match="context budget"):
        m.tokenize_tasks(*inputs)


@pytest.mark.parametrize("value", [True, 0, -1, "32"])
def test_invalid_generation_budget(inputs, value):
    task = inputs[0]["tasks"][0]
    task["candidate_request"]["max_tokens"] = value
    task["request_sha256"] = m._sha(m._canonical(task["candidate_request"]))
    with pytest.raises(ValueError, match="generation budget"):
        m.tokenize_tasks(*inputs)


@pytest.mark.parametrize("ids", [[True] * 32, [-1] * 32, [64] * 32, [0] * 15, "tokens"])
def test_invalid_token_ids(inputs, ids):
    inputs[1].encode_prompt.return_value = ids
    with pytest.raises(ValueError, match="actual prompt token IDs"):
        m.tokenize_tasks(*inputs)


def test_direct_tokenization_mismatch(inputs):
    inputs[1].encode_prompt.return_value = [1] * 32
    with pytest.raises(ValueError, match="disagree"):
        m.tokenize_tasks(*inputs)


def test_source_text_and_request_tampering(inputs):
    task = inputs[0]["tasks"][0]
    task["candidate_request"]["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="Request changed"):
        m.tokenize_tasks(*inputs)
    task["request_sha256"] = m._sha(m._canonical(task["candidate_request"]))
    with pytest.raises(ValueError, match="Source prompt changed"):
        m.tokenize_tasks(*inputs)


@pytest.mark.parametrize("value", [None, {}, {"enable_thinking": True}])
def test_thinking_mode_cannot_silently_change(inputs, value):
    task = inputs[0]["tasks"][0]
    task["candidate_request"]["chat_template_kwargs"] = value
    task["request_sha256"] = m._sha(m._canonical(task["candidate_request"]))
    with pytest.raises(ValueError, match="non-thinking"):
        m.tokenize_tasks(*inputs)


def test_empty_tasks_rejected(inputs):
    inputs[0]["tasks"] = []
    with pytest.raises(ValueError, match="No source tasks"):
        m.tokenize_tasks(*inputs)


@pytest.fixture
def staged(inputs, monkeypatch, tmp_path):
    originals = {name: b"synthetic template" for name in m.TOKENIZER_FILES}
    for name, raw in originals.items():
        (tmp_path / name).write_bytes(raw)
    (tmp_path / "config.json").write_text('{"unverified": true}')
    monkeypatch.setattr(
        m, "TOKENIZER_FILES", {name: m._sha(raw) for name, raw in originals.items()}
    )
    monkeypatch.setattr(m, "prepare_source", lambda *_: inputs[0])
    monkeypatch.setattr(m, "load_scorer", lambda *_: inputs[1])
    monkeypatch.setattr(m.importlib.metadata, "version", lambda _: "test-fixture")
    loaded = []

    def load(directory, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        assert {x.name for x in directory.iterdir()} == set(originals)
        assert all(
            (directory / name).read_bytes() == raw for name, raw in originals.items()
        )
        assert all(
            (directory / name).stat().st_mode & 0o777 == 0o600 for name in originals
        )
        loaded.append(directory)
        (tmp_path / "tokenizer.json").write_bytes(b"changed after verified read")
        return inputs[2]

    loader = Mock(side_effect=load)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader)),
    )
    return inputs, tmp_path, loaded, loader


def test_only_verified_snapshot_is_loaded_and_claims_remain_blocked(staged):
    inputs, directory, loaded, _ = staged
    report = m.prepare_tokens(directory, directory, directory, directory, directory)
    assert report["status"] == "local_tokens_verified_candidate_plan_execution_blocked"
    assert report["source_tasks_sha256"] == inputs[0]["tasks_sha256"]
    assert report["tokenizer_verified"] is True
    assert report["serving_tokenizer_verified"] is False
    assert report["transport_verified"] is False
    assert report["runtime_activation_verified"] is False
    assert report["model_requests_sent"] == 0
    assert report["task_quality"] is None
    assert not any(report["claim_boundary"].values())
    assert not loaded[0].exists()


def test_checksum_mismatch_is_rejected_before_loader(staged):
    _, directory, _, loader = staged
    (directory / "tokenizer.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        m.prepare_tokens(directory, directory, directory, directory, directory)
    loader.assert_not_called()


def test_loaded_template_mismatch(staged):
    inputs, directory, _, _ = staged
    inputs[2].chat_template = "unexpected"
    with pytest.raises(ValueError, match="chat template differs"):
        m.prepare_tokens(directory, directory, directory, directory, directory)


def test_failed_budget_keeps_execution_blocked(staged):
    inputs, directory, _, _ = staged
    inputs[0]["transport_intent"]["context_tokens"] = 63
    report = m.prepare_tokens(directory, directory, directory, directory, directory)
    assert report["status"] == "context_budget_failed_execution_blocked"
    assert len(report["tasks"]) == 2


def test_official_assets_with_pinned_real_tokenizer(tmp_path):
    names = (
        "AQKV_TEST_BENCHMARK_CHECKOUT",
        "AQKV_TEST_MMLU_ASSETS",
        "AQKV_TEST_MMLU_REFERENCE",
        "AQKV_TEST_MMLU_TOKENIZER",
    )
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Opt in with the pinned official assets and tokenizer")
    from scripts.prepare_pujiang_mmlu_source import DATASET_REVISION, SELECTION_SCHEMA

    selection = tmp_path / "selection.json"
    import json

    selection.write_text(
        json.dumps(
            {
                "schema_version": SELECTION_SCHEMA,
                "dataset_revision": DATASET_REVISION,
                "split": "test",
                "review_status": "candidate-unreviewed",
                "selection_reason": "One fixed official test ID; no model outcomes",
                "model_alias": "candidate-alias",
                "question_ids": [70],
            }
        )
    )
    report = m.prepare_tokens(
        Path(os.environ[names[1]]),
        Path(os.environ[names[2]]),
        selection,
        Path(os.environ[names[0]]),
        Path(os.environ[names[3]]),
    )
    assert report["context_audit"]["context_budget_passed"]
    assert report["tasks"][0]["question_id"] == 70
    assert report["tasks"][0]["prompt_tokens"] == len(
        report["tasks"][0]["prompt_token_ids"]
    )
    assert report["tokenizer"]["files_sha256"] == m.TOKENIZER_FILES
    assert report["model_requests_sent"] == 0
