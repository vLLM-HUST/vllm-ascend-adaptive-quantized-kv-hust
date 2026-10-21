import copy
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import validate_pujiang_terminal_plan as m


@pytest.fixture
def stub_plan():
    def job(arm, tasks, api_base):
        return {
            "job_name": "pyramidkv-terminal-" + arm,
            "jobs_dir": "jobs-" + arm,
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "tasks": [{"path": "tasks/" + task} for task in tasks],
            "verifier": {"disable": False},
            "environment": {"delete": True, "force_build": False},
            "agents": [
                {
                    "name": "terminus-2",
                    "model_name": "openai/qwen35-pyramidkv",
                    "kwargs": {
                        "api_base": api_base,
                        "max_turns": 100,
                        "enable_summarize": False,
                        "llm_kwargs": {"num_retries": 0},
                        "llm_call_kwargs": {"max_tokens": 4096},
                    },
                }
            ],
        }

    return SimpleNamespace(
        HARBOR_VERSION="0.24.0",
        HARBOR_REVISION="b53b8134e1241686dca7759af188f987ecc48e8b",
        job_config=job,
    )


def make_payload(plan):
    inventory = [
        {
            "task_id": name,
            "pinned_image": "synthetic/unit-test@sha256:" + "a" * 64,
            **{
                key: m.sha(("synthetic-" + name + key).encode())
                for key in (
                    "source_row_sha256",
                    "staged_files_sha256",
                    "file_modes_sha256",
                    "task_policy_sha256",
                )
            },
        }
        for name in ("synthetic-task-a", "synthetic-task-b")
    ]
    endpoints = {"B0": "http://127.0.0.1:18000/v1", "B1": "http://127.0.0.1:18001/v1"}
    jobs = {}
    for arm in endpoints:
        jobs[arm] = plan.job_config(
            arm.lower(), [row["task_id"] for row in inventory], endpoints[arm]
        )
        jobs[arm]["job_name"] = "aqkv-terminal-" + arm.lower()
        jobs[arm]["agents"][0]["model_name"] = "openai/qwen35-aqkv-synthetic"
    envelope = {
        "schema_version": m.SCHEMA,
        "evidence_kind": "development-fixture",
        "scope": "supplied-task-plan-only",
        "dataset": "terminal-bench-2.1",
        "dataset_revision": m.DATASET_REVISION,
        "benchmark_revision": m.BENCHMARK_REVISION,
        "source_sha256": {name: value[1] for name, value in m.SOURCE_FILES.items()},
        "harbor_version": plan.HARBOR_VERSION,
        "harbor_revision": plan.HARBOR_REVISION,
        "retry_policy": m.RETRY_POLICY.copy(),
        "task_count": len(inventory),
        "serving_intent": {
            "model_name": "openai/qwen35-aqkv-synthetic",
            "api_base": endpoints,
            "aqkv_enabled": {"B0": False, "B1": True},
        },
        "runtime_receipts": {"B0": None, "B1": None},
    }
    return envelope, inventory, jobs


@pytest.fixture
def payload(stub_plan):
    return make_payload(stub_plan)


def write_inputs(tmp_path, payload):
    envelope, inventory, jobs = copy.deepcopy(payload)
    paths = {}
    for name, value in {"inventory": inventory, **jobs}.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(m.canonical(value))
    envelope["files_sha256"] = {
        name: m.sha(path.read_bytes()) for name, path in paths.items()
    }
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_bytes(m.canonical(envelope))
    return envelope_path, paths


def test_matched_jobs_only_validate_intent_not_services(payload, stub_plan):
    report = m.validate_inputs(*payload, stub_plan)
    assert report["status"] == "candidate_plan_valid_execution_blocked"
    assert report["task_count"] == 2
    assert report["tasks_executed"] == 0 and report["model_calls"] == 0
    assert report["task_quality"] is None
    for key in (
        "source_membership_verified",
        "task_assets_verified",
        "harbor_schema_validated_locally",
        "actual_backend_connection_verified",
        "actual_service_switches_verified",
        "ready_for_scored_execution",
    ):
        assert report[key] is False
    assert all(value is False for value in report["claim_boundary"].values())


def test_same_planned_endpoint_is_not_proof_of_sequential_service_switch(
    payload, stub_plan
):
    envelope, inventory, jobs = payload
    endpoint = envelope["serving_intent"]["api_base"]["B0"]
    envelope["serving_intent"]["api_base"]["B1"] = endpoint
    jobs["B1"]["agents"][0]["kwargs"]["api_base"] = endpoint
    report = m.validate_inputs(envelope, inventory, jobs, stub_plan)
    assert report["actual_service_switches_verified"] is False


def test_matching_changed_budgets_in_both_arms_still_break_candidate(
    payload, stub_plan
):
    for job in payload[2].values():
        job["n_attempts"] = 2
    with pytest.raises(ValueError, match="candidate task/policy/budget mismatch"):
        m.validate_inputs(*payload, stub_plan)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "measured/v1"),
        ("evidence_kind", "runtime-measured"),
        ("evidence_kind", []),
        ("scope", "official-evaluation"),
        ("dataset", "terminal-bench"),
        ("dataset_revision", "main"),
        ("benchmark_revision", "main"),
        ("source_sha256", {}),
        ("harbor_version", "latest"),
        ("harbor_revision", "main"),
        ("task_count", True),
        ("task_count", 89),
        ("runtime_receipts", {"B0": {}, "B1": {}}),
    ],
)
def test_candidate_cannot_promote_revision_scope_or_activation(
    payload, stub_plan, field, value
):
    envelope, inventory, jobs = payload
    envelope[field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, inventory, jobs, stub_plan)


@pytest.mark.parametrize("key", list(m.RETRY_POLICY))
def test_nested_upstream_attempts_must_not_be_hidden(payload, stub_plan, key):
    envelope, inventory, jobs = payload
    envelope["retry_policy"][key] = 0 if m.RETRY_POLICY[key] else 1
    with pytest.raises(ValueError, match="Nested retries"):
        m.validate_inputs(envelope, inventory, jobs, stub_plan)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "../escape"),
        ("task_id", "synthetic-task-b"),
        ("task_id", []),
        ("pinned_image", "synthetic/image:latest"),
        ("pinned_image", "synthetic/image@sha256:bad"),
        ("source_row_sha256", None),
        ("staged_files_sha256", "bad"),
        ("file_modes_sha256", "bad"),
        ("task_policy_sha256", "bad"),
    ],
)
def test_tasks_bind_unique_ids_image_and_metadata(payload, stub_plan, field, value):
    envelope, inventory, jobs = payload
    inventory[0][field] = value
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, inventory, jobs, stub_plan)


@pytest.mark.parametrize(
    "mutation",
    ["B0-flag", "B1-flag", "bool-type", "alias", "endpoint", "credential-alias"],
)
def test_planned_switches_are_explicit_not_inferred_from_job_names(
    payload, stub_plan, mutation
):
    envelope, inventory, jobs = payload
    serving = envelope["serving_intent"]
    if mutation == "B0-flag":
        serving["aqkv_enabled"]["B0"] = True
    elif mutation == "B1-flag":
        serving["aqkv_enabled"]["B1"] = False
    elif mutation == "bool-type":
        serving["aqkv_enabled"]["B1"] = 1
    elif mutation == "alias":
        serving["model_name"] = "openai/another-model"
    elif mutation == "endpoint":
        serving["api_base"].pop("B1")
    else:
        serving["model_name"] = "openai/user:key@model"
    with pytest.raises(ValueError):
        m.validate_inputs(envelope, inventory, jobs, stub_plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "attempts",
        "attempt-type",
        "concurrency",
        "trial-retries",
        "sdk-retries",
        "verifier",
        "cleanup",
        "build",
        "budget",
        "turns",
        "summarize",
        "task-list",
        "alias",
        "endpoint",
        "label",
    ],
)
def test_candidate_pair_cannot_change_task_policy_or_budget(
    payload, stub_plan, mutation
):
    envelope, inventory, jobs = payload
    job = jobs["B1"]
    options = job["agents"][0]["kwargs"]
    if mutation == "attempts":
        job["n_attempts"] = 2
    elif mutation == "attempt-type":
        job["n_attempts"] = True
    elif mutation == "concurrency":
        job["n_concurrent_trials"] = 2
    elif mutation == "trial-retries":
        job["retry"]["max_retries"] = 1
    elif mutation == "sdk-retries":
        options["llm_kwargs"]["num_retries"] = 1
    elif mutation == "verifier":
        job["verifier"]["disable"] = True
    elif mutation == "cleanup":
        job["environment"]["delete"] = False
    elif mutation == "build":
        job["environment"]["force_build"] = True
    elif mutation == "budget":
        options["llm_call_kwargs"]["max_tokens"] -= 1
    elif mutation == "turns":
        options["max_turns"] -= 1
    elif mutation == "summarize":
        options["enable_summarize"] = True
    elif mutation == "task-list":
        job["tasks"].pop()
    elif mutation == "alias":
        job["agents"][0]["model_name"] = "openai/another-model"
    elif mutation == "endpoint":
        options["api_base"] = envelope["serving_intent"]["api_base"]["B0"]
    else:
        job["job_name"] = "aqkv-terminal-b0"
    with pytest.raises(ValueError, match="candidate task/policy/budget mismatch"):
        m.validate_inputs(envelope, inventory, jobs, stub_plan)


@pytest.mark.parametrize("name", ["inventory", "B0", "B1"])
def test_tampered_artifact_rejected_before_planner_import(tmp_path, payload, name):
    envelope, paths = write_inputs(tmp_path, payload)
    paths[name].write_bytes(paths[name].read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        m.validate_files(
            envelope, paths["inventory"], paths["B0"], paths["B1"], tmp_path
        )


def test_duplicate_json_key_rejected_even_with_updated_digest(tmp_path, payload):
    envelope, paths = write_inputs(tmp_path, payload)
    paths["B1"].write_bytes(b'{"n_attempts":1,"n_attempts":2}')
    record = json.loads(envelope.read_text())
    record["files_sha256"]["B1"] = m.sha(paths["B1"].read_bytes())
    envelope.write_bytes(m.canonical(record))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        m.validate_files(
            envelope, paths["inventory"], paths["B0"], paths["B1"], tmp_path
        )


def test_invalid_checkout_is_not_an_ambient_plan_dependency(tmp_path):
    with pytest.raises(ValueError, match="exact pinned repository"):
        m.load_plan(tmp_path)


@pytest.mark.parametrize("present", [(True, False, False), (False, True, True)])
def test_partial_asset_binding_rejected_before_input_reads(tmp_path, present):
    with pytest.raises(ValueError, match="must be supplied together"):
        m.validate_files(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            **{
                key: tmp_path if supplied else None
                for key, supplied in zip(
                    ("published_root", "snapshot", "staged_tasks"), present, strict=True
                )
            },
        )


def test_explicit_asset_binding_does_not_promote_services_or_execution(
    tmp_path, payload, stub_plan, monkeypatch
):
    from scripts import audit_pujiang_terminal_source as source

    envelope, paths = write_inputs(tmp_path, payload)
    monkeypatch.setattr(m, "load_plan", lambda path: stub_plan)
    calls = []

    def audit(checkout, published, snapshot, **kwargs):
        calls.append((checkout, published, snapshot, kwargs))
        return {
            "benchmark_revision": m.BENCHMARK_REVISION,
            "source_revision": m.DATASET_REVISION,
            "source_tree_root": "synthetic tree",
            "source_tree_sha256": "a" * 64,
            "report_files_sha256": {},
            "candidate_task_count": 2,
            "staged_files_verified": 10,
            "staged_bytes_verified": 100,
            "staged_task_bindings_sha256": "b" * 64,
            "registry_metadata_bytes_reverified": False,
            "image_layers_pulled": False,
            "rights_review_complete": False,
        }

    monkeypatch.setattr(source, "audit", audit)
    report = m.validate_files(
        envelope,
        paths["inventory"],
        paths["B0"],
        paths["B1"],
        tmp_path,
        published_root=tmp_path / "published",
        snapshot=tmp_path / "source",
        staged_tasks=tmp_path / "tasks",
    )
    assert len(calls) == 1 and calls[0][3]["candidate_inventory"] == payload[1]
    assert report["status"] == "candidate_plan_assets_bound_execution_blocked"
    assert report["source_membership_verified"] and report["task_assets_verified"]
    assert report["local_staged_modes_verified"]
    assert not report["actual_backend_connection_verified"]
    assert not report["actual_service_switches_verified"]
    assert not report["ready_for_scored_execution"]
    assert report["tasks_executed"] == report["model_calls"] == 0
    assert report["task_quality"] is None
    assert all(value is False for value in report["claim_boundary"].values())


def test_asset_audit_failure_propagates_without_success_report(
    tmp_path, payload, stub_plan, monkeypatch
):
    from scripts import audit_pujiang_terminal_source as source

    envelope, paths = write_inputs(tmp_path, payload)
    monkeypatch.setattr(m, "load_plan", lambda path: stub_plan)

    def fail(*args, **kwargs):
        raise ValueError("actual asset mismatch")

    monkeypatch.setattr(source, "audit", fail)
    with pytest.raises(ValueError, match="actual asset mismatch"):
        m.validate_files(
            envelope,
            paths["inventory"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            published_root=tmp_path,
            snapshot=tmp_path,
            staged_tasks=tmp_path,
        )


@pytest.fixture
def real_plan_checkout():
    value = os.environ.get("AQKV_TEST_BENCHMARK_CHECKOUT")
    if not value:
        pytest.skip("Pinned benchmark checkout required for upstream integration")
    return Path(value)


def test_real_planner_preserves_candidate_budgets_and_does_not_validate_services(
    real_plan_checkout,
):
    plan = m.load_plan(real_plan_checkout)
    envelope, inventory, jobs = make_payload(plan)
    report = m.validate_inputs(envelope, inventory, jobs, plan)
    assert jobs["B0"]["environment"]["cpu_enforcement_policy"] == "limit"
    assert jobs["B0"]["agents"][0]["kwargs"]["model_info"]["max_input_tokens"] == 32768
    assert report["retry_policy"]["harbor_llm_layer_attempts"] == 3
    assert report["retry_policy"]["terminus_query_layer_attempts"] == 3
    assert report["actual_service_switches_verified"] is False


@pytest.mark.parametrize(
    "url",
    [
        "file:///v1",
        "https://user:synthetic@example.invalid/v1",
        "https://example.invalid/v1?key=synthetic",
        "https://example.invalid/v1#fragment",
        "https://example.invalid",
        "http:///v1",
    ],
)
def test_real_upstream_rejects_credentials_and_invalid_endpoint(
    real_plan_checkout, url
):
    plan = m.load_plan(real_plan_checkout)
    payload = make_payload(plan)
    payload[0]["serving_intent"]["api_base"]["B0"] = url
    with pytest.raises(ValueError, match="base URL"):
        m.validate_inputs(*payload, plan)


def test_real_plan_loader_restores_ambient_audit_dependency(
    real_plan_checkout, monkeypatch
):
    ambient = ModuleType("audit_pyramidkv_terminal")
    monkeypatch.setitem(sys.modules, ambient.__name__, ambient)
    plan = m.load_plan(real_plan_checkout)
    assert plan.source_audit.REVISION == m.DATASET_REVISION
    assert sys.modules[ambient.__name__] is ambient


def test_real_command_preserves_previous_output(
    tmp_path, real_plan_checkout, monkeypatch, capsys
):
    envelope, paths = write_inputs(
        tmp_path, make_payload(m.load_plan(real_plan_checkout))
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_pujiang_terminal_plan.py",
            "--envelope",
            str(envelope),
            "--inventory",
            str(paths["inventory"]),
            "--b0",
            str(paths["B0"]),
            "--b1",
            str(paths["B1"]),
            "--benchmark-checkout",
            str(real_plan_checkout),
            "--output",
            str(output),
        ],
    )
    assert m.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "candidate_plan_valid_execution_blocked"
    )
    old = output.read_bytes()
    assert m.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert output.read_bytes() == old
