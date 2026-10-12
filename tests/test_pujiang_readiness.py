import copy
import json

import pytest

from scripts.validate_pujiang_readiness import (
    CLAIM_BOUNDARY,
    DEFAULT_DELIVERY,
    DEFAULT_MATRIX,
    DEFAULT_PROGRAM,
    ROOT,
    load_json,
    main,
    validate_files,
    validate_matrix,
)


@pytest.fixture
def matrix():
    return load_json(DEFAULT_MATRIX)


@pytest.fixture
def program():
    return load_json(DEFAULT_PROGRAM)


def test_pinned_planning_record_does_not_authorize_execution():
    report = validate_files(DEFAULT_MATRIX, DEFAULT_PROGRAM)
    assert report["status"] == "planning_valid_execution_blocked"
    assert report["dataset_count"] == 5
    assert report["formal_run_count"] == 0
    assert report["review_status"] == "prepared-unreviewed"
    assert report["claim_boundary"] == CLAIM_BOUNDARY


def test_runtime_declaration_matches_actual_inert_manifest(matrix):
    manifest = load_json(
        ROOT / "src/vllm_ascend_adaptive_quantized_kv/vllm-hust-extension-v0.3.json"
    )
    assert matrix["runtime"]["manifest_schema"] == manifest["schema_version"]
    assert manifest["implementation"][0]["status"] == "import_only"
    assert manifest["activation"]["entry_points"] == []


@pytest.mark.parametrize("field", list(CLAIM_BOUNDARY))
@pytest.mark.parametrize("value", [True, 0])
def test_cannot_promote_planning_to_claim(matrix, program, field, value):
    matrix["claim_boundary"][field] = value
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "substitute", "reorder"])
def test_exact_five_dataset_scope(matrix, program, mutation):
    if mutation == "missing":
        matrix["datasets"].pop()
    elif mutation == "duplicate":
        matrix["datasets"][1] = copy.deepcopy(matrix["datasets"][0])
    elif mutation == "substitute":
        matrix["datasets"][0]["id"] = "mmlu"
    else:
        matrix["datasets"].reverse()
    with pytest.raises(ValueError, match="coverage or order"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("module_status", "applicable"),
        ("data_readiness", "executable"),
        ("source_revision", "0" * 40),
        ("sample_count", 200),
        ("manifest_sha256", "0" * 64),
        ("primary_metric", "token throughput"),
        ("formal_execution_complete", True),
        ("raw_result_artifacts", ["fixture.json"]),
        ("task_quality", 0.9),
        ("supporting_telemetry", {"throughput": 100}),
        ("path_observations", [{"dispatch": "fixture"}]),
    ],
)
def test_row_cannot_drift_or_invent_results(matrix, program, field, value):
    matrix["datasets"][0][field] = value
    with pytest.raises(ValueError):
        validate_matrix(matrix, program)


def test_cannot_drop_canonical_license_blocker(matrix, program):
    matrix["datasets"][1]["canonical_blockers"].pop(0)
    with pytest.raises(ValueError, match="canonical blocker"):
        validate_matrix(matrix, program)


def test_unconfirmed_blocker_owner_remains_explicit(matrix, program):
    matrix["datasets"][1]["canonical_blockers"][0].pop("owner")
    with pytest.raises(ValueError, match="missing owners"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("value", [True, 0])
def test_shared_work_is_not_aqkv_runtime_evidence(matrix, program, index, value):
    matrix["datasets"][index]["shared_progress"]["aqkv_runtime_verified"] = value
    with pytest.raises(ValueError, match="another MOD"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize("index", range(5))
def test_current_blockers_cannot_be_dropped(matrix, program, index):
    matrix["datasets"][index]["current_blockers"] = []
    with pytest.raises(ValueError, match="current blockers"):
        validate_matrix(matrix, program)


def test_shared_progress_requires_source(matrix, program):
    matrix["datasets"][0]["shared_progress"]["source_url"] = "unattributed"
    with pytest.raises(ValueError, match="attributable"):
        validate_matrix(matrix, program)


def test_shared_revision_cannot_be_latest_or_ambient(matrix, program):
    matrix["sources"]["shared_implementation_revision"] = "main"
    with pytest.raises(ValueError, match="shared implementation provenance"):
        validate_matrix(matrix, program)


def test_contract_inputs_are_not_fabricated(matrix, program):
    matrix["datasets"][0]["execution_contract"]["scorer_revision"] = "a" * 40
    with pytest.raises(ValueError, match="invent frozen inputs"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("implementation_status", "active"),
        ("manifest_schema", "0.2-experimental"),
        ("accepted_host_revision", "1" * 40),
        ("asset_access_verified_locally", True),
    ],
)
def test_upstream_assets_are_not_local_runtime_evidence(matrix, program, field, value):
    matrix["runtime"][field] = value
    with pytest.raises(ValueError):
        validate_matrix(matrix, program)


def test_snapshot_tampering_fails_before_semantic_validation(tmp_path):
    path = tmp_path / "program.json"
    path.write_bytes(DEFAULT_PROGRAM.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_files(DEFAULT_MATRIX, path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_plan_version", "V4.6"),
        ("test_plan_version", "V5.4"),
        ("source_revision", "main"),
        ("snapshot_sha256", "0" * 64),
        ("precision", "C8"),
        ("dtype", "int8"),
        ("chip_count", True),
        ("tensor_parallel_size", 4),
        ("baseline_roles", ["B0", "B1", "B2"]),
        ("inactive_optional_deliveries", []),
        ("source_fidelity", "canonical-final"),
        ("canonical_final_document_verified", True),
        ("aqkv_formal_inclusion_status", "accepted"),
        ("optional_measurements_enqueued", 1),
        ("execution_authorized", True),
    ],
)
def test_current_delivery_cannot_revert_to_research_or_invent_admission(
    matrix, program, field, value
):
    matrix["delivery_control"][field] = value
    with pytest.raises(ValueError, match="current delivery control drifted"):
        validate_matrix(matrix, program)


def test_delivery_snapshot_bytes_are_checked(tmp_path):
    path = tmp_path / "delivery.json"
    path.write_bytes(DEFAULT_DELIVERY.read_bytes() + b" ")
    with pytest.raises(ValueError, match="delivery contract checksum mismatch"):
        validate_files(DEFAULT_MATRIX, DEFAULT_PROGRAM, path)


def test_delivery_projection_keeps_implementation_and_authority_separate():
    report = validate_files(DEFAULT_MATRIX, DEFAULT_PROGRAM)
    assert report["shared_implementation_revision"].startswith("8ed7ec9")
    delivery = report["current_delivery_contract"]
    assert delivery["test_plan_version"] == "V5.5"
    assert delivery["source_revision"].startswith("9e7f833")
    assert delivery["precision"] == "BF16"
    assert delivery["optional_measurements_enqueued"] == 0
    assert delivery["execution_authorized"] is False
    assert report["canonical_snapshot_scope"].startswith("historical-")


def test_v54_snapshot_remains_historical_not_current_authority():
    old = ROOT / "docs/evidence/PUJIANG_V54_CONTRACT_SOURCE_20261010.json"
    assert load_json(old)["test_plan_version"] == "V5.4"
    with pytest.raises(ValueError, match="delivery contract checksum mismatch"):
        validate_files(DEFAULT_MATRIX, DEFAULT_PROGRAM, old)


def test_v55_declared_canonical_identity_is_not_verified_original():
    delivery = validate_files(DEFAULT_MATRIX, DEFAULT_PROGRAM)[
        "current_delivery_contract"
    ]
    identity = delivery["canonical_final_identity"]
    assert identity["status"] == "declared-external-identity"
    assert identity["size_bytes"] == 637379
    assert identity["page_count"] == 83
    assert identity["verified_locally"] is False
    assert delivery["canonical_final_document_verified"] is False
    assert load_json(DEFAULT_DELIVERY)["source_sha256"] != identity["sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner", "XilingGao"),
        ("producer", "plugin-fixture-generator"),
        ("value_source", "inferred-from-ci"),
        ("prefill_actual_values", True),
        ("prefill_actual_values", 0),
        ("infer_actual_values", True),
        ("required_materials", []),
    ],
)
def test_v55_receipt_role_and_actual_value_rules_cannot_drift(
    matrix, program, field, value
):
    matrix["delivery_control"]["b0_receipt_gate"][field] = value
    with pytest.raises(ValueError, match="current delivery control drifted"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize("index", range(13))
def test_v55_each_b0_material_is_required(matrix, program, index):
    matrix["delivery_control"]["b0_receipt_gate"]["required_materials"].pop(index)
    with pytest.raises(ValueError, match="current delivery control drifted"):
        validate_matrix(matrix, program)


def test_v55_both_oci_digests_are_separate_requirements():
    gate = validate_files(DEFAULT_MATRIX, DEFAULT_PROGRAM)["current_delivery_contract"][
        "b0_receipt_gate"
    ]
    assert gate["owner"] == "release-and-evaluation-engineering"
    assert len(gate["required_materials"]) == 13
    assert "b0_oci_repo_digest" in gate["required_materials"]
    assert "b1_oci_repo_digest" in gate["required_materials"]
    assert "startup_receipt" in gate["required_materials"]
    assert gate["prefill_actual_values"] is False
    assert gate["infer_actual_values"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("sha256", "0" * 64), ("size_bytes", 571121), ("verified_locally", True)],
)
def test_mirror_and_original_cannot_be_conflated(matrix, program, field, value):
    matrix["delivery_control"]["canonical_final_identity"][field] = value
    with pytest.raises(ValueError, match="current delivery control drifted"):
        validate_matrix(matrix, program)


@pytest.mark.parametrize("text", ['{"id": 1, "id": 2}', '{"value": NaN}'])
def test_json_loader_rejects_ambiguous_input(tmp_path, text):
    path = tmp_path / "ambiguous.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_json(path)


def test_command_reports_planning_only(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["validate_pujiang_readiness.py"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["claim_boundary"] == CLAIM_BOUNDARY


def test_command_fails_closed_for_unreadable_source(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv", ["validate_pujiang_readiness.py", "--program", str(tmp_path / "no")]
    )
    assert main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
