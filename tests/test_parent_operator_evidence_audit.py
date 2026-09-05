import hashlib
import json

from scripts import audit_parent_operator_evidence as audit


def _records():
    contract = {
        "schema_version": "adaptive-quantized-kv-paged-int8-prefill-probe-contract/v1",
        "status": "PREREGISTERED",
        "revision": 2,
        "hardware": {"device_type": "Ascend 910B2", "co_rental_allowed": False},
    }
    nz = {
        "status": "PILOT_DIAGNOSTIC",
        "runtime": {"device": "Ascend 910B2"},
        "native_status": "UNSUPPORTED_OR_ERROR",
        "native_error_code": 561002,
        "raw_receipt": {"sha256": audit.RAW_NZ_RECEIPT_SHA256},
    }
    nd = {
        "status": "PILOT_DIAGNOSTIC",
        "runtime": {"device": "Ascend 910B2"},
        "native_status": "UNSUPPORTED_OR_ERROR",
        "native_error_code": 561002,
        "native_error_summary": "antiquant scale is nullptr",
        "raw_receipt": {"sha256": audit.RAW_ND_RECEIPT_SHA256},
    }
    v2 = {
        "execution_performed": False,
        "npu_allocated": False,
        "source": {
            "parent_commit": audit.PINNED_PARENT_SOURCE,
            "vllm_ascend_commit": audit.PINNED_ASCEND_SOURCE,
        },
        "direct_v2_coverage": {
            "status": "NO_DIRECT_COVERAGE",
            "query_length_covered_dispatches": 0,
            "query_length_total_dispatches": 3840,
            "query_head_pair_supported": False,
        },
    }
    return {
        "probe_contract": json.dumps(contract, sort_keys=True).encode(),
        "raw_nz_pilot_summary": json.dumps(nz, sort_keys=True).encode(),
        "nd_repack_pilot_summary": json.dumps(nd, sort_keys=True).encode(),
        "probe_implementation": b"# pinned probe\n",
        "v2_compatibility_audit": json.dumps(v2, sort_keys=True).encode(),
    }


def _install_fixture(monkeypatch, *, mutate=None):
    blobs = _records()
    if mutate is not None:
        mutate(blobs)
    specs = []
    for identifier, blob in blobs.items():
        revision = (
            audit.M1_REVISION
            if identifier == "v2_compatibility_audit"
            else audit.M0_REVISION
        )
        specs.append(
            audit.RecordSpec(
                identifier,
                revision,
                f"fixture/{identifier}.json",
                hashlib.sha256(blob).hexdigest(),
                json_record=identifier != "probe_implementation",
            )
        )
    monkeypatch.setattr(audit, "RECORDS", tuple(specs))
    monkeypatch.setattr(audit, "_resolve_revision", lambda repo, revision: revision)
    monkeypatch.setattr(
        audit,
        "_read_git_blob",
        lambda repo, revision, path: blobs[
            path.removeprefix("fixture/").removesuffix(".json")
        ],
    )
    monkeypatch.setattr(
        audit, "_origin", lambda repo: "https://example.test/research.git"
    )
    return blobs


def test_accepts_exact_summary_integrity(monkeypatch, tmp_path):
    _install_fixture(monkeypatch)
    receipt = audit.audit_parent_operator_evidence(tmp_path)
    assert receipt["status"] == "SUPPORTED_BY_PINNED_SUMMARIES"
    assert receipt["runtime_acceptance"] is False
    assert len(receipt["records"]) == 5
    assert len(receipt["findings"]) == 3
    assert not receipt["errors"]


def test_hash_drift_fails_closed(monkeypatch, tmp_path):
    blobs = _install_fixture(monkeypatch)
    blobs["probe_implementation"] += b"# drift\n"
    receipt = audit.audit_parent_operator_evidence(tmp_path)
    assert receipt["status"] == "INCONCLUSIVE"
    assert any(
        "hash drift for probe_implementation" in item for item in receipt["errors"]
    )
    assert not receipt["findings"]


def test_semantic_drift_fails_closed_even_with_matching_hash(monkeypatch, tmp_path):
    def mutate(blobs):
        summary = json.loads(blobs["nd_repack_pilot_summary"])
        summary["native_status"] = "PASS"
        blobs["nd_repack_pilot_summary"] = json.dumps(summary, sort_keys=True).encode()

    _install_fixture(monkeypatch, mutate=mutate)
    receipt = audit.audit_parent_operator_evidence(tmp_path)
    assert receipt["status"] == "INCONCLUSIVE"
    assert any("native_status" in item for item in receipt["errors"])


def test_exclusive_receipt_write(monkeypatch, tmp_path):
    _install_fixture(monkeypatch)
    receipt = audit.audit_parent_operator_evidence(tmp_path)
    output = tmp_path / "receipt.json"
    audit._write_exclusive(output, receipt)
    assert json.loads(output.read_text()) == receipt
    try:
        audit._write_exclusive(output, receipt)
    except FileExistsError:
        pass
    else:
        raise AssertionError("receipt writer overwrote an existing file")
