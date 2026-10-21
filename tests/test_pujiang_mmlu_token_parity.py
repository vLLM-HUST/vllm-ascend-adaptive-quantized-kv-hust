import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import pujiang_mmlu_token_parity as m
from scripts import validate_pujiang_mmlu_pair as pair
from scripts.prepare_pujiang_mmlu_source import GENERATION


def store(root, name, value):
    raw = m._canonical(value)
    path = root / name
    path.write_bytes(raw)
    return {"path": name, "sha256": m._sha(raw), "bytes": len(raw)}


@pytest.fixture
def cases():
    result = []
    for index in range(2):
        request = {
            "model": "synthetic-alias",
            "messages": [{"role": "user", "content": f"synthetic question {index}"}],
            **copy.deepcopy(GENERATION),
            "stream": False,
        }
        ids = list(range(16 + index))
        result.append(
            {
                "case_id": f"synthetic-{index}",
                "candidate_request": request,
                "request_sha256": m._sha(m._canonical(request)),
                "prompt": ids,
                "prompt_tokens": len(ids),
                "prompt_sha256": m._sha(m._canonical(ids)),
                "rendered_prompt_sha256": "a" * 64,
                "max_tokens": request["max_tokens"],
                "ignore_eos": False,
                "option_count": 2,
                "answer": "B",
                "category": "fixture",
            }
        )
    return result


@pytest.fixture
def tokenizer():
    return {
        "repository": "Qwen/Qwen3.5-35B-A3B",
        "revision": "1" * 40,
        "files_sha256": {
            name: "2" * 64
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        },
        "class": "SyntheticTokenizer",
        "library_versions": {
            "transformers": "fixture",
            "tokenizers": "fixture",
            "jinja2": "fixture",
        },
        "scope": "local-text-only-not-serving-runtime",
    }


def make_receipt(root, cases, tokenizer, arm="B0", bundle_hash="b" * 64):
    identity = {
        "schema_version": m.IDENTITY_SCHEMA,
        "tokenizer": {field: tokenizer[field] for field in m.TOKENIZER_FIELDS},
        "served_model_name": "synthetic-alias",
        "context_tokens": 32768,
        "server_configuration_sha256": "c" * 64,
        "release_manifest_sha256": "d" * 64,
    }
    rows = []
    for index, case in enumerate(cases):
        rows.append(
            {
                "case_id": case["case_id"],
                "completion_request_sha256": case["request_sha256"],
                "request": store(
                    root, f"request-{index}.json", m.tokenize_request(case)
                ),
                "attempts": 1,
                "result": {
                    "kind": "http",
                    "status": 200,
                    "body": store(
                        root,
                        f"response-{index}.json",
                        {
                            "count": len(case["prompt"]),
                            "tokens": case["prompt"],
                            "max_model_len": 32768,
                            "token_strs": None,
                        },
                    ),
                },
            }
        )
    return {
        "schema_version": m.SCHEMA,
        "arm": arm,
        "evidence_kind": "development-fixture",
        "scorer_bundle_sha256": bundle_hash,
        "tokenize_reference_revision": m.TOKENIZE_REFERENCE,
        "request_method": "POST",
        "request_path": "/tokenize",
        "tokenizer_declaration": store(root, "declaration.json", identity),
        "requests": rows,
    }


def audit(receipt, cases, root, tokenizer):
    return m.audit_stored_token_parity(
        receipt, cases, root, "B0", "b" * 64, "development-fixture", tokenizer, 32768
    )


def rewrite(root, descriptor, mutate):
    value = json.loads((root / descriptor["path"]).read_bytes())
    mutate(value)
    descriptor.update(store(root, descriptor["path"], value))


def test_matching_full_ids_do_not_establish_server_origin(tmp_path, cases, tokenizer):
    receipt = make_receipt(tmp_path, cases, tokenizer)
    original = copy.deepcopy(receipt)
    result = audit(receipt, cases, tmp_path, tokenizer)
    assert receipt == original
    assert result["case_count"] == 2 and result["all_cases_retained"]
    assert result["all_token_ids_match"] and result["failure_counts"] == {}
    assert result["status"] == "stored_token_ids_match_origin_unverified"
    for key in (
        "tokenizer_declaration_origin_verified",
        "release_signature_verified",
        "serving_tokenizer_verified",
        "transport_verified",
        "execution_authorized",
    ):
        assert result[key] is False


@pytest.mark.parametrize("mutation", ["same-length", "shorter", "longer", "reordered"])
def test_count_parity_is_insufficient_and_full_cases_retained(
    tmp_path, cases, tokenizer, mutation
):
    receipt = make_receipt(tmp_path, cases, tokenizer)

    def mutate(value):
        if mutation == "same-length":
            value["tokens"][3] = 999
        elif mutation == "shorter":
            value["tokens"].pop()
        elif mutation == "longer":
            value["tokens"].append(999)
        else:
            value["tokens"].reverse()
        value["count"] = len(value["tokens"])

    rewrite(tmp_path, receipt["requests"][0]["result"]["body"], mutate)
    result = audit(receipt, cases, tmp_path, tokenizer)
    assert result["case_count"] == 2 and len(result["outcomes"]) == 2
    assert result["failure_counts"] == {"token-ids-mismatch": 1}
    assert not result["all_token_ids_match"] and result["outcomes"][1]["match"]
    assert (
        result["outcomes"][0]["first_mismatch_index"]
        == {"same-length": 3, "shorter": 15, "longer": 16, "reordered": 0}[mutation]
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("bool-token", "invalid-token-ids-or-count"),
        ("float-token", "invalid-token-ids-or-count"),
        ("negative-token", "invalid-token-ids-or-count"),
        ("count", "invalid-token-ids-or-count"),
        ("bool-count", "invalid-token-ids-or-count"),
        ("token-strs", "invalid-token-ids-or-count"),
        ("only-count", "invalid-tokenize-response"),
        ("extra", "invalid-tokenize-response"),
        ("context", "context-mismatch"),
        ("bool-context", "context-mismatch"),
    ],
)
def test_bad_response_is_a_retained_failure(
    tmp_path, cases, tokenizer, mutation, error
):
    receipt = make_receipt(tmp_path, cases, tokenizer)

    def mutate(value):
        if mutation in ("bool-token", "float-token", "negative-token"):
            value["tokens"][0] = {
                "bool-token": True,
                "float-token": 0.0,
                "negative-token": -1,
            }[mutation]
        elif mutation in ("count", "bool-count"):
            value["count"] = True if mutation == "bool-count" else 999
        elif mutation == "token-strs":
            value["token_strs"] = ["unexpected"]
        elif mutation == "only-count":
            value.pop("tokens")
        elif mutation == "extra":
            value["serving_tokenizer_verified"] = True
        else:
            value["max_model_len"] = True if mutation == "bool-context" else 8192

    rewrite(tmp_path, receipt["requests"][0]["result"]["body"], mutate)
    result = audit(receipt, cases, tmp_path, tokenizer)
    assert result["failure_counts"] == {error: 1} and result["case_count"] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "timeout",
        "connection-error",
        "not-attempted",
        "response-too-large",
        "http-error",
        "bad-json",
    ],
)
def test_transport_failures_block_parity_without_deleting_tasks(
    tmp_path, cases, tokenizer, kind
):
    receipt = make_receipt(tmp_path, cases, tokenizer)
    row = receipt["requests"][0]
    if kind == "http-error":
        row["result"]["status"] = 503
    elif kind == "bad-json":
        raw = b"not-json"
        descriptor = row["result"]["body"]
        (tmp_path / descriptor["path"]).write_bytes(raw)
        descriptor.update(sha256=m._sha(raw), bytes=len(raw))
    else:
        row["result"] = {"kind": kind}
        row["attempts"] = 0 if kind == "not-attempted" else 1
    result = audit(receipt, cases, tmp_path, tokenizer)
    assert result["failure_counts"] == {
        "invalid-json-response" if kind == "bad-json" else kind: 1
    }
    assert result["case_count"] == 2 and not result["all_token_ids_match"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "arm",
        "reference",
        "measured",
        "bundle",
        "method",
        "path",
        "retry",
        "completion-hash",
    ],
)
def test_invalid_receipt_integrity_fails_closed(tmp_path, cases, tokenizer, mutation):
    receipt = make_receipt(tmp_path, cases, tokenizer)
    if mutation == "missing":
        receipt["requests"].pop()
    elif mutation == "duplicate":
        receipt["requests"][1] = copy.deepcopy(receipt["requests"][0])
    elif mutation in ("arm", "reference", "measured", "bundle", "method", "path"):
        key = {
            "arm": "arm",
            "reference": "tokenize_reference_revision",
            "measured": "evidence_kind",
            "bundle": "scorer_bundle_sha256",
            "method": "request_method",
            "path": "request_path",
        }[mutation]
        receipt[key] = "invalid"
    elif mutation == "retry":
        receipt["requests"][0]["attempts"] = 2
    else:
        receipt["requests"][0]["completion_request_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        audit(receipt, cases, tmp_path, tokenizer)


@pytest.mark.parametrize(
    "mutation",
    ["messages", "thinking", "generation", "special", "continue", "template", "tools"],
)
def test_rehashed_tokenize_request_changes_are_rejected(
    tmp_path, cases, tokenizer, mutation
):
    receipt = make_receipt(tmp_path, cases, tokenizer)

    def mutate(value):
        if mutation == "messages":
            value["messages"][0]["content"] = "changed"
        elif mutation == "thinking":
            value["chat_template_kwargs"]["enable_thinking"] = True
        else:
            key = {
                "generation": "add_generation_prompt",
                "special": "add_special_tokens",
                "continue": "continue_final_message",
                "template": "chat_template",
                "tools": "tools",
            }[mutation]
            value[key] = not value[key] if isinstance(value[key], bool) else "changed"

    rewrite(tmp_path, receipt["requests"][0]["request"], mutate)
    with pytest.raises(ValueError, match="full source-bound projection"):
        audit(receipt, cases, tmp_path, tokenizer)


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "files",
        "libraries",
        "class",
        "alias",
        "context",
        "manifest",
        "claim",
    ],
)
def test_rehashed_declaration_does_not_bypass_identity_binding(
    tmp_path, cases, tokenizer, mutation
):
    receipt = make_receipt(tmp_path, cases, tokenizer)

    def mutate(value):
        if mutation in ("revision", "class"):
            value["tokenizer"][mutation] = "other"
        elif mutation in ("files", "libraries"):
            key = "files_sha256" if mutation == "files" else "library_versions"
            value["tokenizer"][key][next(iter(value["tokenizer"][key]))] = "changed"
        elif mutation == "alias":
            value["served_model_name"] = "other"
        elif mutation == "context":
            value["context_tokens"] = True
        elif mutation == "manifest":
            value["release_manifest_sha256"] = "missing"
        else:
            value["execution_authorized"] = True

    rewrite(tmp_path, receipt["tokenizer_declaration"], mutate)
    with pytest.raises(ValueError):
        audit(receipt, cases, tmp_path, tokenizer)


@pytest.mark.parametrize("mutation", ["corrupt", "symlink", "escape"])
def test_raw_artifact_integrity_is_enforced(tmp_path, cases, tokenizer, mutation):
    receipt = make_receipt(tmp_path, cases, tokenizer)
    descriptor = receipt["requests"][0]["result"]["body"]
    path = tmp_path / descriptor["path"]
    if mutation == "corrupt":
        path.write_bytes(b"changed")
    elif mutation == "symlink":
        other = tmp_path / "other.json"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    else:
        descriptor["path"] = "../escape.json"
    with pytest.raises(ValueError):
        audit(receipt, cases, tmp_path, tokenizer)


@pytest.mark.parametrize("mismatch", [False, True])
def test_pair_integration_blocks_scoring_on_any_token_mismatch(
    tmp_path, cases, tokenizer, monkeypatch, mismatch
):
    envelope, paths, roots, calls = prepare_pair(
        tmp_path, cases, tokenizer, monkeypatch
    )
    if mismatch:
        path = roots / "B1/receipt.json"
        receipt = json.loads(path.read_bytes())
        rewrite(
            roots / "B1",
            receipt["requests"][0]["result"]["body"],
            lambda value: value["tokens"].__setitem__(0, 999),
        )
        path.write_bytes(m._canonical(receipt))
        record = json.loads(envelope.read_bytes())
        record["stored_token_parity_sha256"]["B1"] = m._sha(path.read_bytes())
        envelope.write_bytes(m._canonical(record))
    report = pair.score_pair(
        envelope,
        paths["cases"],
        paths["B0"],
        paths["B1"],
        tmp_path,
        source_inputs={name: tmp_path for name in pair.SOURCE_INPUTS},
        stored_token_parity_root=roots,
    )
    assert len(calls) == (0 if mismatch else 2)
    assert (report["offline_quality"] is None) == mismatch
    assert not report["serving_tokenizer_verified"] and not any(
        report["claim_boundary"].values()
    )
    assert report["stored_token_parity"]["B0"]["case_count"] == 2
    assert report["stored_token_parity"]["B1"]["all_token_ids_match"] is not mismatch


def prepare_pair(tmp_path, cases, tokenizer, monkeypatch):
    # Synthetic source/scorer doubles exercise integration, not source provenance.
    binding = {"tokenizer": tokenizer, "context_audit": {"context_tokens": 32768}}
    monkeypatch.setattr(pair, "verify_source_bundle", lambda *_args: (cases, binding))
    calls = []

    def score(rows, manifest):
        calls.append((rows, manifest))
        return {"eligible": {}, "ineligible": {}, "outcomes": []}

    monkeypatch.setattr(
        pair, "load_scorer", lambda _path: SimpleNamespace(score_results=score)
    )
    outcomes = [
        {
            "case_id": case["case_id"],
            **{field: case[field] for field in pair.SOURCE_REQUEST_FIELDS},
            "success": False,
        }
        for case in cases
    ]
    paths = {}
    for name, value in (("cases", cases), ("B0", outcomes), ("B1", outcomes)):
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(m._canonical(value))
    record = {
        "schema_version": pair.SOURCE_SCHEMA,
        "scope": "verified-local-source-selection",
        "dataset": "mmlu-pro",
        "dataset_revision": pair.DATASET_REVISION,
        "benchmark_revision": pair.BENCHMARK_REVISION,
        "scorer_source_sha256": pair.SCORER_SHA256,
        "task_count": len(cases),
        "generation": GENERATION,
        "evidence_kind": "development-fixture",
        "files_sha256": {
            name: m._sha(path.read_bytes()) for name, path in paths.items()
        },
        "stored_token_parity_sha256": {},
    }
    roots = tmp_path / "parity"
    for arm in ("B0", "B1"):
        root = roots / arm
        root.mkdir(parents=True)
        receipt = make_receipt(
            root, cases, tokenizer, arm, record["files_sha256"]["cases"]
        )
        (root / "receipt.json").write_bytes(m._canonical(receipt))
        record["stored_token_parity_sha256"][arm] = m._sha(
            (root / "receipt.json").read_bytes()
        )
    envelope = tmp_path / "envelope.json"
    envelope.write_bytes(m._canonical(record))
    return envelope, paths, roots, calls


def test_bound_parity_cannot_be_skipped_by_omitting_argument(
    tmp_path, cases, tokenizer, monkeypatch
):
    envelope, paths, _roots, _calls = prepare_pair(
        tmp_path, cases, tokenizer, monkeypatch
    )
    with pytest.raises(ValueError, match="Envelope binds token parity"):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            source_inputs={name: tmp_path for name in pair.SOURCE_INPUTS},
        )


@pytest.mark.parametrize(
    "mutation", ["hash", "missing-arm", "omitted-hashes", "legacy"]
)
def test_pair_requires_bound_source_and_exact_two_arm_receipts(
    tmp_path, cases, tokenizer, monkeypatch, mutation
):
    envelope, paths, roots, calls = prepare_pair(
        tmp_path, cases, tokenizer, monkeypatch
    )
    record = json.loads(envelope.read_bytes())
    if mutation == "hash":
        record["stored_token_parity_sha256"]["B1"] = "0" * 64
    elif mutation == "missing-arm":
        record["stored_token_parity_sha256"].pop("B1")
    elif mutation == "omitted-hashes":
        record.pop("stored_token_parity_sha256")
    else:
        record["schema_version"] = pair.SCHEMA
        record["scope"] = "supplied-manifest-only"
    envelope.write_bytes(m._canonical(record))
    with pytest.raises(ValueError):
        pair.score_pair(
            envelope,
            paths["cases"],
            paths["B0"],
            paths["B1"],
            tmp_path,
            source_inputs=None
            if mutation == "legacy"
            else {name: tmp_path for name in pair.SOURCE_INPUTS},
            stored_token_parity_root=roots,
        )
    assert not calls


@pytest.mark.parametrize(
    "mutation", ["bool", "count", "hash", "alias", "extra-request"]
)
def test_inconsistent_source_cases_rejected(tmp_path, cases, tokenizer, mutation):
    receipt = make_receipt(tmp_path, cases, tokenizer)
    if mutation == "bool":
        cases[0]["prompt"][1] = True
    elif mutation == "count":
        cases[0]["prompt_tokens"] = True
    elif mutation == "hash":
        cases[0]["request_sha256"] = "0" * 64
    elif mutation == "alias":
        cases[1]["candidate_request"]["model"] = "other"
    else:
        cases[0]["candidate_request"]["tools"] = [{"type": "function"}]
        cases[0]["request_sha256"] = m._sha(m._canonical(cases[0]["candidate_request"]))
    with pytest.raises(ValueError):
        audit(receipt, cases, tmp_path, tokenizer)


@pytest.mark.parametrize("existing", [False, True])
def test_cli_parity_failure_saves_diagnostics_and_preserves_existing_output(
    tmp_path, cases, tokenizer, monkeypatch, existing
):
    envelope, paths, roots, _calls = prepare_pair(
        tmp_path, cases, tokenizer, monkeypatch
    )
    receipt_path = roots / "B1/receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    rewrite(
        roots / "B1",
        receipt["requests"][0]["result"]["body"],
        lambda value: value["tokens"].__setitem__(0, 999),
    )
    receipt_path.write_bytes(m._canonical(receipt))
    record = json.loads(envelope.read_bytes())
    record["stored_token_parity_sha256"]["B1"] = m._sha(receipt_path.read_bytes())
    envelope.write_bytes(m._canonical(record))
    output = tmp_path / "report.json"
    if existing:
        output.write_bytes(b"preserve-original")
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
        str(tmp_path),
        "--output",
        str(output),
        "--stored-token-parity-root",
        str(roots),
    ]
    for name in pair.SOURCE_INPUTS:
        argv.extend(("--" + name, str(tmp_path)))
    monkeypatch.setattr("sys.argv", argv)
    assert pair.main() == 1
    if existing:
        assert output.read_bytes() == b"preserve-original"
    else:
        report = json.loads(output.read_bytes())
        assert report["offline_quality"] is None
        assert report["stored_token_parity"]["B1"]["failure_counts"] == {
            "token-ids-mismatch": 1
        }
        assert output.stat().st_mode & 0o777 == 0o600


def test_projection_matches_exact_upstream_protocol_ast(cases):
    root = os.environ.get("AQKV_TEST_TOKENIZE_REFERENCE")
    if not root:
        pytest.skip(
            "Set AQKV_TEST_TOKENIZE_REFERENCE for the pinned upstream source check"
        )
    sources = {}
    for name, digest in m.TOKENIZE_SOURCE_FILES.items():
        raw = (Path(root) / ("vllm-tokenize-" + name)).read_bytes()
        assert m._sha(raw) == digest
        sources[name] = ast.parse(raw)
    classes = {
        node.name: node
        for node in sources["protocol.py"].body
        if isinstance(node, ast.ClassDef)
    }
    fields = {
        node.target.id
        for node in classes["TokenizeChatRequest"].body
        if isinstance(node, ast.AnnAssign)
    }
    assert set(m.tokenize_request(cases[0])) <= fields
    response_fields = {
        node.target.id
        for node in classes["TokenizeResponse"].body
        if isinstance(node, ast.AnnAssign)
    }
    assert response_fields == {"count", "tokens", "max_model_len", "token_strs"}
