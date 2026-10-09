import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_bf16_c8_reachability as audit


@pytest.mark.parametrize(
    "revision",
    ["origin/main", "1138283", "A" * 40, "g" * 40, "a" * 39],
)
def test_revision_must_be_a_full_lowercase_commit_sha(revision: str) -> None:
    with pytest.raises(audit.AuditInconclusive):
        audit.require_exact_revision(revision)


def test_expected_repository_origins_are_accepted() -> None:
    assert (
        audit.canonical_repository_origin(
            "https://github.com/vLLM-HUST/vllm-ascend-hust.git"
        )
        == audit.ASCEND_REPOSITORY
    )
    assert (
        audit.canonical_repository_origin(
            "git@github.com:vLLM-HUST/vllm-ascend-hust.git"
        )
        == audit.ASCEND_REPOSITORY
    )


def test_wrong_repository_origin_fails_closed() -> None:
    with pytest.raises(audit.AuditInconclusive, match="unexpected repository"):
        audit.canonical_repository_origin(
            "https://github.com/example/vllm-ascend-hust.git"
        )


def test_required_source_fragment_is_structural() -> None:
    function = ast.parse(
        """
def configure(config):
    config.calculate_kv_scales = False
"""
    ).body[0]

    audit._require_text(function, "config.calculate_kv_scales = False")
    with pytest.raises(audit.AuditInconclusive, match="source fragments"):
        audit._require_text(function, "config.calculate_kv_scales = True")


def test_retained_bf16_manifest_has_no_modelslim_description() -> None:
    assert "config.json" in audit.BF16_MODEL_FILES
    assert "model.safetensors.index.json" in audit.BF16_MODEL_FILES
    assert "quant_model_description.json" not in audit.BF16_MODEL_FILES


def test_success_receipt_is_deterministic(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        audit,
        "_git",
        lambda repo, *args: b"https://github.com/vLLM-HUST/vllm-ascend-hust.git\n",
    )
    monkeypatch.setattr(
        audit,
        "load_source",
        lambda repo, revision, path: audit.SourceUnit(
            path=path,
            text="",
            sha256=f"sha256:{path}",
        ),
    )
    monkeypatch.setattr(audit, "analyze", lambda units: [])

    first = audit.audit(tmp_path)
    second = audit.audit(tmp_path)

    assert first == second
    assert first["runtime_compatible"] is False
    assert first["model"]["has_modelslim_description"] is False
    assert "generated_at" not in first
