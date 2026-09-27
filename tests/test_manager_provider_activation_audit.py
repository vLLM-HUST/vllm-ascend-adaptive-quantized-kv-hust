import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_manager_provider_activation as audit


@pytest.mark.parametrize(
    "revision",
    ["main", "cf1ea71", "A" * 40, "g" * 40, "a" * 39],
)
def test_revision_must_be_a_full_lowercase_commit_sha(revision: str) -> None:
    with pytest.raises(audit.AuditInconclusive):
        audit.require_exact_revision(revision)


def test_repository_origin_must_match_expected_repository() -> None:
    assert (
        audit.canonical_repository_origin(
            "git@github.com:vLLM-HUST/extension-manager.git",
            audit.MANAGER_REPOSITORY,
        )
        == audit.MANAGER_REPOSITORY
    )
    with pytest.raises(audit.AuditInconclusive, match="unexpected repository"):
        audit.canonical_repository_origin(
            "https://github.com/example/extension-manager.git",
            audit.MANAGER_REPOSITORY,
        )


def test_annotated_fields_are_structural() -> None:
    class_node = ast.parse(
        """
class ProviderConfig:
    layer_name: str
    head_size: int

    def method(self):
        pass
"""
    ).body[0]
    assert audit._annotated_fields(class_node) == {"layer_name", "head_size"}


def test_module_string_set_is_structural() -> None:
    node, values = audit._module_string_set(
        '_REQUIRED_KEYS = {"schema_version", "profile_sha256"}',
        "_REQUIRED_KEYS",
    )

    assert isinstance(node, ast.Assign)
    assert values == {"schema_version", "profile_sha256"}


def test_provider_owned_attestation_has_a_canonical_json_carrier() -> None:
    assert "provider_config_json" in audit.EXPECTED_PROVIDER_CONFIG_FIELDS
    assert "model" in audit.EXPECTED_PROVIDER_CONFIG_FIELDS
    assert "tensor_parallel_size" in audit.EXPECTED_PROVIDER_CONFIG_FIELDS
    assert "profile_sha256" in audit.REQUIRED_PROVIDER_CONFIG_KEYS


def test_success_receipt_is_deterministic(monkeypatch, tmp_path) -> None:
    host_repo = tmp_path / "host"
    manager_repo = tmp_path / "manager"
    plugin_repo = tmp_path / "plugin"
    manifest = plugin_repo / audit.PLUGIN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"implementation":[{"status":"import_only"}],"activation":{}}',
        encoding="utf-8",
    )
    provider_config = plugin_repo / audit.PLUGIN_PROVIDER_CONFIG
    provider_config.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        audit,
        "_git",
        lambda repo, *args: (
            b"https://github.com/vLLM-HUST/vllm-ascend-hust.git\n"
            if repo == host_repo
            else b"https://github.com/vLLM-HUST/extension-manager.git\n"
        ),
    )
    monkeypatch.setattr(
        audit,
        "load_source",
        lambda repo, repository, revision, path: audit.SourceUnit(
            repository, revision, path, "", f"sha256:{path}"
        ),
    )
    monkeypatch.setattr(audit, "analyze", lambda units, payload: [])

    first = audit.audit(host_repo, manager_repo, plugin_repo)
    second = audit.audit(host_repo, manager_repo, plugin_repo)

    assert first == second
    assert first["activation_status"] == (
        "HOST_CONFIG_AND_PLUGIN_SCHEMA_READY_PROFILE_AND_PROVIDER_STILL_MISSING"
    )
    assert first["runtime_compatible"] is False
    assert first["plugin_provider_config"]["path"] == audit.PLUGIN_PROVIDER_CONFIG
    assert "generated_at" not in first
