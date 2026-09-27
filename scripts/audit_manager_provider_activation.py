#!/usr/bin/env python3
"""Audit the Manager-to-Host activation contract for the C8 provider."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOST_REPOSITORY = "vLLM-HUST/vllm-ascend-hust"
HOST_REVISION = "084f70f50dfcdf2daf66b3a31813bc982c2d1d09"
MANAGER_REPOSITORY = "vLLM-HUST/extension-manager"
MANAGER_REVISION = "cf1ea71e3e2cb81ab06267ef05eddb3e580ea20b"
PLUGIN_REPOSITORY = "vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust"
SCHEMA_VERSION = "vllm-ascend-c8-manager-provider-activation-audit/v1"

MANAGER_CLI = "src/vllm_hust_ext/cli.py"
MANAGER_PROVIDER = "src/vllm_hust_ext/providers/vllm.py"
MANAGER_TEST_CLI = "tests/test_cli.py"
HOST_CONFIG = "vllm_ascend/ascend_config.py"
HOST_PROVIDER = "vllm_ascend/attention/continuing_prefill.py"
HOST_ATTENTION = "vllm_ascend/attention/attention_v1.py"
PLUGIN_MANIFEST = "src/vllm_ascend_adaptive_quantized_kv/vllm-hust-extension-v0.2.json"
PLUGIN_PROVIDER_CONFIG = "src/vllm_ascend_adaptive_quantized_kv/provider_config.py"

EXPECTED_PROVIDER_CONFIG_FIELDS = {
    "layer_name",
    "model",
    "model_revision",
    "tensor_parallel_rank",
    "tensor_parallel_size",
    "num_heads",
    "num_kv_heads",
    "head_size",
    "scale",
    "kv_cache_dtype",
    "provider_config_json",
}
REQUIRED_PROVIDER_CONFIG_KEYS = {
    "expected_model",
    "expected_model_revision",
    "model_config_sha256",
    "profile_sha256",
    "zero_offsets_attested",
}
EXPECTED_PLUGIN_PROVIDER_CONFIG_KEYS = {
    "schema_version",
    "expected_model",
    "expected_model_revision",
    "model_config_sha256",
    "profile_path",
    "profile_sha256",
    "calibration_provenance",
    "supported_soc",
    "full_attention_layer_ids",
    "global_channels_per_tensor",
    "tp_size",
    "tp_local_channels_per_tensor",
    "zero_offsets_attested",
    "kv_layout",
    "cache_write_owner",
}


class AuditInconclusive(RuntimeError):
    """Raised when a pinned source no longer has the expected structure."""


@dataclass(frozen=True, slots=True)
class SourceUnit:
    repository: str
    revision: str
    path: str
    text: str
    sha256: str


def require_exact_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AuditInconclusive(
            f"revision must be a full lowercase 40-character commit SHA: {revision!r}"
        )
    return revision


def canonical_repository_origin(origin: str, expected: str) -> str:
    value = origin.strip().rstrip("/")
    repository = None
    for pattern in (
        r"https://github\.com/(?P<repository>[^?#]+)",
        r"ssh://git@github\.com/(?P<repository>[^?#]+)",
        r"git@github\.com:(?P<repository>[^?#]+)",
    ):
        match = re.fullmatch(pattern, value)
        if match is not None:
            repository = match.group("repository").removesuffix(".git")
            break
    if repository is None:
        raise AuditInconclusive(f"unsupported GitHub origin: {origin!r}")
    if repository.lower() != expected.lower():
        raise AuditInconclusive(
            f"unexpected repository origin: expected {expected}, got {repository}"
        )
    return expected


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def load_source(repo: Path, repository: str, revision: str, path: str) -> SourceUnit:
    require_exact_revision(revision)
    resolved = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.decode().strip() != revision:
        raise AuditInconclusive(f"revision did not resolve exactly: {revision}")
    raw = _git(repo, "show", f"{revision}:{path}")
    return SourceUnit(
        repository=repository,
        revision=revision,
        path=path,
        text=raw.decode(),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def find_qualified_def(source: str, qualified_name: str) -> ast.AST:
    body: list[ast.stmt] = ast.parse(source).body
    node: ast.AST | None = None
    for part in qualified_name.split("."):
        matches = [
            item
            for item in body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == part
        ]
        if len(matches) != 1:
            raise AuditInconclusive(
                f"expected exactly one definition for {qualified_name!r}"
            )
        node = matches[0]
        body = node.body
    assert node is not None
    return node


def source_ref(unit: SourceUnit, symbol: str, node: ast.AST) -> dict[str, Any]:
    return {
        "repository": unit.repository,
        "revision": unit.revision,
        "path": unit.path,
        "symbol": symbol,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "blob_sha256": unit.sha256,
    }


def _render(node: ast.AST) -> str:
    return ast.unparse(node)


def _require_text(node: ast.AST, *needles: str) -> None:
    rendered = _render(node)
    missing = [needle for needle in needles if needle not in rendered]
    if missing:
        raise AuditInconclusive(f"expected source fragments are absent: {missing}")


def _annotated_fields(class_node: ast.AST) -> set[str]:
    if not isinstance(class_node, ast.ClassDef):
        raise AuditInconclusive("expected a class definition")
    return {
        item.target.id
        for item in class_node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }


def _module_string_set(source: str, name: str) -> tuple[ast.AST, set[str]]:
    matches: list[ast.AST] = []
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            matches.append(node)
    if len(matches) != 1:
        raise AuditInconclusive(f"expected exactly one assignment for {name!r}")
    node = matches[0]
    value_node = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
    try:
        value = ast.literal_eval(value_node)
    except (TypeError, ValueError) as error:
        raise AuditInconclusive(f"{name} is not a literal value") from error
    if not isinstance(value, set) or not all(isinstance(item, str) for item in value):
        raise AuditInconclusive(f"{name} is not a literal string set")
    return node, value


def analyze(
    units: dict[str, SourceUnit], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    manager_cli = units[f"manager:{MANAGER_CLI}"]
    manager_test_cli = units[f"manager:{MANAGER_TEST_CLI}"]
    activation_config = find_qualified_def(manager_cli.text, "_activation_config")
    merge_command_config = find_qualified_def(manager_cli.text, "_merge_command_config")
    merge_test = find_qualified_def(
        manager_test_cli.text, "test_run_merges_existing_additional_config"
    )
    conflict_test = find_qualified_def(
        manager_test_cli.text, "test_run_rejects_activation_conflict"
    )
    _require_text(
        activation_config,
        "bundle.manifest.activation.additional_config",
        "enabled Bundles disagree on additional_config key",
    )
    _require_text(
        merge_command_config,
        "--additional-config",
        "plugin activation conflicts with additional_config keys",
        "existing.update(activation)",
    )
    _require_text(
        merge_test,
        "_merge_command_config",
        "victim_selector_plugin",
        "bidkv",
    )
    _require_text(
        conflict_test,
        "pytest.raises(ValueError, match='conflicts')",
        "_merge_command_config",
    )
    findings.append(
        {
            "id": "manager-merges-static-activation-with-operator-config",
            "status": "confirmed-in-pinned-source-and-tests",
            "summary": (
                "The Manager merges static manifest activation into an existing "
                "operator-supplied --additional-config object and rejects conflicting "
                "keys. This can carry a fixed provider factory plus user-owned profile "
                "settings without inventing a custom Host Provider."
            ),
            "evidence": [
                source_ref(manager_cli, "_activation_config", activation_config),
                source_ref(manager_cli, "_merge_command_config", merge_command_config),
                source_ref(
                    manager_test_cli,
                    "test_run_merges_existing_additional_config",
                    merge_test,
                ),
                source_ref(
                    manager_test_cli,
                    "test_run_rejects_activation_conflict",
                    conflict_test,
                ),
            ],
        }
    )

    plugin_config = units[f"plugin:{PLUGIN_PROVIDER_CONFIG}"]
    required_keys_node, plugin_config_keys = _module_string_set(
        plugin_config.text, "_REQUIRED_KEYS"
    )
    if plugin_config_keys != EXPECTED_PLUGIN_PROVIDER_CONFIG_KEYS:
        raise AuditInconclusive(
            "plugin provider config keys changed: "
            f"expected {sorted(EXPECTED_PLUGIN_PROVIDER_CONFIG_KEYS)}, "
            f"got {sorted(plugin_config_keys)}"
        )
    from_json = find_qualified_def(
        plugin_config.text, "ProviderActivationConfig.from_json"
    )
    _require_text(
        from_json,
        "object_pairs_hook=_unique_object",
        "parse_constant=_reject_constant",
        "zero_offsets_attested must be true",
        "global channels must equal tp_size * TP-local channels",
        "_require_sha256(payload, 'profile_sha256')",
    )
    validate_runtime = find_qualified_def(
        plugin_config.text, "ProviderActivationConfig.validate_runtime"
    )
    _require_text(
        validate_runtime,
        "runtime model mismatch",
        "runtime model revision",
        "runtime TP size",
        "runtime TP rank",
        "full-attention layer",
        "TP-local KV channels",
        "torch.int8",
    )
    host_adapter = find_qualified_def(
        plugin_config.text, "runtime_identity_from_host_config"
    )
    _require_text(
        host_adapter,
        "HOST_LAYER_NAME_PATTERN.fullmatch(layer_name)",
        "_host_int(config, 'tensor_parallel_rank')",
        "_host_int(config, 'tensor_parallel_size')",
        "_host_int(config, 'num_kv_heads')",
        "_host_int(config, 'head_size')",
        "str(_host_attribute(config, 'kv_cache_dtype'))",
    )
    validate_host_config = find_qualified_def(
        plugin_config.text, "ProviderActivationConfig.validate_host_config"
    )
    _require_text(
        validate_host_config,
        "runtime_identity_from_host_config(config)",
        "self.validate_runtime(runtime)",
    )
    findings.append(
        {
            "id": "plugin-provider-schema-fails-closed-before-runtime",
            "status": "confirmed-in-worktree-source",
            "summary": (
                "The plugin now owns a strict construction-time provider JSON "
                "schema, derives an unambiguous identity from Host's construction "
                "carrier, and compares that identity with the activation contract. It "
                "rejects malformed or non-finite JSON, contract-key drift, bad "
                "digests, nonzero-offset policy, TP/layer/channel mismatch and "
                "non-INT8 cache identity. Profile loading and provider execution "
                "remain unimplemented."
            ),
            "observed_keys": sorted(plugin_config_keys),
            "evidence": [
                source_ref(plugin_config, "_REQUIRED_KEYS", required_keys_node),
                source_ref(
                    plugin_config,
                    "ProviderActivationConfig.from_json",
                    from_json,
                ),
                source_ref(
                    plugin_config,
                    "ProviderActivationConfig.validate_runtime",
                    validate_runtime,
                ),
                source_ref(
                    plugin_config,
                    "runtime_identity_from_host_config",
                    host_adapter,
                ),
                source_ref(
                    plugin_config,
                    "ProviderActivationConfig.validate_host_config",
                    validate_host_config,
                ),
            ],
        }
    )

    manager_provider = units[f"manager:{MANAGER_PROVIDER}"]
    plan = find_qualified_def(manager_provider.text, "VllmProvider.plan")
    _require_text(
        plan,
        "additional_config = dict(manifest.activation.additional_config)",
        "'user_config': configuration",
        "launch_options = configuration.get('launch_options', {})",
    )
    findings.append(
        {
            "id": "manager-saved-user-config-is-not-generic-host-config",
            "status": "confirmed-in-pinned-source",
            "summary": (
                "The built-in vLLM Provider retains arbitrary saved extension "
                "configuration as user_config, but only supported launch_options and "
                "static manifest additional_config affect the launch command. A "
                "profile supplied through 'extension configure' alone will not reach "
                "AscendConfig."
            ),
            "evidence": [source_ref(manager_provider, "VllmProvider.plan", plan)],
        }
    )

    host_provider = units[f"host:{HOST_PROVIDER}"]
    provider_config = find_qualified_def(
        host_provider.text, "C8ContinuingPrefillProviderConfig"
    )
    provider_fields = _annotated_fields(provider_config)
    if provider_fields != EXPECTED_PROVIDER_CONFIG_FIELDS:
        raise AuditInconclusive(
            "C8 provider config fields changed: "
            f"expected {sorted(EXPECTED_PROVIDER_CONFIG_FIELDS)}, "
            f"got {sorted(provider_fields)}"
        )
    findings.append(
        {
            "id": "host-construction-carries-runtime-identity-and-provider-json",
            "status": "confirmed-in-pinned-candidate-source",
            "summary": (
                "The Host factory config now carries the actual model and revision, "
                "TP rank and size, layer shape and dtype data, plus one canonical "
                "provider-owned JSON object. Artifact digests and zero-offset policy "
                "remain plugin-owned values that the provider must validate from that "
                "JSON during construction."
            ),
            "observed_fields": sorted(provider_fields),
            "required_provider_config_keys": sorted(REQUIRED_PROVIDER_CONFIG_KEYS),
            "evidence": [
                source_ref(
                    host_provider,
                    "C8ContinuingPrefillProviderConfig",
                    provider_config,
                )
            ],
        }
    )

    provider_protocol = find_qualified_def(
        host_provider.text, "C8ContinuingPrefillProvider"
    )
    is_eligible = find_qualified_def(
        host_provider.text, "C8ContinuingPrefillProvider.is_eligible"
    )
    _require_text(
        is_eligible,
        "must be side-effect free",
        "launch device work",
        "synchronize the device",
    )
    findings.append(
        {
            "id": "runtime-eligibility-cannot-read-device-offset-values",
            "status": "confirmed-contract-constraint",
            "summary": (
                "is_eligible must be side-effect free and may not launch or "
                "synchronize device work. Zero-offset value validation belongs in "
                "profile loading or provider construction, not per-request eligibility."
            ),
            "evidence": [
                source_ref(
                    host_provider,
                    "C8ContinuingPrefillProvider",
                    provider_protocol,
                ),
                source_ref(
                    host_provider,
                    "C8ContinuingPrefillProvider.is_eligible",
                    is_eligible,
                ),
            ],
        }
    )

    host_config = units[f"host:{HOST_CONFIG}"]
    ascend_config = find_qualified_def(host_config.text, "AscendConfig")
    ascend_fields = _annotated_fields(ascend_config)
    if "c8_continuing_prefill_provider" not in ascend_fields:
        raise AuditInconclusive("Host no longer exposes the provider factory field")
    provider_option_fields = sorted(
        field
        for field in ascend_fields
        if field.startswith("c8_continuing_prefill_provider")
        and field != "c8_continuing_prefill_provider"
    )
    if provider_option_fields != ["c8_continuing_prefill_provider_config"]:
        raise AuditInconclusive(
            f"Host provider option fields changed: {provider_option_fields}"
        )
    validate_user_input = find_qualified_def(
        host_config.text, "AscendConfig._validate_user_input_ranges"
    )
    _require_text(
        validate_user_input,
        "c8_continuing_prefill_provider_config requires c8_continuing_prefill_provider",
        "json.dumps",
        "allow_nan=False",
        "sort_keys=True",
    )

    host_attention = units[f"host:{HOST_ATTENTION}"]
    configure = find_qualified_def(
        host_attention.text,
        "AscendC8AttentionBackendImpl.configure_c8_continuing_prefill_provider",
    )
    _require_text(
        configure,
        "ascend_config = get_ascend_config()",
        "factory_path = ascend_config.c8_continuing_prefill_provider",
        "C8ContinuingPrefillProviderConfig",
        "model=model_config.model",
        "model_revision=model_config.revision",
        "tensor_parallel_rank=get_tensor_model_parallel_rank()",
        "tensor_parallel_size=get_tensor_model_parallel_world_size()",
        "ascend_config.c8_continuing_prefill_provider_config",
        "allow_nan=False",
        "sort_keys=True",
    )
    findings.append(
        {
            "id": "host-activation-carries-immutable-provider-config",
            "status": "confirmed-in-pinned-candidate-source",
            "summary": (
                "AscendConfig exposes a factory path and a finite JSON provider "
                "configuration. The attention layer canonicalizes that object and "
                "combines it with actual runtime identity before constructing the "
                "provider ahead of graph capture."
            ),
            "provider_option_fields": provider_option_fields,
            "evidence": [
                source_ref(host_config, "AscendConfig", ascend_config),
                source_ref(
                    host_config,
                    "AscendConfig._validate_user_input_ranges",
                    validate_user_input,
                ),
                source_ref(
                    host_attention,
                    "AscendC8AttentionBackendImpl.configure_c8_continuing_prefill_provider",
                    configure,
                ),
            ],
        }
    )

    implementation = manifest.get("implementation")
    activation = manifest.get("activation")
    if not isinstance(implementation, list) or not implementation:
        raise AuditInconclusive("plugin manifest has no implementation carrier")
    if implementation[0].get("status") != "import_only":
        raise AuditInconclusive("plugin manifest is no longer import_only")
    if activation != {
        "entry_points": [],
        "environment": {},
        "additional_config": {},
    }:
        raise AuditInconclusive("plugin activation changed from the inert baseline")
    findings.append(
        {
            "id": "plugin-remains-inert-until-contract-closes",
            "status": "confirmed-in-worktree-manifest",
            "summary": (
                "The plugin remains import_only with empty activation, so installation "
                "and discovery cannot select the incomplete Host provider path."
            ),
        }
    )
    return findings


def audit(host_repo: Path, manager_repo: Path, plugin_repo: Path) -> dict[str, Any]:
    repositories = (
        (host_repo, HOST_REPOSITORY, HOST_REVISION),
        (manager_repo, MANAGER_REPOSITORY, MANAGER_REVISION),
    )
    for repo, repository, revision in repositories:
        origin = _git(repo, "remote", "get-url", "origin").decode()
        canonical_repository_origin(origin, repository)
        require_exact_revision(revision)

    units: dict[str, SourceUnit] = {}
    for path in (HOST_CONFIG, HOST_PROVIDER, HOST_ATTENTION):
        units[f"host:{path}"] = load_source(
            host_repo, HOST_REPOSITORY, HOST_REVISION, path
        )
    for path in (MANAGER_CLI, MANAGER_PROVIDER, MANAGER_TEST_CLI):
        units[f"manager:{path}"] = load_source(
            manager_repo, MANAGER_REPOSITORY, MANAGER_REVISION, path
        )

    plugin_config_path = plugin_repo / PLUGIN_PROVIDER_CONFIG
    plugin_config_raw = plugin_config_path.read_bytes()
    units[f"plugin:{PLUGIN_PROVIDER_CONFIG}"] = SourceUnit(
        repository=PLUGIN_REPOSITORY,
        revision="worktree",
        path=PLUGIN_PROVIDER_CONFIG,
        text=plugin_config_raw.decode(),
        sha256=hashlib.sha256(plugin_config_raw).hexdigest(),
    )

    manifest_path = plugin_repo / PLUGIN_MANIFEST
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    findings = analyze(units, manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "read-only-pinned-source-and-worktree-plugin-audit",
        "activation_status": (
            "HOST_CONFIG_AND_PLUGIN_SCHEMA_READY_PROFILE_AND_PROVIDER_STILL_MISSING"
        ),
        "runtime_compatible": False,
        "npu_started": False,
        "performance_claim": False,
        "host": {"repository": HOST_REPOSITORY, "revision": HOST_REVISION},
        "manager": {
            "repository": MANAGER_REPOSITORY,
            "revision": MANAGER_REVISION,
        },
        "plugin_manifest": {
            "path": PLUGIN_MANIFEST,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "implementation_status": manifest["implementation"][0]["status"],
        },
        "plugin_provider_config": {
            "path": PLUGIN_PROVIDER_CONFIG,
            "sha256": hashlib.sha256(plugin_config_raw).hexdigest(),
        },
        "sources": [
            {
                "repository": unit.repository,
                "revision": unit.revision,
                "path": unit.path,
                "blob_sha256": unit.sha256,
            }
            for unit in units.values()
        ],
        "findings": findings,
        "required_closure": [
            "Implement provider construction around the existing strict schema, load "
            "the attested profile before graph capture, and compare its logical "
            "tensor-content digest with profile_sha256.",
            "Supply and validate a revision-bound C8 profile before per-request "
            "eligibility; do not read NPU offset tensors in is_eligible, and prove the "
            "BF16 target reaches C8 cache write rather than default 1/0 parameters.",
            "Use the standard Manager vLLM Provider: keep the factory path in static "
            "activation and supply deployment profile settings through operator-owned "
            "--additional-config, with conflict rejection preserved.",
            "Keep the published manifest import_only until a clean Manager dry-run and "
            "Host contract test prove disabled, eligible, and fail-closed paths.",
        ],
        "limitations": [
            "This audit does not supply Qwen3.5 calibration values.",
            "It does not import Host or plugin runtime modules.",
            "It does not run a service, NPU operator, correctness test, or "
            "performance test.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-repo", type=Path, required=True)
    parser.add_argument("--manager-repo", type=Path, required=True)
    parser.add_argument("--plugin-repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.host_repo, args.manager_repo, args.plugin_repo)
    except (
        AuditInconclusive,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "classification": "read-only-pinned-source-and-worktree-plugin-audit",
            "activation_status": "INCONCLUSIVE",
            "runtime_compatible": None,
            "npu_started": False,
            "performance_claim": False,
            "error": str(error),
        }
        exit_code = 2
    else:
        exit_code = 0

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
