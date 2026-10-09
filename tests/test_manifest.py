import json
from importlib import resources

from vllm_ascend_adaptive_quantized_kv import (
    COMPONENT_ID,
    CONTRACT_ID,
    EXTENSION_ID,
    __version__,
)


def _manifest() -> dict[str, object]:
    path = resources.files("vllm_ascend_adaptive_quantized_kv").joinpath(
        "vllm-hust-extension-v0.3.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_static_manifest_matches_package_identity() -> None:
    manifest = _manifest()
    component = manifest["components"][0]

    assert manifest["schema_version"] == "0.3-experimental"
    assert manifest["host"]["api_range"] == ">=1,<2"
    assert manifest["requires_extensions"] == []
    assert manifest["resource_claims"] == [
        {
            "resource": "vllm-ascend.continuing-prefill.observer",
            "scope": "vllm-process",
            "mode": "shared",
        }
    ]
    assert manifest["extension_id"] == EXTENSION_ID
    assert manifest["extension_version"] == __version__
    assert manifest["kind"] == "in_process_plugin"
    assert manifest["host"]["provider"] == "vllm"
    assert manifest["runtime"]["process_scope"] == "vllm-ascend-worker"
    assert manifest["lifecycle_owner"] == "vllm"
    assert component["component_id"] == COMPONENT_ID
    assert component["contracts"] == [CONTRACT_ID]


def test_p1_has_no_runtime_activation() -> None:
    manifest = _manifest()

    assert manifest["implementation"][0]["status"] == "import_only"
    assert manifest["activation"] == {
        "entry_points": [],
        "environment": {},
        "additional_config": {},
    }
