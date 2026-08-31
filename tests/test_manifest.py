import json
from importlib import resources

from vllm_ascend_adaptive_quantized_kv import (
    BUNDLE_ID,
    COMPONENT_ID,
    CONTRACT_ID,
    __version__,
)


def _manifest() -> dict[str, object]:
    path = resources.files("vllm_ascend_adaptive_quantized_kv").joinpath(
        "vllm-hust-extension-v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_static_manifest_matches_package_identity() -> None:
    manifest = _manifest()
    component = manifest["components"][0]

    assert manifest["schema_version"] == "1.0"
    assert manifest["bundle_id"] == BUNDLE_ID
    assert manifest["bundle_version"] == __version__
    assert component["component_id"] == COMPONENT_ID
    assert component["contracts"] == [CONTRACT_ID]


def test_p1_has_no_runtime_activation() -> None:
    manifest = _manifest()

    assert "activation" not in manifest
