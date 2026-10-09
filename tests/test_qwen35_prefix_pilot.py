import copy
import json
from pathlib import Path

import pytest

from scripts.validate_qwen35_prefix_pilot import validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "docs" / "evidence" / "QWEN35_PREFIX_REPETITION_PILOT_CONTRACT_20260928.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_frozen_qwen35_prefix_pilot_contract_passes() -> None:
    receipt = validate_contract(_contract())

    assert receipt["status"] == "PASS"
    assert receipt["execution_authorized"] is False
    assert receipt["scenario"] == "prefix-repetition-online-2chip"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("target", "tensor_parallel_size"), 4, "target configuration"),
        (("benchmark", "scenario"), "random-online-2chip", "scenario"),
        (("execution", "max_minutes_per_arm"), 120, "resource bound"),
        (
            ("gates", "guardrails", "error_rate_percent_max"),
            1.0,
            "error gate",
        ),
    ],
)
def test_contract_drift_fails_closed(
    path: tuple[str, ...], value: object, match: str
) -> None:
    contract = copy.deepcopy(_contract())
    current = contract
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        validate_contract(contract)
