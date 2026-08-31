import pytest

from vllm_ascend_adaptive_quantized_kv import CONTRACT_ID
from vllm_ascend_adaptive_quantized_kv.contracts import (
    HostContractError,
    evaluate_host_contract,
    require_host_contract,
)


def test_exact_contract_and_revision_pass() -> None:
    receipt = evaluate_host_contract(
        [CONTRACT_ID],
        host_source_revision="40f9834ee82aadfa4656ec65e5bd84f4d6241b5f",
    )

    assert receipt.accepted
    assert receipt.status == "PASS"
    require_host_contract(receipt)


def test_missing_contract_fails_closed() -> None:
    receipt = evaluate_host_contract(
        [],
        host_source_revision="40f9834ee82aadfa4656ec65e5bd84f4d6241b5f",
    )

    assert not receipt.accepted
    assert receipt.status == "FAIL_CLOSED"
    with pytest.raises(HostContractError, match="does not advertise"):
        require_host_contract(receipt)


def test_missing_revision_fails_closed() -> None:
    receipt = evaluate_host_contract([CONTRACT_ID], host_source_revision="")

    assert not receipt.accepted
    assert "host source revision is required" in receipt.errors
