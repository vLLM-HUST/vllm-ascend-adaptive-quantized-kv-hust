"""Proposed continuing-prefill observer component."""

from __future__ import annotations

from collections.abc import Iterable

from . import CONTRACT_ID
from .contracts import (
    HostContractReceipt,
    evaluate_host_contract,
    require_host_contract,
)


class ContinuingPrefillObserver:
    """Compatibility-gated component placeholder for the reviewed host seam.

    P1 deliberately provides no vLLM entry point. The lifecycle manager may
    inspect this implementation reference, but runtime activation starts only
    after the host advertises the proposed contract.
    """

    contract_id = CONTRACT_ID

    def __init__(
        self,
        *,
        advertised_contracts: Iterable[str],
        host_source_revision: str,
    ) -> None:
        self.receipt = evaluate_host_contract(
            advertised_contracts,
            host_source_revision=host_source_revision,
        )

    def require_compatible_host(self) -> HostContractReceipt:
        require_host_contract(self.receipt)
        return self.receipt
