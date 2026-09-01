"""Fail-closed compatibility contracts independent of vLLM imports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from . import CONTRACT_ID

COMPATIBILITY_SCHEMA: Final = "adaptive-quantized-kv-host-compatibility/v1"


class HostContractError(RuntimeError):
    """The enabled extension cannot execute against the advertised host."""


@dataclass(frozen=True, slots=True)
class HostContractReceipt:
    """Auditable decision for one exact host revision."""

    schema_version: str
    status: str
    required_contract: str
    advertised_contracts: tuple[str, ...]
    host_source_revision: str
    errors: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status == "PASS" and not self.errors


def evaluate_host_contract(
    advertised_contracts: Iterable[str],
    *,
    host_source_revision: str,
) -> HostContractReceipt:
    """Accept only an exact, attributable implementation contract."""

    advertised = tuple(sorted(set(advertised_contracts)))
    errors: list[str] = []
    if not host_source_revision.strip():
        errors.append("host source revision is required")
    if CONTRACT_ID not in advertised:
        errors.append(f"host does not advertise required contract {CONTRACT_ID}")
    return HostContractReceipt(
        schema_version=COMPATIBILITY_SCHEMA,
        status="PASS" if not errors else "FAIL_CLOSED",
        required_contract=CONTRACT_ID,
        advertised_contracts=advertised,
        host_source_revision=host_source_revision,
        errors=tuple(errors),
    )


def require_host_contract(receipt: HostContractReceipt) -> None:
    """Raise before runtime activation when compatibility is not proved."""

    if not receipt.accepted:
        detail = "; ".join(receipt.errors) or "compatibility receipt did not pass"
        raise HostContractError(detail)
