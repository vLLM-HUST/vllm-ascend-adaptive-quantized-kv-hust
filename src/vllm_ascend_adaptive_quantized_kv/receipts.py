"""Validation for retained graph-lifecycle experiment receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

GRAPH_RESET_SCHEMA: Final = "vllm-ascend-c8-graph-reset/v3"
GRAPH_RECAPTURE_SCHEMA: Final = "vllm-ascend-c8-graph-recapture/v1"


@dataclass(frozen=True, slots=True)
class ReceiptValidation:
    """Result of validating an untrusted runtime receipt."""

    schema_version: str
    accepted: bool
    errors: tuple[str, ...]


def _positive_int(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_graph_reset(payload: Mapping[str, object]) -> ReceiptValidation:
    """Validate the full old reset contract without importing backend code."""

    errors: list[str] = []
    if payload.get("schema_version") != GRAPH_RESET_SCHEMA:
        errors.append("unexpected graph reset schema")
    if payload.get("status") != "PASS":
        errors.append("graph reset status is not PASS")
    for key in ("wrapper_count", "graph_entries_before", "graph_objects_reset"):
        if not _positive_int(payload, key):
            errors.append(f"{key} must be a positive integer")
    if payload.get("graph_objects_reset") != payload.get("graph_entries_before"):
        errors.append("not every graph entry was reset")
    if payload.get("graph_entries_after") != 0:
        errors.append("graph entries remain after reset")
    if payload.get("graph_pool_rebound") is not True:
        errors.append("graph pool was not rebound")
    if payload.get("reset_error") is not None:
        errors.append("reset reported an error")
    if payload.get("cudagraph_capturing_enabled_after") is not True:
        errors.append("capture was not enabled for the requested recapture")
    return ReceiptValidation(
        GRAPH_RESET_SCHEMA,
        not errors,
        tuple(errors),
    )


def validate_graph_recapture(payload: Mapping[str, object]) -> ReceiptValidation:
    """Validate that recapture produced graphs and was sealed afterward."""

    errors: list[str] = []
    if payload.get("schema_version") != GRAPH_RECAPTURE_SCHEMA:
        errors.append("unexpected graph recapture schema")
    if payload.get("status") != "PASS":
        errors.append("graph recapture status is not PASS")
    if not _positive_int(payload, "wrapper_count"):
        errors.append("wrapper_count must be a positive integer")
    if not _positive_int(payload, "graph_entries_after_recapture"):
        errors.append("recapture produced no graph entries")
    if payload.get("cudagraph_capturing_enabled_after") is not False:
        errors.append("capture remained enabled after sealing")
    return ReceiptValidation(
        GRAPH_RECAPTURE_SCHEMA,
        not errors,
        tuple(errors),
    )
