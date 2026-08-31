from vllm_ascend_adaptive_quantized_kv.receipts import (
    GRAPH_RECAPTURE_SCHEMA,
    GRAPH_RESET_SCHEMA,
    validate_graph_recapture,
    validate_graph_reset,
)


def test_complete_graph_reset_receipt_passes() -> None:
    result = validate_graph_reset(
        {
            "schema_version": GRAPH_RESET_SCHEMA,
            "status": "PASS",
            "wrapper_count": 2,
            "graph_entries_before": 5,
            "graph_objects_reset": 5,
            "graph_entries_after": 0,
            "graph_pool_rebound": True,
            "reset_error": None,
            "cudagraph_capturing_enabled_after": True,
        }
    )

    assert result.accepted
    assert result.errors == ()


def test_partial_graph_reset_fails_closed() -> None:
    result = validate_graph_reset(
        {
            "schema_version": GRAPH_RESET_SCHEMA,
            "status": "PASS",
            "wrapper_count": 1,
            "graph_entries_before": 2,
            "graph_objects_reset": 1,
            "graph_entries_after": 1,
            "graph_pool_rebound": False,
            "reset_error": None,
            "cudagraph_capturing_enabled_after": False,
        }
    )

    assert not result.accepted
    assert "not every graph entry was reset" in result.errors


def test_complete_graph_recapture_receipt_passes() -> None:
    result = validate_graph_recapture(
        {
            "schema_version": GRAPH_RECAPTURE_SCHEMA,
            "status": "PASS",
            "wrapper_count": 2,
            "graph_entries_after_recapture": 5,
            "cudagraph_capturing_enabled_after": False,
        }
    )

    assert result.accepted
