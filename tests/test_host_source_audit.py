import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_host_execution import (
    AuditInconclusive,
    calls_with_suffix,
    find_if,
    find_qualified_def,
    has_class_rewrite,
    literal_assignments,
    loaded_name_count,
)


def test_qualified_lookup_and_unused_parameter_are_structural() -> None:
    source = """
class Platform:
    def choose(cls, selected_backend, config):
        return config.backend
"""
    method = find_qualified_def(source, "Platform.choose")

    assert loaded_name_count(method, "selected_backend") == 0
    assert loaded_name_count(method, "config") == 1


def test_used_backend_parameter_is_not_reported_unused() -> None:
    source = """
class Platform:
    def choose(cls, selected_backend, config):
        return selected_backend or config.backend
"""
    method = find_qualified_def(source, "Platform.choose")

    assert loaded_name_count(method, "selected_backend") == 1


def test_class_rewrite_requires_the_exact_target_and_value() -> None:
    expected = ast.parse("layer.impl.__class__ = AscendC8AttentionBackendImpl")
    wrong_target = ast.parse("layer.other.__class__ = AscendC8AttentionBackendImpl")
    wrong_value = ast.parse("layer.impl.__class__ = OtherImpl")

    assert has_class_rewrite(expected)
    assert not has_class_rewrite(wrong_target)
    assert not has_class_rewrite(wrong_value)


def test_capture_branch_fails_closed_when_required_call_moves() -> None:
    source = """
def forward(self):
    if _EXTRA_CTX.capturing:
        return self.full_graph_fia()
"""
    function = find_qualified_def(source, "forward")
    branch = find_if(function, "_EXTRA_CTX.capturing", "full_graph_fia")
    assert branch.lineno == 3

    changed = source.replace("self.full_graph_fia()", "self.other_path()")
    with pytest.raises(AuditInconclusive):
        find_if(
            find_qualified_def(changed, "forward"),
            "_EXTRA_CTX.capturing",
            "full_graph_fia",
        )


def test_literal_assignments_and_call_lookup_ignore_comments() -> None:
    source = """
def capture(query):
    # input_layout = 'TND'; fake_call()
    input_layout = "BNSD"
    sparse_mode = 0
    query = query.unsqueeze(2)
    return query
"""
    function = find_qualified_def(source, "capture")

    assert literal_assignments(function) == {
        "input_layout": "BNSD",
        "sparse_mode": 0,
    }
    assert len(calls_with_suffix(function, "unsqueeze")) == 1
    assert calls_with_suffix(function, "fake_call") == []
