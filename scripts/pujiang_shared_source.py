"""Load pinned shared evaluator bytes without ambient benchmark dependencies."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

BENCHMARK_REVISION = "8ed7ec9a16075c5bce5644a68a712139a6e5d403"


def load_verified_module(
    checkout: Path,
    sources: dict[str, tuple[str, str]],
    target: str,
    dependency: tuple[str, str] | None = None,
) -> ModuleType:
    checkout = checkout.resolve()
    for option, expected in (
        ("--show-toplevel", str(checkout)),
        ("HEAD", BENCHMARK_REVISION),
    ):
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", option],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ValueError(
                "Benchmark checkout must be the exact pinned repository root and HEAD"
            )
    verified = {}
    for name, (relative, expected_hash) in sources.items():
        raw = (checkout / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Pinned {name} source checksum mismatch")
        verified[name] = raw
    bindings = {}
    if dependency is not None:
        key, import_name = dependency
        parent, _, _ = import_name.rpartition(".")
        if parent:
            package = ModuleType(parent)
            package.__path__ = []
            bindings[parent] = package
        bindings[import_name] = ModuleType(import_name)
        bindings[import_name].__file__ = str(checkout / sources[key][0])
    previous = {name: sys.modules.get(name) for name in bindings}
    try:
        # All sources are checked before executing any dependency or target.
        sys.modules.update(bindings)
        if dependency is not None:
            key, import_name = dependency
            module = bindings[import_name]
            exec(compile(verified[key], module.__file__, "exec"), module.__dict__)
        module = ModuleType("pujiang_pinned_" + target)
        module.__file__ = str(checkout / sources[target][0])
        exec(compile(verified[target], module.__file__, "exec"), module.__dict__)
        return module
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
