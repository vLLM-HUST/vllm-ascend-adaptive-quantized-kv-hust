import os
import subprocess
import sys
from pathlib import Path


def test_package_import_does_not_import_vllm() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "before=set(sys.modules); "
            "import vllm_ascend_adaptive_quantized_kv; "
            "loaded=set(sys.modules)-before; "
            "assert 'vllm' not in loaded; "
            "assert not any(name.startswith('vllm.') for name in loaded)"
        ),
    ]

    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
