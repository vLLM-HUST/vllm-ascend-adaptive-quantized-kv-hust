import os
import subprocess
import sys
from pathlib import Path


def test_package_import_does_not_import_runtime_or_reference() -> None:
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
            "import vllm_ascend_adaptive_quantized_kv.observer; "
            "loaded=set(sys.modules)-before; "
            "forbidden=('vllm', 'vllm_ascend', 'torch', 'torch_npu', "
            "'vllm_ascend_adaptive_quantized_kv.reference'); "
            "assert not any(name == prefix or name.startswith(prefix+'.') "
            "for name in loaded for prefix in forbidden), loaded"
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
