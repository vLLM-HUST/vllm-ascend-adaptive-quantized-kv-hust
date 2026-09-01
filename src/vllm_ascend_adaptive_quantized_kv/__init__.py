"""Static package identity for the Adaptive Quantized KV extension."""

EXTENSION_ID = "org.vllm-hust.ascend-adaptive-quantized-kv"
# Compatibility alias for the pre-alpha Bundle prototype.
BUNDLE_ID = EXTENSION_ID
COMPONENT_ID = "continuing-prefill-profiler"
CONTRACT_ID = "vllm.ascend.continuing-prefill.observer.v1"
__version__ = "0.1.0.dev0"

__all__ = [
    "BUNDLE_ID",
    "COMPONENT_ID",
    "CONTRACT_ID",
    "EXTENSION_ID",
    "__version__",
]
