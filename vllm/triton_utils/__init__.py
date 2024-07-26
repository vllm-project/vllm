from vllm.triton_utils.importing import HAS_TRITON

if HAS_TRITON:
    from vllm.triton_utils.custom_cache_manager import (
        maybe_set_triton_cache_manager)

__all__ = ["HAS_TRITON", "maybe_set_triton_cache_manager"]

if not HAS_TRITON:
    # need to do this afterwards due to ruff complaining
    __all__.pop()
