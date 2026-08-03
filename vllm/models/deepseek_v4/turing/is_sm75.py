from __future__ import annotations


def is_turing_target(capability) -> bool:
    """True when we should use the Turing (SM75) DeepSeek-V4 backend."""
    return capability is not None and capability.major == 7
