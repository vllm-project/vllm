from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

from vllm.config import VllmConfig


@dataclass
class ForwardContext:
    static_forward_context: Dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    dynamic_forward_context: Any


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    """Get the current forward context."""
    return _forward_context


@contextmanager
def set_forward_context(context: Any, vllm_config: VllmConfig):
    """A context manager that stores the current forward context,
    can be attention metadata, etc."""
    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        static_forward_context=vllm_config.compilation_config.
        static_forward_context,
        dynamic_forward_context=context)
    try:
        yield
    finally:
        _forward_context = prev_context
