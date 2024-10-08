from contextlib import contextmanager
from typing import Any

_forward_context: Any = None


def get_forward_context() -> Any:
    """Get the current forward context."""
    return _forward_context


@contextmanager
def set_forward_context(context: Any):
    """A context manager that stores the current forward context,
    can be attention metadata, etc."""
    global _forward_context
    prev_context = _forward_context
    _forward_context = context
    try:
        yield
    finally:
        _forward_context = prev_context
