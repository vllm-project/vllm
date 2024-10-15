from contextlib import contextmanager
from typing import Any

_compile_context: Any = None


def get_compile_context() -> Any:
    """Get the current compile context."""
    return _compile_context


@contextmanager
def set_compile_context(context: Any):
    """A context manager that stores the current compile context,
    usually it is a list of sizes to specialize.
    """
    global _compile_context
    prev_context = _compile_context
    _compile_context = context
    try:
        yield
    finally:
        _compile_context = prev_context
