# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for serializing arbitrary Python values to
JSON-compatible structures.

These utilities are used by HTTP API endpoints and runtime
plugin launchers that need to emit server configs (which are
dataclasses potentially containing non-JSON-native values) as
plain JSON payloads.
"""

# Standard
from dataclasses import asdict, is_dataclass
from typing import Any


def make_json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-serializable value.

    Dicts, lists and tuples are traversed; primitive JSON types are
    returned unchanged; any other value falls back to ``str(obj)``.

    Args:
        obj: Arbitrary Python object to sanitize.

    Returns:
        A value composed solely of ``dict`` / ``list`` /
        ``str`` / ``int`` / ``float`` / ``bool`` / ``None``.

    Exceptions:
        None.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def safe_asdict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-safe ``dict``.

    Args:
        obj: A dataclass instance.

    Returns:
        Dictionary with all fields recursively sanitized via
        :func:`make_json_safe`.

    Exceptions:
        TypeError: If ``obj`` is not a dataclass instance.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("Expected a dataclass instance, got %s" % type(obj).__name__)
    return make_json_safe(asdict(obj))  # type: ignore[return-value]
