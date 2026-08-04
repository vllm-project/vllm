# SPDX-License-Identifier: Apache-2.0
"""Metrics formatters — control *how* metrics are rendered.

A formatter converts a title + sections into a string (or dict).
Formatters are attached to handlers, separating rendering from destination.
"""

# Standard
from typing import Any
import abc
import inspect
import json

# First Party
from lmcache.cli.metrics.section import Section, sections_to_dict


class MetricsFormatter(abc.ABC):
    """Base class for metrics formatters."""

    @abc.abstractmethod
    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics into a string.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.

        Returns:
            The formatted string.
        """


# ---------------------------------------------------------------------------
# Formatter registry
# ---------------------------------------------------------------------------

_FORMATTER_REGISTRY: dict[str, type[MetricsFormatter]] = {}


def register_formatter(name: str):
    """Decorator that registers a ``MetricsFormatter`` subclass under *name*.

    Args:
        name: The format name used for CLI lookup (e.g. ``"json"``).

    Returns:
        A class decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[MetricsFormatter]) -> type[MetricsFormatter]:
        _FORMATTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_formatter(name: str, **kwargs: Any) -> MetricsFormatter:
    """Instantiate a formatter by its registered name.

    Args:
        name: Registered format name (e.g. ``"terminal"``, ``"json"``).
        **kwargs: Forwarded to the formatter constructor (e.g. ``width``).

    Returns:
        A new formatter instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    cls = _FORMATTER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_FORMATTER_REGISTRY))
        raise ValueError(f"Unknown format {name!r}. Available: {available}")
    # Only forward kwargs that the constructor accepts.
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid)


# ---------------------------------------------------------------------------
# Built-in formatters
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a metric value for terminal display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


@register_formatter("terminal")
class TerminalFormatter(MetricsFormatter):
    """Plain ASCII table formatter for terminal output.

    Title is centered in ``=`` borders, section headers are centered in
    ``-`` borders, key-value lines have left-aligned labels and
    right-aligned values.

    Args:
        width: Target total character width for the output.
    """

    def __init__(self, width: int = 48) -> None:
        self._width = width

    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics as an ASCII table.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.

        Returns:
            Rendered multi-line string.
        """
        width = self._width
        lines: list[str] = []

        # Title bar
        title_text = f" {title} "
        lines.append(title_text.center(width, "="))

        for section in sections:
            # Section header (skip for unnamed section)
            if section.label is not None:
                header_text = f" {section.label} "
                lines.append(header_text.center(width, "-"))

            for _key, label, value in section.entries:
                formatted = _format_value(value)
                label_part = f"{label}:"
                padding = width - len(label_part) - len(formatted)
                if padding < 1:
                    padding = 1
                lines.append(f"{label_part}{' ' * padding}{formatted}")

        # Footer
        lines.append("=" * width)

        return "\n".join(lines)


@register_formatter("json")
class JsonFormatter(MetricsFormatter):
    """Renders metrics as a JSON string.

    Args:
        indent: JSON indentation level.
    """

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics as indented JSON.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.

        Returns:
            JSON string.
        """
        return json.dumps(
            sections_to_dict(title, sections),
            indent=self._indent,
        )
