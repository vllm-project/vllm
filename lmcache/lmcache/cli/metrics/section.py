# SPDX-License-Identifier: Apache-2.0
"""Section — a named group of metric entries."""

# Standard
from typing import Any, Optional


class Section:
    """A named group of metrics.

    Each entry has a machine ``key`` (used in JSON), a human-readable
    ``label`` (used in terminal output), and a ``value``.

    Sections with the same :attr:`list_group` are collected into a
    JSON list under that key (e.g., ``"models": [{...}, {...}]``).
    In terminal output they render as normal independent sections.
    """

    def __init__(
        self,
        key: Optional[str],
        label: Optional[str],
        list_group: Optional[str] = None,
    ) -> None:
        self.key = key
        self.label = label
        self.list_group = list_group
        self.entries: list[tuple[str, str, Any]] = []

    def add(self, key: str, label: str, value: Any) -> None:
        """Record a metric in this section.

        Args:
            key: Machine-readable key (used in JSON output).
            label: Human-readable label (used in terminal output).
            value: Metric value. Floats are formatted to 2 decimal
                places on terminal output; strings are printed as-is.
        """
        self.entries.append((key, label, value))


def sections_to_dict(
    title: str,
    sections: list[Section],
) -> dict[str, Any]:
    """Convert a title and sections to a JSON-serialisable dictionary.

    Named sections become nested dicts keyed by machine key. The
    unnamed default section's entries are placed at the top level
    of ``"metrics"``.

    Args:
        title: The report title.
        sections: Ordered list of ``Section`` objects.

    Returns:
        A dict with ``"title"`` and ``"metrics"`` keys.
    """
    metrics: dict[str, Any] = {}
    list_groups: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        if section.key is None:
            for key, _label, value in section.entries:
                metrics[key] = value
        elif section.list_group is not None:
            section_dict: dict[str, Any] = {}
            for key, _label, value in section.entries:
                section_dict[key] = value
            list_groups.setdefault(section.list_group, []).append(section_dict)
        else:
            section_dict = {}
            for key, _label, value in section.entries:
                section_dict[key] = value
            metrics[section.key] = section_dict
    for group_key, items in list_groups.items():
        metrics[group_key] = items
    return {"title": title, "metrics": metrics}
