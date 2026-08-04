# SPDX-License-Identifier: Apache-2.0
"""Hierarchical metrics collector with handler-based output.

Example usage::

    from lmcache.cli.metrics import Metrics

    metrics = Metrics(title="Bench KV Cache Result (30s)")

    # Sectioned metrics
    metrics.add_section("ops", "Operations (ops/s)")
    metrics["ops"].add("store", "Store", 41.3)
    metrics["ops"].add("retrieve", "Retrieve", 127.3)

    # Top-level (flat) metrics
    metrics.add("status", "Status", "OK")

    metrics.emit()
"""

# Standard
from typing import Any, Optional

# First Party
from lmcache.cli.metrics.handler import FileHandler, MetricsHandler
from lmcache.cli.metrics.section import Section, sections_to_dict
from lmcache.logging import init_logger

logger = init_logger(__name__)


class Metrics:
    """Hierarchical metrics collector with handler-based output.

    Handlers are registered via :meth:`add_handler` and triggered
    together by :meth:`emit`.  ``BaseCommand`` sets up default
    handlers automatically, so command authors typically only need
    to build metrics and call :meth:`emit`.

    Args:
        title: Report title shown in the header.
    """

    def __init__(self, title: str) -> None:
        self._title = title
        self._sections: list[Section] = []
        self._section_map: dict[Optional[str], Section] = {}
        self._handlers: list[MetricsHandler] = []

    def title(self, title: str) -> None:
        """Set the report title.

        Args:
            title: New report title shown in the header.
        """
        self._title = title

    # -- Handler management -------------------------------------------------

    def add_handler(self, handler: MetricsHandler) -> None:
        """Register a handler.

        Args:
            handler: The handler to add.
        """
        self._handlers.append(handler)

    # -- Section management -------------------------------------------------

    def add_section(self, key: str, label: str) -> Section:
        """Add a named section.

        Args:
            key: Machine-readable section key (used in JSON output and
                for ``metrics["key"]`` access).
            label: Human-readable label (used in terminal output).

        Returns:
            The newly created ``Section``.

        Raises:
            ValueError: If a section with the same *key* already exists.
        """
        if key in self._section_map:
            raise ValueError(f"Section {key!r} already exists")
        section = Section(key, label)
        self._sections.append(section)
        self._section_map[key] = section
        return section

    def add_list_section(self, group: str, key: str, label: str) -> Section:
        """Add a section that belongs to a list group.

        In terminal output, renders as a normal section with *label* as
        the header. In JSON output, sections sharing the same *group*
        are collected into a list: ``"group": [{...}, {...}]``.

        Args:
            group: The JSON key for the list (e.g., ``"models"``).
            key: Unique machine key for this section instance.
            label: Human-readable label (used in terminal output).

        Returns:
            The newly created ``Section``.
        """
        if key in self._section_map:
            raise ValueError(f"Section {key!r} already exists")
        section = Section(key, label, list_group=group)
        self._sections.append(section)
        self._section_map[key] = section
        return section

    def __getitem__(self, key: str) -> Section:
        """Return the section registered under *key*.

        Raises:
            KeyError: If ``add_section(key, ...)`` was not called first.
        """
        return self._section_map[key]

    # -- Flat (top-level) metrics -------------------------------------------

    def _default_section(self) -> Section:
        """Return the unnamed default section, creating it on first use."""
        if None not in self._section_map:
            section = Section(None, None)
            # Insert at the beginning so flat metrics appear first
            self._sections.insert(0, section)
            self._section_map[None] = section
        return self._section_map[None]

    def add(self, key: str, label: str, value: Any) -> None:
        """Record a top-level metric (not inside any named section).

        Args:
            key: Machine-readable key (used in JSON output).
            label: Human-readable label (used in terminal output).
            value: Metric value.
        """
        self._default_section().add(key, label, value)

    # -- Output -------------------------------------------------------------

    def emit(self) -> None:
        """Trigger all registered handlers."""
        for handler in self._handlers:
            handler.emit(self._title, self._sections)
        for handler in self._handlers:
            if isinstance(handler, FileHandler):
                logger.info("Results saved to %s", handler.path)

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a JSON-serialisable dictionary.

        Returns:
            A dict with ``"title"`` and ``"metrics"`` keys. Named
            sections become nested dicts keyed by machine key. The
            unnamed default section's entries are placed at the top
            level of ``"metrics"``.
        """
        return sections_to_dict(self._title, self._sections)
