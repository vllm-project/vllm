# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File search plugin interface and loader.

Plugins are registered as Python entry points under the
``vllm.file_search_plugins`` group.  Each entry point should resolve to a
callable (class or factory function) that accepts no arguments and returns a
:class:`FileSearchHandler` instance.

Example ``pyproject.toml`` entry::

    [project.entry-points."vllm.file_search_plugins"]
    my_handler = "my_package.file_search:create_handler"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from vllm.plugins import FILE_SEARCH_PLUGINS_GROUP, load_plugins_by_group

logger = logging.getLogger(__name__)

# Module-level cached handler so we only load once.
_cached_handler: FileSearchHandler | None = None
_handler_loaded: bool = False


class FileSearchHandler(ABC):
    """Abstract base class for file search handlers.

    Implementations must override :meth:`search` to perform the actual
    vector-store lookup and return results in the OpenAI-compatible format.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        vector_store_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a file search and return results.

        Returns:
            A dict with a ``"results"`` key containing a list of result dicts.
            Each result should have ``file_id``, ``filename``, ``score``,
            ``attributes``, and ``content`` fields.
        """
        ...


def get_file_search_handler() -> FileSearchHandler | None:
    """Return the installed file search handler, or ``None`` if absent.

    The handler is discovered from the ``vllm.file_search_plugins`` entry
    point group.  Only the first discovered plugin is used.  The result is
    cached so subsequent calls are cheap.
    """
    global _cached_handler, _handler_loaded

    if _handler_loaded:
        return _cached_handler

    _handler_loaded = True

    plugins = load_plugins_by_group(FILE_SEARCH_PLUGINS_GROUP)
    if not plugins:
        logger.info(
            "No file_search plugins installed. "
            "file_search tool calls will return empty results."
        )
        return None

    if len(plugins) > 1:
        logger.warning(
            "Multiple file_search plugins found: %s. Using the first one.",
            list(plugins.keys()),
        )

    name, factory = next(iter(plugins.items()))
    try:
        handler = factory()
        if not isinstance(handler, FileSearchHandler):
            logger.error(
                "file_search plugin '%s' returned %s, expected FileSearchHandler",
                name,
                type(handler),
            )
            return None
        _cached_handler = handler
        logger.info("Loaded file_search plugin: %s", name)
        return _cached_handler
    except Exception:
        logger.exception("Failed to instantiate file_search plugin '%s'", name)
        return None
