# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
from unittest.mock import patch

import pytest

from vllm.entrypoints.openai.responses.context import _run_file_search_handler
from vllm.plugins.file_search import FileSearchHandler


class OkHandler(FileSearchHandler):
    async def search(
        self,
        query: str,
        vector_store_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "results": [{"file_id": "file_test", "score": 1.0, "content": []}]
        }


class BadHandler(FileSearchHandler):
    async def search(self, **kwargs) -> dict[str, Any]:
        return ["not", "a", "dict"]  # type: ignore[return-value]


class RaisingHandler(FileSearchHandler):
    async def search(self, **kwargs) -> dict[str, Any]:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_no_plugin_returns_empty():
    with patch(
        "vllm.plugins.file_search.get_file_search_handler", return_value=None
    ):
        payload = await _run_file_search_handler({"query": "test"})
    assert payload == {"results": []}


@pytest.mark.asyncio
async def test_plugin_returns_results():
    with patch(
        "vllm.plugins.file_search.get_file_search_handler",
        return_value=OkHandler(),
    ):
        payload = await _run_file_search_handler({"query": "test"})
    assert isinstance(payload, dict)
    assert len(payload["results"]) == 1
    assert payload["results"][0]["file_id"] == "file_test"


@pytest.mark.asyncio
async def test_plugin_non_dict_returns_empty():
    with patch(
        "vllm.plugins.file_search.get_file_search_handler",
        return_value=BadHandler(),
    ):
        payload = await _run_file_search_handler({"query": "test"})
    assert payload == {"results": []}


@pytest.mark.asyncio
async def test_plugin_exception_returns_empty():
    with patch(
        "vllm.plugins.file_search.get_file_search_handler",
        return_value=RaisingHandler(),
    ):
        payload = await _run_file_search_handler({"query": "test"})
    assert payload == {"results": []}


@pytest.mark.asyncio
async def test_plugin_discovery():
    """Test that get_file_search_handler discovers plugins via entry points."""
    from unittest.mock import MagicMock

    from vllm.plugins import file_search

    # Reset cached state
    file_search._handler_loaded = False
    file_search._cached_handler = None

    mock_handler = OkHandler()
    mock_factory = MagicMock(return_value=mock_handler)

    with patch(
        "vllm.plugins.file_search.load_plugins_by_group",
        return_value={"test_handler": mock_factory},
    ):
        handler = file_search.get_file_search_handler()

    assert handler is mock_handler
    mock_factory.assert_called_once()

    # Reset state
    file_search._handler_loaded = False
    file_search._cached_handler = None


@pytest.mark.asyncio
async def test_no_plugins_installed():
    """Test graceful fallback when no plugins are installed."""
    from vllm.plugins import file_search

    # Reset cached state
    file_search._handler_loaded = False
    file_search._cached_handler = None

    with patch(
        "vllm.plugins.file_search.load_plugins_by_group",
        return_value={},
    ):
        handler = file_search.get_file_search_handler()

    assert handler is None

    # Reset state
    file_search._handler_loaded = False
    file_search._cached_handler = None
