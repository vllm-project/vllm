# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vllm.entrypoints.openai.responses.context import _run_file_search_handler
from vllm.plugins.file_search import FileSearchHandler
from vllm.plugins.file_search.ogx_handler import OGXFileSearchHandler

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def reset_file_search_handler_cache():
    from vllm.plugins import file_search

    original_loaded = file_search._handler_loaded
    original_handler = file_search._cached_handler
    file_search._handler_loaded = False
    file_search._cached_handler = None
    yield
    file_search._handler_loaded = original_loaded
    file_search._cached_handler = original_handler


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
            "results": [{"file_id": "file_test", "score": 1.0, "text": "matched text"}]
        }


class BadHandler(FileSearchHandler):
    async def search(self, **kwargs) -> dict[str, Any]:
        return ["not", "a", "dict"]  # type: ignore[return-value]


class RaisingHandler(FileSearchHandler):
    async def search(self, **kwargs) -> dict[str, Any]:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_no_plugin_raises():
    with (
        patch("vllm.plugins.file_search.get_file_search_handler", return_value=None),
        pytest.raises(RuntimeError, match="No file_search plugin"),
    ):
        await _run_file_search_handler({"query": "test"})


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
async def test_plugin_non_dict_raises():
    with (
        patch(
            "vllm.plugins.file_search.get_file_search_handler",
            return_value=BadHandler(),
        ),
        pytest.raises(RuntimeError, match="invalid results payload"),
    ):
        await _run_file_search_handler({"query": "test"})


@pytest.mark.asyncio
async def test_plugin_exception_raises():
    with (
        patch(
            "vllm.plugins.file_search.get_file_search_handler",
            return_value=RaisingHandler(),
        ),
        pytest.raises(RuntimeError, match="handler failed"),
    ):
        await _run_file_search_handler({"query": "test"})


@pytest.mark.asyncio
async def test_ogx_http_failure_is_not_reported_as_empty_results():
    request = httpx.Request("POST", "http://localhost/search")
    response = httpx.Response(503, request=request)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response),
    )

    with (
        patch(
            "vllm.plugins.file_search.ogx_handler.httpx.AsyncClient",
            return_value=client,
        ),
        pytest.raises(RuntimeError, match="status 503"),
    ):
        await OGXFileSearchHandler().search(
            query="test",
            vector_store_ids=["vs_test"],
        )


@pytest.mark.asyncio
async def test_plugin_discovery():
    """Test that get_file_search_handler discovers plugins via entry points."""
    from vllm.plugins import file_search

    mock_handler = OkHandler()
    mock_factory = MagicMock(return_value=mock_handler)

    with patch(
        "vllm.plugins.file_search.load_plugins_by_group",
        return_value={"test_handler": mock_factory},
    ):
        handler = file_search.get_file_search_handler()

    assert handler is mock_handler
    mock_factory.assert_called_once()


@pytest.mark.asyncio
async def test_no_plugins_installed():
    """Test graceful fallback when no plugins are installed."""
    from vllm.plugins import file_search

    with patch(
        "vllm.plugins.file_search.load_plugins_by_group",
        return_value={},
    ):
        handler = file_search.get_file_search_handler()

    assert handler is None


@pytest.mark.asyncio
async def test_multiple_plugins_require_explicit_selection():
    from vllm.plugins import file_search

    first_factory = MagicMock(return_value=OkHandler())
    second_factory = MagicMock(return_value=OkHandler())

    with patch(
        "vllm.plugins.file_search.load_plugins_by_group",
        return_value={"first": first_factory, "second": second_factory},
    ):
        handler = file_search.get_file_search_handler()

    assert handler is None
    first_factory.assert_not_called()
    second_factory.assert_not_called()
