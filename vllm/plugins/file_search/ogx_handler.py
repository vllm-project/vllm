# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OGX file_search plugin for vLLM.

The handler is registered as a vLLM plugin by the vLLM package.

Environment variables:
  OGX_URL  - Base URL of the OGX server (default: http://localhost:8321)
  OGX_TIMEOUT - HTTP timeout in seconds (default: 10)
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from vllm.logger import init_logger
from vllm.plugins.file_search import FileSearchError, FileSearchHandler

logger = init_logger(__name__)


def _get_base_url() -> str:
    return os.getenv("OGX_URL", "http://localhost:8321").rstrip("/")


def _get_timeout() -> float:
    try:
        return float(os.getenv("OGX_TIMEOUT", "10"))
    except ValueError:
        return 10.0


def _get_vector_store_id(
    vector_store_ids: list[str] | None,
) -> str | None:
    if isinstance(vector_store_ids, list) and vector_store_ids:
        return str(vector_store_ids[0])
    return None


def _to_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        results.append(
            {
                "file_id": item.get("file_id") or item.get("document_id"),
                "filename": item.get("filename"),
                "score": item.get("score"),
                "attributes": item.get("attributes") or {},
                "text": _content_text(item.get("content")),
            }
        )
    return results


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts) if texts else None


class OGXFileSearchHandler(FileSearchHandler):
    """File search handler that delegates to an OGX vector store."""

    async def search(
        self,
        query: str,
        vector_store_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query:
            return {"results": []}

        vector_store_id = _get_vector_store_id(vector_store_ids)
        if not vector_store_id:
            return {"results": []}

        payload: dict[str, Any] = {"query": query}
        if filters is not None:
            payload["filters"] = filters
        if max_num_results is not None:
            payload["max_num_results"] = max_num_results
        if ranking_options is not None:
            payload["ranking_options"] = ranking_options

        encoded_vector_store_id = quote(vector_store_id, safe="")
        url = f"{_get_base_url()}/v1/vector_stores/{encoded_vector_store_id}/search"
        timeout = _get_timeout()

        try:
            logger.info("[ogx_file_search] POST %s", url)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                logger.info("[ogx_file_search] status=%s", response.status_code)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise FileSearchError(
                f"OGX file_search request failed with status {status}"
            ) from exc
        except Exception as exc:
            raise FileSearchError("OGX file_search request failed") from exc

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise FileSearchError(
                "OGX file_search returned an invalid response payload"
            )

        logger.info("[ogx_file_search] data_len=%s", len(items))
        results = _to_results(items)
        if max_num_results is not None:
            results = results[:max_num_results]
        return {"results": results}


def create_handler() -> OGXFileSearchHandler:
    """Entry point factory for the vLLM plugin system."""
    return OGXFileSearchHandler()
