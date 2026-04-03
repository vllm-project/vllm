# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama Stack file_search plugin for vLLM.

Register as a vLLM plugin by adding the following to your ``pyproject.toml``::

    [project.entry-points."vllm.file_search_plugins"]
    llama_stack = "vllm.plugins.file_search.llama_stack_handler:create_handler"

Environment variables:
  LLAMA_STACK_URL  - Base URL of the Llama Stack server (default: http://localhost:8321)
  LLAMA_STACK_TIMEOUT - HTTP timeout in seconds (default: 10)
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from vllm.logger import init_logger
from vllm.plugins.file_search import FileSearchHandler

logger = init_logger(__name__)


def _get_base_url() -> str:
    return os.getenv("LLAMA_STACK_URL", "http://localhost:8321").rstrip("/")


def _get_timeout() -> float:
    try:
        return float(os.getenv("LLAMA_STACK_TIMEOUT", "10"))
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
                "content": item.get("content") or [],
            }
        )
    return results


class LlamaStackFileSearchHandler(FileSearchHandler):
    """File search handler that delegates to a Llama Stack vector store."""

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

        url = f"{_get_base_url()}/v1/vector_stores/{vector_store_id}/search"
        timeout = _get_timeout()

        try:
            logger.info("[llama_stack_file_search] POST %s", url)
            logger.info("[llama_stack_file_search] payload=%s", payload)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                logger.info(
                    "[llama_stack_file_search] status=%s", response.status_code
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text
            logger.warning(
                "[llama_stack_file_search] request failed; status=%s body=%s",
                status,
                body,
            )
            return {"results": []}
        except Exception as exc:
            logger.exception(
                "[llama_stack_file_search] request failed; error=%s message=%s",
                type(exc).__name__,
                exc,
            )
            return {"results": []}

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            logger.warning(
                "[llama_stack_file_search] unexpected response shape; type=%s keys=%s",
                type(data),
                list(data) if isinstance(data, dict) else None,
            )
            return {"results": []}

        logger.info("[llama_stack_file_search] data_len=%s", len(items))
        return {"results": _to_results(items)}


def create_handler() -> LlamaStackFileSearchHandler:
    """Entry point factory for the vLLM plugin system."""
    return LlamaStackFileSearchHandler()
