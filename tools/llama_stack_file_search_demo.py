"""Llama Stack file_search demo handler for vLLM.

Use:
  export VLLM_GPT_OSS_FILE_SEARCH_HANDLER="tools.llama_stack_file_search_demo:handle"
  export LLAMA_STACK_URL="http://localhost:8321"
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def _get_base_url() -> str:
    return os.getenv("LLAMA_STACK_URL", "http://localhost:8321").rstrip("/")


def _get_timeout() -> float:
    try:
        return float(os.getenv("LLAMA_STACK_TIMEOUT", "10"))
    except ValueError:
        return 10.0


def _get_vector_store_id(args: dict[str, Any]) -> str | None:
    vector_store_ids = args.get("vector_store_ids") or []
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


def handle(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a Llama Stack vector store search and return file_search results."""
    query = args.get("query")
    if not query:
        return {"results": []}

    vector_store_id = _get_vector_store_id(args)
    if not vector_store_id:
        return {"results": []}

    payload: dict[str, Any] = {"query": query}
    if "filters" in args:
        payload["filters"] = args["filters"]
    if "max_num_results" in args:
        payload["max_num_results"] = args["max_num_results"]
    if "ranking_options" in args:
        payload["ranking_options"] = args["ranking_options"]

    url = f"{_get_base_url()}/v1/vector_stores/{vector_store_id}/search"
    timeout = _get_timeout()

    try:
        print(f"[llama_stack_file_search_demo] POST {url}")
        print(f"[llama_stack_file_search_demo] payload={payload}")
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            print(
                f"[llama_stack_file_search_demo] status={response.status_code}"
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        body = exc.response.text
        print(
            "[llama_stack_file_search_demo] request failed; "
            f"status={status} body={body}"
        )
        return {"results": []}
    except Exception as exc:
        print(
            "[llama_stack_file_search_demo] request failed; "
            f"error={type(exc).__name__} message={exc}"
        )
        return {"results": []}

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        print(
            "[llama_stack_file_search_demo] unexpected response shape; "
            f"type={type(data)} keys={list(data) if isinstance(data, dict) else None}"
        )
        return {"results": []}

    print(
        f"[llama_stack_file_search_demo] data_len={len(items)}"
    )
    return {"results": _to_results(items)}
