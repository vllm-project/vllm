# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import envs
from vllm.entrypoints.openai.responses.context import _run_file_search_handler


def handler_ok(args: dict) -> dict:
    return {"results": [{"file_id": "file_test", "score": 1.0, "content": []}]}


def handler_bad(args: dict):
    return ["not", "a", "dict"]


@pytest.mark.asyncio
async def test_file_search_handler_default(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_GPT_OSS_FILE_SEARCH_HANDLER", raising=False)
    payload = await _run_file_search_handler({"query": "test"})
    assert payload == {"results": []}


@pytest.mark.asyncio
async def test_file_search_handler_custom(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv(
        "VLLM_GPT_OSS_FILE_SEARCH_HANDLER",
        "tests.entrypoints.openai.responses.test_file_search_handler:handler_ok",
    )
    payload = await _run_file_search_handler({"query": "test"})
    assert isinstance(payload, dict)
    assert payload.get("results")


@pytest.mark.asyncio
async def test_file_search_handler_non_dict(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv(
        "VLLM_GPT_OSS_FILE_SEARCH_HANDLER",
        "tests.entrypoints.openai.responses.test_file_search_handler:handler_bad",
    )
    payload = await _run_file_search_handler({"query": "test"})
    assert payload == {"results": []}
