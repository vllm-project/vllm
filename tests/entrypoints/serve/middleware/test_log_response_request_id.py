# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from types import SimpleNamespace

import pytest
from fastapi import Request

from vllm.entrypoints.serve.middleware.log_response import log_response


@pytest.mark.asyncio
async def test_streaming_response_logs_keep_assigned_external_id(caplog, monkeypatch):
    request = Request({"type": "http", "method": "GET", "headers": []})

    async def async_iter(items):
        for item in items:
            yield item

    monkeypatch.setattr(
        "vllm.entrypoints.serve.middleware.log_response.iterate_in_threadpool",
        async_iter,
    )

    async def call_next(req):
        req.state.request_metadata = SimpleNamespace(request_id="resp-external")
        return SimpleNamespace(
            body_iterator=async_iter(
                [b'data: {"object": "response"}\n\n', b"data: [DONE]\n\n"]
            ),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    with caplog.at_level(logging.INFO):
        response = await log_response(request, call_next)
        _ = [chunk async for chunk in response.body_iterator]

    records = [r for r in caplog.records if r.getMessage().startswith("response_body=")]
    assert [r.request_id for r in records] == ["resp-external"] * 2
