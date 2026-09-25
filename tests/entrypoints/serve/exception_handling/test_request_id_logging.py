# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.serve.exception_handling.handlers.exception import (
    exception_handler,
)
from vllm.entrypoints.serve.exception_handling.handlers.http import (
    http_exception_handler,
)
from vllm.entrypoints.serve.exception_handling.handlers.validation import (
    validation_exception_handler,
)
from vllm.entrypoints.serve.exception_handling.handlers.vllm_error import (
    engine_error_handler,
)
from vllm.v1.engine.exceptions import EngineGenerateError


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", ["chatcmpl-external", None])
@pytest.mark.parametrize(
    ("handler", "exc"),
    [
        (exception_handler, ValueError("bad request")),
        (http_exception_handler, HTTPException(400, "bad request")),
        (
            validation_exception_handler,
            RequestValidationError(
                [{"type": "missing", "loc": ("body", "prompt"), "msg": "required"}]
            ),
        ),
        (engine_error_handler, EngineGenerateError("engine failure")),
    ],
)
async def test_error_log_uses_assigned_external_id_only(
    handler, exc, request_id, caplog, monkeypatch
):
    monkeypatch.setattr(
        "vllm.entrypoints.serve.exception_handling.handlers.vllm_error."
        "terminate_if_errored",
        lambda **kwargs: None,
    )
    state = SimpleNamespace()
    if request_id is not None:
        state.request_metadata = SimpleNamespace(request_id=request_id)
    req = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                args=SimpleNamespace(log_error_stack=True),
                server=None,
                engine_client=None,
            )
        ),
        state=state,
        headers={"X-Request-Id": "raw-header-id"},
    )

    with caplog.at_level(logging.ERROR):
        await handler(req, exc)

    record = next(r for r in caplog.records if "caught. Request id:" in r.getMessage())
    assert record.getMessage().endswith(f"Request id: {request_id}")
    if request_id is None:
        assert not hasattr(record, "request_id")
    else:
        assert record.request_id == request_id
