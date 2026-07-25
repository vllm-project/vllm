# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that validation_exception_handler populates the `param` field
in its error response using the Pydantic error's `loc`, even when no
custom VLLMValidationError context is present.

Previously, `param` was only populated for errors carrying a custom
VLLMValidationError in their Pydantic `ctx`. Plain validation failures
(missing fields, wrong types) left `param` as None, even though the
field name was readily available from `error['loc']`.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.serve.utils.server_utils import (
    clean_loc_for_param,
    validation_exception_handler,
)


def _fake_request(log_error_stack: bool = False) -> SimpleNamespace:
    """Minimal stand-in for a FastAPI Request - just enough for the
    handler to read req.app.state.args.log_error_stack."""
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(args=SimpleNamespace(log_error_stack=log_error_stack))
        ),
        state=SimpleNamespace(),  # no request_metadata -> hasattr(...) is False
    )


class TestValidationErrorParamFallback:
    """Ensure `param` falls back to the Pydantic error's `loc` when no
    custom VLLMValidationError context is present."""

    @pytest.mark.parametrize(
        ("error_type", "msg"),
        [
            ("missing", "Field required"),
            ("list_type", "Input should be a valid list"),
        ],
        ids=["missing-field", "wrong-type"],
    )
    @pytest.mark.asyncio
    async def test_param_falls_back_to_loc(self, error_type: str, msg: str):
        errors = [{"type": error_type, "loc": ("body", "messages"), "msg": msg}]
        exc = RequestValidationError(errors)

        response = await validation_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)

        assert body["error"]["param"] == "body.messages"

    @pytest.mark.asyncio
    async def test_param_fallback_does_not_crash_on_non_dict_error(self):
        """Schemathesis fuzzing found that errors[0] isn't always a dict.
        The fallback must not crash in that case - it should just leave
        param as None instead of raising."""
        exc = RequestValidationError(["some unexpected non-dict error"])

        response = await validation_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)

        assert body["error"]["param"] is None


class TestCleanLocForParam:
    """Guards against PR #1's naive dot-joined `loc` fallback leaking
    Pydantic-internal wrapper/union-branch markers into `param`, e.g.
    'body.function-wrap[__log_extra_fields__()].prompt' instead of the
    clean 'body.prompt' an API consumer would recognize.
    """

    @pytest.mark.parametrize(
        "loc,expected",
        [
            (("body", "prompt"), "body.prompt"),
            (("body", "messages", 2, "content"), "body.messages.2.content"),
            (
                ("body", "function-wrap[__log_extra_fields__()]", "prompt"),
                "body.prompt",
            ),
            (("body", "stop", "str"), "body.stop"),
            (("body", "stop", "list[str]"), "body.stop"),
            (
                ("body", "prompt", "list[constrained-int]"),
                "body.prompt",
            ),
        ],
    )
    def test_strips_internal_markers(self, loc, expected):
        assert clean_loc_for_param(loc) == expected

    def test_all_internal_falls_back_to_raw_join(self):
        """If every loc segment looks internal, fall back to the raw
        dot-join rather than returning an empty string."""
        loc = ("function-wrap[__log_extra_fields__()]",)
        assert clean_loc_for_param(loc) == "function-wrap[__log_extra_fields__()]"


class TestValidationErrorDoesNotLeakServerPaths:
    """FastAPI sets endpoint_file/line/function/path on the exception
    during routing, not in the constructor - so we set them manually
    here to reproduce the real leak. See #31683."""

    @pytest.mark.asyncio
    async def test_handler_strips_endpoint_file_context(self):
        errors = [
            {
                "type": "list_type",
                "loc": ("body", "messages"),
                "msg": "Input should be a valid list",
                "input": "not-a-list",
            }
        ]
        exc = RequestValidationError(errors)
        exc.endpoint_file = (
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "entrypoints/serve/utils/api_utils.py"
        )
        exc.endpoint_line = 40
        exc.endpoint_function = "create_chat_completion"
        exc.endpoint_path = "POST /v1/chat/completions"

        response = await validation_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)
        message = body["error"]["message"]

        assert "/usr/local/" not in message
        assert "api_utils.py" not in message
        assert "create_chat_completion" not in message
        assert "list_type" in message
        assert "Input should be a valid list" in message

    @pytest.mark.asyncio
    async def test_handler_strips_endpoint_path_only_variant(self):
        """Covers the branch where only endpoint_path is set."""
        errors = [
            {"type": "missing", "loc": ("body", "messages"), "msg": "Field required"}
        ]
        exc = RequestValidationError(errors)
        exc.endpoint_path = "POST /v1/chat/completions"

        response = await validation_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)
        message = body["error"]["message"]

        assert "missing" in message
        assert "Field required" in message
