# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    load_aware_call,
    sanitize_message,
    should_include_usage,
    with_cancellation,
)


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )


def _make_load_tracking_request(disconnect_delay: float):
    state = SimpleNamespace(
        enable_server_load_tracking=True,
        server_load_metrics=0,
    )
    app = SimpleNamespace(state=state)

    async def receive():
        await asyncio.sleep(disconnect_delay)
        return {"type": "http.disconnect"}

    return SimpleNamespace(app=app, receive=receive), state


async def _run_background(response: JSONResponse | None) -> None:
    if response is None or response.background is None:
        return
    await response.background()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("disconnect_delay", "handler_delay"),
    [
        (0.05, 0.0),  # handler completes first
        (0.0, 0.05),  # disconnect wins
        (0.0, 0.0),  # simultaneous completion (previously double-decremented)
    ],
    ids=["handler-first", "disconnect-first", "simultaneous"],
)
async def test_server_load_never_negative_on_disconnect(
    disconnect_delay: float, handler_delay: float
):
    """with_cancellation + load_aware_call must not double-decrement load.

    When the handler and disconnect tasks finish in the same wait(), the old
    listen_for_disconnect path decremented load and the JSONResponse
    background task decremented again, driving server_load_metrics to -1.
    """
    raw_request, state = _make_load_tracking_request(disconnect_delay)

    @with_cancellation
    @load_aware_call
    async def handler(_self, raw_request):
        await asyncio.sleep(handler_delay)
        return JSONResponse({"ok": True})

    response = await handler(None, raw_request)
    await _run_background(response)

    assert state.server_load_metrics == 0


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )


class TestSanitizeMessageFilePaths:
    """sanitize_message should also strip file paths and traceback
    frames, not just memory addresses - see #31683."""

    def test_strips_traceback_style_frame(self):
        msg = (
            "1 validation error:\n"
            "  {'type': 'list_type', 'loc': ('body', 'messages')}\n"
            '\n  File "/usr/local/lib/python3.12/dist-packages/vllm/'
            'entrypoints/serve/utils/api_utils.py", line 40, '
            "in create_chat_completion\n"
            "    POST /v1/chat/completions"
        )
        result = sanitize_message(msg)
        assert "/usr/local/" not in result
        assert "api_utils.py" not in result
        assert "list_type" in result

    def test_strips_arbitrary_absolute_path(self):
        result = sanitize_message("Error in /home/user/project/vllm/server.py")
        assert "/home/user" not in result

    def test_strips_single_parent_container_path(self):
        """Regression: /app/server.py and /workspace/server.py (common in
        container deployments) were missed by the original {2,} quantifier."""
        assert "/app/" not in sanitize_message("Error in /app/server.py")
        assert "/workspace/" not in sanitize_message("Error in /workspace/server.py")

    def test_preserves_api_endpoint_paths(self):
        msg = "POST /v1/chat/completions failed"
        assert "/v1/chat/completions" in sanitize_message(msg)

    def test_preserves_short_field_references(self):
        msg = "Invalid value for field 'body.messages'"
        assert sanitize_message(msg) == msg

    def test_strips_both_address_and_path(self):
        msg = (
            "<Request at 0x7f123> failed at "
            "/usr/local/lib/python3.12/dist-packages/vllm/server.py"
        )
        result = sanitize_message(msg)
        assert "0x" not in result
        assert "/usr/local/" not in result
