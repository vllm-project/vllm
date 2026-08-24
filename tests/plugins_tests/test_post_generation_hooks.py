# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for optional EndpointPlugin.post_generation hooks (RFC #43999)."""

import asyncio

import pytest

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.plugins.endpoint_plugins.interface import EndpointPlugin
from vllm.plugins.endpoint_plugins.post_generation import (
    PostGenerationOutcome,
    apply_post_generation_hooks,
    refusal_message,
)


def _request_output(text: str = "hello world") -> RequestOutput:
    return RequestOutput(
        request_id="req-1",
        prompt="hi",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[3, 4],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
    )


class _AnnotatePlugin:
    name = "annotate"
    blocking = False

    async def post_generation(self, ctx):
        return {"score": 0.25, "generated_text": ctx.generated_text}


class _BlockPlugin:
    name = "blocker"
    blocking = True

    async def post_generation(self, ctx):
        if "secret" in ctx.generated_text:
            return {"block": True, "replacement": "refused"}
        return {"block": False}


class _RefusePlugin:
    name = "refuser"
    blocking = True

    async def post_generation(self, ctx):
        return {"block": True, "message": "not allowed"}


class _TimeoutPlugin:
    name = "slow"
    blocking = True
    timeout_ms = 10

    async def post_generation(self, ctx):
        await asyncio.sleep(1)
        return {"block": True}


class _TimeoutAnnotatePlugin:
    name = "slow_annotate"
    blocking = False
    timeout_ms = 10

    async def post_generation(self, ctx):
        await asyncio.sleep(1)
        return {"score": 1}


class _RouteOnlyPlugin:
    name = "route_only"
    required_tasks = None

    def attach_router(self, app):
        return

    async def init_state(self, engine_client, state, args):
        return


def test_route_only_plugin_still_satisfies_protocol():
    assert isinstance(_RouteOnlyPlugin(), EndpointPlugin)


@pytest.mark.asyncio
async def test_no_plugins_is_noop():
    output = _request_output()
    outcome = await apply_post_generation_hooks([], output)
    assert not outcome.blocked
    assert output.metadata is None
    assert output.outputs[0].text == "hello world"


@pytest.mark.asyncio
async def test_route_only_plugin_is_skipped():
    output = _request_output()
    outcome = await apply_post_generation_hooks(
        [_RouteOnlyPlugin()], output
    )
    assert not outcome.blocked
    assert output.metadata is None


@pytest.mark.asyncio
async def test_annotation_scores_are_stored_on_metadata():
    output = _request_output()
    outcome = await apply_post_generation_hooks([_AnnotatePlugin()], output)
    assert not outcome.blocked
    assert output.metadata["external_scores"]["annotate"]["score"] == 0.25
    assert (
        output.metadata["external_scores"]["annotate"]["generated_text"]
        == "hello world"
    )


@pytest.mark.asyncio
async def test_blocking_plugin_replaces_text():
    output = _request_output("leaked secret")
    outcome = await apply_post_generation_hooks([_BlockPlugin()], output)
    assert outcome.blocked
    assert outcome.replacement == "refused"
    assert output.outputs[0].text == "refused"


@pytest.mark.asyncio
async def test_blocking_plugin_without_replacement_refuses():
    output = _request_output()
    outcome = await apply_post_generation_hooks([_RefusePlugin()], output)
    assert outcome.blocked
    assert outcome.replacement is None
    assert outcome.error_message == "not allowed"


@pytest.mark.asyncio
async def test_blocking_timeout_fail_closed():
    output = _request_output()
    outcome = await apply_post_generation_hooks([_TimeoutPlugin()], output)
    assert outcome.blocked
    assert "timed out" in (outcome.error_message or "")


@pytest.mark.asyncio
async def test_annotation_timeout_fail_open():
    output = _request_output()
    outcome = await apply_post_generation_hooks(
        [_TimeoutAnnotatePlugin()], output
    )
    assert not outcome.blocked
    assert output.outputs[0].text == "hello world"


def test_non_streaming_replacement_is_not_a_refusal():
    outcome = PostGenerationOutcome(blocked=True, replacement="refused")
    assert refusal_message(outcome, streaming=False) is None


def test_streaming_block_with_replacement_is_a_refusal():
    outcome = PostGenerationOutcome(blocked=True, replacement="refused")
    message = refusal_message(outcome, streaming=True)
    assert message is not None
    assert "already-streamed" in message


def test_streaming_block_without_replacement_uses_error_message():
    outcome = PostGenerationOutcome(blocked=True, error_message="not allowed")
    assert refusal_message(outcome, streaming=True) == "not allowed"
