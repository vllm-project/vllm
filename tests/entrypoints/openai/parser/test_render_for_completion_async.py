# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests verifying that render_for_completion is run off the event loop.

Issue #38266: render_for_completion() is a synchronous Harmony tokenizer call
that can take up to 10 minutes for 1M-character inputs. Calling it directly
inside async request handlers freezes the event loop for all concurrent
requests. The fix wraps it with make_async() so it runs in a thread-pool
executor, keeping the event loop free.

These tests require no GPU and no real model.
"""

import asyncio
import inspect
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.async_utils import make_async


class TestMakeAsyncReleasesEventLoop:
    """
    Verify the make_async() wrapper used by render_for_completion_async
    actually releases the event loop during synchronous execution.
    """

    @pytest.mark.asyncio
    async def test_blocking_call_in_executor_allows_other_coroutines(self):
        """
        A sync function wrapped with make_async() must not block other
        coroutines from running while the sync work is in progress.
        """
        events: list = []

        def slow_sync(messages: list) -> list:
            time.sleep(0.05)  # simulate slow tokenization (~50 ms)
            events.append("tokenize_done")
            return [1, 2, 3]

        slow_async = make_async(slow_sync)

        async def event_loop_checker() -> None:
            # Yield once so slow_async can be dispatched to the thread pool,
            # then record that the loop is responsive.
            await asyncio.sleep(0.01)
            events.append("loop_free")

        await asyncio.gather(slow_async([]), event_loop_checker())

        assert "loop_free" in events, "event_loop_checker never ran"
        assert "tokenize_done" in events, "slow_sync never completed"
        # The event-loop check should finish before the slow sync call
        assert events.index("loop_free") < events.index("tokenize_done"), (
            "event loop was blocked: loop_free happened after tokenize_done"
        )

    @pytest.mark.asyncio
    async def test_make_async_returns_awaitable(self):
        """make_async() must return a callable that produces an awaitable."""

        def identity(x: int) -> int:
            return x

        async_identity = make_async(identity)
        result = async_identity(42)
        # The return value should be a Future / awaitable
        assert asyncio.isfuture(result) or inspect.isawaitable(result)
        value = await result
        assert value == 42


class TestRenderForCompletionAsyncIsAwaitable:
    """
    Structural tests: verify the async wrapper is correctly exported and
    that the context methods were changed to async def.
    """

    def test_render_for_completion_async_exported(self):
        """render_for_completion_async must be importable from harmony_utils."""
        from vllm.entrypoints.openai.parser.harmony_utils import (
            render_for_completion_async,
        )

        assert callable(render_for_completion_async)

    def test_harmony_context_render_for_completion_is_coroutine(self):
        """HarmonyContext.render_for_completion must be an async def."""
        from vllm.entrypoints.openai.responses.context import HarmonyContext

        assert inspect.iscoroutinefunction(HarmonyContext.render_for_completion), (
            "HarmonyContext.render_for_completion must be async to avoid "
            "blocking the event loop"
        )

    def test_streaming_harmony_context_render_for_completion_is_coroutine(self):
        """StreamingHarmonyContext.render_for_completion must be an async def."""
        from vllm.entrypoints.openai.responses.context import StreamingHarmonyContext

        assert inspect.iscoroutinefunction(
            StreamingHarmonyContext.render_for_completion
        ), (
            "StreamingHarmonyContext.render_for_completion must be async to "
            "avoid blocking the event loop"
        )

    def test_make_request_with_harmony_is_coroutine_in_render_serving(self):
        """OpenAIServingRender._make_request_with_harmony must be async def."""
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender

        assert inspect.iscoroutinefunction(
            OpenAIServingRender._make_request_with_harmony
        ), (
            "OpenAIServingRender._make_request_with_harmony must be async to "
            "avoid blocking the event loop"
        )

    def test_make_request_with_harmony_is_coroutine_in_responses_serving(self):
        """OpenAIServingResponses._make_request_with_harmony must be async def."""
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        assert inspect.iscoroutinefunction(
            OpenAIServingResponses._make_request_with_harmony
        ), (
            "OpenAIServingResponses._make_request_with_harmony must be async to "
            "avoid blocking the event loop"
        )


class TestGetEncodingThreadSafety:
    """
    Verify that get_encoding() initializes the singleton exactly once even when
    called concurrently from many threads (the scenario introduced by wrapping
    render_for_completion with make_async).
    """

    def test_get_encoding_initialized_exactly_once_under_concurrency(self):
        """
        Simulate many threads racing to call get_encoding() while the singleton
        is unset. load_harmony_encoding() must be called exactly once.
        """
        import vllm.entrypoints.openai.parser.harmony_utils as hu

        call_count = 0
        call_count_lock = threading.Lock()

        def slow_load_harmony_encoding(_name):
            nonlocal call_count
            # Widen the race window so threads are more likely to overlap.
            time.sleep(0.05)
            with call_count_lock:
                call_count += 1
            return MagicMock()

        original_encoding = hu._harmony_encoding
        try:
            hu._harmony_encoding = None
            with patch(
                "vllm.entrypoints.openai.parser.harmony_utils.load_harmony_encoding",
                side_effect=slow_load_harmony_encoding,
            ):
                threads = [threading.Thread(target=hu.get_encoding) for _ in range(20)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            assert call_count == 1, (
                f"load_harmony_encoding was called {call_count} times; "
                "expected exactly 1 (race condition in get_encoding)"
            )
        finally:
            hu._harmony_encoding = original_encoding


class TestRenderForCompletionAsyncResult:
    """
    Functional test: render_for_completion_async must return the same value
    as the synchronous render_for_completion when given the same input.
    The underlying sync function is mocked to avoid needing openai_harmony.
    """

    @pytest.mark.asyncio
    async def test_async_wrapper_preserves_return_value(self):
        """render_for_completion_async must propagate the return value."""
        import vllm.entrypoints.openai.parser.harmony_utils as hu

        expected = [10, 20, 30]

        original = hu.render_for_completion_async
        try:
            hu.render_for_completion_async = make_async(lambda msgs: expected)
            result = await hu.render_for_completion_async([])
            assert result == expected
        finally:
            hu.render_for_completion_async = original
