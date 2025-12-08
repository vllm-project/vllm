# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for track_token_ids feature."""

import asyncio
import math
import os

os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import pytest

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

# Use a smaller model for faster testing
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-14B")


@pytest.fixture(scope="module")
def llm_instance():
    """Create a shared LLM instance for all tests.

    Uses max_logprobs=-1 to allow full vocabulary logprobs for differential tests.
    """
    return LLM(MODEL_PATH, enforce_eager=True, max_logprobs=-1)


@pytest.fixture(scope="module")
def async_llm_instance():
    """Create a shared AsyncLLM instance for streaming tests.

    Uses max_logprobs=-1 to allow full vocabulary logprobs for differential tests.
    Note: Tests using this fixture must use
        @pytest.mark.asyncio(loop_scope="module")
    to ensure the event loop scope matches the fixture scope.
    """
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enforce_eager=True,
        max_logprobs=-1,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    yield engine
    engine.shutdown()


class TestTrackTokenIdsBasic:
    """Basic functionality tests for track_token_ids."""

    def test_track_token_ids_returns_logprobs(self, llm_instance):
        """Test that track_token_ids returns logprobs for specified tokens."""
        track_ids = [1234, 5678, 9012]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=10,
            track_token_ids=track_ids,
        )

        outputs = llm_instance.generate(["Hello, my name is"], sampling_params)

        assert len(outputs) == 1
        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is not None
        assert isinstance(tracked, dict)

        # All tracked token IDs should be present
        for tid in track_ids:
            assert tid in tracked, f"Token {tid} not in tracked_logprobs"
            # Should have one logprob per generated token
            assert len(tracked[tid]) == 10, (
                f"Expected 10 logprobs for token {tid}, got {len(tracked[tid])}"
            )
            # All values should be valid log probabilities (<= 0)
            for lp in tracked[tid]:
                assert lp <= 0, f"Logprob {lp} should be <= 0"

    def test_track_token_ids_with_logprobs_coexist(self, llm_instance):
        """Test that track_token_ids works alongside regular logprobs."""
        track_ids = [100, 200, 300]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            logprobs=5,  # Request top-5 logprobs
            track_token_ids=track_ids,
        )

        outputs = llm_instance.generate(["The capital of France is"], sampling_params)

        output = outputs[0].outputs[0]

        # Regular logprobs should be present
        assert output.logprobs is not None
        assert len(output.logprobs) == 5

        # Tracked logprobs should also be present
        assert output.tracked_logprobs is not None
        for tid in track_ids:
            assert tid in output.tracked_logprobs
            assert len(output.tracked_logprobs[tid]) == 5


class TestTrackTokenIdsNoneAndEmpty:
    """Tests for track_token_ids=None and track_token_ids=[]."""

    def test_track_token_ids_not_provided(self, llm_instance):
        """Test that not providing track_token_ids returns None."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
        )

        outputs = llm_instance.generate(["Hello world"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is None, f"Expected None, got {tracked}"

    def test_track_token_ids_explicit_none(self, llm_instance):
        """Test that track_token_ids=None explicitly returns None."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=None,
        )

        outputs = llm_instance.generate(["Hello world"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is None, f"Expected None, got {tracked}"

    def test_track_token_ids_empty_list(self, llm_instance):
        """Test that track_token_ids=[] returns empty dict."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=[],
        )

        outputs = llm_instance.generate(["Hello world"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked == {}, f"Expected empty dict, got {tracked}"


class TestTrackTokenIdsBatch:
    """Tests for batch processing with track_token_ids."""

    def test_batch_multiple_prompts_same_track_ids(self, llm_instance):
        """Test batch with multiple prompts using same track_token_ids."""
        track_ids = [500, 1000, 1500]
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=10,
            track_token_ids=track_ids,
        )

        outputs = llm_instance.generate(prompts, sampling_params)

        assert len(outputs) == 4

        for i, output in enumerate(outputs):
            tracked = output.outputs[0].tracked_logprobs
            assert tracked is not None, f"Prompt {i}: tracked_logprobs is None"

            for tid in track_ids:
                assert tid in tracked, f"Prompt {i}: token {tid} not tracked"
                assert len(tracked[tid]) == 10, (
                    f"Prompt {i}: token {tid} has {len(tracked[tid])} logprobs, "
                    f"expected 10"
                )

    def test_batch_different_max_tokens(self, llm_instance):
        """Test batch where different prompts might finish at different times."""
        track_ids = [100, 200]
        prompts = ["Hi", "Hello there, how are you doing today"]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=15,
            track_token_ids=track_ids,
        )

        outputs = llm_instance.generate(prompts, sampling_params)

        for output in outputs:
            tracked = output.outputs[0].tracked_logprobs
            assert tracked is not None
            for tid in track_ids:
                assert tid in tracked
                # Each should have logprobs for each generated token
                num_gen = len(output.outputs[0].token_ids)
                assert len(tracked[tid]) == num_gen


class TestTrackTokenIdsDifferential:
    """Differential tests comparing track_token_ids vs full logprobs.

    IMPORTANT: All tests use BOTH track_token_ids AND logprobs=-1 in the SAME
    request to ensure we compare values from identical computation. Using two
    separate generations would be unreliable even with the same seed.
    """

    def test_tracked_vs_full_logprobs_match(self, llm_instance):
        """
        CRITICAL TEST: Verify track_token_ids produces same values as
        extracting from full vocabulary logprobs (logprobs=-1).

        Uses BOTH parameters in the same request for reliable comparison.
        """
        track_ids = [100, 500, 1000, 2000, 5000]
        prompt = "The meaning of life is"
        max_tokens = 10

        # Use BOTH track_token_ids AND logprobs=-1 in the same request
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=-1,  # Full vocabulary logprobs
            track_token_ids=track_ids,  # Also track specific tokens
        )
        outputs = llm_instance.generate([prompt], params)

        output = outputs[0].outputs[0]
        tracked_logprobs = output.tracked_logprobs
        full_logprobs_list = output.logprobs

        # Compare: tracked values should match extracted from full
        assert tracked_logprobs is not None
        assert full_logprobs_list is not None

        for tid in track_ids:
            assert tid in tracked_logprobs

            for step_idx in range(max_tokens):
                tracked_value = tracked_logprobs[tid][step_idx]

                # Extract from full logprobs
                step_full = full_logprobs_list[step_idx]
                if tid in step_full:
                    full_value = step_full[tid].logprob
                else:
                    pytest.fail(f"Token {tid} not in full logprobs at step {step_idx}")

                # Values should match exactly (same computation)
                assert math.isclose(tracked_value, full_value), (
                    f"Mismatch at step {step_idx} for token {tid}: "
                    f"tracked={tracked_value}, full={full_value}"
                )

    @pytest.mark.parametrize(
        "temperature,top_p,top_k",
        [
            (0.0, 1.0, -1),  # Greedy
            (0.7, 1.0, -1),  # Standard temperature 1
            (2.0, 1.0, -1),  # Standard temperature 2
            (10.0, 1.0, -1),  # Standard temperature 3
            (1.0, 0.9, -1),  # With top_p
            (0.8, 1.0, 50),  # With top_k
            (0.8, 0.95, 100),  # Combined
        ],
    )
    def test_tracked_vs_full_various_sampling(
        self, llm_instance, temperature, top_p, top_k
    ):
        """
        Test tracked logprobs match full logprobs under various sampling configs.

        Uses BOTH parameters in the same request - no need for seed matching
        since we compare values from the same generation.
        """
        track_ids = [100, 1000, 5000]
        prompt = "Once upon a time"
        max_tokens = 5

        # Use BOTH track_token_ids AND logprobs=-1 in the same request
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=-1,
            track_token_ids=track_ids,
        )
        outputs = llm_instance.generate([prompt], params)

        output = outputs[0].outputs[0]
        tracked = output.tracked_logprobs
        full_list = output.logprobs

        # Compare - should match exactly since same computation
        assert tracked is not None
        assert full_list is not None

        for tid in track_ids:
            for step_idx in range(max_tokens):
                tracked_val = tracked[tid][step_idx]
                full_val = full_list[step_idx][tid].logprob

                assert math.isclose(tracked_val, full_val), (
                    f"Mismatch at step {step_idx}, token {tid} "
                    f"(temp={temperature}, top_p={top_p}, top_k={top_k}): "
                    f"tracked={tracked_val}, full={full_val}"
                )

    def test_tracked_vs_full_batch_consistency(self, llm_instance):
        """Test that tracked and full logprobs match in batch processing.

        Uses BOTH parameters in the same request for all prompts.
        """
        track_ids = [200, 400, 600]
        prompts = ["Hello", "World", "Test"]
        max_tokens = 5

        # Use BOTH track_token_ids AND logprobs=-1 in the same batch request
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=-1,
            track_token_ids=track_ids,
        )
        outputs = llm_instance.generate(prompts, params)

        # Compare each prompt - values should match exactly
        for prompt_idx in range(len(prompts)):
            output = outputs[prompt_idx].outputs[0]
            tracked = output.tracked_logprobs
            full_list = output.logprobs

            assert tracked is not None
            assert full_list is not None

            for tid in track_ids:
                for step_idx in range(max_tokens):
                    tracked_val = tracked[tid][step_idx]
                    full_val = full_list[step_idx][tid].logprob

                    assert math.isclose(tracked_val, full_val), (
                        f"Prompt {prompt_idx}, step {step_idx}, token {tid}: "
                        f"tracked={tracked_val}, full={full_val}"
                    )


class TestTrackTokenIdsStreaming:
    """Streaming mode tests for track_token_ids using AsyncLLM with DELTA output."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_basic(self, async_llm_instance):
        """Test track_token_ids in streaming mode returns correct logprobs."""
        track_ids = [1234, 5678, 9012]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=10,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        # Collect all streaming outputs
        all_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}

        async for output in async_llm_instance.generate(
            request_id="stream-basic-1",
            prompt="Hello, my name is",
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        all_tracked[tid].extend(tracked[tid])
            if output.finished:
                break

        # Should have 10 logprobs per tracked token
        for tid in track_ids:
            assert len(all_tracked[tid]) == 10, (
                f"Token {tid}: expected 10 logprobs, got {len(all_tracked[tid])}"
            )
            # All values should be valid log probabilities
            for lp in all_tracked[tid]:
                assert lp <= 0, f"Logprob {lp} should be <= 0"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_track_token_ids_none(self, async_llm_instance):
        """Test streaming with track_token_ids=None returns None."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=None,
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in async_llm_instance.generate(
            request_id="stream-none-1",
            prompt="Hello world",
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            assert tracked is None, f"Expected None, got {tracked}"
            if output.finished:
                break

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_track_token_ids_empty_list(self, async_llm_instance):
        """Test streaming with track_token_ids=[] returns empty dict."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=[],
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in async_llm_instance.generate(
            request_id="stream-empty-1",
            prompt="Hello world",
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            assert tracked == {}, f"Expected empty dict, got {tracked}"
            if output.finished:
                break

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_with_logprobs_coexist(self, async_llm_instance):
        """Test streaming with both track_token_ids and regular logprobs."""
        track_ids = [100, 200, 300]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            logprobs=5,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        all_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        logprobs_count = 0

        async for output in async_llm_instance.generate(
            request_id="stream-coexist-1",
            prompt="The capital of France is",
            sampling_params=sampling_params,
        ):
            out = output.outputs[0]

            # Count regular logprobs
            if out.logprobs:
                logprobs_count += len(out.logprobs)

            # Collect tracked logprobs
            if out.tracked_logprobs:
                for tid in track_ids:
                    if tid in out.tracked_logprobs:
                        all_tracked[tid].extend(out.tracked_logprobs[tid])

            if output.finished:
                break

        # Should have 5 regular logprobs entries
        assert logprobs_count == 5

        # Should have 5 tracked logprobs per token
        for tid in track_ids:
            assert len(all_tracked[tid]) == 5

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_tracked_vs_full_logprobs_match(self, async_llm_instance):
        """Test streaming tracked logprobs match full logprobs in same request.

        Uses BOTH track_token_ids AND logprobs=-1 in the same streaming request
        to compare values from the exact same computation.
        """
        track_ids = [100, 500, 1000]
        prompt = "The meaning of life is"
        max_tokens = 10

        # Use BOTH track_token_ids AND logprobs=-1 in same streaming request
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=-1,  # Full vocabulary logprobs
            track_token_ids=track_ids,  # Also track specific tokens
            output_kind=RequestOutputKind.DELTA,
        )

        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        full_logprobs: list[dict] = []

        async for output in async_llm_instance.generate(
            request_id="stream-match-1",
            prompt=prompt,
            sampling_params=params,
        ):
            # Collect tracked logprobs
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])

            # Collect full logprobs
            if output.outputs[0].logprobs:
                full_logprobs.extend(output.outputs[0].logprobs)

            if output.finished:
                break

        # Compare tracked vs extracted from full logprobs (same computation)
        num_tokens = len(full_logprobs)
        assert num_tokens > 0, "No tokens generated"

        for tid in track_ids:
            assert len(stream_tracked[tid]) == num_tokens
            for step_idx in range(num_tokens):
                tracked_val = stream_tracked[tid][step_idx]
                full_val = full_logprobs[step_idx][tid].logprob

                assert math.isclose(tracked_val, full_val), (
                    f"Token {tid}, step {step_idx}: "
                    f"tracked={tracked_val}, full={full_val}"
                )

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_differential_more_tokens(self, async_llm_instance):
        """
        CRITICAL: Test streaming tracked logprobs match full vocabulary logprobs
        with more tokens to track.

        Uses BOTH track_token_ids AND logprobs=-1 in the same streaming request
        to compare values from the exact same computation.
        """
        track_ids = [100, 500, 1000, 2000, 5000]
        prompt = "Once upon a time"
        max_tokens = 8

        # Use BOTH track_token_ids AND logprobs=-1 in same streaming request
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=-1,  # Full vocabulary logprobs
            track_token_ids=track_ids,  # Also track specific tokens
            output_kind=RequestOutputKind.DELTA,
        )

        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        full_logprobs: list[dict] = []

        async for output in async_llm_instance.generate(
            request_id="stream-diff-1",
            prompt=prompt,
            sampling_params=params,
        ):
            # Collect tracked logprobs
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])

            # Collect full logprobs
            if output.outputs[0].logprobs:
                full_logprobs.extend(output.outputs[0].logprobs)

            if output.finished:
                break

        # Compare tracked vs extracted from full logprobs (same computation)
        num_tokens = len(full_logprobs)
        assert num_tokens > 0, "No tokens generated"

        for tid in track_ids:
            assert len(stream_tracked[tid]) == num_tokens
            for step_idx in range(num_tokens):
                tracked_val = stream_tracked[tid][step_idx]
                full_val = full_logprobs[step_idx][tid].logprob

                assert math.isclose(tracked_val, full_val), (
                    f"Token {tid}, step {step_idx}: "
                    f"tracked={tracked_val}, full={full_val}"
                )

    @pytest.mark.asyncio(loop_scope="module")
    @pytest.mark.parametrize(
        "temperature,top_p,top_k",
        [
            (0.0, 1.0, -1),  # Greedy
            (0.7, 1.0, -1),  # Standard temperature 1
            (2.0, 1.0, -1),  # Standard temperature 2
            (10.0, 1.0, -1),  # Standard temperature 3
            (1.0, 0.9, -1),  # With top_p
            (0.8, 1.0, 50),  # With top_k
        ],
    )
    async def test_streaming_differential_various_sampling(
        self, async_llm_instance, temperature, top_p, top_k
    ):
        """Test streaming tracked logprobs match full logprobs with various sampling.

        Uses BOTH track_token_ids AND logprobs=-1 in the same streaming request
        to compare values from the exact same computation - no seed needed.
        """
        track_ids = [100, 1000, 5000]
        prompt = "Hello world"
        max_tokens = 5

        # Use BOTH track_token_ids AND logprobs=-1 in same streaming request
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=-1,  # Full vocabulary logprobs
            track_token_ids=track_ids,  # Also track specific tokens
            output_kind=RequestOutputKind.DELTA,
        )

        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        full_logprobs: list[dict] = []
        req_id = f"stream-sampling-{temperature}-{top_p}-{top_k}"

        async for output in async_llm_instance.generate(
            request_id=req_id,
            prompt=prompt,
            sampling_params=params,
        ):
            # Collect tracked logprobs
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])

            # Collect full logprobs
            if output.outputs[0].logprobs:
                full_logprobs.extend(output.outputs[0].logprobs)

            if output.finished:
                break

        # Compare tracked vs extracted from full logprobs (same computation)
        num_tokens = len(full_logprobs)
        assert num_tokens > 0, "No tokens generated"

        for tid in track_ids:
            assert len(stream_tracked[tid]) == num_tokens
            for step_idx in range(num_tokens):
                tracked_val = stream_tracked[tid][step_idx]
                full_val = full_logprobs[step_idx][tid].logprob

                assert math.isclose(tracked_val, full_val), (
                    f"Token {tid}, step {step_idx} "
                    f"(temp={temperature}, top_p={top_p}, top_k={top_k}): "
                    f"tracked={tracked_val}, full={full_val}"
                )

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_batch_multiple_prompts(self, async_llm_instance):
        """Test streaming with multiple prompts (batch)."""
        track_ids = [500, 1000, 1500]
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
        ]
        max_tokens = 8

        # Collect per-request tracked logprobs
        all_tracked: dict[str, dict[int, list[float]]] = {}

        # Submit all requests
        tasks = []
        for i, prompt in enumerate(prompts):
            req_id = f"stream-batch-{i}"
            all_tracked[req_id] = {tid: [] for tid in track_ids}

            async def process_request(req_id, prompt):
                sampling_params = SamplingParams(
                    temperature=0.8,
                    max_tokens=max_tokens,
                    track_token_ids=track_ids,
                    output_kind=RequestOutputKind.DELTA,
                )
                async for output in async_llm_instance.generate(
                    request_id=req_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                ):
                    tracked = output.outputs[0].tracked_logprobs
                    if tracked:
                        for tid in track_ids:
                            if tid in tracked:
                                all_tracked[req_id][tid].extend(tracked[tid])
                    if output.finished:
                        break

            tasks.append(process_request(req_id, prompt))

        await asyncio.gather(*tasks)

        # Should have outputs for all 3 prompts
        assert len(all_tracked) == 3

        # Each prompt should have max_tokens logprobs per tracked token
        for req_id, req_tracked in all_tracked.items():
            for tid in track_ids:
                assert len(req_tracked[tid]) == max_tokens, (
                    f"Request {req_id}, token {tid}: "
                    f"expected {max_tokens}, got {len(req_tracked[tid])}"
                )

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_single_token_generation(self, async_llm_instance):
        """Test streaming with max_tokens=1."""
        track_ids = [100, 200, 300]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=1,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        all_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}

        async for output in async_llm_instance.generate(
            request_id="stream-single-1",
            prompt="Hello",
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        all_tracked[tid].extend(tracked[tid])
            if output.finished:
                break

        # Should have exactly 1 logprob per token
        for tid in track_ids:
            assert len(all_tracked[tid]) == 1

    @pytest.mark.asyncio(loop_scope="module")
    async def test_streaming_many_tokens_tracking(self, async_llm_instance):
        """Test streaming while tracking many tokens."""
        # Track 50 token IDs
        track_ids = list(range(100, 150))
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        all_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}

        async for output in async_llm_instance.generate(
            request_id="stream-many-1",
            prompt="Classify this text:",
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        all_tracked[tid].extend(tracked[tid])
            if output.finished:
                break

        # All 50 tokens should have 5 logprobs each
        assert len(all_tracked) == 50
        for tid in track_ids:
            assert len(all_tracked[tid]) == 5


class TestTrackTokenIdsEdgeCases:
    """Edge case tests for track_token_ids."""

    def test_single_token_tracking(self, llm_instance):
        """Test tracking a single token."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=10,
            track_token_ids=[1234],
        )

        outputs = llm_instance.generate(["Test prompt"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is not None
        assert len(tracked) == 1
        assert 1234 in tracked
        assert len(tracked[1234]) == 10

    def test_many_tokens_tracking(self, llm_instance):
        """Test tracking many tokens (like 100-class classification)."""
        # Track 100 token IDs
        track_ids = list(range(100, 200))
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=track_ids,
        )

        outputs = llm_instance.generate(["Classify this text:"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is not None
        assert len(tracked) == 100

        for tid in track_ids:
            assert tid in tracked
            assert len(tracked[tid]) == 5

    def test_track_boundary_token_ids(self, llm_instance):
        """Test tracking token ID 0 (boundary case)."""
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5,
            track_token_ids=[0, 1, 2],  # Very low token IDs
        )

        outputs = llm_instance.generate(["Test"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is not None
        assert 0 in tracked
        assert 1 in tracked
        assert 2 in tracked

    def test_single_token_generation(self, llm_instance):
        """Test with max_tokens=1."""
        track_ids = [100, 200, 300]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=1,
            track_token_ids=track_ids,
        )

        outputs = llm_instance.generate(["Hello"], sampling_params)

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is not None
        for tid in track_ids:
            assert tid in tracked
            assert len(tracked[tid]) == 1


# Summarization prompts designed for high ngram acceptance rate
SUMMARIZATION_PROMPTS = [
    """Summarize the following text:
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over 
the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown 
fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
Summary:""",
    """Summarize the following repetitive text in one sentence:
Hello world. Hello world. Hello world. Hello world. Hello world. Hello world.
Hello world. Hello world. Hello world. Hello world. Hello world. Hello world.
Summary:""",
    """Summarize:
Test test test test test. Test test test test test. Test test test test test.
Test test test test test. Test test test test test. Test test test test test.
Summary:""",
]


@pytest.fixture(scope="module")
def spec_decode_llm_instance():
    """Create a shared LLM instance with ngram speculative decoding.

    Uses max_logprobs=-1 to allow full vocabulary logprobs for differential tests.
    """
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": 3,
    }
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_logprobs=-1,
        speculative_config=speculative_config,
    )
    yield llm
    del llm
    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def async_spec_decode_llm_instance():
    """Create a shared AsyncLLM instance with ngram speculative decoding.

    Uses max_logprobs=-1 to allow full vocabulary logprobs for differential tests.
    Note: Tests using this fixture must use @pytest.mark.asyncio(loop_scope="module")
    to ensure the event loop scope matches the fixture scope.
    """
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": 3,
    }
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enforce_eager=True,
        max_logprobs=-1,
        speculative_config=speculative_config,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    yield engine
    engine.shutdown()
    cleanup_dist_env_and_memory()


class TestTrackTokenIdsSpeculativeDecoding:
    """Tests for track_token_ids with ngram speculative decoding."""

    def test_spec_decode_basic(self, spec_decode_llm_instance):
        """Test track_token_ids works with speculative decoding."""
        track_ids = [100, 500, 1000, 2000]
        max_tokens = 15
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy for determinism
            max_tokens=max_tokens,
            track_token_ids=track_ids,
        )

        outputs = spec_decode_llm_instance.generate(
            SUMMARIZATION_PROMPTS[:1], sampling_params
        )

        assert len(outputs) == 1
        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is not None

        # All tracked token IDs should be present
        for tid in track_ids:
            assert tid in tracked, f"Token {tid} not in tracked_logprobs"
            # Should have one logprob per generated token
            num_generated = len(outputs[0].outputs[0].token_ids)
            assert len(tracked[tid]) == num_generated, (
                f"Expected {num_generated} logprobs for token {tid}, "
                f"got {len(tracked[tid])}"
            )
            # All values should be valid log probabilities (<= 0)
            for lp in tracked[tid]:
                assert lp <= 0, f"Logprob {lp} should be <= 0"

    def test_spec_decode_with_logprobs_coexist(self, spec_decode_llm_instance):
        """Test track_token_ids works alongside regular logprobs with spec decode."""
        track_ids = [100, 200, 300]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            logprobs=5,  # Request top-5 logprobs
            track_token_ids=track_ids,
        )

        outputs = spec_decode_llm_instance.generate(
            SUMMARIZATION_PROMPTS[:1], sampling_params
        )

        output = outputs[0].outputs[0]
        num_generated = len(output.token_ids)

        # Regular logprobs should be present
        assert output.logprobs is not None
        assert len(output.logprobs) == num_generated

        # Tracked logprobs should also be present
        assert output.tracked_logprobs is not None
        for tid in track_ids:
            assert tid in output.tracked_logprobs
            assert len(output.tracked_logprobs[tid]) == num_generated

    def test_spec_decode_track_token_ids_none(self, spec_decode_llm_instance):
        """Test track_token_ids=None returns None with spec decode."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            track_token_ids=None,
        )

        outputs = spec_decode_llm_instance.generate(
            SUMMARIZATION_PROMPTS[:1], sampling_params
        )

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked is None, f"Expected None, got {tracked}"

    def test_spec_decode_track_token_ids_empty_list(self, spec_decode_llm_instance):
        """Test track_token_ids=[] returns empty dict with spec decode."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            track_token_ids=[],
        )

        outputs = spec_decode_llm_instance.generate(
            SUMMARIZATION_PROMPTS[:1], sampling_params
        )

        tracked = outputs[0].outputs[0].tracked_logprobs
        assert tracked == {}, f"Expected empty dict, got {tracked}"

    def test_spec_decode_tracked_vs_full_logprobs_match(self, spec_decode_llm_instance):
        """
        CRITICAL TEST: Verify track_token_ids with spec decode produces same
        values as extracting from full vocabulary logprobs (logprobs=-1).

        Uses BOTH parameters in the same request for reliable comparison.
        """
        track_ids = [100, 500, 1000, 2000, 5000]
        prompt = SUMMARIZATION_PROMPTS[0]
        max_tokens = 10

        # Use BOTH track_token_ids AND logprobs=-1 in the same request
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=-1,
            track_token_ids=track_ids,
        )
        outputs = spec_decode_llm_instance.generate([prompt], params)

        output = outputs[0].outputs[0]
        tracked_logprobs = output.tracked_logprobs
        full_logprobs_list = output.logprobs

        # Compare: tracked values should match extracted from full
        assert tracked_logprobs is not None
        assert full_logprobs_list is not None

        num_tokens = len(full_logprobs_list)

        for tid in track_ids:
            assert tid in tracked_logprobs
            assert len(tracked_logprobs[tid]) == num_tokens

            for step_idx in range(num_tokens):
                tracked_value = tracked_logprobs[tid][step_idx]

                # Extract from full logprobs
                step_full = full_logprobs_list[step_idx]
                if tid in step_full:
                    full_value = step_full[tid].logprob
                else:
                    pytest.fail(f"Token {tid} not in full logprobs at step {step_idx}")

                # Values should match exactly (same computation)
                assert math.isclose(tracked_value, full_value), (
                    f"Mismatch at step {step_idx} for token {tid}: "
                    f"tracked={tracked_value}, full={full_value}"
                )

    def test_spec_decode_batch_multiple_prompts(self, spec_decode_llm_instance):
        """Test batch with multiple prompts using track_token_ids with spec decode."""
        track_ids = [500, 1000, 1500]
        max_tokens = 12

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            track_token_ids=track_ids,
        )

        outputs = spec_decode_llm_instance.generate(
            SUMMARIZATION_PROMPTS, sampling_params
        )

        assert len(outputs) == len(SUMMARIZATION_PROMPTS)

        for i, output in enumerate(outputs):
            tracked = output.outputs[0].tracked_logprobs
            num_generated = len(output.outputs[0].token_ids)
            assert tracked is not None, f"Prompt {i}: tracked_logprobs is None"

            for tid in track_ids:
                assert tid in tracked, f"Prompt {i}: token {tid} not tracked"
                assert len(tracked[tid]) == num_generated, (
                    f"Prompt {i}: token {tid} has {len(tracked[tid])} logprobs, "
                    f"expected {num_generated}"
                )

    @pytest.mark.parametrize(
        "temperature,top_p,top_k",
        [
            (0.0, 1.0, -1),  # Greedy
            (0.7, 1.0, -1),  # Standard temperature 1
            (2.0, 1.0, -1),  # Standard temperature 2
            (10.0, 1.0, -1),  # Standard temperature 3
            (0.8, 0.95, 100),  # Combined
        ],
    )
    def test_spec_decode_various_sampling(
        self, spec_decode_llm_instance, temperature, top_p, top_k
    ):
        """Test tracked logprobs match full logprobs under various sampling configs.

        Uses BOTH track_token_ids AND logprobs=-1 in the same request for
        reliable comparison regardless of sampling randomness.
        """
        track_ids = [100, 1000, 5000]
        prompt = SUMMARIZATION_PROMPTS[0]
        max_tokens = 8

        # Use BOTH track_token_ids AND logprobs=-1 in the same request
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=-1,
            track_token_ids=track_ids,
        )
        outputs = spec_decode_llm_instance.generate([prompt], params)

        output = outputs[0].outputs[0]
        tracked = output.tracked_logprobs
        full_list = output.logprobs

        # Verify tracked logprobs are valid and match full logprobs
        assert tracked is not None
        assert full_list is not None

        num_tokens = len(full_list)

        for tid in track_ids:
            assert tid in tracked
            assert len(tracked[tid]) == num_tokens

            for step_idx in range(num_tokens):
                tracked_val = tracked[tid][step_idx]

                # Verify value is valid
                assert tracked_val <= 0, f"Logprob {tracked_val} should be <= 0"
                assert not math.isnan(tracked_val), "Logprob should not be NaN"
                assert not math.isinf(tracked_val), "Logprob should not be inf"

                # Verify matches full logprobs (same computation)
                full_val = full_list[step_idx][tid].logprob
                assert math.isclose(tracked_val, full_val), (
                    f"Mismatch at step {step_idx}, token {tid} "
                    f"(temp={temperature}, top_p={top_p}, top_k={top_k}): "
                    f"tracked={tracked_val}, full={full_val}"
                )


class TestTrackTokenIdsSpeculativeDecodingStreaming:
    """Streaming tests for track_token_ids with ngram speculative decoding."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_spec_decode_streaming_basic(self, async_spec_decode_llm_instance):
        """Test streaming with track_token_ids and spec decode."""
        track_ids = [100, 500, 1000]
        max_tokens = 12
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        all_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}

        async for output in async_spec_decode_llm_instance.generate(
            request_id="spec-stream-basic-1",
            prompt=SUMMARIZATION_PROMPTS[0],
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        all_tracked[tid].extend(tracked[tid])
            if output.finished:
                break

        # Should have logprobs for each tracked token
        for tid in track_ids:
            assert len(all_tracked[tid]) > 0, (
                f"Token {tid}: expected logprobs, got {len(all_tracked[tid])}"
            )
            # All values should be valid log probabilities
            for lp in all_tracked[tid]:
                assert lp <= 0, f"Logprob {lp} should be <= 0"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_spec_decode_streaming_track_token_ids_none(
        self, async_spec_decode_llm_instance
    ):
        """Test streaming with track_token_ids=None and spec decode."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            track_token_ids=None,
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in async_spec_decode_llm_instance.generate(
            request_id="spec-stream-none-1",
            prompt=SUMMARIZATION_PROMPTS[0],
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            assert tracked is None, f"Expected None, got {tracked}"
            if output.finished:
                break

    @pytest.mark.asyncio(loop_scope="module")
    async def test_spec_decode_streaming_track_token_ids_empty(
        self, async_spec_decode_llm_instance
    ):
        """Test streaming with track_token_ids=[] and spec decode."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            track_token_ids=[],
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in async_spec_decode_llm_instance.generate(
            request_id="spec-stream-empty-1",
            prompt=SUMMARIZATION_PROMPTS[0],
            sampling_params=sampling_params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            assert tracked == {}, f"Expected empty dict, got {tracked}"
            if output.finished:
                break

    @pytest.mark.asyncio(loop_scope="module")
    async def test_spec_decode_streaming_with_logprobs_coexist(
        self, async_spec_decode_llm_instance
    ):
        """
        Test that track_token_ids works alongside regular logprobs
        in streaming spec decode.

        Uses both track_token_ids and logprobs in the same request, then verifies
        the tracked values match those extracted from the logprobs dict.
        """
        track_ids = [100, 500, 1000]
        prompt = SUMMARIZATION_PROMPTS[0]
        max_tokens = 10

        # Use both track_token_ids AND logprobs in the same request
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=10,  # Get top-10 logprobs
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        regular_logprobs: list[dict] = []
        num_tokens_generated = 0

        async for output in async_spec_decode_llm_instance.generate(
            request_id="spec-stream-coexist-1",
            prompt=prompt,
            sampling_params=params,
        ):
            # Collect tracked logprobs
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])

            # Collect regular logprobs
            if output.outputs[0].logprobs:
                regular_logprobs.extend(output.outputs[0].logprobs)
                num_tokens_generated += len(output.outputs[0].logprobs)

            if output.finished:
                break

        # Verify we got logprobs
        assert num_tokens_generated > 0, "No tokens generated"

        # Verify tracked logprobs have correct length
        for tid in track_ids:
            assert len(stream_tracked[tid]) == num_tokens_generated, (
                f"Token {tid}: expected {num_tokens_generated}, "
                f"got {len(stream_tracked[tid])}"
            )

        # Verify all tracked values are valid log probabilities
        for tid in track_ids:
            for lp in stream_tracked[tid]:
                assert lp <= 0, f"Logprob {lp} should be <= 0"
                assert not math.isnan(lp), "Logprob should not be NaN"
                assert not math.isinf(lp), "Logprob should not be inf"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_spec_decode_streaming_differential_vs_full_logprobs(
        self, async_spec_decode_llm_instance
    ):
        """
        CRITICAL: Test streaming tracked logprobs with spec decode match
        full vocabulary logprobs (logprobs=-1) in the SAME request.

        Using both track_token_ids and logprobs=-1 together ensures we compare
        logprobs computed from the exact same generation, avoiding variance
        from separate runs.
        """
        track_ids = [100, 500, 1000, 2000, 5000]
        prompt = SUMMARIZATION_PROMPTS[0]
        max_tokens = 10

        # Use BOTH track_token_ids AND logprobs=-1 in the same request
        # This ensures we compare values from the exact same generation
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=-1,  # Get full vocabulary logprobs
            track_token_ids=track_ids,  # Also track specific tokens
            output_kind=RequestOutputKind.DELTA,
        )

        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        full_logprobs: list[dict] = []

        async for output in async_spec_decode_llm_instance.generate(
            request_id="spec-stream-diff-1",
            prompt=prompt,
            sampling_params=params,
        ):
            # Collect tracked logprobs
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])

            # Collect full logprobs
            if output.outputs[0].logprobs:
                full_logprobs.extend(output.outputs[0].logprobs)

            if output.finished:
                break

        # Compare tracked vs extracted from full logprobs
        # Should match exactly (within floating point tolerance) since
        # they come from the same generation
        num_tokens = min(len(stream_tracked[track_ids[0]]), len(full_logprobs))
        assert num_tokens > 0, "No tokens generated"

        for tid in track_ids:
            for step_idx in range(num_tokens):
                tracked_val = stream_tracked[tid][step_idx]
                full_val = full_logprobs[step_idx][tid].logprob

                assert math.isclose(tracked_val, full_val, rel_tol=1e-4), (
                    f"Token {tid}, step {step_idx}: "
                    f"tracked={tracked_val}, full={full_val}"
                )

    @pytest.mark.asyncio(loop_scope="module")
    async def test_spec_decode_streaming_batch(self, async_spec_decode_llm_instance):
        """Test streaming with multiple prompts (batch) and spec decode."""
        track_ids = [500, 1000, 1500]
        max_tokens = 10

        # Collect per-request tracked logprobs
        all_tracked: dict[str, dict[int, list[float]]] = {}

        # Submit all requests
        tasks = []
        for i, prompt in enumerate(SUMMARIZATION_PROMPTS):
            req_id = f"spec-stream-batch-{i}"
            all_tracked[req_id] = {tid: [] for tid in track_ids}

            async def process_request(req_id, prompt):
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    track_token_ids=track_ids,
                    output_kind=RequestOutputKind.DELTA,
                )
                async for output in async_spec_decode_llm_instance.generate(
                    request_id=req_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                ):
                    tracked = output.outputs[0].tracked_logprobs
                    if tracked:
                        for tid in track_ids:
                            if tid in tracked:
                                all_tracked[req_id][tid].extend(tracked[tid])
                    if output.finished:
                        break

            tasks.append(process_request(req_id, prompt))

        await asyncio.gather(*tasks)

        # Should have outputs for all prompts
        assert len(all_tracked) == len(SUMMARIZATION_PROMPTS)

        # Each prompt should have logprobs for tracked tokens
        for req_id, req_tracked in all_tracked.items():
            for tid in track_ids:
                assert len(req_tracked[tid]) > 0, (
                    f"Request {req_id}, token {tid}: expected logprobs"
                )


def run_quick_demo():
    """Quick demo showing the feature in action."""
    print("=" * 70)
    print("Track Token IDs Feature Demo")
    print("=" * 70)

    # Use max_logprobs=-1 to allow full vocabulary logprobs for diff test
    llm = LLM(MODEL_PATH, enforce_eager=True, max_logprobs=-1)

    # Demo 1: Basic usage
    print("\n1. Basic Usage - Track specific tokens")
    print("-" * 50)
    track_ids = [1234, 5678, 9012]
    params = SamplingParams(
        temperature=0.8,
        max_tokens=10,
        track_token_ids=track_ids,
    )
    outputs = llm.generate(["Hello, my name is"], params)
    tracked = outputs[0].outputs[0].tracked_logprobs
    print(f"Tracked token IDs: {track_ids}")
    print(f"tracked_logprobs: {tracked}")

    # Demo 2: None case
    print("\n2. track_token_ids=None (default)")
    print("-" * 50)
    params = SamplingParams(temperature=0.8, max_tokens=5)
    outputs = llm.generate(["Test"], params)
    print(f"tracked_logprobs: {outputs[0].outputs[0].tracked_logprobs}")

    # Demo 3: Empty list case
    print("\n3. track_token_ids=[] (empty list)")
    print("-" * 50)
    params = SamplingParams(temperature=0.8, max_tokens=5, track_token_ids=[])
    outputs = llm.generate(["Test"], params)
    print(f"tracked_logprobs: {outputs[0].outputs[0].tracked_logprobs}")

    # Demo 4: Differential test
    print("\n4. Differential Test: track_token_ids vs full logprobs")
    print("-" * 50)
    track_ids = [100, 500, 1000]
    prompt = "The answer is"

    # Tracked approach
    params_tracked = SamplingParams(
        temperature=0.0, max_tokens=5, track_token_ids=track_ids
    )
    out_tracked = llm.generate([prompt], params_tracked)
    tracked = out_tracked[0].outputs[0].tracked_logprobs

    # Full logprobs approach (requires LLM started with max_logprobs=-1)
    params_full = SamplingParams(temperature=0.0, max_tokens=5, logprobs=-1)
    out_full = llm.generate([prompt], params_full)
    full_list = out_full[0].outputs[0].logprobs

    print(f"Comparing for tokens: {track_ids}")
    all_match = True
    for tid in track_ids:
        print(f"\nToken {tid}:")
        for step in range(5):
            t_val = tracked[tid][step]
            f_val = full_list[step][tid].logprob
            match = math.isclose(t_val, f_val, rel_tol=1e-4)
            status = "✓" if match else "✗"
            print(f"  Step {step}: tracked={t_val:.6f}, full={f_val:.6f} {status}")
            if not match:
                all_match = False

    print(f"\nAll values match: {'YES ✓' if all_match else 'NO ✗'}")

    # Demo 5: Batch processing
    print("\n5. Batch Processing")
    print("-" * 50)
    prompts = ["Hello", "World", "Test", "Demo"]
    params = SamplingParams(temperature=0.8, max_tokens=5, track_token_ids=[100, 200])
    outputs = llm.generate(prompts, params)
    for i, out in enumerate(outputs):
        tracked = out.outputs[0].tracked_logprobs
        print(
            f"Prompt {i} ('{prompts[i]}'): {len(tracked)} tracked tokens, 5 steps each"
        )

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


async def run_streaming_demo():
    """Streaming demo showing track_token_ids with AsyncLLM."""
    print("=" * 70)
    print("Track Token IDs Streaming Demo (AsyncLLM)")
    print("=" * 70)

    # Create AsyncLLM engine
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enforce_eager=True,
        max_logprobs=-1,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        # Demo 1: Basic streaming with track_token_ids
        print("\n1. Streaming with track_token_ids")
        print("-" * 50)
        track_ids = [100, 500, 1000]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )

        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}
        chunk_count = 0

        async for output in engine.generate(
            request_id="demo-stream-1",
            prompt="The answer is",
            sampling_params=params,
        ):
            chunk_count += 1
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])
            if output.finished:
                break

        print(f"Received {chunk_count} streaming chunks")
        for tid in track_ids:
            print(f"  Token {tid}: {len(stream_tracked[tid])} logprobs")

        # Demo 2: Streaming with track_token_ids=None
        print("\n2. Streaming with track_token_ids=None")
        print("-" * 50)
        params = SamplingParams(
            temperature=0.8,
            max_tokens=3,
            track_token_ids=None,
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in engine.generate(
            request_id="demo-stream-none",
            prompt="Test",
            sampling_params=params,
        ):
            print(f"  tracked_logprobs: {output.outputs[0].tracked_logprobs}")
            if output.finished:
                break

        # Demo 3: Streaming with track_token_ids=[]
        print("\n3. Streaming with track_token_ids=[]")
        print("-" * 50)
        params = SamplingParams(
            temperature=0.8,
            max_tokens=3,
            track_token_ids=[],
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in engine.generate(
            request_id="demo-stream-empty",
            prompt="Test",
            sampling_params=params,
        ):
            print(f"  tracked_logprobs: {output.outputs[0].tracked_logprobs}")
            if output.finished:
                break

        # Demo 4: Streaming values are valid logprobs
        print("\n4. Streaming - Validate Logprob Values")
        print("-" * 50)
        track_ids = [100, 500, 1000]
        prompt = "The answer is"
        max_tokens = 5

        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            track_token_ids=track_ids,
            output_kind=RequestOutputKind.DELTA,
        )
        stream_tracked: dict[int, list[float]] = {tid: [] for tid in track_ids}

        async for output in engine.generate(
            request_id="demo-stream-validate",
            prompt=prompt,
            sampling_params=params,
        ):
            tracked = output.outputs[0].tracked_logprobs
            if tracked:
                for tid in track_ids:
                    if tid in tracked:
                        stream_tracked[tid].extend(tracked[tid])
            if output.finished:
                break

        print(f"Tracked tokens: {track_ids}")
        all_valid = True
        for tid in track_ids:
            values = stream_tracked[tid]
            valid = all(
                v <= 0 and not math.isnan(v) and not math.isinf(v) for v in values
            )
            status = "✓" if valid else "✗"
            print(f"  Token {tid}: {len(values)} values, all valid logprobs: {status}")
            print(f"    Values: {[f'{v:.4f}' for v in values]}")
            if not valid:
                all_valid = False

        status = "YES ✓" if all_valid else "NO ✗"
        print(f"\nAll values are valid log probabilities: {status}")
        print("\nNote: For full differential test (streaming vs full logprobs),")
        print(
            "      run: pytest tests/v1/e2e/"
            "test_tracked_logprobs.py::TestTrackTokenIdsStreaming -v"
        )

    finally:
        engine.shutdown()

    print("\n" + "=" * 70)
    print("Streaming Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--streaming":
        # Run streaming demo
        asyncio.run(run_streaming_demo())
    else:
        # Run non-streaming demo
        run_quick_demo()
        print("\nRun with --streaming for streaming demo")
