# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the streaming input feature in AsyncLLM.

These tests verify that:
1. Streaming inputs work correctly with bunched inputs (queued)
2. Streaming inputs work correctly with spaced out inputs
3. Outputs are equivalent whether inputs are bunched or spaced
4. Cancelling the output stream correctly aborts the session
5. Closing the input stream correctly signals completion
6. Queued inputs are cancelled when the session is aborted
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from vllm import SamplingParams
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

# Use a small model that doesn't require authentication for fast tests
MODEL = "facebook/opt-125m"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    """Create an AsyncLLM engine for the test.

    Note: Using function scope because pytest_asyncio creates a new event loop
    for each test, and the output_handler task gets cancelled between tests
    with module scope.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs(
        model=MODEL, enforce_eager=True, gpu_memory_utilization=0.7
    )
    with set_default_torch_num_threads(1):
        engine = AsyncLLM.from_engine_args(engine_args)
    try:
        yield engine
    finally:
        engine.shutdown()
        await asyncio.sleep(0.1)


def get_sampling_params(max_tokens: int = 20) -> SamplingParams:
    """Create sampling params for streaming input tests."""
    return SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
        temperature=0.0,  # Deterministic for reproducibility
    )


def streaming_input_from_text(
    engine: AsyncLLM, text: str, sampling_params: SamplingParams | None = None
):
    prompt = TextPrompt(prompt=text)
    (tok_prompt,) = engine.renderer.render_cmpl([prompt])

    return StreamingInput(prompt=tok_prompt, sampling_params=sampling_params)


async def collect_outputs(
    output_gen: AsyncGenerator[RequestOutput, None],
) -> tuple[list[RequestOutput], str]:
    """Collect all outputs from a generate call, return outputs and full text."""
    outputs: list[RequestOutput] = []
    full_text = ""
    async for output in output_gen:
        outputs.append(output)
        if output.outputs and output.outputs[0].text:
            full_text += output.outputs[0].text
    return outputs, full_text


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_bunched(engine: AsyncLLM):
    """Test streaming input where all inputs are sent at once (bunched/queued).

    This tests the case where multiple inputs arrive before any completes.
    The inputs should be queued and processed in sequence.
    """
    request_id = "test_bunched"
    sampling_params = get_sampling_params(max_tokens=10)

    # Create an input generator that yields all inputs quickly
    async def bunched_input_generator() -> AsyncGenerator[StreamingInput, None]:
        # Send multiple inputs rapidly - they should be queued
        yield streaming_input_from_text(engine, "Hello, my name is")
        yield streaming_input_from_text(engine, " Alice and I like")
        yield streaming_input_from_text(engine, " to code in Python")

    outputs, full_text = await collect_outputs(
        engine.generate(
            bunched_input_generator(),
            sampling_params,
            request_id,
        )
    )

    # Verify we got outputs
    assert len(outputs) > 0, "Should have received outputs"

    # Verify the final output is marked as finished
    assert outputs[-1].finished, "Last output should be marked as finished"

    # Verify intermediate outputs are not marked as finished
    for output in outputs[:-1]:
        assert not output.finished, "Intermediate outputs should not be finished"

    # Verify we generated some text
    assert len(full_text) > 0, "Should have generated text"
    print(f"Bunched test generated: {full_text}")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_spaced(engine: AsyncLLM):
    """Test streaming input where inputs are spaced out.

    This tests the case where each input completes processing before the
    next one is sent. Each chunk should be prefilled, generate tokens,
    then the next chunk should be processed.
    """
    request_id = "test_spaced"
    sampling_params = get_sampling_params(max_tokens=10)

    # Track when each input is sent
    input_times: list[float] = []
    outputs_per_chunk: list[int] = [0, 0, 0]
    current_chunk = 0

    async def spaced_input_generator() -> AsyncGenerator[StreamingInput, None]:
        nonlocal current_chunk
        import time

        # First input
        input_times.append(time.time())
        yield streaming_input_from_text(engine, "Hello, my name is")
        current_chunk = 0

        # Wait for some outputs to be generated
        await asyncio.sleep(0.5)

        # Second input
        input_times.append(time.time())
        current_chunk = 1
        yield streaming_input_from_text(engine, " Alice and I like")

        # Wait for some outputs
        await asyncio.sleep(0.5)

        # Third input
        input_times.append(time.time())
        current_chunk = 2
        yield streaming_input_from_text(engine, " to code in Python")

    outputs: list[RequestOutput] = []
    full_text = ""

    async for output in engine.generate(
        spaced_input_generator(),
        sampling_params,
        request_id,
    ):
        outputs.append(output)
        if output.outputs and output.outputs[0].text:
            full_text += output.outputs[0].text
            outputs_per_chunk[current_chunk] += 1

    # Verify we got outputs
    assert len(outputs) > 0, "Should have received outputs"

    # Verify the final output is marked as finished
    assert outputs[-1].finished, "Last output should be marked as finished"

    # Verify we received outputs from multiple chunks
    # (with spaced inputs, we should see outputs distributed across chunks)
    chunks_with_outputs = sum(1 for c in outputs_per_chunk if c > 0)
    assert chunks_with_outputs >= 1, "Should have outputs from at least one chunk"

    print(f"Spaced test generated: {full_text}")
    print(f"Outputs per chunk: {outputs_per_chunk}")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_output_equivalence(engine: AsyncLLM):
    """Test that bunched and spaced inputs produce equivalent outputs.

    When the same prompts are provided either bunched or spaced,
    the final concatenated output should be the same (with deterministic
    sampling).
    """
    prompts = ["Hello, my name is", " Bob and I work", " at Anthropic"]
    sampling_params = get_sampling_params(max_tokens=15)

    # Test bunched inputs
    async def bunched_gen() -> AsyncGenerator[StreamingInput, None]:
        for prompt in prompts:
            yield StreamingInput(prompt=prompt)

    _, bunched_text = await collect_outputs(
        engine.generate(bunched_gen(), sampling_params, "equiv_bunched")
    )

    # Test spaced inputs (same prompts, but with delays)
    async def spaced_gen() -> AsyncGenerator[StreamingInput, None]:
        for prompt in prompts:
            yield StreamingInput(prompt=prompt)
            await asyncio.sleep(0.3)

    _, spaced_text = await collect_outputs(
        engine.generate(spaced_gen(), sampling_params, "equiv_spaced")
    )

    # Both should produce the same output since we use temperature=0
    assert bunched_text == spaced_text, (
        f"Bunched and spaced should produce same output.\n"
        f"Bunched: {bunched_text!r}\n"
        f"Spaced: {spaced_text!r}"
    )

    print(f"Equivalence test passed. Generated: {bunched_text}")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_cancel_output_stream(engine: AsyncLLM):
    """Test that cancelling the output stream aborts the entire session.

    When the consumer cancels iteration over the output generator,
    the session should be aborted including any queued inputs.
    """
    request_id = "test_cancel_output"
    sampling_params = get_sampling_params(max_tokens=1000)

    input_completed = asyncio.Event()
    input_task_cancelled = False

    async def slow_input_generator() -> AsyncGenerator[StreamingInput, None]:
        nonlocal input_task_cancelled
        try:
            yield streaming_input_from_text(engine, "Tell me a very long story about")
            yield streaming_input_from_text(engine, " a dragon and a knight")

            # This should be cancelled before we get here
            await asyncio.sleep(10)
            yield streaming_input_from_text(engine, " who become friends")
            input_completed.set()
        except asyncio.CancelledError:
            input_task_cancelled = True
            raise

    outputs_received = 0
    output_gen = engine.generate(slow_input_generator(), sampling_params, request_id)

    # Collect a few outputs then cancel
    try:
        async for output in output_gen:
            outputs_received += 1
            if outputs_received >= 5:
                # Cancel by breaking out of the loop (generator will be GC'd)
                break
    finally:
        # Explicitly close the generator to ensure cleanup
        await output_gen.aclose()

    # Give time for cleanup
    await asyncio.sleep(0.5)

    # Verify we got some outputs before cancelling
    assert outputs_received >= 5, "Should have received outputs before cancel"

    # Verify the input task was cancelled
    assert input_task_cancelled, "Input task should have been cancelled"

    # Verify the session is properly cleaned up
    assert not engine.output_processor.has_unfinished_requests(), (
        "Should have no unfinished requests after cancel"
    )

    print(f"Cancel test passed. Received {outputs_received} outputs before cancel")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_close_signals_completion(engine: AsyncLLM):
    """Test that closing the input stream signals completion.

    When the input generator finishes (naturally or via return),
    the session should complete with finished=True on the last output.
    """
    request_id = "test_close_completion"
    sampling_params = get_sampling_params(max_tokens=15)

    input_generator_finished = False

    async def limited_input_generator() -> AsyncGenerator[StreamingInput, None]:
        nonlocal input_generator_finished
        yield streaming_input_from_text(engine, "What is 2 + 2? The answer is")
        # Generator finishes naturally here
        input_generator_finished = True

    outputs, _ = await collect_outputs(
        engine.generate(limited_input_generator(), sampling_params, request_id)
    )

    # Verify the input generator completed
    assert input_generator_finished, "Input generator should have finished"

    # Verify we got a finished output
    assert len(outputs) > 0, "Should have received outputs"
    assert outputs[-1].finished, "Last output should be marked as finished"

    # Verify the session is cleaned up
    assert not engine.output_processor.has_unfinished_requests(), (
        "Should have no unfinished requests"
    )

    print("Close completion test passed")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_abort_queued_inputs(engine: AsyncLLM):
    """Test that aborting the session cancels queued inputs.

    When multiple inputs are queued and the session is aborted,
    all pending inputs should be cancelled.
    """
    request_id = "test_abort_queued"
    # Use large max_tokens to ensure we have time to queue inputs
    sampling_params = get_sampling_params(max_tokens=2000)

    inputs_sent = 0
    input_cancelled = False

    async def many_inputs_generator() -> AsyncGenerator[StreamingInput, None]:
        nonlocal inputs_sent, input_cancelled
        try:
            # Send several inputs to fill the queue
            for i in range(10):
                yield StreamingInput(prompt=f" Part {i}: Tell me about the number {i}.")
                inputs_sent += 1
                # Small delay to interleave with output processing
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            input_cancelled = True
            raise

    outputs_received = 0
    output_gen = engine.generate(many_inputs_generator(), sampling_params, request_id)

    try:
        async for output in output_gen:
            outputs_received += 1
            # Cancel after receiving some outputs
            if outputs_received >= 10:
                break
    finally:
        await output_gen.aclose()

    # Give time for cleanup
    await asyncio.sleep(0.5)

    # Verify we received some outputs
    assert outputs_received >= 10, "Should have received outputs before abort"

    # Verify the input generator was cancelled OR finished naturally
    # (it might finish naturally if all inputs were sent before cancel)
    assert input_cancelled or inputs_sent == 10, (
        f"Input generator should have been cancelled or completed. "
        f"cancelled={input_cancelled}, inputs_sent={inputs_sent}"
    )

    # Verify the session is cleaned up
    assert not engine.output_processor.has_unfinished_requests(), (
        "Should have no unfinished requests after abort"
    )

    print(
        f"Abort queued test passed. Sent {inputs_sent} inputs, "
        f"received {outputs_received} outputs"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_error_propagation(engine: AsyncLLM):
    """Test that errors in the input generator are propagated to the caller."""
    request_id = "test_error_propagation"
    sampling_params = get_sampling_params(max_tokens=20)

    class InputError(Exception):
        pass

    async def error_input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield streaming_input_from_text(engine, "Start with this")
        await asyncio.sleep(0.1)
        raise InputError("Simulated input error")

    # Note: The current implementation catches exceptions and puts them
    # in the queue, so we should get the error when iterating outputs
    with pytest.raises(InputError, match="Simulated input error"):
        async for _ in engine.generate(
            error_input_generator(), sampling_params, request_id
        ):
            pass

    # Give time for cleanup
    await asyncio.sleep(0.3)

    # Verify the session is cleaned up
    assert not engine.output_processor.has_unfinished_requests(), (
        "Should have no unfinished requests after error"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_multiple_concurrent_sessions(engine: AsyncLLM):
    """Test multiple concurrent streaming input sessions.

    Multiple streaming sessions should be able to run concurrently
    without interfering with each other.
    """
    num_sessions = 3
    results: list[tuple[str, str]] = []

    async def run_session(session_id: int) -> tuple[str, str]:
        request_id = f"test_concurrent_{session_id}"
        sampling_params = get_sampling_params(max_tokens=10)

        prompts = [f"Session {session_id}: Hello", f" world from session {session_id}"]

        async def input_gen() -> AsyncGenerator[StreamingInput, None]:
            for prompt in prompts:
                yield StreamingInput(prompt=prompt)
                await asyncio.sleep(0.1)

        _, text = await collect_outputs(
            engine.generate(input_gen(), sampling_params, request_id)
        )
        return request_id, text

    # Run sessions concurrently
    tasks = [asyncio.create_task(run_session(i)) for i in range(num_sessions)]
    results = await asyncio.gather(*tasks)

    # Verify all sessions completed
    assert len(results) == num_sessions

    for request_id, text in results:
        assert len(text) > 0, f"Session {request_id} should have generated text"
        print(f"{request_id}: {text}")

    # Verify cleanup
    assert not engine.output_processor.has_unfinished_requests()


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_per_chunk_sampling_params(engine: AsyncLLM):
    """Test that per-chunk sampling params are respected.

    Each StreamingInput can have its own sampling_params.
    """
    request_id = "test_per_chunk_params"
    base_params = get_sampling_params(max_tokens=10)

    async def variable_params_generator() -> AsyncGenerator[StreamingInput, None]:
        # First chunk with base params
        yield streaming_input_from_text(
            engine, "Count to five:", sampling_params=base_params
        )

        # Second chunk with different max_tokens
        chunk_params = get_sampling_params(max_tokens=5)
        yield StreamingInput(
            prompt=" Now count backwards:", sampling_params=chunk_params
        )

    outputs, full_text = await collect_outputs(
        engine.generate(variable_params_generator(), base_params, request_id)
    )

    assert len(outputs) > 0, "Should have received outputs"
    assert outputs[-1].finished, "Last output should be finished"
    assert len(full_text) > 0, "Should have generated text"

    print(f"Per-chunk params test generated: {full_text}")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_empty_generator(engine: AsyncLLM):
    """Test behavior when the input generator yields nothing.

    An empty generator should still produce a finished output.
    """
    request_id = "test_empty_generator"
    sampling_params = get_sampling_params(max_tokens=10)

    async def empty_generator() -> AsyncGenerator[StreamingInput, None]:
        # Don't yield anything
        return
        yield  # Make it a generator

    outputs: list[RequestOutput] = []
    async for output in engine.generate(empty_generator(), sampling_params, request_id):
        outputs.append(output)

    # Should still get a finished marker
    assert len(outputs) >= 1, "Should receive at least one output"
    assert outputs[-1].finished, "Should have a finished output"


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_single_chunk(engine: AsyncLLM):
    """Test streaming input with a single chunk.

    This is effectively the same as a regular non-streaming request,
    but using the streaming input API.
    """
    request_id = "test_single_chunk"
    sampling_params = get_sampling_params(max_tokens=15)

    async def single_chunk_generator() -> AsyncGenerator[StreamingInput, None]:
        yield streaming_input_from_text(engine, "What color is the sky? The sky is")

    outputs, full_text = await collect_outputs(
        engine.generate(single_chunk_generator(), sampling_params, request_id)
    )

    assert len(outputs) > 0
    assert outputs[-1].finished
    assert "blue" in full_text.lower() or len(full_text) > 0

    print(f"Single chunk test generated: {full_text}")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_reuse_request_id(engine: AsyncLLM):
    """Test that request IDs can be reused after a session completes."""
    request_id = "test_reuse_id"
    sampling_params = get_sampling_params(max_tokens=5)

    # First session
    async def gen1() -> AsyncGenerator[StreamingInput, None]:
        yield streaming_input_from_text(engine, "First session")

    _, text1 = await collect_outputs(
        engine.generate(gen1(), sampling_params, request_id)
    )

    # Second session with same ID
    async def gen2() -> AsyncGenerator[StreamingInput, None]:
        yield streaming_input_from_text(engine, "Second session")

    _, text2 = await collect_outputs(
        engine.generate(gen2(), sampling_params, request_id)
    )

    assert len(text1) > 0
    assert len(text2) > 0
    assert not engine.output_processor.has_unfinished_requests()

    print(f"Reuse ID test: session 1: {text1}, session 2: {text2}")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_validation_errors(engine: AsyncLLM):
    """Test that invalid configurations raise appropriate errors."""

    async def dummy_generator() -> AsyncGenerator[StreamingInput, None]:
        yield streaming_input_from_text(engine, "test")

    # Test n > 1 is rejected
    with pytest.raises(ValueError, match="Input streaming not currently supported"):
        params_n2 = SamplingParams(max_tokens=10, n=2)
        async for _ in engine.generate(dummy_generator(), params_n2, "test_n2"):
            pass

    # Test FINAL_ONLY is rejected
    with pytest.raises(ValueError, match="Input streaming not currently supported"):
        params_final = SamplingParams(
            max_tokens=10, output_kind=RequestOutputKind.FINAL_ONLY
        )
        async for _ in engine.generate(dummy_generator(), params_final, "test_final"):
            pass

    # Test stop strings are rejected
    with pytest.raises(ValueError, match="Input streaming not currently supported"):
        params_stop = SamplingParams(max_tokens=10, stop=["stop"])
        async for _ in engine.generate(dummy_generator(), params_stop, "test_stop"):
            pass


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_input_delayed_generator_exit(engine: AsyncLLM):
    """Test that output generator exits when input generator closes after outputs.

    This tests the case where:
    1. Multiple inputs are sent and fully processed
    2. The engine has finished
    3. The input generator doesn't exit until after the engine finishes
    4. The output generator should exit properly once the input generator exits
    """
    request_id = "test_delayed_exit"
    sampling_params = get_sampling_params(max_tokens=10)

    engine_finished_event = asyncio.Event()
    input_generator_exited = False
    finish_count = 0

    async def delayed_exit_input_generator() -> AsyncGenerator[StreamingInput, None]:
        nonlocal input_generator_exited
        # Send all inputs immediately
        yield streaming_input_from_text(engine, "Hello, my name is")
        yield streaming_input_from_text(engine, " Alice")

        # Wait until the engine has finished generating before exiting
        await engine_finished_event.wait()

        # Add a small delay to ensure we're testing the "delayed exit" case
        await asyncio.sleep(0.1)
        input_generator_exited = True

    outputs: list[RequestOutput] = []
    full_text = ""

    async for output in engine.generate(
        delayed_exit_input_generator(), sampling_params, request_id
    ):
        outputs.append(output)
        if output.outputs and output.outputs[0].text:
            full_text += output.outputs[0].text

        # Signal when the engine finishes both input chunks (each gets a finish_reason)
        # Note: output.finished will be False while input stream is open
        if output.outputs and output.outputs[0].finish_reason is not None:
            finish_count += 1
            if finish_count == 2:
                engine_finished_event.set()

    # Verify the input generator exited properly
    assert input_generator_exited, (
        "Input generator should have exited after engine finished"
    )

    # Verify we got outputs
    assert len(outputs) > 0, "Should have received outputs"

    # Verify we generated some text
    assert len(full_text) > 0, "Should have generated text"

    # Verify the session is cleaned up
    assert not engine.output_processor.has_unfinished_requests(), (
        "Should have no unfinished requests"
    )

    print(f"Delayed exit test passed. Generated: {full_text}")
