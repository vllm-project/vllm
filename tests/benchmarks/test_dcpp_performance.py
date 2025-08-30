# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#python -m pytest -o log_cli=true -o log_cli_level=INFO -q test_dcpp_performance.py
import os
from statistics import mean, stdev

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

# Allow function objects to be serialized in collective_rpc for this test
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _enable_worker_timing(worker) -> bool:  # executed inside worker via RPC

    if getattr(worker, "_orig_execute_model", None) is not None:
        return True

    worker._timed_steps = []
    orig_execute_model = worker.execute_model

    def wrapped_execute_model(scheduler_output):
        total_tokens = getattr(scheduler_output,
                               "total_num_scheduled_tokens", 0) or 0
        is_prefill_step = total_tokens > 1
        if is_prefill_step and torch.cuda.is_available():
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()
            result = orig_execute_model(scheduler_output)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)
            worker._timed_steps.append((int(total_tokens), float(elapsed_ms)
                                        / 1000.0))
            return result
        return orig_execute_model(scheduler_output)

    worker._orig_execute_model = orig_execute_model
    worker.execute_model = wrapped_execute_model
    return True


def _get_worker_timing(worker):  # executed inside worker via RPC
    return list(getattr(worker, "_timed_steps", []))


def _disable_worker_timing(worker):  # executed inside worker via RPC
    data = list(getattr(worker, "_timed_steps", []))
    if getattr(worker, "_orig_execute_model", None) is not None:
        worker.execute_model = worker._orig_execute_model
        delattr(worker, "_orig_execute_model")
    worker._timed_steps = []
    return data


# Mark this module as a performance-related test
pytestmark = pytest.mark.performance

logger = init_logger(__name__)

# A small model that is quick to load and run
MODEL_NAME = "qwen/Qwen2.5-3B"
# Use a sequence length that is long enough to show the performance trend
# but not too long to make the test run for ages.
LONG_SEQ_LEN = 16384
# The "large" sequence threshold to activate DCPP
LARGE_SEQ_THRESHOLD = 1024
# max batched tokens
MAX_BATCHED_TOKENS = 4096
# The baseline chunk size for prefill (set high so DCPP can shrink it)
PREFILL_CHUNK_SIZE = MAX_BATCHED_TOKENS


@pytest.fixture(scope="module")
def long_prompt_text():
    """Create a long prompt for testing; token expansion is handled later."""
    return "a " * LONG_SEQ_LEN


def _run_and_profile(llm: LLM, prompt: str, warmup_runs: int = 1) -> tuple[list[tuple[int, float]], float]:
    """
    Run generation on the LLM engine while profiling the execution time
    of each prefill step. In current vLLM, hook into the engine's
    model executor rather than a worker attribute.
    """
    # Ensure prompt has enough tokens to trigger multiple prefill steps.
    # Some models (e.g., Qwen) compress repeated characters heavily.
    # Expand the prompt based on tokenizer token count.
    try:
        tok = llm.get_tokenizer()
        def token_count(p: str) -> int:
            return len(tok.encode(p, add_special_tokens=False))

        # Target at least two chunks plus a small buffer, and over large-seq threshold
        min_tokens = max(LARGE_SEQ_THRESHOLD + PREFILL_CHUNK_SIZE,
                         PREFILL_CHUNK_SIZE * 2 + 256)
        if token_count(prompt) < min_tokens:
            filler = " foo123"
            # Append in blocks to avoid too many tokenizer calls
            while token_count(prompt) < min_tokens:
                prompt += filler * 64
                # Safety cap to avoid runaway strings
                if len(prompt) > LONG_SEQ_LEN * 8:
                    break
    except Exception:
        # If tokenizer access fails, proceed with original prompt
        pass

    # Optional warmup to avoid cold-start bias in first measured chunk
    if warmup_runs > 0:
        try:
            warmup_prompt = "warmup"
            for _ in range(warmup_runs):
                llm.generate(warmup_prompt,
                             SamplingParams(max_tokens=1,
                                            output_kind=RequestOutputKind.FINAL_ONLY))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    # Enable timing hook inside V1 workers via RPC
    llm.collective_rpc(_enable_worker_timing)
    try:
        # Run generation to trigger the wrapped execute_model in workers
        import time as _time
        _host_t0 = _time.monotonic()
        llm.generate(prompt, SamplingParams(max_tokens=1, output_kind=RequestOutputKind.FINAL_ONLY))
        _host_t1 = _time.monotonic()
        # Retrieve timing data from workers
        timing_data_per_worker = llm.collective_rpc(_get_worker_timing)
    finally:
        # Clean up hooks
        llm.collective_rpc(_disable_worker_timing)

    # Use the first worker's data (single-GPU typical). Merge if multiple.
    for data in timing_data_per_worker:
        if data:
            return data, (_host_t1 - _host_t0)
    return [], (_host_t1 - _host_t0)


def _print_results(title: str, steps: list[tuple[int, float]], wall_time_s: float | None = None):
    """Helper function to log formatted results via vLLM logger."""
    logger.info("\n--- %s ---", title)
    header = f"{'Step':<5} | {'Chunk Size':<12} | {'Exec Time (ms)':<15}"
    logger.info("%s", header)
    logger.info("%s", "-" * 45)
    for i, (chunk_size, exec_time) in enumerate(steps):
        logger.info("%s", f"{i:<5} | {chunk_size:<12} | {exec_time * 1000:<15.2f}")

    times = [t for _, t in steps]
    if len(times) > 1:
        logger.info("%s", "-" * 45)
        logger.info("Avg Time: %.2f ms", mean(times) * 1000)
        logger.info("Std Dev:  %.2f ms", stdev(times) * 1000)
        logger.info("CoV:      %.2f%%", (stdev(times) / mean(times) * 100))
    if times:
        total_s = sum(times)
        logger.info("Total Prefill Time: %.2f ms", total_s * 1000)
        # CPP chunk bubble time: sum of positive deltas between consecutive steps
        if len(times) > 1:
            bubble_cpp_s = 0.0
            prev = times[0]
            for t in times[1:]:
                if t > prev:
                    bubble_cpp_s += (t - prev)
                prev = t
            logger.info("CPP Bubble Time: %.2f ms", bubble_cpp_s * 1000)
        if wall_time_s is not None and wall_time_s > 0:
            bubble_wall_s = max(0.0, wall_time_s - total_s)
            logger.info("E2E Wall Time: %.2f ms", wall_time_s * 1000)
            logger.info("Wall Bubble: %.2f ms (%.2f%%)", bubble_wall_s * 1000,
                        (bubble_wall_s / wall_time_s * 100) if wall_time_s else 0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 1,
                    reason="Need at least 1 GPU to run this performance test.")
def test_dcpp_execution_time_stability(long_prompt_text):
    """
    Tests the end-to-end effect of DCPP on execution time stability.

    This test runs a long prefill with and without DCPP enabled and
    compares the wall-clock time of each model execution step.

    Without DCPP, execution time per step should increase as the KV cache grows.
    With DCPP, execution time per step should remain relatively stable.
    """
    # Case 1: DCPP Disabled (baseline)
    llm_no_dcpp = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        # Set a large token threshold to ensure it's not hit
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        # Force chunked prefill in baseline so we see multiple steps
        enable_chunked_prefill=True,
        long_prefill_token_threshold=PREFILL_CHUNK_SIZE,
    )
    no_dcpp_steps, no_dcpp_wall = _run_and_profile(llm_no_dcpp, long_prompt_text)
    _print_results("DCPP Disabled", no_dcpp_steps, no_dcpp_wall)
    del llm_no_dcpp

    # Case 2: DCPP Enabled
    llm_dcpp = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        enable_chunked_prefill=True,
        # DCPP specific settings
        enable_dcpp=True,
        dcpp_length_threshold=LARGE_SEQ_THRESHOLD,
        long_prefill_token_threshold=PREFILL_CHUNK_SIZE,
        dcpp_min_chunk=256,
    )
    dcpp_steps, dcpp_wall = _run_and_profile(llm_dcpp, long_prompt_text)
    _print_results("DCPP Enabled", dcpp_steps, dcpp_wall)
    del llm_dcpp

    # --- Analysis and Assertions ---
    assert len(no_dcpp_steps) > 1, "Should have multiple prefill steps"
    assert len(dcpp_steps) > 1, "Should have multiple prefill steps"

    # Verify that without DCPP, chunk size is constant
    no_dcpp_chunks = [c for c, _ in no_dcpp_steps]
    assert all(c == no_dcpp_chunks[0] for c in no_dcpp_chunks)

    # Verify that with DCPP, chunk size decreases
    dcpp_chunks = [c for c, _ in dcpp_steps]
    assert dcpp_chunks[-1] < dcpp_chunks[0]

    no_dcpp_times = [t for _, t in no_dcpp_steps]
    dcpp_times = [t for _, t in dcpp_steps]

    # Verify that without DCPP, execution time increases
    # We check if the last step is at least 20% slower than the first.
    # This is a robust way to check for the increasing trend.
    assert no_dcpp_times[-1] > no_dcpp_times[0] * 1.2

    # The core assertion: DCPP execution times are more stable.
    # We measure this using the Coefficient of Variation (CoV = stdev / mean).
    # A lower CoV means more stability.
    if len(no_dcpp_times) > 1 and len(dcpp_times) > 1:
        no_dcpp_cov = stdev(no_dcpp_times) / mean(no_dcpp_times)
        dcpp_cov = stdev(dcpp_times) / mean(dcpp_times)
        # We expect the CoV of DCPP to be at least 2x smaller than without.
        assert dcpp_cov < no_dcpp_cov / 2
