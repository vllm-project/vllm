# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end CUDA coverage for NaN fault tolerance.

The suite uses Qwen3.5-0.8B and needs only one CUDA GPU. It bounds the KV cache
to 32 blocks, so it is suitable for an H200 (and substantially smaller GPUs).

Run all scenarios from the repository root:

    CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -m pytest -sv \
      tests/v1/e2e/general/test_nan_fault_tolerance.py

Run only the production-like async CUDA-graph scenarios:

    CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -m pytest -sv \
      tests/v1/e2e/general/test_nan_fault_tolerance.py \
      -k "cudagraph and async"
"""

import pytest
import torch

from tests.conftest import VllmRunner
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

MODEL = "Qwen/Qwen3.5-0.8B-Base"
PROMPT_TOKEN_IDS = [42] * 48
FORCED_TOKEN_ID = 42
MAX_TOKENS = 8


def _metric_value(runner: VllmRunner, name: str) -> float:
    return sum(
        metric.value for metric in runner.llm.get_metrics() if metric.name == name
    )


def _model_runner(runner: VllmRunner):
    engine_core = runner.llm.llm_engine.engine_core.engine_core
    return engine_core.model_executor.driver_worker.worker.model_runner


def _assert_fault_tolerance_config(runner: VllmRunner) -> None:
    config = runner.llm.llm_engine.vllm_config
    assert config.use_v2_model_runner
    assert config.fault_tolerance_config.enable_nan_fault_tolerance
    assert config.observability_config.enable_detect_nans_in_logits
    assert _model_runner(runner).kv_cache_config.needs_kv_cache_zeroing


def _record_full_cudagraph_replays(
    runner: VllmRunner, monkeypatch: pytest.MonkeyPatch
) -> list[int]:
    manager = _model_runner(runner).cudagraph_manager
    assert manager is not None
    assert 1 in manager.captured_token_counts()

    calls = [0]
    original = manager.run_fullgraph

    def record_replay(batch_desc):
        calls[0] += 1
        return original(batch_desc)

    monkeypatch.setattr(manager, "run_fullgraph", record_replay)
    return calls


def _inject_one_nan(
    runner: VllmRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    require_speculative_rows: bool,
) -> list[bool]:
    """Inject into raw target logits, after graph replay and before sampling."""
    model = _model_runner(runner).model
    original = model.compute_logits
    injected = [False]

    def compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        logits = original(hidden_states)
        is_speculative_step = logits.shape[0] > 1
        if not injected[0] and (not require_speculative_rows or is_speculative_step):
            # Use the final row so speculative aggregation must associate a
            # non-leading target-logit row with the correct request.
            logits[-1, 0] = float("nan")
            injected[0] = True
        return logits

    monkeypatch.setattr(model, "compute_logits", compute_logits)
    return injected


def _run_abort_cache_recovery(
    runner: VllmRunner,
    *,
    expect_partial_output: bool,
) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        ignore_eos=True,
        allowed_token_ids=[FORCED_TOKEN_ID],
    )
    prompt = TokensPrompt(prompt_token_ids=PROMPT_TOKEN_IDS)

    corrupted_before = _metric_value(runner, "vllm:corrupted_requests")
    hits_before = _metric_value(runner, "vllm:prefix_cache_hits")

    corrupted = runner.llm.generate(prompt, sampling_params)[0]
    assert corrupted.finished
    assert corrupted.outputs[0].finish_reason == "error"
    assert len(corrupted.outputs[0].token_ids) < MAX_TOKENS
    if expect_partial_output:
        assert corrupted.outputs[0].token_ids
        assert set(corrupted.outputs[0].token_ids) == {FORCED_TOKEN_ID}
    else:
        assert not corrupted.outputs[0].token_ids

    assert _metric_value(runner, "vllm:corrupted_requests") == corrupted_before + 1
    hits_after_abort = _metric_value(runner, "vllm:prefix_cache_hits")
    assert hits_after_abort == hits_before

    # The first clean retry must recompute the prompt: the corrupted request's
    # KV blocks must not have entered, or remained in, the prefix cache.
    recovered = runner.llm.generate(prompt, sampling_params)[0]
    assert recovered.outputs[0].finish_reason == "length"
    assert list(recovered.outputs[0].token_ids) == [FORCED_TOKEN_ID] * MAX_TOKENS
    hits_after_recovery = _metric_value(runner, "vllm:prefix_cache_hits")
    assert hits_after_recovery == hits_after_abort

    # A subsequent request must reuse the now-validated prefix.
    cached = runner.llm.generate(prompt, sampling_params)[0]
    assert cached.outputs[0].finish_reason == "length"
    assert list(cached.outputs[0].token_ids) == [FORCED_TOKEN_ID] * MAX_TOKENS
    assert _metric_value(runner, "vllm:prefix_cache_hits") > hits_after_recovery


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("async_scheduling", "cudagraph_mode"),
    [
        (False, CUDAGraphMode.NONE),
        (True, CUDAGraphMode.NONE),
        (False, CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, CUDAGraphMode.FULL_AND_PIECEWISE),
    ],
    ids=["eager-sync", "eager-async", "cudagraph-sync", "cudagraph-async"],
)
@single_gpu_only
def test_nan_abort_cache_recovery(
    monkeypatch: pytest.MonkeyPatch,
    async_scheduling: bool,
    cudagraph_mode: CUDAGraphMode,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_RAISE_ON_LOGIT_NANS", "0")

    with VllmRunner(
        MODEL,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_gpu_blocks_override=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_nan_fault_tolerance=True,
        async_scheduling=async_scheduling,
        distributed_executor_backend="uni",
        disable_log_stats=False,
        enforce_eager=cudagraph_mode == CUDAGraphMode.NONE,
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config={
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_capture_sizes": [1, 4],
        },
    ) as runner:
        _assert_fault_tolerance_config(runner)
        injected = _inject_one_nan(runner, monkeypatch, require_speculative_rows=False)
        replay_calls = (
            _record_full_cudagraph_replays(runner, monkeypatch)
            if cudagraph_mode != CUDAGraphMode.NONE
            else None
        )
        _run_abort_cache_recovery(runner, expect_partial_output=False)
        assert injected[0]
        if replay_calls is not None:
            assert replay_calls[0] > 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires NVIDIA CUDA")
@single_gpu_only
def test_async_cudagraph_speculative_nan_abort_cache_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise per-request NaN aggregation across speculative logit rows."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_RAISE_ON_LOGIT_NANS", "0")

    with VllmRunner(
        MODEL,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_gpu_blocks_override=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_nan_fault_tolerance=True,
        async_scheduling=True,
        distributed_executor_backend="uni",
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
        compilation_config={
            "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
            "cudagraph_capture_sizes": [1, 4],
        },
    ) as runner:
        _assert_fault_tolerance_config(runner)
        injected = _inject_one_nan(runner, monkeypatch, require_speculative_rows=True)
        replay_calls = _record_full_cudagraph_replays(runner, monkeypatch)
        _run_abort_cache_recovery(runner, expect_partial_output=True)
        assert injected[0]
        assert replay_calls[0] > 0
