# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from inspect import signature
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
import torch

from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    """Minimal MoERunner stand-in for token-count discovery tests."""

    moe_config: Any


def _make_moe(*, max_deferred_tokens: int = 128, enabled: bool = True):
    """Create a fake MoE layer with a deferred-finalize token limit."""
    moe = _FakeMoERunner()
    moe.moe_config = SimpleNamespace(
        use_deferred_moe_finalize=enabled,
        defer_moe_finalize_max_num_tokens=max_deferred_tokens,
    )
    return moe


def _make_runner(modules, *, max_tokens: int = 8192, linear_backend: str = "auto"):
    """Create a runner carrying only the state used by FlashInfer warmup."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=linear_backend)
        ),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
        _dummy_run=Mock(),
    )


def test_flashinfer_autotune_token_counts_include_deferred_moe_limits():
    runner = _make_runner(
        [
            object(),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=64, enabled=False),
            _make_moe(max_deferred_tokens=-1),
        ],
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (8192, 32, 128)


def test_flashinfer_autotune_token_counts_are_bounded_and_deduplicated():
    runner = _make_runner(
        [_make_moe(max_deferred_tokens=4096)],
        max_tokens=32,
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (32,)


@pytest.mark.parametrize(
    "linear_cutoff,silu_cutoff,max_tokens,expected",
    [
        (4, 16, 8192, (8192, 1, 2, 4, 8, 16)),
        (4, None, 8192, (8192, 1, 2, 4)),
        (0, 16, 8, (8, 1, 2, 4)),
        (0, None, 8192, (8192,)),
    ],
)
def test_dynamic_nvfp4_warmup_covers_fused_cutoff(
    linear_cutoff, silu_cutoff, max_tokens, expected
):
    """Fused down projections may use W4A16 beyond the ordinary linear cutoff."""
    runner = _make_runner(
        [], max_tokens=max_tokens, linear_backend="flashinfer_cutedsl_dynamic"
    )
    runner.vllm_config.kernel_config.nvfp4_dynamic_max_tokens = linear_cutoff
    runner.vllm_config.kernel_config.nvfp4_dynamic_silu_max_tokens = silu_cutoff
    runner.vllm_config.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32]
    )
    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        assert _flashinfer_autotune_token_counts(runner) == expected


@pytest.mark.parametrize("skip_attn", [False, True])
def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run(skip_attn):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1Runner

    runner = _make_runner([])
    dummy_run_signature = signature((V2Runner if skip_attn else V1Runner)._dummy_run)
    runner._dummy_run.side_effect = lambda **kwargs: dummy_run_signature.bind(
        None, **kwargs
    )
    max_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192)
    deferred_buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            side_effect=(max_buckets, deferred_buckets),
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)

    assert get_buckets.call_args_list == [call(8192), call(128)]
    assert autotune.call_args_list == [
        call(tuning_buckets=max_buckets),
        call(tuning_buckets=deferred_buckets),
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=8192,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
    ]


def test_dynamic_nvfp4_logits_tuning_is_bounded_by_sampled_sequences():
    """Large prefill buckets must not allocate equally large logits buffers."""
    runner = _make_runner(
        [], max_tokens=2048, linear_backend="flashinfer_cutedsl_dynamic"
    )
    sampled = torch.empty((16, 32))
    active_buckets = []
    observed = {}

    @contextmanager
    def autotune(*, tuning_buckets):
        active_buckets.append(tuning_buckets)
        try:
            yield
        finally:
            active_buckets.pop()

    def backbone(**kwargs):
        observed["prefill_max"] = max(active_buckets[-1])
        return None, sampled

    def logits(hidden_states):
        observed["sampled_rows"] = hidden_states.shape[0]
        observed["logits_buckets"] = active_buckets[-1]

    runner._dummy_run.side_effect = backbone
    runner.get_model.return_value.compute_logits = logits
    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(2048,),
        ),
        patch("vllm.utils.flashinfer.autotune", autotune),
    ):
        _run_flashinfer_autotune_dummy_runs(runner)

    assert observed == {
        "prefill_max": 2048,
        "sampled_rows": 16,
        "logits_buckets": (1, 2, 4, 8, 16),
    }
    assert not active_buckets
