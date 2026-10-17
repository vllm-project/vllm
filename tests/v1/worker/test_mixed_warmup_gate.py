# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_mixed_warmup_disables_watermarking():
    outputs: list[SchedulerOutput] = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=128,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=0),
        max_model_len=128,
        num_speculative_steps=0,
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )

    assert run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=outputs.append,
        worker_sample_tokens=lambda grammar_output: None,
        num_tokens=32,
    )

    sampling_params = [
        request.sampling_params
        for output in outputs
        for request in output.scheduled_new_reqs
    ]
    assert len(sampling_params) == 2
    assert all(params.watermarking is False for params in sampling_params)


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_kernel_warmup_restores_uncalibrated_adaptive_manager(monkeypatch, fail_warmup):
    """Startup must warm fixed drafts before calibration and retain its manager."""
    manager = SimpleNamespace(cost_tables=None)
    rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner = SimpleNamespace(
        adaptive_verification=manager,
        rejection_sampler=rejection_sampler,
    )

    def run_steps(model_runner, execute, sample, greedy=False, seeded=False):
        assert model_runner.adaptive_verification is None
        assert not model_runner.rejection_sampler.enable_adaptive_verification
        if fail_warmup:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(warmup, "_warmup_kernels", run_steps)
    if fail_warmup:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.warmup_kernels(runner, _fail, _fail)
    else:
        warmup.warmup_kernels(runner, _fail, _fail)
    assert runner.adaptive_verification is manager
    assert manager.cost_tables is None
    assert rejection_sampler.enable_adaptive_verification


@pytest.mark.parametrize("fused_backend", ["flashinfer", "xpu", None])
def test_warmup_kernels_dispatches_seeded_pass_when_fused(monkeypatch, fused_backend):
    """Verify that warmup_kernels dispatches a dedicated seeded pass (#54425)
    when fused sampling (FlashInfer or XPU) is enabled, and skips it otherwise."""
    passes_called = []

    def mock_warmup_kernels(model_runner, execute, sample, greedy=False, seeded=False):
        passes_called.append({"greedy": greedy, "seeded": seeded})

    monkeypatch.setattr(warmup, "_warmup_kernels", mock_warmup_kernels)

    sampler = SimpleNamespace(
        use_flashinfer=(fused_backend == "flashinfer"),
        use_xpu_sampler=(fused_backend == "xpu"),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        adaptive_verification=None,
        rejection_sampler=None,
        sampler=sampler,
    )

    warmup.warmup_kernels(runner, _fail, _fail)

    if fused_backend in ("flashinfer", "xpu"):
        assert passes_called == [
            {"greedy": False, "seeded": False},
            {"greedy": False, "seeded": True},
        ]
    else:
        assert passes_called == [
            {"greedy": False, "seeded": False},
        ]
