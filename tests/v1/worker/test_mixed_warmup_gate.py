# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _kv_cache_spec():
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
        dcp_sharded=False,
    )


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
        max_num_tokens=32,
        speculative_config=None,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=_kv_cache_spec())],
            num_blocks=128,
        ),
        vllm_config=SimpleNamespace(
            num_lookahead_tokens=0,
            num_scheduled_spec_tokens=0,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        max_model_len=128,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
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
    assert all(
        params is not None and params.watermarking is False
        for params in sampling_params
    )


def test_mixed_warmup_reserves_drafting_slots():
    outputs: list[SchedulerOutput] = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        max_num_tokens=32,
        speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=1),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=_kv_cache_spec())],
            num_blocks=128,
        ),
        vllm_config=SimpleNamespace(
            num_lookahead_tokens=0,
            num_scheduled_spec_tokens=0,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        max_model_len=128,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        num_speculative_steps=0,
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )

    assert run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=outputs.append,
        worker_sample_tokens=lambda grammar_output: None,
        num_tokens=runner.max_num_tokens,
    )

    assert outputs[1].total_num_scheduled_tokens == runner.max_num_tokens - 2


def test_mixed_warmup_skips_when_drafting_slots_exhaust_budget():
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        max_num_tokens=4,
        speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=1),
    )

    assert not run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=_fail,
        worker_sample_tokens=_fail,
        num_tokens=runner.max_num_tokens,
    )


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_kernel_warmup_restores_uncalibrated_adaptive_manager(monkeypatch, fail_warmup):
    """Startup must warm fixed drafts before calibration and retain its manager."""
    manager = SimpleNamespace(cost_tables=None)
    rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner = SimpleNamespace(
        adaptive_verification=manager,
        speculator=None,
        rejection_sampler=rejection_sampler,
        speculative_config=SimpleNamespace(),
    )
    budget_warmups = []
    manager.warmup_budget_kernel = lambda: budget_warmups.append(
        runner.adaptive_verification
    )

    def run_steps(model_runner, execute, sample, greedy=False):
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
    # The budget kernel is warmed once, after the manager is restored.
    assert budget_warmups == ([] if fail_warmup else [manager])


@pytest.mark.parametrize("speculative", [False, True])
def test_kernel_warmup_runs_same_rounds_on_every_pp_stage(monkeypatch, speculative):
    """PP receivers need a matching sender for every speculative warmup round."""
    rounds = []
    monkeypatch.setattr(
        warmup,
        "_warmup_kernels",
        lambda runner, execute, sample, greedy=False: rounds.append(greedy),
    )
    for is_last_pp_rank in (False, True):
        rounds.clear()
        runner = SimpleNamespace(
            adaptive_verification=None,
            speculator=None,
            speculative_config=SimpleNamespace() if speculative else None,
            rejection_sampler=(
                SimpleNamespace(enable_adaptive_verification=False)
                if is_last_pp_rank and speculative
                else None
            ),
        )
        warmup.warmup_kernels(runner, _fail, _fail)
        assert rounds == ([False, True] if speculative else [False])
