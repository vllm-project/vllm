# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for MRv2 sampler memory profiling.

``profile_run`` sizes the KV cache from the peak it observes, and
``warmup_kernels`` runs *after* the KV cache is allocated, sampling with
``SamplingParams.for_sampler_warmup()``. If profiling samples through a cheaper
path than the warm-up does, the budget handed out is one the warm-up cannot
live within and startup OOMs with nothing left to give back.

These wire the real ``Sampler``, the real ``InputBuffers`` and the real CUDA
caching allocator into ``GPUModelRunner._dummy_sampler_run`` and assert the
profiled peak actually contains the logits-processing work -- the fp32 logits
copy in ``Sampler.apply_sampling_params`` and everything downstream of it. A
mocked sampler cannot show that, and a full engine init would have to assert on
free GPU memory, which is not stable enough to gate on.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 sampler profiling tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 4096
NUM_REQS = 8
HIDDEN_SIZE = 8
LOGITS_DTYPE = torch.float16

# apply_sampling_params copies logits into a fresh fp32 tensor before any
# processor runs. That copy is the floor of what profiling used to miss.
FP32_UPCAST_BYTES = NUM_REQS * VOCAB_SIZE * 4


class _MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


class _PipelineProbe(LogitsProcessor):
    """Records that the logits-processing pipeline ran, without being the
    reason it ran: ``add_request`` returns False, so the flag that admits the
    pipeline has to come from the built-in processors."""

    def __init__(self) -> None:
        self.applied_dtypes: list[torch.dtype] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return False

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self.applied_dtypes.append(logits.dtype)
        return logits


def _make_runner(probe: _PipelineProbe) -> Any:
    """A GPUModelRunner carrying only what _dummy_sampler_run touches:
    ``model.compute_logits``, ``input_buffers`` and a real ``sampler``."""
    req_states = RequestState(
        max_num_reqs=NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=NUM_REQS,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = Sampler(
        vllm_config=SimpleNamespace(reasoning_config=_MockReasoningConfig()),
        max_num_reqs=NUM_REQS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=[probe],
    )
    runner.input_buffers = InputBuffers(
        max_num_reqs=NUM_REQS, max_num_tokens=NUM_REQS, device=DEVICE
    )
    runner.model = SimpleNamespace(
        compute_logits=lambda hidden: torch.zeros(
            hidden.shape[0], VOCAB_SIZE, dtype=LOGITS_DTYPE, device=DEVICE
        )
    )
    return runner


def _peak_bytes(run) -> int:
    """Extra bytes the allocator had to hand out while ``run`` executed.

    ``max_memory_allocated`` counts requested bytes, not reserved ones, so
    block reuse between the two measurements cannot hide an allocation.
    """
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    run()
    torch.accelerator.synchronize()
    return torch.accelerator.max_memory_allocated() - before


def test_dummy_sampler_run_profiles_the_path_the_warmup_takes():
    """The profiled peak includes the logits-processing work, so the KV cache
    is not sized against a sampler cheaper than the one warmup_kernels runs."""
    probe = _PipelineProbe()
    runner = _make_runner(probe)
    hidden = torch.zeros(NUM_REQS, HIDDEN_SIZE, dtype=LOGITS_DTYPE, device=DEVICE)

    def baseline_run() -> None:
        # _dummy_sampler_run as it behaved before: make_dummy registers no
        # sampling params, so apply_sampling_params early-exits.
        with torch.inference_mode():
            logits = runner.model.compute_logits(hidden)
            batch = InputBatch.make_dummy(NUM_REQS, NUM_REQS, runner.input_buffers)
            runner.sampler(logits, batch)

    baseline_peak = _peak_bytes(baseline_run)
    assert not runner.sampler.needs_logits_processing.any()
    assert probe.applied_dtypes == []

    profiled_peak = _peak_bytes(
        lambda: GPUModelRunner._dummy_sampler_run(runner, hidden)
    )

    # The pipeline ran, on the fp32 copy rather than the raw fp16 logits.
    assert probe.applied_dtypes == [torch.float32]
    # ...and that copy is inside the peak the KV cache is sized against.
    assert profiled_peak - baseline_peak >= FP32_UPCAST_BYTES


def test_profiling_residue_does_not_put_serving_on_the_expensive_path():
    """Profiling leaves its warm-up params in the slots it touched, exactly as
    ``warmup_kernels`` does through the real ``add_requests`` path. That is
    harmless, and this pins down why: a slot is rewritten from the arriving
    request's own params before anything reads it, so serving does not inherit
    the expensive path. Restoring the flags here would be undone by
    ``warmup_kernels`` moments later anyway."""
    probe = _PipelineProbe()
    runner = _make_runner(probe)
    hidden = torch.zeros(NUM_REQS, HIDDEN_SIZE, dtype=LOGITS_DTYPE, device=DEVICE)

    GPUModelRunner._dummy_sampler_run(runner, hidden)

    # The residue is real, and deliberately left in place.
    assert runner.sampler.needs_logits_processing.any()

    # A greedy request lands on slot 0 and rewrites the flag from its own params.
    runner.sampler.add_request(0, SamplingParams(temperature=0.0))
    runner.sampler.apply_staged_writes()
    assert not runner.sampler.needs_logits_processing[0]

    probe.applied_dtypes.clear()
    with torch.inference_mode():
        batch = InputBatch.make_dummy(1, 1, runner.input_buffers)
        runner.sampler(runner.model.compute_logits(hidden[:1]), batch)

    assert probe.applied_dtypes == []
