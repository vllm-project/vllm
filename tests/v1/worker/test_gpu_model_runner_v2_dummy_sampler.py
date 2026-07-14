# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MRv2 GPUModelRunner._dummy_sampler_run memory profiling."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import flashinfer_sampler_supported
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

NUM_REQS = 4
VOCAB_SIZE = 32000
HIDDEN_SIZE = 64


@pytest.mark.skipif(
    not current_platform.is_cuda() or not flashinfer_sampler_supported(),
    reason="Requires CUDA + flashinfer top-k/top-p sampler.",
)
def test_dummy_sampler_run_exercises_flashinfer(monkeypatch):
    """_dummy_sampler_run must take the flashinfer top-k/top-p path.

    The dummy run sizes the KV cache via memory profiling. If it runs the
    greedy path, profiling never allocates flashinfer's logits-sized transient
    buffer, the KV-cache budget is oversized, and warmup_kernels (which forces
    this path) OOMs. See model_runner._dummy_sampler_run.
    """
    device = torch.device("cuda")

    req_states = RequestState(
        max_num_reqs=NUM_REQS,
        max_model_len=HIDDEN_SIZE,
        max_num_batched_tokens=NUM_REQS,
        num_speculative_steps=0,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    sampler = Sampler(
        max_num_reqs=NUM_REQS,
        vocab_size=VOCAB_SIZE,
        device=device,
        req_states=req_states,
    )

    seen_shapes: list[torch.Size] = []

    def fake_flashinfer_sample(logits, _k, _p, generators=None):
        seen_shapes.append(logits.shape)
        return torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.sampler.flashinfer_sample",
        fake_flashinfer_sample,
    )

    # Minimal runner stub: only the attributes _dummy_sampler_run touches.
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = sampler
    runner.input_buffers = InputBuffers(
        max_num_reqs=NUM_REQS, max_num_tokens=NUM_REQS, device=device
    )
    runner.model = SimpleNamespace(
        compute_logits=lambda hs: torch.randn(hs.shape[0], VOCAB_SIZE, device=device)
    )

    hidden_states = torch.randn(NUM_REQS, HIDDEN_SIZE, device=device)
    GPUModelRunner._dummy_sampler_run(runner, hidden_states)

    assert seen_shapes, (
        "flashinfer top-k/top-p sampler was not invoked during the dummy run; "
        "memory profiling under-counts and warmup_kernels can OOM."
    )
    assert seen_shapes[0] == torch.Size([NUM_REQS, VOCAB_SIZE])
