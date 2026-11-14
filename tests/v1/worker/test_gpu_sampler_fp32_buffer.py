# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sampler's fp32 working copy of the logits comes from one buffer sized
for the largest step, not a fresh tensor per step."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 sampler tests", allow_module_level=True
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 16384
MAX_NUM_REQS = 4
NUM_SPEC = 1
TEMPERATURE = 2.0


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=64,
        num_speculative_steps=NUM_SPEC,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    sampler = Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=MAX_NUM_REQS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        num_speculative_tokens=NUM_SPEC,
    )
    for i in range(MAX_NUM_REQS):
        sampler.add_request(i, SamplingParams(temperature=TEMPERATURE))
    sampler.apply_staged_writes()
    return sampler


def _apply(sampler: Sampler, raw: torch.Tensor) -> torch.Tensor:
    rows = raw.shape[0]
    expanded = torch.arange(rows, dtype=torch.int32, device=DEVICE) % MAX_NUM_REQS
    idx = torch.arange(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    return sampler.apply_sampling_params(
        raw,
        expanded_idx_mapping=expanded,
        idx_mapping=idx,
        idx_mapping_np=np.arange(MAX_NUM_REQS, dtype=np.intp),
        pos=torch.zeros(rows, dtype=torch.int32, device=DEVICE),
        input_ids=torch.zeros(rows, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.zeros(rows, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.ones(MAX_NUM_REQS, dtype=np.int64),
        skip_top_k_top_p=True,
    )


@pytest.mark.parametrize("rows", [MAX_NUM_REQS * (1 + NUM_SPEC), 3])
def test_fp32_copy_comes_from_the_buffer(rows: int) -> None:
    sampler = _make_sampler()
    gen = torch.Generator(device=DEVICE).manual_seed(rows)
    raw = torch.randn(
        rows, VOCAB_SIZE, dtype=torch.bfloat16, device=DEVICE, generator=gen
    )
    # A larger previous step leaves values behind in the buffer.
    _apply(
        sampler, torch.ones(MAX_NUM_REQS * (1 + NUM_SPEC), VOCAB_SIZE, device=DEVICE)
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = _apply(sampler, raw)
    torch.cuda.synchronize()
    peak_growth = torch.cuda.max_memory_allocated() - before

    assert out.dtype == torch.float32
    assert torch.equal(out, raw.float() / TEMPERATURE)
    # No rows x vocab fp32 tensor was allocated for this step.
    assert peak_growth < rows * VOCAB_SIZE * 4


def test_step_larger_than_the_buffer_still_samples() -> None:
    sampler = _make_sampler()
    rows = MAX_NUM_REQS * (1 + NUM_SPEC) + 1
    raw = torch.randn(rows, VOCAB_SIZE, dtype=torch.bfloat16, device=DEVICE)
    out = _apply(sampler, raw)
    assert out.dtype == torch.float32
    assert torch.equal(out, raw.float() / TEMPERATURE)
