# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Model Runner V2 sampler wants FP32 logits from the gather only where it
would copy them to FP32 and no returned logprobs read the raw logits, and
processes them in place only when told they are writable."""

from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

if not current_platform.is_cuda():
    pytest.skip("Model Runner V2 sampler tests need CUDA", allow_module_level=True)

from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 256


def _make_sampler(logprobs_mode: str) -> Sampler:
    req_states = RequestState(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=VllmConfig(),
        max_num_reqs=2,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        logprobs_mode=logprobs_mode,
    )


@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
@pytest.mark.parametrize("num_logprobs", [None, 3])
def test_logits_processed_in_place_only_when_raw_logits_unread(
    logprobs_mode: str, num_logprobs: int | None
):
    sampler = _make_sampler(logprobs_mode)
    sampler.add_request(0, SamplingParams(temperature=0.0, logprobs=num_logprobs))
    sampler.add_request(1, SamplingParams(temperature=2.0, logprobs=num_logprobs))
    sampler.apply_staged_writes()

    # A greedy request is not processed, so there is no FP32 copy to save.
    assert not sampler.wants_fp32_logits(np.array([0]))
    idx_mapping_np = np.array([1])
    writable = sampler.wants_fp32_logits(idx_mapping_np)
    assert writable == (
        num_logprobs is None or logprobs_mode in PROCESSED_LOGPROBS_MODES
    )
    # Verification processes the logits through the same sampler.
    rejection_sampler = RejectionSampler(
        sampler, SpeculativeConfig(method="ngram", num_speculative_tokens=1), DEVICE
    )
    for idx in (np.array([0]), idx_mapping_np):
        assert rejection_sampler.wants_fp32_logits(idx) == sampler.wants_fp32_logits(
            idx
        )

    idx_mapping = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(1, VOCAB_SIZE, device=DEVICE).to(dtype)
        raw = logits.clone()
        processed = sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping=idx_mapping,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            pos=torch.zeros(1, dtype=torch.int32, device=DEVICE),
            input_ids=torch.zeros(1, dtype=torch.int32, device=DEVICE),
            expanded_local_pos=torch.zeros(1, dtype=torch.int32, device=DEVICE),
            seq_lens_upper_bound_np=np.ones(1, dtype=np.int64),
            logits_writable=writable,
        )
        assert processed.dtype == torch.float32
        assert torch.equal(processed, raw.float() / 2)
        # FP32 logits are overwritten exactly when they may be; others never.
        in_place = writable and dtype == torch.float32
        assert (processed.data_ptr() == logits.data_ptr()) == in_place
        if not in_place:
            assert torch.equal(logits, raw)
