# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.sampler import Sampler

DEVICE = current_platform.device_type


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available on this platform")
@pytest.mark.parametrize("num_logprobs", [0, 1, 3])
def test_compute_topk_logprobs_matches_gather_logprobs(num_logprobs: int):
    """Top-k logprobs should match the full log_softmax + gather path."""
    batch_size = 4
    vocab_size = 32
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    logits = torch.randn(
        batch_size,
        vocab_size,
        device=DEVICE,
        generator=generator,
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size,),
        device=DEVICE,
        dtype=torch.int64,
        generator=generator,
    )

    full_logprobs = Sampler.compute_logprobs(logits)
    expected = Sampler.gather_logprobs(full_logprobs, num_logprobs, token_ids)
    actual = Sampler.compute_topk_logprobs(logits, num_logprobs, token_ids)

    torch.testing.assert_close(
        actual.logprob_token_ids.to(expected.logprob_token_ids.dtype),
        expected.logprob_token_ids,
    )
    torch.testing.assert_close(actual.logprobs, expected.logprobs)
    torch.testing.assert_close(
        actual.selected_token_ranks,
        expected.selected_token_ranks,
    )
