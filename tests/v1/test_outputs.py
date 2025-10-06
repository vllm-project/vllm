# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.outputs import SamplerOutput


def test_sampler_output():
    # fmt: off
    # -1 is the padding token
    sampled_token_ids = torch.tensor([
        [1, 2, 3, -1],
        [1, -1, -1, -1],
        [3, 2, -1, -1]
    ])
    # fmt: on
    so = SamplerOutput(sampled_token_ids=sampled_token_ids, logprobs_tensors=None)
    expected_n_sampled_tokens = torch.tensor([3, 1, 2])
    assert so.n_sampled_tokens().eq(expected_n_sampled_tokens).all()
