# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking.prfs import HMACSHA256PRF, PhiloxPRF, WatermarkPRF


@pytest.mark.parametrize(
    ("prf", "expected_mantissas"),
    [
        (PhiloxPRF(42), [[1661833, 10571167], [9742331, 13375724]]),
        (HMACSHA256PRF(42), [[6202050, 16483231], [11455363, 10557370]]),
    ],
)
def test_prf_compatibility_vector(
    prf: WatermarkPRF, expected_mantissas: list[list[int]]
):
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([7, 8])
    expected = (
        (torch.tensor(expected_mantissas, dtype=torch.float64) + 1) / (2**24 + 1)
    ).to(torch.float32)

    assert torch.equal(prf.uniform(contexts, token_ids), expected)


@pytest.mark.parametrize("prf", [PhiloxPRF(42), HMACSHA256PRF(42)])
def test_prf_pairs_contexts_with_target_tokens(prf: WatermarkPRF):
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([[7], [8]])

    assert prf.uniform(contexts, token_ids).shape == (2, 1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("key", [42, 15726070495360670683])
@pytest.mark.parametrize("context_width", [1, 4, 16])
def test_philox_accelerator_matches_cpu(key: int, context_width: int):
    contexts = torch.arange(2 * context_width, dtype=torch.int64).reshape(
        2, context_width
    )
    token_ids = torch.arange(1024, dtype=torch.int64)
    prf = PhiloxPRF(key)

    expected = prf.uniform(contexts, token_ids)
    actual = prf.uniform(contexts.cuda(), token_ids.cuda()).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
