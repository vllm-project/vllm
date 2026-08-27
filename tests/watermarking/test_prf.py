# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.watermarking.prfs import HMACSHA256PRF, PhiloxPRF, WatermarkPRF


@pytest.mark.parametrize(
    ("prf", "expected_mantissas"),
    [
        (PhiloxPRF(42), [[7512786, 8202112], [2919935, 2915985]]),
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
