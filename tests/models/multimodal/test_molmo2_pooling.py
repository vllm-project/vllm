# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.molmo2_pooling import Molmo2PoolingPreparation

pytestmark = pytest.mark.skip_global_cleanup


def _reference_pooling_preparation(
    image_features: torch.Tensor,
    token_pooling: torch.Tensor,
    masked_average: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, _, dim = image_features.shape
    valid = token_pooling >= 0
    batch_idx = torch.arange(
        token_pooling.shape[0],
        dtype=torch.long,
        device=token_pooling.device,
    )
    batch_idx = torch.tile(
        batch_idx.view(batch_size, 1, 1),
        [1, token_pooling.shape[1], token_pooling.shape[2]],
    )
    to_pool = image_features.reshape(batch_size, -1, dim)[
        batch_idx, torch.clamp(token_pooling, min=0)
    ]
    to_pool = to_pool * valid.to(image_features.dtype)[..., None]
    to_pool = to_pool.reshape(-1, token_pooling.shape[-1], dim)

    if masked_average:
        denom = valid.reshape(-1, valid.shape[-1]).float().sum(-1).clamp_min(1)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(
            to_pool.dtype
        )
    else:
        query = to_pool.mean(-2, keepdim=True)

    return (
        to_pool,
        query,
        valid.reshape(-1, 1, 1, valid.shape[-1]),
        valid.any(-1),
    )


@pytest.mark.parametrize("masked_average", [False, True])
def test_molmo2_pooling_native_contract(
    default_vllm_config, masked_average: bool
) -> None:
    image_features = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(
        2, 3, 5, 7
    )
    token_pooling = torch.tensor(
        [
            [[0, 1, 2, 3], [5, -1, 5, 14], [-1, -1, -1, -1]],
            [[14, 0, 7, 7], [3, 4, -1, 6], [8, 9, 10, 11]],
        ],
        dtype=torch.long,
    )

    expected = _reference_pooling_preparation(
        image_features, token_pooling, masked_average
    )
    actual = Molmo2PoolingPreparation(
        masked_average=masked_average
    ).forward_native(image_features, token_pooling)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
