# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("molmo2_pooling_preparation")
class Molmo2PoolingPreparation(CustomOp):
    """Prepare gathered patch features for Molmo2 pooling attention."""

    def __init__(self, *, masked_average: bool) -> None:
        super().__init__(enforce_enable=True)
        self.masked_average = masked_average

    def forward_native(
        self,
        image_features: torch.Tensor,
        token_pooling: torch.Tensor,
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
            batch_idx, torch.clip(token_pooling, min=0)
        ]
        to_pool = to_pool * valid.to(image_features.dtype)[..., None]
        to_pool = to_pool.reshape(-1, token_pooling.shape[-1], dim)

        if self.masked_average:
            denom = valid.reshape(-1, valid.shape[-1]).float().sum(-1)
            denom = denom.clamp_min(1)
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

    def forward_cuda(
        self,
        image_features: torch.Tensor,
        token_pooling: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    forward_hip = forward_native
    forward_xpu = forward_native
    forward_cpu = forward_native
    forward_oot = forward_native
