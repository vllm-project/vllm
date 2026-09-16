# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.triton_utils import tl, triton


@triton.jit
def _molmo2_pooling_preparation_kernel(
    image_features_ptr,
    token_pooling_ptr,
    to_pool_ptr,
    query_ptr,
    valid_ptr,
    valid_token_ptr,
    image_stride_batch,
    image_stride_crop,
    image_stride_patch,
    image_stride_dim,
    pooling_stride_batch,
    pooling_stride_group,
    pooling_stride_item,
    num_groups: tl.constexpr,
    num_patches: tl.constexpr,
    pool_size: tl.constexpr,
    dim: tl.constexpr,
    masked_average: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)
    batch_id = group_id // num_groups
    group_in_batch = group_id % num_groups

    item_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = dim_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    item_mask = item_offsets < pool_size
    dim_mask = dim_offsets < dim

    pooling_offsets = (
        batch_id * pooling_stride_batch
        + group_in_batch * pooling_stride_group
        + item_offsets * pooling_stride_item
    )
    patch_indices = tl.load(
        token_pooling_ptr + pooling_offsets,
        mask=item_mask,
        other=-1,
    )
    valid_items = item_mask & (patch_indices >= 0)
    safe_indices = tl.maximum(patch_indices, 0)
    crop_indices = safe_indices // num_patches
    patch_indices = safe_indices % num_patches

    image_offsets = (
        batch_id * image_stride_batch
        + crop_indices[:, None] * image_stride_crop
        + patch_indices[:, None] * image_stride_patch
        + dim_offsets[None, :] * image_stride_dim
    )
    values = tl.load(
        image_features_ptr + image_offsets,
        mask=valid_items[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offsets = (
        group_id * pool_size * dim + item_offsets[:, None] * dim + dim_offsets[None, :]
    )
    tl.store(
        to_pool_ptr + output_offsets,
        values,
        mask=item_mask[:, None] & dim_mask[None, :],
    )

    summed = tl.sum(values.to(tl.float32), axis=0)
    if masked_average:
        divisor = tl.maximum(tl.sum(valid_items.to(tl.int32), axis=0), 1)
    else:
        divisor = pool_size
    tl.store(
        query_ptr + group_id * dim + dim_offsets,
        summed / divisor,
        mask=dim_mask,
    )

    metadata_mask = (dim_block_id == 0) & item_mask
    tl.store(
        valid_ptr + group_id * pool_size + item_offsets,
        valid_items,
        mask=metadata_mask,
    )
    if dim_block_id == 0:
        tl.store(
            valid_token_ptr + batch_id * num_groups + group_in_batch,
            tl.sum(valid_items.to(tl.int32), axis=0) > 0,
        )


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
        if (
            image_features.dtype not in (torch.float16, torch.bfloat16)
            or token_pooling.dtype not in (torch.int32, torch.int64)
            or image_features.ndim != 4
            or token_pooling.ndim != 3
            or image_features.shape[0] != token_pooling.shape[0]
            or 0 in image_features.shape
            or 0 in token_pooling.shape
        ):
            return self.forward_native(image_features, token_pooling)

        batch_size, _, num_patches, dim = image_features.shape
        _, num_groups, pool_size = token_pooling.shape
        total_groups = batch_size * num_groups
        to_pool = torch.empty(
            (total_groups, pool_size, dim),
            dtype=image_features.dtype,
            device=image_features.device,
        )
        query = torch.empty(
            (total_groups, 1, dim),
            dtype=image_features.dtype,
            device=image_features.device,
        )
        valid = torch.empty(
            (total_groups, 1, 1, pool_size),
            dtype=torch.bool,
            device=token_pooling.device,
        )
        valid_token = torch.empty(
            (batch_size, num_groups),
            dtype=torch.bool,
            device=token_pooling.device,
        )

        block_k = triton.next_power_of_2(pool_size)
        block_d = 128
        _molmo2_pooling_preparation_kernel[(total_groups, triton.cdiv(dim, block_d))](
            image_features,
            token_pooling,
            to_pool,
            query,
            valid,
            valid_token,
            *image_features.stride(),
            *token_pooling.stride(),
            num_groups=num_groups,
            num_patches=num_patches,
            pool_size=pool_size,
            dim=dim,
            masked_average=self.masked_average,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return to_pool, query, valid, valid_token

    forward_hip = forward_native
    forward_xpu = forward_native
    forward_cpu = forward_native
    forward_oot = forward_native
