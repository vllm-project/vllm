# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12 Humming NVFP4 GEMMs using Inkling's quantized expert rows."""

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton

from .block_scales import _scale_offsets
from .moe_sm90 import HummingExperts
from .trtllm_bmm import ExpertPlan


@triton.jit
def _prepare_rows(
    Scales,
    Tiles,
    OutScales,
    Ids,
    GROUPS: tl.constexpr,
    TILE: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = row // TILE
    active = tile < tl.load(Tiles + 2 * MAX_CTAS)
    row_end = tl.load(Tiles + MAX_CTAS + tile, active, other=0)
    valid = active & (row < row_end)
    expert = tl.load(Tiles + tile, active, other=-1)
    tl.store(Ids + row, tl.where(valid, expert, -1))
    groups = tl.arange(0, BLOCK)
    scales = tl.load(
        Scales + _scale_offsets(row, groups, GROUPS, TILE),
        valid & (groups < GROUPS),
        other=0,
    )
    tl.store(OutScales + row * GROUPS + groups, scales, groups < GROUPS)


class HummingNVFP4Experts(HummingExperts):
    """Keep quantized activations and their outer scales across both GEMMs.

    Weight outer scales are already folded into the routed row scales, so the
    prepared weights use unit global scales. Repacking preserves FP4 payloads
    and E4M3 scale bytes; the existing router and SiTU quantizer remain in use.
    """

    def __init__(self, weights, shapes) -> None:
        from vllm.utils.humming import ModeloptNvfp4InputSchema

        super().__init__(
            weights,
            shapes,
            nvfp4=True,
            input_schema=ModeloptNvfp4InputSchema(
                quant_algo="NVFP4", dynamic=True
            ).to_humming_schema(torch.bfloat16),
        )

    def bmm(
        self,
        quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        name: str,
        tiles: torch.Tensor,
        plan: ExpertPlan,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        act, scale, row_scale = quantized
        config, tensors = self.configs[name], self.tensors[name]
        rows, groups = act.shape[0], config.shape_k // 16
        scales = torch.empty((rows, groups), device=act.device, dtype=torch.uint8)
        ids = torch.empty((rows, 1), device=act.device, dtype=torch.int32)
        _prepare_rows[(rows,)](
            scale.view(torch.uint8),
            tiles,
            scales,
            ids,
            GROUPS=groups,
            TILE=plan.tile,
            MAX_CTAS=plan.max_ctas,
            BLOCK=triton.next_power_of_2(groups),
        )
        sorted_ids, expert_ids, count = moe_align_block_size(
            ids,
            self._block_m(name, rows),
            self.num_experts,
            ignore_invalid_experts=True,
        )[:3]
        return humming_forward(
            config,
            inputs=act,
            weight=tensors["weight"],
            weight_scale=tensors["weight_scale"],
            weight_scale_2=tensors["weight_scale_2"],
            input_scale=scales.view(torch.int32),
            input_scale_2=row_scale.reshape(-1, 1),
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=count,
            top_k=1,
            valid_shape_m=rows,
            locks=self.locks,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )
