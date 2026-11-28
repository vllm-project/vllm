# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the Qwen3-VL vision tower's Triton kernels."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.kernel_warmup import KernelWarmup
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _warm_vision(model: torch.nn.Module, device: torch.device) -> None:
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    for visual in model.modules():
        if not isinstance(visual, Qwen3_VisionTransformer):
            continue
        # CPU offloading keeps tower weights off the accelerator until the
        # module forward hook copies them back, so warming here would launch
        # Triton on host pointers.
        if visual.pos_embed.weight.device.type != device.type:
            continue
        attention = visual.blocks[0].attn
        num_heads = int(attention.num_attention_heads_per_partition)
        head_size = int(attention.hidden_size_per_attention_head)
        merge_size = int(visual.spatial_merge_size)
        divisible = round_up(16, merge_size)
        while divisible % 16:
            divisible += merge_size
        spatial_sizes = [divisible]
        if merge_size % 16:
            spatial_sizes.append(merge_size)
            if merge_size == 1:
                spatial_sizes.append(2)
        grids = [(1, h, w) for h in spatial_sizes for w in spatial_sizes]

        for grid in grids:
            grid_thw = [list(grid)]
            visual.fast_pos_embed_interpolate(grid_thw)
            cos, sin = visual.rot_pos_emb(grid_thw)
            qk = torch.empty(
                (2, grid[0] * grid[1] * grid[2], num_heads, head_size),
                dtype=cos.dtype,
                device=cos.device,
            )
            attention.apply_rotary_emb(qk, cos, sin)

        logger.info(
            "Warmed position embedding and vision rotary kernels on grids=%s.",
            grids,
        )


class Qwen3VLKernelWarmup(KernelWarmup):
    """Kernel warmup for models with a Qwen3-VL vision tower."""

    def warmup_jit_kernels(self, worker: "Worker") -> None:
        super().warmup_jit_kernels(worker)
        device = self.runner.device
        with torch.inference_mode():
            _warm_vision(self.model, device)
        if device.type == "cuda":
            torch.accelerator.synchronize(device)
