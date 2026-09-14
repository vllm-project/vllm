# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm Qwen3-VL ViT kernels and M-RoPE for SupportsMRoPE models."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

# Covers Triton's ==1 / %16==0 / other integer buckets.
_MROPE_TOKEN_COUNTS = (1, 2, 16)


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


def _warm_mrope(runner: "GPUModelRunner", model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    from vllm.model_executor.models.interfaces import supports_mrope

    is_mrope_model = supports_mrope(model)
    if not is_mrope_model:
        return

    rope = next(
        (module for module in model.modules() if isinstance(module, MRotaryEmbedding)),
        None,
    )
    if rope is None:
        return

    model_config = runner.model_config
    parallel_config = runner.parallel_config
    num_query_heads = int(model_config.get_num_attention_heads(parallel_config))
    num_kv_heads = int(model_config.get_num_kv_heads(parallel_config))
    for num_tokens in _MROPE_TOKEN_COUNTS:
        positions = torch.arange(
            num_tokens, dtype=torch.long, device=runner.device
        ).expand(3, -1)
        query = torch.empty(
            (num_tokens, num_query_heads * rope.head_size),
            dtype=runner.dtype,
            device=runner.device,
        )
        key_tensor = torch.empty(
            (num_tokens, num_kv_heads * rope.head_size),
            dtype=runner.dtype,
            device=runner.device,
        )
        rope(positions, query, key_tensor)

    logger.info("Warmed M-RoPE Triton kernels.")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_vl_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm Qwen3-VL ViT kernels and M-RoPE for ``SupportsMRoPE`` models."""
    model = runner.get_model()
    _warm_vision(model, runner.device)
    _warm_mrope(runner, model)
    _synchronize_device(runner.device)
