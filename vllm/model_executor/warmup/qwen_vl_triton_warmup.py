# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm Qwen3-VL / Qwen3.5-VL Triton kernels (ViT interpolate, rotary, M-RoPE)."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

# Covers Triton's ==1 / %16==0 / other integer buckets.
_MROPE_TOKEN_COUNTS = (1, 2, 16)


def _warm_vision(model: torch.nn.Module) -> None:
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    for visual in model.modules():
        if not isinstance(visual, Qwen3_VisionTransformer):
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


def _runner_uses_mrope(runner: object) -> bool:
    uses_mrope = getattr(runner, "uses_mrope", None)
    if uses_mrope is not None:
        return bool(uses_mrope)
    model_config = getattr(runner, "model_config", None)
    return bool(getattr(model_config, "uses_mrope", False))


def _runner_num_query_heads(runner: object) -> int:
    num_query_heads = getattr(runner, "num_query_heads", None)
    if num_query_heads is not None:
        return int(num_query_heads)
    model_config = getattr(runner, "model_config", None)
    parallel_config = getattr(runner, "parallel_config", None)
    return int(model_config.get_num_attention_heads(parallel_config))


def _warm_mrope(runner: "GPUModelRunner", model: torch.nn.Module) -> None:
    if not _runner_uses_mrope(runner):
        return

    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    num_query_heads = _runner_num_query_heads(runner)
    num_kv_heads = int(runner.model_config.get_num_kv_heads(runner.parallel_config))
    seen: set[tuple[object, ...]] = set()

    for rope in model.modules():
        if not isinstance(rope, MRotaryEmbedding):
            continue
        key = (
            rope.head_size,
            rope.rotary_dim,
            tuple(rope.mrope_section or ()),
            rope.mrope_interleaved,
            rope.is_neox_style,
            runner.dtype,
        )
        if key in seen:
            continue
        seen.add(key)

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

    if seen:
        logger.info(
            "Warmed M-RoPE Triton kernels for %d compile-key shape(s).", len(seen)
        )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_vl_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm Qwen3-VL / Qwen3.5-VL kernels dummy runs do not compile.

    Detects ``Qwen3_VisionTransformer`` (Qwen3-VL, Qwen3-VL-MoE, Qwen3.5-VL)
    and ``MRotaryEmbedding``. Not gated on GDN model types.
    """
    model = runner.get_model()
    _warm_vision(model)
    _warm_mrope(runner, model)
    _synchronize_device(getattr(runner, "device", torch.device("cuda")))
