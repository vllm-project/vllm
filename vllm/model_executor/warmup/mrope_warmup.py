# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm M-RoPE Triton kernels for `SupportsMRoPE` models."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Covers Triton's ==1 / %16==0 / other integer buckets.
_MROPE_TOKEN_COUNTS = (1, 2, 16)


@torch.inference_mode()
def mrope_triton_warmup(runner: "GPUModelRunner") -> None:
    _warm_mrope(runner, runner.get_model())
    if runner.device.type == "cuda":
        torch.accelerator.synchronize(runner.device)


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
    if rope is None or all(
        getattr(module, "use_fused_qk_norm_rope_gate", False)
        for module in model.modules()
        if any(child is rope for child in module.children())
    ):
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
