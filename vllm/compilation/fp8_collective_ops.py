# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    input_to_float8)
from vllm.utils import direct_register_custom_op


def vllm_quantize_fp8_impl(x: torch.Tensor) -> tuple[torch.Tensor,
                                                      torch.Tensor]:
    """Quantize tensor to FP8 with per-tensor scaling"""
    return input_to_float8(x)


def vllm_quantize_fp8_fake(x: torch.Tensor) -> tuple[torch.Tensor,
                                                      torch.Tensor]:
    """Fake implementation for torch.compile tracing"""
    fp8_dtype = torch.float8_e4m3fn
    scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    return x.to(fp8_dtype), scale


def vllm_all_gather_fp8_impl(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    """All-gather FP8 tensor"""
    return get_tp_group().all_gather(x, dim)


def vllm_all_gather_fp8_fake(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    """Fake implementation - just replicate along dimension"""
    return x.repeat_interleave(world_size, dim=dim)


# Register custom ops
direct_register_custom_op(
    op_name="vllm_quantize_fp8",
    op_func=vllm_quantize_fp8_impl,
    mutates_args=[],
    fake_impl=vllm_quantize_fp8_fake,
)

direct_register_custom_op(
    op_name="vllm_all_gather_fp8",
    op_func=vllm_all_gather_fp8_impl,
    mutates_args=[],
    fake_impl=vllm_all_gather_fp8_fake,
)

# Export ops
vllm_quantize_fp8 = torch.ops.vllm.vllm_quantize_fp8.default
vllm_all_gather_fp8 = torch.ops.vllm.vllm_all_gather_fp8.default
