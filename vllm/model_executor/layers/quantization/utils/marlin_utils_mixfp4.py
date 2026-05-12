# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin helpers for MixFP4 W4A16 weight-only quantization."""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_global_scale,
)
from vllm.scalar_type import scalar_types

_GROUP_SIZE = 16


def mixfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Reorder FP8 scale byte lanes while preserving MixFP4 flags."""
    raw = marlin_scales.contiguous().view(torch.uint8)
    if raw.dim() != 2:
        raise ValueError(
            "MixFP4 Marlin scales must be a 2D tensor after permutation, "
            f"got shape {tuple(raw.shape)}."
        )
    if raw.size(1) % 4 != 0:
        raise ValueError(
            "MixFP4 Marlin scale layout requires the N dimension to be "
            f"divisible by 4 after permutation, got {raw.size(1)}."
        )

    raw = raw.view(-1, 4)[:, [0, 2, 1, 3]].reshape(raw.size(0), -1)
    return raw.contiguous().view(torch.float8_e4m3fn)


def prepare_mixfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    """Repack MixFP4 compressed-tensors weights into Marlin layout."""
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype
    if param_dtype != torch.bfloat16:
        raise RuntimeError("MixFP4 Marlin currently supports bfloat16 models only.")
    if part_size_k % (2 * _GROUP_SIZE) != 0:
        raise RuntimeError(
            "MixFP4 Marlin requires input_size_per_partition to be divisible "
            f"by {2 * _GROUP_SIZE}, got {part_size_k}."
        )

    assert layer.weight.shape == (part_size_n, part_size_k // 2)
    device = layer.weight.device
    layer.workspace = marlin_make_workspace_new(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    weight_scale = layer.weight_scale.T.contiguous()
    scale_bf16 = weight_scale.to(torch.bfloat16)
    scale_perm_bf16 = marlin_permute_scales(
        s=scale_bf16,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=_GROUP_SIZE,
    )
    flag_permuted = torch.signbit(scale_perm_bf16).to(torch.uint8)
    mag_fp8 = scale_perm_bf16.abs().to(torch.float8_e4m3fn)
    mag_bytes = mag_fp8.contiguous().view(torch.uint8)
    scales_flagged = ((mag_bytes & 0x7F) | (flag_permuted << 7)).view(
        torch.float8_e4m3fn
    )
    layer.weight_scale = torch.nn.Parameter(
        mixfp4_marlin_process_scales(scales_flagged), requires_grad=False
    )

    weight_global_scale = layer.weight_global_scale.to(torch.float32)
    weight_global_scale = nvfp4_marlin_process_global_scale(
        weight_global_scale, param_dtype
    )
    layer.weight_global_scale = torch.nn.Parameter(
        weight_global_scale, requires_grad=False
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        layer.bias = torch.nn.Parameter(
            marlin_permute_bias(layer.bias), requires_grad=False
        )


def apply_mixfp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the MixFP4 Marlin GEMM kernel."""
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    output = ops.mixfp4_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_bias=bias,
        b_scales=weight_scale,
        global_scale=weight_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    )
    return output.reshape(out_shape)
