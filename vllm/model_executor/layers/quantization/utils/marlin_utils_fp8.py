# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .marlin_utils import marlin_make_workspace, marlin_permute_scales

logger = init_logger(__name__)


def is_fp8_marlin_supported():
    return current_platform.has_device_capability(80)


def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n, )

    output = ops.fp8_marlin_gemm(
        a=reshaped_x,
        b_q_weight=weight,
        b_scales=weight_scale,
        workspace=workspace,
        num_bits=8,
        size_m=reshaped_x.shape[0],
        size_n=size_n,
        size_k=size_k,
    )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def prepare_fp8_layer_for_marlin(layer: torch.nn.Module,
                                 strategy: str = "tensor") -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace(part_size_n, device)

    # WEIGHT
    # Repack weights to marlin format
    marlin_qweight = ops.gptq_marlin_repack(b_q_weight=pack_fp8_to_int32(
        layer.weight),
                                            perm=torch.empty(0,
                                                             dtype=torch.int,
                                                             device=device),
                                            size_k=part_size_k,
                                            size_n=part_size_n,
                                            num_bits=8)
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    scales = layer.weight_scale.to(layer.orig_dtype)
    # Permute scales
    marlin_scales = marlin_permute_scales(s=scales,
                                          size_k=part_size_k,
                                          size_n=part_size_n,
                                          group_size=-1)
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)


def pack_fp8_to_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.shape[0] % 4 == 0

    # Reshape to prepare for packing
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

    # Convert fp8 to uint8 (byte) representation
    byte_tensor = reshaped.view(torch.uint8)

    # Pack 4 uint8 values into one int32
    packed = (byte_tensor[:, 0].to(torch.int32) |
              (byte_tensor[:, 1].to(torch.int32) << 8) |
              (byte_tensor[:, 2].to(torch.int32) << 16) |
              (byte_tensor[:, 3].to(torch.int32) << 24))

    return packed.view(fp8_tensor.shape[0] // 4,
                       *fp8_tensor.shape[1:]).contiguous()
