# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hw-agnostic FP8 weight + Triton kernel helpers."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _get_fp8_min_max() -> tuple[float, float]:
    finfo = torch.finfo(current_platform.fp8_dtype())
    return finfo.min, finfo.max


# ---------------------------------------------------------------------------
# Per-token-group FP8 quantization (Triton fallback + CUDA fast-path)
# ---------------------------------------------------------------------------


@triton.jit
def _per_token_group_quant_fp8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax * (1.0 / fp8_max)
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    out_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP8 quantization (row-major scales)."""
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"x.shape[-1]={x.shape[-1]} must be divisible by group_size={group_size}"
    )
    assert x.stride(-1) == 1, "x groups must be contiguous"

    fp8_min, fp8_max = _get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = (
        out_q
        if out_q is not None
        else torch.empty(x.shape, device=x.device, dtype=dtype)
    )

    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # Prefer the registered C++ op when available (CUDA/XPU); fall back to
    # the Triton kernel otherwise.
    if (
        current_platform.is_cuda_alike() or current_platform.is_xpu()
    ) and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            False,  # use_ue8m0
            False,  # column_major_scales
            False,  # tma_aligned_scales
        )
        return x_q, x_s

    M = x.numel() // group_size
    BLOCK = triton.next_power_of_2(group_size)
    num_warps = min(max(BLOCK // 256, 1), 8)
    _per_token_group_quant_fp8_kernel[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, x_s


# ---------------------------------------------------------------------------
# W8A8 block-scaled MM (Triton)
# ---------------------------------------------------------------------------


@triton.jit
def _w8a8_triton_block_scaled_mm_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache
def _get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, dict] | None:
    """Load tuned configs for the w8a8 block FP8 kernel if available.

    Reuses the shared tuned-config directory so drop-in JSON tunings keep
    working without duplicating them under this tree.
    """
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )
    configs_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "layers",
        "quantization",
        "utils",
        "configs",
    )
    config_file_path = os.path.normpath(os.path.join(configs_dir, json_file_name))
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block FP8 kernel.",
                config_file_path,
            )
            return {int(key): val for key, val in json.load(f).items()}

    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        config_file_path,
    )
    return None


def w8a8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Block-scaled w8a8 matmul on (A, B) with per-block scales (As, Bs)."""
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = _get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def create_fp8_weight_parameter(
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader: Callable | None,
) -> torch.nn.Parameter:
    return ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )


def create_fp8_scale_parameter(
    parameter_type: torch.nn.Parameter,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    block_size: list[int] | None,
    weight_loader: Callable | None,
    scale_dtype: torch.dtype | None = None,
) -> torch.nn.Parameter:
    dtype = scale_dtype if scale_dtype is not None else torch.float32
    if parameter_type == ChannelQuantScaleParameter:
        scale = parameter_type(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
    elif parameter_type == BlockQuantScaleParameter:
        assert block_size is not None
        block_n, block_k = block_size[0], block_size[1]
        output_size_per_partition = sum(output_partition_sizes)
        scale = parameter_type(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
    elif parameter_type == PerTensorScaleParameter:
        scale = parameter_type(
            data=torch.empty(len(output_partition_sizes), dtype=dtype),
            weight_loader=weight_loader,
        )
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")

    if dtype == torch.float32:
        scale[:] = torch.finfo(torch.float32).min
    set_weight_attrs(scale, {"scale_type": "weight_scale"})
    return scale


def create_fp8_input_scale(
    output_partition_sizes: list[int], weight_loader: Callable | None
) -> torch.nn.Parameter:
    scale = PerTensorScaleParameter(
        data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        weight_loader=weight_loader,
    )
    scale[:] = torch.finfo(torch.float32).min
    return scale


def validate_fp8_block_shape(
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    block_size: list[int],
) -> None:
    """Validate block-quantization shapes against TP partitioning."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    if getattr(layer, "allow_fp8_block_shape_mismatch", False):
        logger.debug(
            "Skipping FP8 block shape validation for layer %s "
            "due to detected mismatch allowance.",
            getattr(layer, "prefix", "<unknown>"),
        )
        return

    tp_size = getattr(layer, "tp_size", get_tensor_model_parallel_world_size())
    block_n, block_k = block_size[0], block_size[1]

    if (
        tp_size > 1
        and input_size // input_size_per_partition == tp_size
        and input_size_per_partition % block_k != 0
    ):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by weight quantization block_k = {block_k}."
        )

    is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size
    is_merged_gemm = len(output_partition_sizes) > 1
    if is_tp_split or is_merged_gemm:
        sizes_to_check = output_partition_sizes
        if not is_tp_split and is_merged_gemm:
            sizes_to_check = output_partition_sizes[:-1]
        for output_partition_size in sizes_to_check:
            if output_partition_size % block_n != 0:
                raise ValueError(
                    f"Weight output_partition_size = {output_partition_size} "
                    f"is not divisible by weight quantization block_n = {block_n}."
                )


# ---------------------------------------------------------------------------
# Weight requantization (linear)
# ---------------------------------------------------------------------------


def _per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: float | torch.Tensor
) -> torch.Tensor:
    return tensor.to(torch.float16) * inv_scale


def _requantize_with_max_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize a fused weight to the max of its per-shard scales."""
    max_w_scale = weight_scale.max()

    # If the on-disk checkpoint had a fused QKV/MLP module, all but the
    # first scale stay at the float32-min sentinel and we can skip the
    # requantization (already quantized with the single scale).
    unfused_module_in_checkpoint = (
        weight_scale.ndim != 0
        and weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = _per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = ops.scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def process_fp8_weight_tensor_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: list[int],
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Tensor-wise FP8 weight requantization (single max scale)."""
    weight_scale, weight = _requantize_with_max_scale(
        weight=weight,
        weight_scale=weight_scale,
        logical_widths=logical_widths,
    )
    return weight, weight_scale, input_scale


def process_fp8_weight_block_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 weight processing (passthrough)."""
    return weight, weight_scale
