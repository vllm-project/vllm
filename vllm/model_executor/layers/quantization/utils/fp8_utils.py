# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import functools
import json
import os
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    group_broadcast)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv, direct_register_custom_op, has_deep_gemm
from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used

logger = init_logger(__name__)


def is_fp8(x: Union[torch.dtype, torch.Tensor]) -> bool:
    if isinstance(x, torch.Tensor):
        x = x.dtype
    return x == torch.float8_e4m3fn or x == torch.float8_e4m3fnuz


def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(A,
                                 B.T,
                                 out_dtype=output_dtype,
                                 scale_a=As,
                                 scale_b=Bs.T)


def rocm_aiter_gemm_w8a8_blockscale_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    import aiter as rocm_aiter

    return rocm_aiter.gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)


def rocm_aiter_gemm_w8a8_blockscale_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:

    m = A.shape[0]
    n = B.shape[0]
    Y = torch.empty(m, n, dtype=output_dtype, device=A.device)
    return Y


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_gemm_w8a8_blockscale",
        op_func=rocm_aiter_gemm_w8a8_blockscale_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_gemm_w8a8_blockscale_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    if (envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_LINEAR
            and current_platform.is_fp8_fnuz()):

        import aiter as rocm_aiter
        from aiter import get_hip_quant

        aiter_per1x128_quant = get_hip_quant(rocm_aiter.QuantType.per_1x128)


def dispatch_w8a8_blockscale_func(
    use_cutlass: bool, use_aiter_and_is_supported: bool
) -> Callable[[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.dtype,
], torch.Tensor]:
    if use_cutlass:
        return cutlass_scaled_mm
    if (use_aiter_and_is_supported):
        return torch.ops.vllm.rocm_aiter_gemm_w8a8_blockscale
    return w8a8_block_fp8_matmul


def should_use_deepgemm(output_dtype: torch.dtype, weight: torch.Tensor):
    """
    Check if DeepGEMM should be used based on the output dtype and weight shape.
    DeepGEMM is only supported for bfloat16 output dtype and weights with shape
    divisible by 128.
    """

    return (current_platform.is_cuda()
            and current_platform.is_device_capability(90) and has_deep_gemm()
            and envs.VLLM_USE_DEEP_GEMM and output_dtype == torch.bfloat16
            and weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0)


# TODO fix ROCm->Triton custom path:
#  https://github.com/vllm-project/vllm/issues/14397
def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
    use_aiter_and_is_supported: bool = False,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    output_dtype = input.dtype

    if should_use_deepgemm(output_dtype, weight):

        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]

        q_input, x_scale = per_token_group_quant_fp8(
            input_2d,
            block_size[1],
            column_major_scales=True,
        )

        import vllm.model_executor.layers.quantization.deepgemm  # noqa: F401
        output = torch.ops.vllm.w8a8_block_fp8_matmul_deepgemm(
            q_input,
            weight,
            x_scale,
            weight_scale,
            block_size,
            output_dtype=output_dtype)
        if bias is not None:
            output += bias
        return output.to(dtype=output_dtype).view(*output_shape)

    if current_platform.is_cuda():
        if current_platform.has_device_capability(100):

            use_cutlass = cutlass_block_fp8_supported and (
                cdiv(weight.shape[0], 128) == weight_scale.shape[0]
                and cdiv(weight.shape[1], 128) == weight_scale.shape[1])
        else:
            # TODO: update this after switching to public sm90 block scale gemm
            # as it also supports weight.shape % 128 != 0
            use_cutlass = cutlass_block_fp8_supported and (
                weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0)
    else:
        use_cutlass = False

    w8a8_blockscale_func = dispatch_w8a8_blockscale_func(
        use_cutlass, use_aiter_and_is_supported)
    if use_cutlass:
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=use_cutlass)
        output = w8a8_blockscale_func(q_input, weight, x_scale, weight_scale,
                                      block_size, input.dtype)

    else:
        if use_aiter_and_is_supported:
            q_input, x_scale = aiter_per1x128_quant(
                input_2d.contiguous(), quant_dtype=rocm_aiter.dtypes.fp8)
        else:
            q_input, x_scale = per_token_group_quant_fp8(
                input_2d, block_size[1], column_major_scales=use_cutlass)

        output = w8a8_blockscale_func(q_input, weight, x_scale, weight_scale,
                                      block_size, input.dtype)

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def apply_w8a8_block_fp8_linear_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
    use_aiter_and_is_supported: bool = False,
) -> torch.Tensor:
    output_shape = [*input.shape[:-1], weight.shape[0]]
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


if not current_platform.is_cpu():
    direct_register_custom_op(
        op_name="apply_w8a8_block_fp8_linear",
        op_func=apply_w8a8_block_fp8_linear,
        mutates_args=[],
        fake_impl=apply_w8a8_block_fp8_linear_fake,
    )


def input_to_float8(
        x: torch.Tensor,
        dtype: Optional[torch.dtype] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values "
    "with tensor-wise quantization."""
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise
    quantization. The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise
    quantization scale. Note only float8 is supported for now.
    """
    x_dq_block = group_broadcast(x_q_block, x_s)
    x_q_tensor, scale = input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    return x_q_tensor, scale


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) *
                                                        group_size)
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) *
                                                        group_size)
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    # Ensure offset calculation uses int64 for y_s_ptr
    y_s_ptr_offset = (scale_col.to(tl.int64) * y_s_col_stride) + scale_row.to(
        tl.int64)
    y_s_ptr += y_s_ptr_offset

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
    use_ue8m0: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_blackwell_deep_gemm_e8m0_used()
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert (x.shape[-1] % group_size == 0), (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    # Allocate the scale tensor in either row- or column-major format.
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # prefer CUDA kernel if available
    if current_platform.is_cuda() and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant(x, x_q, x_s, group_size, eps,
                                               fp8_min, fp8_max, use_ue8m0)
        return x_q, x_s

    # TRITON FALLBACK
    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

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
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)

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
def get_w8a8_block_fp8_configs(N: int, K: int, block_n: int,
                               block_k: int) -> Optional[dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.
    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json"  # noqa: E501

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block FP8 kernel.",
                config_file_path,
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        config_file_path,
    )
    return None


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
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

    C_shape = A.shape[:-1] + (N, )
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if configs:
        # Get the optimal config if there is one
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_size[0]
        # BLOCK_SIZE_K must be divisible by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) *
                triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    _w8a8_block_fp8_matmul[grid](
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


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/0c88cd01392c1073c7049a97d6328c7bba9b3947
# TODO(wentao): remove this function when DeepGEMM exposes this function
def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of
        16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return cdiv(x, alignment) * alignment


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/0c88cd01392c1073c7049a97d6328c7bba9b3947
# TODO(wentao): remove this function when DeepGEMM exposes this function
def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose`
        will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along
        the M axis (thus meets the requirement of LHS scaling tensor in
        DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in
    # CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    m, n = x.shape[-2], x.shape[-1]
    aligned_m = get_tma_aligned_size(m, x.element_size())
    if x.dim() == 2:
        if x.stride(0) == 1 and x.stride(1) == aligned_m:
            return x
        x, remove_dim = x.unsqueeze(0), True

    b = x.shape[0]

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(
            2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(
        torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def requant_weight_ue8m0_inplace(
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: Sequence[int] = (128, 128),
) -> None:
    """Re-quantise *weight* so that its per-block scaling factors are in the
    UE8M0 (power-of-two) format expected by the new DeepGEMM kernels inplace.

    Args:
        weight: Block-quantised weight tensor stored in ``torch.float8_e4m3fn``.
            Expected shape ``(..., M, K)``.
        weight_scale: Corresponding per-block scale tensor (``torch.float32``)
            with shape ``(..., M // block_size[0], K // block_size[1])``.
        block_size: 2-element iterable ``[block_m, block_k]`` describing the
            block quantisation granularity.
    """
    if weight.numel() == 0:
        return

    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected *weight* to be torch.float8_e4m3fn, got "
                         f"{weight.dtype} instead.")

    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    block_m, block_k = int(block_size[0]), int(block_size[1])

    # Flatten leading dimensions so we can iterate over the last two dims.
    leading_shape = weight.shape[:-2]
    if len(leading_shape) == 0:
        w_view = weight.unsqueeze(0)
        s_view = weight_scale.unsqueeze(0)
    else:
        w_view = weight.reshape(-1, weight.shape[-2], weight.shape[-1])
        s_view = weight_scale.reshape(-1, *weight_scale.shape[-2:])

    num_mats = w_view.size(0)
    for idx in range(num_mats):
        w_q = w_view[idx]
        s_old = s_view[idx]

        # De-quantise with the *old* scaling factors (float32).
        m_cur, k_cur = w_q.shape
        s_float = s_old.to(torch.float32)
        # Expand scales along rows and cols by block size, then crop.
        s_exp_r = torch.repeat_interleave(s_float, block_m, dim=0)
        s_exp = torch.repeat_interleave(s_exp_r, block_k, dim=1)
        s_exp = s_exp[:m_cur, :k_cur]
        w_dq = w_q.to(torch.float32) * s_exp
        # Re-quantise using power-of-two scaling (UE8M0).
        w_requant, s_requant = per_block_cast_to_fp8(w_dq, [block_m, block_k],
                                                     use_ue8m0=True)

        # Write back the results in-place.
        w_q.copy_(w_requant)
        s_old.copy_(s_requant)
