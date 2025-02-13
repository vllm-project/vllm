# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import functools
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    _normalize_quant_group_shape, scaled_dequantize)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED, CUTLASS_FP8_SUPPORTED, apply_fp8_linear)
from vllm.platforms import current_platform

logger = init_logger(__name__)

current_platform_fp8_dtype = (torch.float8_e4m3fnuz
                              if current_platform.is_rocm() else
                              torch.float8_e4m3fn)


def is_fp8(x: Union[torch.dtype, torch.Tensor]) -> bool:
    if isinstance(x, torch.Tensor):
        x = x.dtype
    return x == torch.float8_e4m3fn or x == torch.float8_e4m3fnuz


def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    shape_supported_by_cutlass = (weight.shape[0] % 128 == 0
                                  and weight.shape[1] % 128 == 0)
    if current_platform.is_rocm():
        scale_a_shape = ((input_2d.shape[-1] // block_size[1], ) +
                         input_2d.shape[:-1])[::-1]
        scale_b_shape = (weight_scale.view(-1, 1)
                         if weight_scale.dim() <= 1 else weight_scale.T).shape
        ar, ac = scale_a_shape
        br, bc = scale_b_shape
        if (ac > 1 or bc > 1 or ar not in (1, input_2d.shape[0])
                or br not in (1, weight.shape[0])):
            shape_supported_by_cutlass = False
    if cutlass_block_fp8_supported and shape_supported_by_cutlass:
        q_input, x_scale = per_token_group_quant_fp8(input_2d,
                                                     block_size[1],
                                                     column_major_scales=True)
        output = ops.cutlass_scaled_mm(q_input,
                                       weight.T,
                                       out_dtype=input.dtype,
                                       scale_a=x_scale,
                                       scale_b=weight_scale.T)
    else:
        q_input, x_scale = per_token_group_quant_fp8(input_2d,
                                                     block_size[1],
                                                     column_major_scales=False)
        output = w8a8_block_fp8_matmul(q_input,
                                       weight,
                                       x_scale,
                                       weight_scale,
                                       block_size,
                                       output_dtype=input.dtype)
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


# Unify the interface between `apply_w8a8_block_fp8_linear` and
# `apply_fp8_linear`
# NOTE(lucas): this is quite messy, we should think through this more formally
def apply_fp8_linear_generic(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_group_shape: Tuple[int, int],
    weight_group_shape: Tuple[int, int],
    input_scale: Optional[torch.Tensor] = None,  # static scale if one
    cutlass_fp8_supported: bool = CUTLASS_FP8_SUPPORTED,
    cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
) -> torch.Tensor:
    # View input as 2D matrix for fp8 methods
    input = input.view(-1, input.shape[-1])

    weight_group_shape = _normalize_quant_group_shape(\
        weight, weight_group_shape)
    input_group_shape = _normalize_quant_group_shape(input, input_group_shape)

    def is_dim_blocked(dim, shape, group_shape):
        return group_shape < shape[dim] and group_shape > 1

    if is_dim_blocked(0, weight.shape, weight_group_shape[0])\
     and is_dim_blocked(1, weight.shape, weight_group_shape[1]) and\
     input_group_shape == (1, weight_group_shape[1]):
        return apply_w8a8_block_fp8_linear(
            input,
            weight,
            list(weight_group_shape),
            weight_scale,
            cutlass_block_fp8_supported=cutlass_block_fp8_supported)
    else:
        # Despite having linear in the it doesn't conform to
        # `torch.nn.functional.linear` which is defined as `input @ weight.T`
        # so we explicitly transpose the weight matrix here
        return apply_fp8_linear(input, weight.T, weight_scale.T,
                    cutlass_fp8_supported=cutlass_fp8_supported,
                         use_per_token_if_dynamic=\
                             (input_group_shape == (1, input.shape[1])))
def pad_weight(weight, block_size):
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode='constant', value=0)
    padded_weight = torch.nn.Parameter(padded_weight, requires_grad=False)
    return padded_weight, M, N  # Return original dimensions for unpadding

def unpad_weight(weight, original_M, original_N, keep_first_dim=False):
    """Removes padding from the matrix to restore its original shape."""
    # if weight.shape[-2] == original_M and weight.shape[-1] == original_N:
    #     return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]

def pad_block_fp8_weight_naive(weight, weight_scale, block_size):

    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


def dequant_block_fp8_weight_naive(weight, weight_scale, block_size, dtype, original_M, original_N, do_unpad=False):

    assert len(block_size) == 2
    
    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    if do_unpad:
        dequant_weight = unpad_weight(dequant_weight, original_M, original_N, keep_first_dim=keep_first_dim)

    return dequant_weight


def apply_block_fp8_linear_hpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    original_M: Optional[torch.Tensor] = None,
    original_N: Optional[torch.Tensor] = None,
    do_unpad: bool = False,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    original_M = original_M.data
    original_N = original_N.data
    output_shape = [*input.shape[:-1], original_M]
    dequant_weight = dequant_block_fp8_weight_naive(weight, weight_scale, block_size, input_2d.dtype, original_M, original_N, do_unpad)
    output = torch.nn.functional.linear(input_2d, dequant_weight, bias=None)
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def input_to_float8(
        x: torch.Tensor,
        dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values "
    "with tensor-wise quantization."""
    if dtype is None:
        dtype = (torch.float8_e4m3fnuz
                 if current_platform.is_rocm() else torch.float8_e4m3fn)
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise
    quantization. The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise
    quantization scale. Note only float8 is supported for now.
    """
    x_dq_block = scaled_dequantize(x_q_block, x_s)
    x_q_tensor, scale = input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    return x_q_tensor, scale


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
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
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    if dtype is None:
        dtype = (torch.float8_e4m3fnuz
                 if current_platform.is_rocm() else torch.float8_e4m3fn)
    assert (x.shape[-1] % group_size == 0), (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.is_contiguous(), "`x` must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

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
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
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
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
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
                               block_k: int) -> Optional[Dict[int, Any]]:
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
    block_size: List[int],
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

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
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
