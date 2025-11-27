# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale
from triton.tools.tensor_descriptor import TensorDescriptor
from vllm.logger import init_logger
logger = init_logger(__name__)


def mxfp8_e4m3_quantize_python(data: torch.Tensor):
    assert len(data.shape) == 2, "Only 2d input tensor is supported"
    block_size1 = 32
    block_size0 = 1
    shape_before_padding = data.shape
    # pad data to make its shape a multiple of weight_block_size with the last element of data
    assert data.shape[1] % block_size1 == 0 and data.shape[0] % block_size0 == 0, "Data shape must be a multiple of tile size [1, 32]"

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    shape_after_padding = data.shape
    blk_m, blk_n = data.shape[0] // block_size0, data.shape[1] // block_size1
    data = data.reshape(blk_m, block_size0, blk_n, block_size1)
    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data = data.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data = data.to(torch.float32).contiguous().flatten(start_dim=2)
    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data), dim=-1, keepdim=True)

    # Calculate scales
    descale = max_abs / max_dtype
    exponent = torch.ceil(torch.log2(descale))
    # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
    exponent = torch.clamp(exponent, min=-127, max=127) + 127
    # Convert to uint8 container
    exponent = exponent.to(torch.uint8)
    # Calculate descale_fp to apply to data_hp
    scale_fp = torch.where(
        # If exponent is 0, descale_fp is 1.0 rather than 2^127
        exponent == 0,
        1.0,
        torch.exp2(127 - exponent.to(torch.float32)),
    )
    exponent = exponent.reshape(blk_m, blk_n)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(shape_after_padding)
    )

    # remove the padding
    if data.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    # Convert to target format, but still in original precision container
    return fp_data, exponent


def mxfp8_e4m3_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from flashinfer import mxfp8_quantize as mxfp8_e4m3_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP8 quantization. Please install it with"
            "`pip install flashinfer`"
        ) from err

    x_q, x_scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    if x_scales.ndim == 1:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def _cast_mxfp8_scales_to_bf16(scales: torch.Tensor) -> torch.Tensor:
    """
    Cast MXFP8 scales from uint8 to BF16.
    
    The scales are stored in uint8 format and need to be converted to BF16
    by left-shifting by 7 bits (to form the exponent) and reinterpreting
    as bfloat16.
    
    Args:
        scales: uint8 tensor containing MXFP8 scales
        
    Returns:
        BF16 tensor with the converted scales
    """
    return (scales.to(torch.int16) << 7).view(torch.bfloat16)


def dequant_mxfp8_to_bf16(
    x: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize MXFP8 tensor to BF16.
    
    Args:
        x: FP8 E4M3 tensor to dequantize
        scales: uint8 tensor containing MXFP8 scales
        
    Returns:
        BF16 dequantized tensor
    """
    scales_bf16 = _cast_mxfp8_scales_to_bf16(scales)
    # Repeat scales along the last dimension to match the block size
    scales_expanded = scales_bf16.reshape(*x.shape[:-1], -1).repeat_interleave(32, dim=-1)
    return x.to(torch.bfloat16) * scales_expanded

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret

@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.bfloat16
    elif output_type == 3:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a, a_scale, b, b_scale, dtype_dst, block_scale_type="mxfp8", is_swizzled=False):
    assert block_scale_type in ["mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"

    M = a.shape[0]
    N = b.shape[0]
    K = a.shape[1]
    assert b.shape[1] == K

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 32
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2
    NUM_STAGES = 4

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])
   
    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # Use 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements.
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.
    def _round_up(x: int, m: int) -> int:
        return (x + m - 1) // m
    
    a_scale_shape = [1,_round_up(M,128), _round_up(K // VEC_SIZE, 4), 2, 256]
    b_scale_shape = [1, _round_up(N,128), _round_up(K // VEC_SIZE, 4), 2, 256]
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]

    if is_swizzled:
        a_scale = a_scale.view(a_scale_shape)
        b_scale = b_scale.view(b_scale_shape)
    else:
        a_scale = swizzle_blockscale(a_scale).view(a_scale_shape)
        b_scale = swizzle_blockscale(b_scale).view(b_scale_shape)

    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.bfloat16:
        dtype_dst = 2
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 3
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        ELEM_PER_BYTE_A,
        ELEM_PER_BYTE_B,
        VEC_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        rep_m,
        rep_n,
        rep_k,
        NUM_STAGES,
    )

    return output
