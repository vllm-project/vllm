# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl

from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

__all__ = [
    "break_fp4_bytes",
    "dequant_nvfp4",
    "dequantize_to_dtype",
    "ref_nvfp4_quant",
    "run_nvfp4_emulations",
    "fused_moe_nvfp4_kernel",
    "invoke_fused_moe_nvfp4_kernel",
    "NVFP4_BLOCK_SIZE",
]

NVFP4_BLOCK_SIZE = 16

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@triton.jit
def fused_moe_nvfp4_kernel(
    a_ptr,
    b_packed_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_bpe,
    stride_bpn,
    stride_bpk,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K % 16 == 0, "BLOCK_SIZE_K must be multiple of 16")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_experts == -1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_packed_ptrs = (
        b_packed_ptr
        + off_experts * stride_bpe
        + offs_bn[None, :] * stride_bpn
        + (offs_k[:, None] // 2) * stride_bpk
    )

    b_scale_ptrs = (
        b_scale_ptr
        + off_experts * stride_bse
        + offs_bn[None, :] * stride_bsn
        + (offs_k[:, None] // 16) * stride_bsk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_remaining = K - k_start

        k_mask = offs_k < k_remaining

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        k_packed_mask = (offs_k // 2) < ((k_remaining + 1) // 2)
        b_packed = tl.load(
            b_packed_ptrs,
            mask=k_packed_mask[:, None],
            other=0,
        )

        k_scale_mask = (offs_k // 16) < tl.cdiv(k_remaining, 16)
        b_scale = tl.load(
            b_scale_ptrs,
            mask=k_scale_mask[:, None],
            other=0.0,
        )

        is_high_nibble = (offs_k % 2) == 1
        nibble = tl.where(
            is_high_nibble[:, None],
            (b_packed >> 4) & 0x0F,
            b_packed & 0x0F
        )

        n = nibble.to(tl.int32)
        sign_bit = (n >> 3) & 1
        mag = n & 0x07
        exp = 14 + (mag >> 1)
        mant_bit = (mag & 1) & (mag > 1)
        fp16_bits = (sign_bit << 15) | (exp << 10) | (mant_bit << 9)
        fp16_bits = fp16_bits * (mag != 0)
        fp_val = fp16_bits.to(tl.uint16).to(tl.float16, bitcast=True)

        b = (fp_val * b_scale).to(compute_type)

        b = tl.where(k_mask[:, None], b, 0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_packed_ptrs += (BLOCK_SIZE_K // 2) * stride_bpk
        b_scale_ptrs += (BLOCK_SIZE_K // 16) * stride_bsk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_nvfp4_kernel(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type: "tl.dtype",
) -> None:
    assert B_packed.dtype == torch.uint8, "Weights must be packed uint8"
    assert topk_weights is not None or not mul_routed_weight

    M = A.size(0)
    num_tokens = M * top_k
    E, N, packed_K = B_packed.shape
    K = packed_K * 2

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    block_k = config.get("BLOCK_SIZE_K", 64)
    if block_k % 16 != 0:
        block_k = ((block_k + 15) // 16) * 16

    fused_moe_nvfp4_kernel[grid](
        A,
        B_packed,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B_packed.stride(0),
        B_packed.stride(1),
        B_packed.stride(2),
        C.stride(-2),
        C.stride(-1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 8),
    )


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles
    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()
    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)
    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype)


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def ref_nvfp4_quant(x, global_scale, block_size):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // block_size, block_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = torch.clamp(scale, max=448, min=-448)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    # both outputs are float32
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
):
    group_size = 16
    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, input_global_scale, group_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // group_size, group_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / input_global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    del x_fp4, x_blockscale

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        x.device,
        group_size,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    del w_dq, x_dq
    return out


def _dequant_nvfp4(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:

    return dequantize_to_dtype(
        tensor_fp4,
        tensor_sf,
        global_scale,
        dtype,
        tensor_fp4.device,
        block_size,
    )

def _dequant_nvfp4_fake(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    return torch.empty((m, k), dtype=dtype, device=tensor_fp4.device)


try:
    direct_register_custom_op(
        op_name="dequant_nvfp4",
        op_func=_dequant_nvfp4,
        fake_impl=_dequant_nvfp4_fake,
    )
    dequant_nvfp4 = torch.ops.vllm.dequant_nvfp4
except AttributeError as error:
    raise error
