# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused inverse RoPE + block-scaled FP8 quantization kernel for DeepseekV4 attention.

Output scale format is pre-transformed (MN-major TMA-aligned; FP32 on SM90,
INT32-packed UE8M0 on SM100) so fp8_einsum skips transform_sf_into_required_layout.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit(do_not_specialize=["num_tokens"])
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
    PERMUTED_OUTPUT: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,  # triton metadata
):
    # Cast every stride to int64 — without this, Python-int strides are
    # inferred as int32 and `pid_token(int64) × stride(int32)` can lower to
    # int32 arithmetic, wrapping past 2³¹ for large prefill batches → IMA.
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)
    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    out_stride_group = out_stride_group.to(tl.int64)
    out_stride_token = out_stride_token.to(tl.int64)
    scale_stride_group = scale_stride_group.to(tl.int64)
    scale_stride_k = scale_stride_k.to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    # PERMUTED_OUTPUT: chunk (h, c) of a wo_a group lands at chunk position
    # c * G + h (FlashMLA fused-kernel layout); scale_ptr is then a uint8 view
    # and the scale strides are in bytes.
    chunk_ids = tl.arange(0, CHUNKS_PER_HEAD) * heads_per_group + head_in_group
    if pid_token >= num_tokens:
        if not QUANTIZE:
            return
        if TMA_ALIGNED_SCALES and PERMUTED_OUTPUT:
            byte_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token * 4
                + (chunk_ids // 4) * scale_stride_k
                + chunk_ids % 4
            )
            tl.store(byte_addr, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.uint8))
        elif TMA_ALIGNED_SCALES:
            packed_offsets = tl.arange(0, CHUNKS_PER_HEAD // 4)
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + (head_in_group * (CHUNKS_PER_HEAD // 4) + packed_offsets)
                * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((CHUNKS_PER_HEAD // 4,), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = NOPE_DIM
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    out_base = out_ptr + g * out_stride_group + pid_token * out_stride_token
    if PERMUTED_OUTPUT:
        out_offsets = tl.reshape(
            tl.reshape(chunk_ids, (CHUNKS_PER_HEAD, 1)) * QUANT_GROUP_SIZE
            + tl.reshape(tl.arange(0, QUANT_GROUP_SIZE), (1, QUANT_GROUP_SIZE)),
            (HEAD_DIM,),
        )
    else:
        out_offsets = qb_start * QUANT_GROUP_SIZE + offsets

    if not QUANTIZE:
        tl.store(out_base + out_offsets, x)
        return

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    tl.store(out_base + out_offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES and PERMUTED_OUTPUT:
        ue8m0_bytes = (scales.to(tl.int32, bitcast=True) >> 23) & 0xFF
        byte_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token * 4
            + (chunk_ids // 4) * scale_stride_k
            + chunk_ids % 4
        )
        tl.store(byte_addr, ue8m0_bytes.to(tl.uint8))
    elif TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(
            tl.reshape(ue8m0_bytes, (CHUNKS_PER_HEAD // 4, 4))
            << (tl.arange(0, 4)[None, :] * 8),
            axis=1,
        )
        packed_offsets = tl.arange(0, CHUNKS_PER_HEAD // 4)
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + (head_in_group * (CHUNKS_PER_HEAD // 4) + packed_offsets) * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
    quantize: bool = True,
    permuted_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + block-scaled FP8 quantization.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] with cos||sin.
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        tma_aligned_scales: Output INT32 packed UE8M0 for SM100 (True)
                            or FP32 for SM90 (False).
        quantize: Quantize the rotated output to FP8 and return its scales.
        permuted_output: Store each group in the FlashMLA fused-kernel layout:
            32-element chunk ``(h, c)`` at chunk position ``c * G + h`` (values
            and scale bytes alike). Requires ``quant_group_size == 32``,
            ``tma_aligned_scales`` and ``quantize``.

    Returns:
        Rotated output in [T, G, D] and its FP8 scales. The scale tensor is
        empty when quantization is disabled.
    """
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32
    if permuted_output:
        assert quantize and tma_aligned_scales and quant_group_size == 32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    tma_aligned_T = get_tma_aligned_size(num_tokens, 4) if quantize else num_tokens
    if quantize and tma_aligned_scales:
        assert chunks_per_head % 4 == 0
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_inner = packed_sf_k
    elif quantize:
        scale_inner = num_scale_blocks
    else:
        scale_inner = 0

    # Run kernel through a custom op so inductor sees an opaque boundary.
    # It's a pytorch bug, see https://github.com/vllm-project/vllm/issues/41106
    out_buf, scale_buf = torch.ops.vllm.fused_inv_rope_fp8_quant_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        quant_group_size,
        chunks_per_head,
        nope_dim,
        rope_dim // 2,
        tma_aligned_scales,
        fp8_max,
        tma_aligned_T,
        num_tokens,
        n_groups,
        d,
        scale_inner,
        quantize,
        permuted_output,
    )
    output = out_buf.transpose(0, 1)
    scales = scale_buf.transpose(0, 1) if quantize else scale_buf
    return output, scales


def _fused_inv_rope_fp8_quant_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    nope_dim: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
    quantize: bool,
    permuted_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    out_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn if quantize else o.dtype,
        device=o.device,
    )
    if quantize:
        scale_storage = torch.empty(
            n_groups * scale_inner * tma_aligned_T,
            dtype=scale_dtype,
            device=o.device,
        )
        scale_buf = scale_storage.as_strided(
            (n_groups, num_tokens, scale_inner),
            (scale_inner * tma_aligned_T, 1, tma_aligned_T),
        )
        scale_stride_group = scale_buf.stride(0)
        scale_stride_k = scale_buf.stride(2)
        scale_arg = scale_buf
        if permuted_output:
            # Byte-addressed stores into the packed int32 ue8m0 buffer.
            scale_arg = scale_storage.view(torch.uint8)
            scale_stride_group *= 4
            scale_stride_k *= 4
    else:
        scale_buf = torch.empty(0, dtype=scale_dtype, device=o.device)
        scale_arg = scale_buf
        scale_stride_group = 0
        scale_stride_k = 0
    grid = (tma_aligned_T, n_groups * heads_per_group)
    use_gdc = current_platform.is_arch_support_pdl()
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        out_buf,
        scale_arg,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_group=out_buf.stride(0),
        out_stride_token=out_buf.stride(1),
        scale_stride_group=scale_stride_group,
        scale_stride_k=scale_stride_k,
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        NOPE_DIM=nope_dim,
        HALF_ROPE=half_rope,
        QUANTIZE=quantize,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        PERMUTED_OUTPUT=permuted_output,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        num_stages=1,
        num_warps=1,
    )
    return out_buf, scale_buf


def _fused_inv_rope_fp8_quant_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    nope_dim: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
    quantize: bool,
    permuted_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    out_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn if quantize else o.dtype,
        device=o.device,
    )
    if not quantize:
        return out_buf, torch.empty(0, dtype=scale_dtype, device=o.device)
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    return out_buf, scale_buf


direct_register_custom_op(
    op_name="fused_inv_rope_fp8_quant_kernel",
    op_func=_fused_inv_rope_fp8_quant_kernel_impl,
    fake_impl=_fused_inv_rope_fp8_quant_kernel_fake,
)
