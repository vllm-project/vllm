# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused inverse RoPE + block-scaled FP8 quantization kernel for DeepseekV4 attention.

Output scale format is pre-transformed (MN-major TMA-aligned; FP32 on SM90,
INT32-packed UE8M0 on SM100) so fp8_einsum skips transform_sf_into_required_layout.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


class FusedInvRopeFP8QuantKernel(
    VllmJitKernel["FusedInvRopeFP8QuantKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        heads_per_group: int
        fp8_max: float
        quant_group_size: int
        chunks_per_head: int
        rope_start: int
        half_rope: int
        tma_aligned_scales: bool
        use_gdc: bool

    @staticmethod
    @triton.jit(do_not_specialize=["num_tokens"])
    def kernel(
        o_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        fp8_ptr,
        scale_ptr,
        num_tokens,
        heads_per_group: tl.constexpr,
        o_stride_token,
        o_stride_head,
        cache_stride_pos,
        fp8_stride_group,
        fp8_stride_token,
        scale_stride_group,
        scale_stride_k,
        fp8_max: tl.constexpr,
        eps: tl.constexpr,
        QUANT_GROUP_SIZE: tl.constexpr,
        CHUNKS_PER_HEAD: tl.constexpr,
        ROPE_START: tl.constexpr,
        HALF_ROPE: tl.constexpr,
        TMA_ALIGNED_SCALES: tl.constexpr,
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
        fp8_stride_group = fp8_stride_group.to(tl.int64)
        fp8_stride_token = fp8_stride_token.to(tl.int64)
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
        if pid_token >= num_tokens:
            if TMA_ALIGNED_SCALES:
                scale_addr = (
                    scale_ptr
                    + g * scale_stride_group
                    + pid_token
                    + head_in_group * scale_stride_k
                )
                tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
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

        rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
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

        fp8_base = (
            fp8_ptr
            + g * fp8_stride_group
            + pid_token * fp8_stride_token
            + qb_start * QUANT_GROUP_SIZE
        )
        tl.store(fp8_base + offsets, x_quant)

        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        if TMA_ALIGNED_SCALES:
            scale_bits = scales.to(tl.int32, bitcast=True)
            ue8m0_bytes = (scale_bits >> 23) & 0xFF
            packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, packed_val)
        else:
            scale_addrs = (
                scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, scales)

    def dispatch(  # type: ignore[override]
        self,
        *,
        heads_per_group: int,
        head_dim: int,
        nope_dim: int,
        rope_dim: int,
        quant_group_size: int,
        tma_aligned_scales: bool,
        use_gdc: bool,
    ) -> CompileKey:
        return self.CompileKey(
            heads_per_group=heads_per_group,
            fp8_max=torch.finfo(torch.float8_e4m3fn).max,
            quant_group_size=quant_group_size,
            chunks_per_head=head_dim // quant_group_size,
            rope_start=nope_dim % quant_group_size,
            half_rope=rope_dim // 2,
            tma_aligned_scales=tma_aligned_scales,
            use_gdc=use_gdc,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return []

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        num_heads = int(getattr(hf_config, "num_attention_heads", 0) or 0)
        num_groups = int(getattr(hf_config, "o_groups", 0) or 0)
        if num_heads <= 0 or num_groups <= 0:
            return []

        local_heads = num_heads // tp_size
        local_groups = num_groups // tp_size
        if local_groups <= 0:
            return []

        capability = current_platform.get_device_capability()
        if capability is None:
            return []

        head_dim = int(getattr(hf_config, "head_dim", 512) or 512)
        rope_dim = int(getattr(hf_config, "qk_rope_head_dim", 64) or 64)
        return self._trace_dispatch(self.dispatch)(
            heads_per_group=local_heads // local_groups,
            head_dim=head_dim,
            nope_dim=head_dim - rope_dim,
            rope_dim=rope_dim,
            quant_group_size=128,
            tma_aligned_scales=capability.major >= 10,
            use_gdc=current_platform.is_arch_support_pdl(),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        head_dim = compile_key.chunks_per_head * compile_key.quant_group_size
        fp8_dim = compile_key.heads_per_group * head_dim
        scale_dtype = torch.int32 if compile_key.tma_aligned_scales else torch.float32
        warmup(
            TritonWarmupTensor(torch.bfloat16),
            TritonWarmupTensor(torch.int64),
            TritonWarmupTensor(torch.float32),
            TritonWarmupTensor(torch.float8_e4m3fn),
            TritonWarmupTensor(scale_dtype),
            1,  # do not specialize num_tokens
            heads_per_group=compile_key.heads_per_group,
            o_stride_token=head_dim * compile_key.heads_per_group,
            o_stride_head=head_dim,
            cache_stride_pos=compile_key.half_rope * 2,
            fp8_stride_group=fp8_dim,
            fp8_stride_token=fp8_dim,
            scale_stride_group=1,
            scale_stride_k=1,
            fp8_max=compile_key.fp8_max,
            eps=1e-10,
            QUANT_GROUP_SIZE=compile_key.quant_group_size,
            CHUNKS_PER_HEAD=compile_key.chunks_per_head,
            ROPE_START=compile_key.rope_start,
            HALF_ROPE=compile_key.half_rope,
            TMA_ALIGNED_SCALES=compile_key.tma_aligned_scales,
            USE_GDC=compile_key.use_gdc,
            launch_pdl=compile_key.use_gdc,
            grid=(1, compile_key.heads_per_group),
            num_stages=1,
            num_warps=1,
        )

    def __call__(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        fp8_buf: torch.Tensor,
        scale_buf: torch.Tensor,
        num_tokens: int,
        *,
        heads_per_group: int,
        quant_group_size: int,
        chunks_per_head: int,
        rope_start: int,
        half_rope: int,
        tma_aligned_scales: bool,
        fp8_max: float,
        use_gdc: bool,
        grid: tuple[int, int],
    ) -> None:
        self.kernel[grid](
            o,
            positions,
            cos_sin_cache,
            fp8_buf,
            scale_buf,
            num_tokens,
            heads_per_group=heads_per_group,
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            fp8_stride_group=fp8_buf.stride(0),
            fp8_stride_token=fp8_buf.stride(1),
            scale_stride_group=scale_buf.stride(0),
            scale_stride_k=scale_buf.stride(2),
            fp8_max=fp8_max,
            eps=1e-10,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=rope_start,
            HALF_ROPE=half_rope,
            TMA_ALIGNED_SCALES=tma_aligned_scales,
            USE_GDC=use_gdc,
            launch_pdl=use_gdc,
            num_stages=1,
            num_warps=1,
        )


_FUSED_INV_ROPE_FP8_QUANT_KERNEL = FusedInvRopeFP8QuantKernel()

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

    Returns:
        o_fp8: [T, G, D] float8_e4m3fn, strides (D, T*D, 1).
        o_scale: Pre-transformed scale tensor for fp8_einsum.
    """
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
    if tma_aligned_scales:
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_inner = packed_sf_k
    else:
        scale_inner = num_scale_blocks

    # Run kernel through a custom op so inductor sees an opaque boundary.
    # It's a pytorch bug, see https://github.com/vllm-project/vllm/issues/41106
    fp8_buf, scale_buf = torch.ops.vllm.fused_inv_rope_fp8_quant_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        quant_group_size,
        chunks_per_head,
        nope_dim % quant_group_size,
        rope_dim // 2,
        tma_aligned_scales,
        fp8_max,
        tma_aligned_T,
        num_tokens,
        n_groups,
        d,
        scale_inner,
    )
    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)


def _fused_inv_rope_fp8_quant_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    rope_start: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    grid = (tma_aligned_T, n_groups * heads_per_group)
    use_gdc = current_platform.is_arch_support_pdl()
    _FUSED_INV_ROPE_FP8_QUANT_KERNEL(
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        quant_group_size=quant_group_size,
        chunks_per_head=chunks_per_head,
        rope_start=rope_start,
        half_rope=half_rope,
        tma_aligned_scales=tma_aligned_scales,
        fp8_max=fp8_max,
        use_gdc=use_gdc,
        grid=grid,
    )
    return fp8_buf, scale_buf


def _fused_inv_rope_fp8_quant_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    rope_start: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    return fp8_buf, scale_buf


direct_register_custom_op(
    op_name="fused_inv_rope_fp8_quant_kernel",
    op_func=_fused_inv_rope_fp8_quant_kernel_impl,
    fake_impl=_fused_inv_rope_fp8_quant_kernel_fake,
)
