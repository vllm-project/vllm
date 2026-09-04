# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float8E4M3FN, Float32, Int64, Uint8, Uint16, Uint32

from vllm.cute_utils import _TORCH_TO_CUTE_DTYPE, cvt
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import compile_cutedsl
from vllm.platforms import current_platform


def _make_fake_tensor(dtype, shape, divisibility):
    stride = tuple(
        1 if i == len(shape) - 1 else cute.sym_int64(divisibility=divisibility)
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride,
        assumed_align=divisibility * dtype.width // 8,
    )


def is_fused_q_cutedsl_supported(
    q_pe: torch.Tensor,
    index_q: torch.Tensor | None,
    ql_nope: torch.Tensor,
    *,
    has_indexer: bool,
    quantize_mqa: bool,
) -> bool:
    if not (
        current_platform.has_device_capability(100)
        and quantize_mqa
        and q_pe.dtype == ql_nope.dtype == torch.bfloat16
        and q_pe.shape[-1] == 64
        and ql_nope.shape[-1] == 512
        # One warp per head group; a non-multiple would trip the kernel's own
        # assert instead of falling back to Triton.
        and q_pe.shape[1] % 4 == 0
    ):
        return False
    return not has_indexer or (
        index_q is not None
        and index_q.dtype == torch.bfloat16
        and index_q.shape[1] % 16 == 0
        and index_q.shape[-1] == 128
    )


def is_fused_q_cutedsl_geometry_supported(
    *,
    num_q_heads: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    index_n_head: int,
    index_head_dim: int,
    has_indexer: bool,
    quantize_mqa: bool,
    act_dtype: torch.dtype,
) -> bool:
    """Tensor-free twin of :func:`is_fused_q_cutedsl_supported`.

    Used at model-load warmup registration, where the runtime tensors do not
    exist yet, to decide whether the CuTeDSL path *will* be the runtime path for
    this layer. The runtime predicate's dtype checks (q_pe/ql_nope/index_q all
    bf16) are approximated by ``act_dtype``: q_pe/ql_nope are the model compute
    dtype and index_q is ``wq_b``'s bf16 output, both tracked by the attention
    layernorm weight dtype. When this returns False the ``fused_q`` wrapper falls
    back to the Triton owner, which is warmed separately.
    """
    if not (
        current_platform.has_device_capability(100)
        and quantize_mqa
        and act_dtype == torch.bfloat16
        and qk_rope_head_dim == 64
        and kv_lora_rank == 512
        # One warp per head group; mirrors q_pe.shape[1] % 4 == 0.
        and num_q_heads % 4 == 0
    ):
        return False
    return not has_indexer or (index_head_dim == 128 and index_n_head % 16 == 0)


class FusedQKernel:
    def __init__(
        self,
        rope_dim: int,
        nope_dim: int,
        num_heads: int,
        idx_dim: int,
        num_idx_heads: int,
        index_rope_interleave: bool,
    ) -> None:
        assert rope_dim == 64
        assert nope_dim == 512
        assert idx_dim in (128, 0)

        self.rope_dim = rope_dim
        self.nope_dim = nope_dim
        self.num_heads = num_heads
        self.idx_dim = idx_dim
        self.num_idx_heads = num_idx_heads
        self.index_rope_interleave = index_rope_interleave

        # mqa:     rope_dim=64, nope_dim=512, num_heads=64/TP
        # indexer: rope_dim=64, nope_dim=64,  num_heads=32

        self.num_warps = 4
        assert num_heads % self.num_warps == 0
        assert num_idx_heads % (4 * self.num_warps) == 0
        self.num_ctas_per_tok = num_heads // self.num_warps
        self.num_ctas_per_idx_tok = num_idx_heads // (4 * self.num_warps)

    @cute.jit
    def __call__(
        self,
        positions: cute.Tensor,
        q_pe: cute.Tensor,
        rope_cache: cute.Tensor,
        ql_nope: cute.Tensor,
        q_scale: cute.Tensor,
        mqa_output: cute.Tensor,
        idx_q: cute.Tensor,
        idx_rope_cache: cute.Tensor,
        idx_weights: cute.Tensor,
        idx_q_fp8: cute.Tensor,
        idx_weights_out: cute.Tensor,
        weight_scale: Float32,
        stream: CUstream,
    ):
        num_tokens = positions.shape[0]
        if cutlass.const_expr(self.idx_dim == 0):
            grid = (num_tokens, self.num_ctas_per_tok, 1)
        else:
            num_mqa_ctas = num_tokens * self.num_ctas_per_tok
            num_idx_ctas = num_tokens * self.num_ctas_per_idx_tok
            grid = (num_mqa_ctas + num_idx_ctas, 1, 1)

        self.kernel(
            positions,
            q_pe,
            rope_cache,
            ql_nope,
            q_scale,
            mqa_output,
            idx_q,
            idx_rope_cache,
            idx_weights,
            idx_q_fp8,
            idx_weights_out,
            weight_scale,
        ).launch(
            grid=grid,
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        positions: cute.Tensor,
        q_pe: cute.Tensor,
        q_pe_rope_cache: cute.Tensor,
        ql_nope: cute.Tensor,
        q_scale: cute.Tensor,
        mqa_output: cute.Tensor,
        idx_q: cute.Tensor,
        idx_q_rope_cache: cute.Tensor,
        idx_weights: cute.Tensor,
        idx_q_fp8: cute.Tensor,
        idx_weights_out: cute.Tensor,
        weight_scale: Float32,
    ):
        if cutlass.const_expr(self.idx_dim == 0):
            token_id, group_id, _ = cute.arch.block_idx()
            self.mqa(
                positions,
                q_pe,
                q_pe_rope_cache,
                ql_nope,
                q_scale,
                mqa_output,
                token_id,
                group_id,
            )
        else:
            # CTA-specialization
            bid, _, _ = cute.arch.block_idx()
            num_mqa_ctas = positions.shape[0] * self.num_ctas_per_tok
            if bid < num_mqa_ctas:
                self.mqa(
                    positions,
                    q_pe,
                    q_pe_rope_cache,
                    ql_nope,
                    q_scale,
                    mqa_output,
                    bid // self.num_ctas_per_tok,
                    bid % self.num_ctas_per_tok,
                )
            else:
                bid -= num_mqa_ctas
                self.indexer(
                    positions,
                    idx_q,
                    idx_q_rope_cache,
                    idx_weights,
                    idx_q_fp8,
                    idx_weights_out,
                    weight_scale,
                    bid // self.num_ctas_per_idx_tok,
                    bid % self.num_ctas_per_idx_tok,
                )

    @cute.jit
    def mqa(
        self,
        positions: cute.Tensor,
        q_pe: cute.Tensor,
        q_pe_rope_cache: cute.Tensor,
        ql_nope: cute.Tensor,
        q_scale: cute.Tensor,
        mqa_output: cute.Tensor,
        token_id,
        group_id,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        head_id = group_id * self.num_warps + warp_id

        cute.arch.griddepcontrol_wait()

        pos = positions[token_id]
        inv_scale = 1.0 / q_scale[0]

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_32B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=256)
        cp_16B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)
        cp_4B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=32)
        cp_2B = cute.make_copy_atom(cp_op, Uint8, num_bits_per_copy=16)

        ##### issue all loads asap #####
        rQ_nope_bf16 = cute.make_rmem_tensor(16, BFloat16)
        rQ_rope_bf16 = cute.make_rmem_tensor(2, BFloat16)

        src_ql_nope = cute.local_tile(
            ql_nope[token_id, head_id, None], (16,), (lane_id,)
        )
        src_q_rope = cute.local_tile(q_pe[token_id, head_id, None], (2,), (lane_id,))
        cute.copy(cp_32B, src_ql_nope, rQ_nope_bf16)
        cute.copy(cp_4B, src_q_rope, rQ_rope_bf16)

        rCos_raw = q_pe_rope_cache[pos, 0 + lane_id]
        rSin_raw = q_pe_rope_cache[pos, 32 + lane_id]

        ##### process NoPE #####
        rQ_nope_f32 = cvt.bf16x2_to_fp32x2(rQ_nope_bf16).load() * inv_scale
        rQ_nope_f8 = cute.make_rmem_tensor(16, Float8E4M3FN)
        rQ_nope_f8.store(rQ_nope_f32.to(Float8E4M3FN))
        dst_Q_nope = cute.local_tile(
            mqa_output[token_id, head_id, None], (16,), (lane_id,)
        )
        cute.copy(cp_16B, rQ_nope_f8, dst_Q_nope)

        ##### process RoPE ######
        rQ_rope_f32 = cvt.bf16x2_to_fp32x2(rQ_rope_bf16)
        rCos = rCos_raw.to(Float32)
        rSin = rSin_raw.to(Float32)
        r0 = (rQ_rope_f32[0] * rCos - rQ_rope_f32[1] * rSin) * inv_scale
        r1 = (rQ_rope_f32[1] * rCos + rQ_rope_f32[0] * rSin) * inv_scale

        cute.arch.griddepcontrol_launch_dependents()

        # TensorSSA fp32->fp8 cvt has a bug. rely on direct PTX
        rQ_rope_f8 = cute.make_rmem_tensor(2, Float8E4M3FN)
        cute.recast_tensor(rQ_rope_f8, Uint16)[0] = cvt.fp32x2_to_fp8x2(r0, r1)
        dst_Q_rope = cute.local_tile(
            mqa_output[token_id, head_id, None], (2,), (256 + lane_id,)
        )
        cute.copy(cp_2B, rQ_rope_f8, dst_Q_rope)

    @cute.jit
    def indexer(
        self,
        positions: cute.Tensor,
        idx_q: cute.Tensor,
        idx_q_rope_cache: cute.Tensor,
        idx_weights: cute.Tensor,
        idx_q_fp8: cute.Tensor,
        idx_weights_out: cute.Tensor,
        weight_scale: Float32,
        token_id,
        group_id,
    ):
        tid, _, _ = cute.arch.thread_idx()
        subwarp_id = tid // 8
        sublane_id = tid % 8
        head_id = group_id * (4 * self.num_warps) + subwarp_id
        cute.arch.griddepcontrol_wait()

        pos = positions[token_id]

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_16B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)
        cp_8B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=64)
        cp_4B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=32)

        ##### issue all loads first #####
        rQ_rope_bf16 = cute.make_rmem_tensor(8, BFloat16)
        if cutlass.const_expr(self.index_rope_interleave):
            src_idx_q_rope = cute.local_tile(
                idx_q[token_id, head_id, None], (8,), (sublane_id,)
            )
            cute.copy(cp_16B, src_idx_q_rope, rQ_rope_bf16)
        else:
            src_idx_q_rope = cute.zipped_divide(
                idx_q[token_id, head_id, None], (4,)
            )  # (4,32)
            cute.copy(
                cp_8B,
                src_idx_q_rope[None, 0 + sublane_id],
                cute.local_tile(rQ_rope_bf16, (4,), (0,)),
            )
            cute.copy(
                cp_8B,
                src_idx_q_rope[None, 8 + sublane_id],
                cute.local_tile(rQ_rope_bf16, (4,), (1,)),
            )

        rQ_nope_bf16 = cute.make_rmem_tensor(8, BFloat16)
        src_idx_q_nope = cute.local_tile(
            idx_q[token_id, head_id, None], (8,), (8 + sublane_id,)
        )
        cute.copy(cp_16B, src_idx_q_nope, rQ_nope_bf16)

        rCos_raw = cute.make_rmem_tensor(4, idx_q_rope_cache.element_type)
        rSin_raw = cute.make_rmem_tensor(4, idx_q_rope_cache.element_type)
        rope_cache_view = cute.zipped_divide(idx_q_rope_cache[pos, None], (4,))

        if cutlass.const_expr(idx_q_rope_cache.element_type == Float32):
            cute.copy(cp_16B, rope_cache_view[None, sublane_id], rCos_raw)
            cute.copy(cp_16B, rope_cache_view[None, 8 + sublane_id], rSin_raw)
        elif cutlass.const_expr(idx_q_rope_cache.element_type == BFloat16):
            cute.copy(cp_8B, rope_cache_view[None, sublane_id], rCos_raw)
            cute.copy(cp_8B, rope_cache_view[None, 8 + sublane_id], rSin_raw)

        # unpack to FP32
        rQ_rope_f32 = cvt.bf16x2_to_fp32x2(rQ_rope_bf16)
        rQ_nope_f32 = cvt.bf16x2_to_fp32x2(rQ_nope_bf16)
        if cutlass.const_expr(idx_q_rope_cache.element_type == Float32):
            rCos = rCos_raw
            rSin = rSin_raw
        elif cutlass.const_expr(idx_q_rope_cache.element_type == BFloat16):
            rCos = cvt.bf16x2_to_fp32x2(rCos_raw)
            rSin = cvt.bf16x2_to_fp32x2(rSin_raw)

        # apply rope
        for i in cutlass.range_constexpr(4):
            if cutlass.const_expr(self.index_rope_interleave):
                r0 = rQ_rope_f32[i * 2 + 0] * rCos[i] - rQ_rope_f32[i * 2 + 1] * rSin[i]
                r1 = rQ_rope_f32[i * 2 + 1] * rCos[i] + rQ_rope_f32[i * 2 + 0] * rSin[i]
                rQ_rope_f32[i * 2 + 0] = r0
                rQ_rope_f32[i * 2 + 1] = r1
            else:
                r0 = rQ_rope_f32[0 + i] * rCos[i] - rQ_rope_f32[4 + i] * rSin[i]
                r1 = rQ_rope_f32[4 + i] * rCos[i] + rQ_rope_f32[0 + i] * rSin[i]
                rQ_rope_f32[0 + i] = r0
                rQ_rope_f32[4 + i] = r1

        # amax
        amax = Float32(1e-4)
        for i in cutlass.range_constexpr(8):
            amax = cute.arch.fmax(amax, cute.math.absf(rQ_rope_f32[i]))
            amax = cute.arch.fmax(amax, cute.math.absf(rQ_nope_f32[i]))

        # warp reduction among 8 lanes
        for i in cutlass.range_constexpr(3):
            other = cute.arch.shuffle_sync_bfly(amax, 1 << i)
            amax = cute.arch.fmax(amax, other)

        # compute scale from amax
        # exp2(ceil(log2(scale))) via bit manipulation
        scale = amax * (1.0 / 448.0)
        bits = scale.bitcast(Uint32)
        exp_bits = (bits + Uint32(0x007FFFFF)) & Uint32(0x7F800000)
        scale = exp_bits.bitcast(Float32)
        inv_scale = (Uint32(0x7F000000) - exp_bits).bitcast(Float32)

        for i in cutlass.range_constexpr(8):
            rQ_nope_f32[i] *= inv_scale
            rQ_rope_f32[i] *= inv_scale

        cute.arch.griddepcontrol_launch_dependents()

        # quantize and store
        rQ_nope_f8 = cute.make_rmem_tensor(8, Float8E4M3FN)
        rQ_nope_f8.store(rQ_nope_f32.load().to(Float8E4M3FN))
        dst_idx_q_nope = cute.local_tile(
            idx_q_fp8[token_id, head_id, None], (8,), (8 + sublane_id,)
        )
        cute.copy(cp_8B, rQ_nope_f8, dst_idx_q_nope)

        rQ_rope_f8 = cute.make_rmem_tensor(8, Float8E4M3FN)
        rQ_rope_f8.store(rQ_rope_f32.load().to(Float8E4M3FN))
        if cutlass.const_expr(self.index_rope_interleave):
            dst_idx_q_rope = cute.local_tile(
                idx_q_fp8[token_id, head_id, None], (8,), (sublane_id,)
            )
            cute.copy(cp_8B, rQ_rope_f8, dst_idx_q_rope)
        else:
            dst_idx_q_rope = cute.zipped_divide(
                idx_q_fp8[token_id, head_id, None], (4,)
            )  # (4,32)
            cute.copy(
                cp_4B,
                cute.local_tile(rQ_rope_f8, (4,), (0,)),
                dst_idx_q_rope[None, 0 + sublane_id],
            )
            cute.copy(
                cp_4B,
                cute.local_tile(rQ_rope_f8, (4,), (1,)),
                dst_idx_q_rope[None, 8 + sublane_id],
            )

        # scale indexer weights
        if sublane_id == 0:
            w = idx_weights[token_id, head_id].to(Float32)
            idx_weights_out[token_id, head_id] = w * scale * weight_scale


class FusedQCuteDSLKernel(VllmJitKernel["FusedQCuteDSLKernel.CompileKey"]):
    """JIT-warmup owner for the fused MQA-query + indexer-query CuTeDSL kernel.

    Extends :class:`VllmJitKernel` directly (like the DSv4 CuTeDSL owners
    ``IndexerQMxFp4Kernel`` / ``DequantGatherKCacheKernel``) rather than the
    ``VllmCuTeDSLJitKernel`` helper base: the compile-key is built from a small,
    fixed set of runtime axes so ``dispatch``/``get_warmup_keys`` stay AST
    traceable. The compiled executor is cached in ``_compiled_cache`` and shared
    by warmup and runtime, so a first request never triggers a JIT compile.

    The device kernel itself is the unchanged :class:`FusedQKernel`; this owner
    only wraps compilation and dispatch, mirroring the Triton
    ``FusedQTritonKernel`` in ``common/kernels.py``.
    """

    @dataclass(frozen=True)
    class CompileKey:
        # Mirrors the arguments of the former ``FusedQKernel.compile``: the axes
        # a compiled executor specializes on.
        rope_dim: int
        nope_dim: int
        num_heads: int
        rope_type: type[cutlass.Numeric]
        idx_dim: int
        num_idx_heads: int
        idx_rope_type: type[cutlass.Numeric] | None
        idx_weights_type: type[cutlass.Numeric] | None
        index_rope_interleave: bool

    @staticmethod
    def kernel(compile_key: "FusedQCuteDSLKernel.CompileKey") -> Any:
        return FusedQKernel(
            compile_key.rope_dim,
            compile_key.nope_dim,
            compile_key.num_heads,
            compile_key.idx_dim,
            compile_key.num_idx_heads,
            compile_key.index_rope_interleave,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        rope_dim: int,
        nope_dim: int,
        num_heads: int,
        rope_type: type[cutlass.Numeric],
        idx_dim: int,
        num_idx_heads: int,
        idx_rope_type: type[cutlass.Numeric] | None,
        idx_weights_type: type[cutlass.Numeric] | None,
        index_rope_interleave: bool,
    ) -> "FusedQCuteDSLKernel.CompileKey":
        # Pure forwarding: the torch->cute dtype conversion and the no-indexer
        # collapse happen in ``_key_args`` (real Python), so this stays trivially
        # AST-traceable for get_warmup_keys -- mirrors the DSv4 CuTeDSL owners,
        # which also pass already-cute dtypes into dispatch.
        return self.CompileKey(
            rope_dim=rope_dim,
            nope_dim=nope_dim,
            num_heads=num_heads,
            rope_type=rope_type,
            idx_dim=idx_dim,
            num_idx_heads=num_idx_heads,
            idx_rope_type=idx_rope_type,
            idx_weights_type=idx_weights_type,
            index_rope_interleave=index_rope_interleave,
        )

    def _key_args(
        self,
        *,
        num_q_heads: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        index_n_head: int,
        index_head_dim: int,
        has_indexer: bool,
        index_rope_interleave: bool,
        rope_cache_dtype: torch.dtype,
        idx_rope_cache_dtype: torch.dtype | None,
        idx_weights_dtype: torch.dtype | None,
    ) -> dict[str, Any]:
        """Runtime-values -> dispatch kwargs (dtype conversion + indexer collapse).

        Shared verbatim by ``__call__`` (runtime) and ``get_warmup_keys``
        (registration) so the warmup key can never drift from the runtime key.
        """
        if has_indexer:
            idx_dim = index_head_dim
            num_idx_heads = index_n_head
            idx_rope_type = _TORCH_TO_CUTE_DTYPE[idx_rope_cache_dtype]
            idx_weights_type = _TORCH_TO_CUTE_DTYPE[idx_weights_dtype]
        else:
            idx_dim = num_idx_heads = 0
            idx_rope_type = idx_weights_type = None
        return dict(
            rope_dim=qk_rope_head_dim,
            nope_dim=kv_lora_rank,
            num_heads=num_q_heads,
            rope_type=_TORCH_TO_CUTE_DTYPE[rope_cache_dtype],
            idx_dim=idx_dim,
            num_idx_heads=num_idx_heads,
            idx_rope_type=idx_rope_type,
            idx_weights_type=idx_weights_type,
            index_rope_interleave=index_rope_interleave,
        )

    def get_warmup_keys(self, **kwargs: Any) -> list["FusedQCuteDSLKernel.CompileKey"]:
        return self._trace_dispatch(self.dispatch)(**self._key_args(**kwargs))

    def compile(self, compile_key: "FusedQCuteDSLKernel.CompileKey") -> None:
        if compile_key in self._compiled_cache:
            return

        num_tokens = cute.sym_int()
        max_pos = cute.sym_int()
        rope_dim = compile_key.rope_dim
        nope_dim = compile_key.nope_dim
        num_heads = compile_key.num_heads
        idx_dim = compile_key.idx_dim
        num_idx_heads = compile_key.num_idx_heads

        positions = _make_fake_tensor(Int64, (num_tokens,), divisibility=1)
        q_pe = _make_fake_tensor(
            BFloat16, (num_tokens, num_heads, rope_dim), divisibility=16
        )
        rope_cache = _make_fake_tensor(
            compile_key.rope_type, (max_pos, rope_dim), divisibility=8
        )
        ql_nope = _make_fake_tensor(
            BFloat16, (num_tokens, num_heads, nope_dim), divisibility=16
        )
        q_scale = _make_fake_tensor(Float32, (1,), divisibility=4)
        mqa_output = _make_fake_tensor(
            Float8E4M3FN,
            (num_tokens, num_heads, nope_dim + rope_dim),
            divisibility=16,
        )

        if compile_key.idx_rope_type is not None:
            index_q = _make_fake_tensor(
                BFloat16, (num_tokens, num_idx_heads, idx_dim), divisibility=16
            )
            index_rope_cache = _make_fake_tensor(
                compile_key.idx_rope_type, (max_pos, rope_dim), divisibility=8
            )
            index_weights = _make_fake_tensor(
                compile_key.idx_weights_type,
                (num_tokens, num_idx_heads),
                divisibility=8,
            )
            index_q_fp8 = _make_fake_tensor(
                Float8E4M3FN, (num_tokens, num_idx_heads, idx_dim), divisibility=16
            )
            index_weights_out = _make_fake_tensor(
                Float32, (num_tokens, num_idx_heads), divisibility=4
            )
        else:
            index_q = index_rope_cache = index_q_fp8 = None
            index_weights = index_weights_out = None

        self._compiled_cache[compile_key] = compile_cutedsl(
            self.kernel(compile_key),
            positions,
            q_pe,
            rope_cache,
            ql_nope,
            q_scale,
            mqa_output,
            index_q,
            index_rope_cache,
            index_weights,
            index_q_fp8,
            index_weights_out,
            Float32(0.0),
        )

    def __call__(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        rope_cache: torch.Tensor,
        ql_nope: torch.Tensor,
        q_scale: torch.Tensor,
        mqa_output: torch.Tensor,
        idx_q: torch.Tensor,
        idx_rope_cache: torch.Tensor,
        idx_weights: torch.Tensor,
        idx_weights_softmax_scale: float,
        idx_weights_head_scale: float,
        idx_q_fp8: torch.Tensor,
        idx_weights_out: torch.Tensor,
        *,
        has_indexer: bool = True,
        index_rope_interleave: bool = True,
    ) -> None:
        _, num_heads, rope_dim = q_pe.shape
        _, _, nope_dim = ql_nope.shape
        _, num_idx_heads, idx_dim = idx_q.shape

        key_args = self._key_args(
            num_q_heads=num_heads,
            qk_rope_head_dim=rope_dim,
            kv_lora_rank=nope_dim,
            index_n_head=num_idx_heads,
            index_head_dim=idx_dim,
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
            rope_cache_dtype=rope_cache.dtype,
            idx_rope_cache_dtype=idx_rope_cache.dtype if has_indexer else None,
            idx_weights_dtype=idx_weights.dtype if has_indexer else None,
        )
        if not has_indexer:
            # Shared layer: the kernel skips the indexer CTAs, so the dummy
            # indexer tensors are never dereferenced -- pass None through.
            idx_q = idx_rope_cache = idx_q_fp8 = None
            idx_weights = idx_weights_out = None

        compile_key = self.dispatch(**key_args)
        executor = self._get_or_compile(
            compile_key,
            runtime_context={**key_args, "has_indexer": has_indexer},
        )
        executor(
            positions,
            q_pe,
            rope_cache,
            ql_nope,
            q_scale.view(1),
            mqa_output,
            idx_q,
            idx_rope_cache,
            idx_weights,
            idx_q_fp8,
            idx_weights_out,
            float(idx_weights_softmax_scale * idx_weights_head_scale),
        )


_FUSED_Q_CUTEDSL_KERNEL = FusedQCuteDSLKernel()
