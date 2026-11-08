# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTeDSL fused MoE finalize with MuseSpark1x's post-expert RMSNorm."""

import operator
from typing import Any

import torch

from vllm.platforms import current_platform

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64
    from cutlass.cute.runtime import make_ptr

    HAS_CUTEDSL = True
except ImportError:  # pragma: no cover - exercised only without nvidia-cutlass-dsl
    HAS_CUTEDSL = False


_TORCH_TO_CUTLASS_DTYPE = (
    {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if HAS_CUTEDSL
    else {}
)

COPY_BITS = 128


if HAS_CUTEDSL:

    class _FinalizeConfig:
        """Launch shape and thread-value layout: one token per CTA."""

        def __init__(self, dtype: Any, hidden_dim: int, threads_per_row=None):
            self.dtype = dtype
            self.N = hidden_dim
            self.vec_size = COPY_BITS // dtype.width
            self.threads_per_row = (
                threads_per_row
                if threads_per_row is not None
                else self._threads_per_row(hidden_dim, self.vec_size)
            )
            self.num_threads = self.threads_per_row
            self.rows_per_block = 1
            self.warps_per_row = max(self.threads_per_row // 32, 1)

        @staticmethod
        def _threads_per_row(N: int, vec_size: int) -> int:
            if N <= 64:
                tpr = 8
            elif N <= 128:
                tpr = 16
            elif N <= 2048:
                tpr = 32
            elif N <= 2560:
                tpr = 64
            elif N <= 16384:
                tpr = 128
            else:
                tpr = 256
            # The tile has to divide the row exactly: there is no tail path.
            while tpr > 1 and N % (tpr * vec_size) != 0:
                tpr //= 2
            return tpr

        def supported(self) -> bool:
            return self.N % (self.threads_per_row * self.vec_size) == 0

        def tv_layout(self):
            vec_size = self.vec_size
            tpr = self.threads_per_row
            num_blocks_n = cute.ceil_div(cute.ceil_div(self.N, vec_size), tpr)
            cols_per_block = self.num_threads // tpr
            tiler_mn = (cols_per_block, vec_size * num_blocks_n * tpr)
            tv_layout = cute.make_layout(
                ((tpr, cols_per_block), (vec_size, num_blocks_n)),
                stride=(
                    (vec_size * cols_per_block, 1),
                    (cols_per_block, cols_per_block * vec_size * tpr),
                ),
            )
            return tiler_mn, tv_layout

        def smem_size_in_bytes(self) -> int:
            return self.rows_per_block * self.warps_per_row * 4

    class MoEFinalizeRMSNormKernel:
        """Unpermute + weight-first RMSNorm + fp32 weighted accumulate, per token."""

        def __init__(
            self,
            dtype: Any,
            hidden_dim: int,
            topk: int,
            hidden_dim_padded: int,
            *,
            weight_before_norm: bool = True,
            column_major_idx: bool = False,
            expert_filter: bool = False,
            threads_per_row=None,
            enable_pdl: bool = False,
        ):
            self.cfg = _FinalizeConfig(dtype, hidden_dim, threads_per_row)
            self.topk = topk
            self.hidden_dim_padded = hidden_dim_padded
            self.weight_before_norm = weight_before_norm
            self.column_major_idx = column_major_idx
            self.expert_filter = expert_filter
            self.enable_pdl = enable_pdl

        @cute.jit
        def __call__(
            self,
            gemm2_ptr: cute.Pointer,
            expert_weights_ptr: cute.Pointer,
            idx_map_ptr: cute.Pointer,
            topk_ids_ptr: cute.Pointer,
            rms_weight_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            num_tokens: Int32,
            num_rows: Int32,
            start_expert: Int32,
            num_local_experts: Int32,
            eps: Float32,
            gain_center: Float32,
            stream: cuda.CUstream,
        ):
            cfg = self.cfg
            hidden_dim = cfg.N
            topk = self.topk

            mGemm2 = cute.make_tensor(
                gemm2_ptr,
                cute.make_layout(
                    (num_tokens * topk, hidden_dim),
                    stride=(self.hidden_dim_padded, 1),
                ),
            )
            mOut = cute.make_tensor(
                out_ptr,
                cute.make_layout((num_tokens, hidden_dim), stride=(hidden_dim, 1)),
            )
            mW = cute.make_tensor(
                rms_weight_ptr,
                cute.make_layout((1, hidden_dim), stride=(hidden_dim, 1)),
            )
            flat = cute.make_layout((num_tokens * topk,), stride=(1,))
            mExpW = cute.make_tensor(expert_weights_ptr, flat)
            mIdx = cute.make_tensor(idx_map_ptr, flat)
            mTopkIds = cute.make_tensor(topk_ids_ptr, flat)

            self.kernel(
                mGemm2,
                mOut,
                mW,
                mExpW,
                mIdx,
                mTopkIds,
                num_tokens,
                num_rows,
                start_expert,
                num_local_experts,
                eps,
                gain_center,
            ).launch(
                grid=[num_tokens, 1, 1],
                block=[cfg.num_threads, 1, 1],
                smem=cfg.smem_size_in_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mGemm2: cute.Tensor,
            mOut: cute.Tensor,
            mW: cute.Tensor,
            mExpW: cute.Tensor,
            mIdx: cute.Tensor,
            mTopkIds: cute.Tensor,
            num_tokens: Int32,
            num_rows: Int32,
            start_expert: Int32,
            num_local_experts: Int32,
            eps: Float32,
            gain_center: Float32,
        ):
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()
            cfg = self.cfg
            topk = self.topk
            hidden_dim = cfg.N
            threads_per_row = cfg.threads_per_row
            warps_per_row = cfg.warps_per_row

            tidx = cute.arch.thread_idx()[0]
            token_idx = cute.arch.block_idx()[0]

            tiler_mn, tv_layout = cfg.tv_layout()

            smem = utils.SmemAllocator()
            reduction_buffer = smem.allocate_tensor(
                Float32, cute.make_layout((1, warps_per_row)), byte_alignment=4
            )

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mGemm2.element_type,
                num_bits_per_copy=COPY_BITS,
            )
            thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(
                tidx
            )

            tXgW = thr_copy.partition_S(cute.local_tile(mW, tiler_mn, (0, 0)))
            w_frag = cute.make_fragment_like(tXgW)
            cute.copy(copy_atom, tXgW, w_frag)
            w_f32 = w_frag.load().to(cute.Float32) + gain_center

            # Hoisted out of the lane-0 branch: warp_idx() is a full-warp
            # shuffle and deadlocks if only one lane arrives.
            lane_idx = cute.arch.lane_idx()
            warp_col = cute.arch.warp_idx() % warps_per_row

            tXgO = thr_copy.partition_D(cute.local_tile(mOut, tiler_mn, (token_idx, 0)))
            input_frag = cute.make_fragment_like(tXgW)
            # fp32 accumulator: rounding each top-k partial would cost accuracy.
            acc_frag = cute.make_fragment_like(tXgW, Float32)
            acc_frag.fill(0.0)

            # Non-local slots still run the body: skipping it would put the
            # cross-warp barriers in a data-dependent branch and hang. Their row
            # is zeroed on load and discarded after the final barrier.
            for k in range(topk):
                expanded_idx = token_idx * topk + k
                if cutlass.const_expr(self.column_major_idx):
                    permuted_idx = Int32(mIdx[token_idx + Int32(k) * num_tokens])
                else:
                    permuted_idx = Int32(mIdx[expanded_idx])

                is_local = permuted_idx >= 0 and permuted_idx < num_rows
                if cutlass.const_expr(self.expert_filter):
                    expert_id = Int32(mTopkIds[expanded_idx])
                    is_local = (
                        is_local
                        and expert_id >= start_expert
                        and expert_id < start_expert + num_local_experts
                    )

                row = Int32(0)
                weight = Float32(0.0)
                if is_local:
                    row = permuted_idx
                    weight = Float32(mExpW[expanded_idx])

                gInput = cute.local_tile(mGemm2, tiler_mn, (Int64(row), 0))
                # Zero, don't read row 0: the workspace may be unwritten, and
                # NaN * 0 spreads through the cross-rank reduce. The predicate
                # is block-uniform, so every thread still reaches the barriers.
                if is_local:
                    cute.copy(copy_atom, thr_copy.partition_S(gInput), input_frag)
                else:
                    input_frag.fill(0.0)

                scaled = input_frag.load().to(cute.Float32)
                if cutlass.const_expr(self.weight_before_norm):
                    scaled = scaled * w_f32
                sum_sq_local = (scaled * scaled).reduce(
                    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                )
                warp_sum = cute.arch.warp_reduction(
                    sum_sq_local,
                    operator.add,
                    threads_in_group=min(threads_per_row, 32),
                )

                if cutlass.const_expr(warps_per_row > 1):
                    # The previous slot's readers are done with the buffer.
                    cute.arch.sync_threads()
                    if lane_idx == 0:
                        reduction_buffer[0, warp_col] = warp_sum
                    cute.arch.sync_threads()
                    sum_sq = Float32(0.0)
                    for w in cutlass.range_constexpr(warps_per_row):
                        sum_sq = sum_sq + reduction_buffer[0, w]
                else:
                    sum_sq = warp_sum

                inv_rms = cute.math.rsqrt(sum_sq / hidden_dim + eps, fastmath=True)
                normalized = scaled * inv_rms
                if cutlass.const_expr(not self.weight_before_norm):
                    normalized = normalized * w_f32
                # NaN * 0 is NaN, so a zero weight alone cannot mask a slot.
                if is_local:
                    acc_frag.store(acc_frag.load() + normalized * weight)

            out_frag = cute.make_fragment_like(tXgO)
            out_frag.store(acc_frag.load().to(out_frag.element_type))
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    mOut.element_type,
                    num_bits_per_copy=COPY_BITS,
                ),
                out_frag,
                tXgO,
            )

    _compile_cache: dict = {}

    def _compiled_kernel(
        dtype: Any,
        hidden_dim: int,
        topk: int,
        hidden_dim_padded: int,
        weight_before_norm: bool,
        column_major_idx: bool,
        expert_filter: bool,
        threads_per_row,
        stream,
        w_dtype: Any = None,
    ):
        # Weights are read as fp32, so the all2all path's fp32 stays unrounded.
        w_dtype = w_dtype if w_dtype is not None else dtype
        enable_pdl = (
            current_platform.is_cuda() and current_platform.is_arch_support_pdl()
        )
        key = (
            enable_pdl,
            dtype,
            hidden_dim,
            topk,
            hidden_dim_padded,
            weight_before_norm,
            column_major_idx,
            expert_filter,
            threads_per_row,
            w_dtype,
        )
        compiled = _compile_cache.get(key)
        if compiled is None:
            gmem_ptr = lambda t: make_ptr(  # noqa: E731
                t, 16, cute.AddressSpace.gmem, assumed_align=16
            )
            compiled = cute.compile(
                MoEFinalizeRMSNormKernel(
                    dtype,
                    hidden_dim,
                    topk,
                    hidden_dim_padded,
                    weight_before_norm=weight_before_norm,
                    column_major_idx=column_major_idx,
                    expert_filter=expert_filter,
                    threads_per_row=threads_per_row,
                    enable_pdl=enable_pdl,
                ),
                gmem_ptr(dtype),
                gmem_ptr(w_dtype),
                gmem_ptr(cutlass.Int32),
                gmem_ptr(cutlass.Int32),
                gmem_ptr(dtype),
                gmem_ptr(dtype),
                Int32(1),
                Int32(1),
                Int32(0),
                Int32(1),
                Float32(1e-6),
                Float32(0.0),
                stream,
            )
            _compile_cache[key] = compiled
        return compiled


def moe_finalize_with_rmsnorm(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    output: torch.Tensor,
    *,
    topk: int,
    eps: float,
    gain_center: float = 0.0,
    weight_before_norm: bool = True,
    column_major_idx: bool = False,
    topk_ids: torch.Tensor | None = None,
    start_expert: int = 0,
    num_local_experts: int = 0,
) -> None:
    """Unpermute, norm each expert's row, and reduce into ``output``."""
    assert HAS_CUTEDSL, "the fused MoE finalize requires nvidia-cutlass-dsl"
    num_tokens, hidden_dim = output.shape
    if num_tokens == 0:
        return

    dtype = _TORCH_TO_CUTLASS_DTYPE[gemm2_output.dtype]
    assert expert_weights.dtype in (gemm2_output.dtype, torch.float32), (
        "expert_weights must be the activation dtype or fp32, got "
        f"{expert_weights.dtype} vs {gemm2_output.dtype}"
    )
    w_dtype = _TORCH_TO_CUTLASS_DTYPE[expert_weights.dtype]
    assert rms_norm_weight.dtype == gemm2_output.dtype
    assert permuted_idx.dtype == torch.int32
    for name, t in (
        ("expert_weights", expert_weights),
        ("permuted_idx", permuted_idx),
        ("rms_norm_weight", rms_norm_weight),
        ("output", output),
    ):
        assert t.is_contiguous(), f"{name} must be contiguous"

    if topk_ids is None:
        expert_ids = permuted_idx  # unused; the kernel compiles the filter away
    else:
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        expert_ids = topk_ids

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _compiled_kernel(
        dtype,
        hidden_dim,
        topk,
        gemm2_output.shape[1],
        weight_before_norm,
        column_major_idx,
        topk_ids is not None,
        None,
        stream,
        w_dtype=w_dtype,
    )
    compiled(
        make_ptr(dtype, gemm2_output.data_ptr()),
        make_ptr(w_dtype, expert_weights.data_ptr()),
        make_ptr(cutlass.Int32, permuted_idx.data_ptr()),
        make_ptr(cutlass.Int32, expert_ids.data_ptr()),
        make_ptr(dtype, rms_norm_weight.data_ptr()),
        make_ptr(dtype, output.data_ptr()),
        Int32(num_tokens),
        Int32(gemm2_output.shape[0]),
        Int32(start_expert),
        Int32(num_local_experts),
        Float32(eps),
        Float32(gain_center),
        stream,
    )


def supports_moe_finalize(dtype: torch.dtype, hidden_dim: int) -> bool:
    """Whether the fused finalize can serve this activation dtype and width."""
    if not HAS_CUTEDSL or dtype not in _TORCH_TO_CUTLASS_DTYPE:
        return False
    return _FinalizeConfig(_TORCH_TO_CUTLASS_DTYPE[dtype], hidden_dim).supported()
