# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# PublishingLowLatencyBlockscaledGemmKernel.epilog_warp is modified from
# LowLatencyBlockscaledGemmKernel.epilog_warp in FlashInfer
# (flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py,
# flashinfer-python 0.7.0.post1), which carries the notice below. Modified by
# the vLLM contributors: the epilogue also stores every output value into the
# fused all-reduce + mHC kernel's Lamport mailbox on all ranks.

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""DSV4.1 attention output projection that publishes into the mHC all-reduce.

At a low-batch attention boundary the output projection ``wo_b`` (FlashInfer's
``cutedsl_low_latency`` MXFP8 GEMM) writes this rank's partial sum, then the
publish kernel of ``BoundaryAllReduce`` reads it back and multicasts it into the
Lamport mailbox before the consumer can collect the peers' contributions.

:class:`PublishingMxfp8Linear` runs the same GEMM, with the tactic FlashInfer's
autotuner selected for the plain call, but its epilogue also does the publish
kernel's work: every output value goes, rounded to BF16 with negative zero
cleared, into this rank's slot of the mailbox generation ``stage_state[0]`` on
all ranks (``multimem.st``), and one thread sets ``stage_state[1]`` to that
generation. The caller then calls the same ``BoundaryAllReduce`` instance with
the returned tensor, which launches only the consumer
(``BoundaryAllReduce.mark_published``). The consumer is unchanged, so its
outputs are bit-identical to the unfused path's.

GEMM structure. The kernel computes ``C[n, m] = sum_k W[n, k] x[m, k]`` with the
weight as operand A (128 output columns per CTA, ``N / 128`` CTAs) and the at
most 8 activation rows as operand B. With split-K ``S`` (the tactic's last
field), the ``S`` CTAs of a cluster each multiply a K slice, exchange FP32
partials through distributed shared memory and each reduce ``1 / S`` of the
register elements. The epilogue's TMEM load is 16x256b for this BF16 output
(32x32b is also handled): for one register element, the eight consecutive
output columns of one row sit in eight lanes four apart, all owned by the same
CTA. Warp shuffles pack them into one 16-byte multicast store issued by the lane
whose column is a multiple of 8, so every store is aligned like the
consumer's 16-byte mailbox fragments.

Ordering. The epilogue reads ``stage_state[0]`` after its MMA completed, hence
after the GEMM's programmatic grid wait (the B loads follow it), exactly as the
publish kernel reads it after its own grid wait; the previous consumer wrote it.
The consumer, launched as the GEMM's programmatic dependent, reads
``stage_state[1]`` only after its grid wait, that is, after the whole GEMM grid
and its stores completed.

Addresses. The mailbox multicast address, the stage-state address, the rank,
the TP size, the mailbox capacity and the hidden size are compile-time
constants of each compiled kernel: each publisher serves one
``BoundaryAllReduce``, whose buffers live as long as it does. Kernels compile
per tactic, before CUDA graph capture (:meth:`PublishingMxfp8Linear.precompile`).
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_fake_stream, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import PipelineUserType, make_pipeline_state
from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
    LowLatencyBlockscaledGemmKernel,
    WorkTileInfo,
)

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.models.deepseek_v41.common.ops.query_quant import mxfp8_scale_bytes

from ...ops.cute_dsl.all_reduce_mhc import ACTIVE_STAGE, NEXT_STAGE
from ...ops.cute_dsl.primitives import (
    VEC_BF16,
    _asm,
    current_cu_stream,
    f32_to_bf16_bits,
    sanitize_negative_zero_u32x4,
    shuffle_sync_idx_u32,
    stmc_bf16x8,
)
from .. import LOW_BATCH_MAX_ROWS
from ..attention.mxfp8 import Tactic, mxfp8_eligible
from .all_reduce import BoundaryAllReduce


@dsl_user_op
def _load_volatile_u32_at(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_u32_at(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


class PublishingLowLatencyBlockscaledGemmKernel(LowLatencyBlockscaledGemmKernel):
    """FlashInfer's low-latency block-scaled GEMM whose epilogue publishes C.

    Only ``epilog_warp`` differs from the base class (the publish is marked
    there); the main loop, the split-K reduction and C are unchanged.
    """

    def __init__(
        self,
        *,
        mailbox_multicast: int,
        stage_state: int,
        tp: int,
        rank: int,
        capacity_m: int,
        hidden: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if hidden % self.cta_m:
            raise ValueError("hidden must tile into whole 128-column CTAs")
        # The 16-byte multicast stores and the stage words' 4-byte accesses.
        if mailbox_multicast % 16 or stage_state % 4:
            raise ValueError("misaligned mailbox or stage-state address")
        self.mailbox_multicast = mailbox_multicast
        self.stage_state = stage_state
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.hidden = hidden

    @cute.jit
    def publish_value(
        self,
        bits: Uint32,
        column: Int32,
        token: Int32,
        in_bounds,
        stage: Uint32,
        column_lanes: cutlass.Constexpr[int],
    ) -> None:
        """Pack eight lanes' BF16 bits and multicast them from the lane that
        holds the 8-aligned column.

        Every lane of the warp calls this for the same register element. The
        lanes holding columns ``c .. c + 7`` of one token (``c % 8 == 0``) are
        ``l, l + s, .., l + 7s`` for the TMEM load's lane stride ``s`` =
        ``column_lanes``; the store address comes from the storing lane's own
        column, so it is 16-byte aligned whatever the layout.
        """
        lane = cute.arch.lane_idx()
        partner = shuffle_sync_idx_u32(bits, lane + Int32(column_lanes))
        word = bits | (partner << Uint32(16))
        pack = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        pack[0] = word
        for index in cutlass.range_constexpr(1, 4):
            pack[index] = shuffle_sync_idx_u32(
                word, lane + Int32(2 * index * column_lanes)
            )
        if in_bounds:  # noqa: SIM102
            if (column & Int32(VEC_BF16 - 1)) == Int32(0):
                clean = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                clean.store(sanitize_negative_zero_u32x4(pack))
                element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(column)
                stmc_bf16x8(Int64(self.mailbox_multicast) + element * 2, clean)

    @cute.jit
    def epilog_warp(
        self,
        tma_epilog_full_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        mC_mnl: cute.Tensor,
        mScale: Optional[cute.Tensor],  # noqa: UP045 (as the base class)
        mBias_mnl: Optional[cute.Tensor],  # noqa: UP045
        tmem_base_smem_ptr,
        mailbox_mbar,
        dsmem_mailbox_ptr,
        epi_tid: cutlass.Int32,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        split_rank: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        """FlashInfer's epilogue (modified from FlashInfer 0.7.0.post1; see the
        file header), plus the Lamport publish of C."""
        gC_mnl = cute.local_tile(mC_mnl, (self.cta_m, self.cta_n), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgC = thr_mma.partition_C(gC_mnl)

        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            (self.cta_m, self.cta_n),
            self.use_2cta_instrs,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc[((None, None), 0, 0)]
        )
        # PUBLISH: lane distance between consecutive output columns (TMEM
        # lanes) in one register element. 32x32b: thread t holds lane t.
        # 16x256b (what FlashInfer picks for the M-major BF16 C): thread t
        # holds lanes t / 4 and t / 4 + 8, two N columns each.
        if cutlass.const_expr(isinstance(copy_atom_t2r.op, tcgen05.Ld32x32bOp)):
            column_lanes = 1
        elif cutlass.const_expr(isinstance(copy_atom_t2r.op, tcgen05.Ld16x256bOp)):
            column_lanes = 4
        else:
            raise NotImplementedError(f"TMEM load {copy_atom_t2r.op} for publish")
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[((None, None), 0, 0)])
        tDgC = thr_copy_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])
        epi_frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        tDrAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)
        tDrC = cute.make_rmem_tensor(epi_frag_shape, self.c_dtype)
        tDgC = tDgC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]

        coordC = cute.make_identity_tensor(mC_mnl.shape)
        gCcoord = cute.local_tile(coordC, (self.cta_m, self.cta_n), (None, None, None))
        tCcC = thr_mma.partition_C(gCcoord)
        tDcC = thr_copy_t2r.partition_D(tCcC[((None, None), 0, 0, None, None, None)])
        tDcC = tDcC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]
        tDpredC = cute.make_rmem_tensor(tDcC.shape, cutlass.Boolean)
        for i in range(cute.size(tDpredC)):
            tDpredC[i] = cute.elem_less(tDcC[i], mC_mnl.shape)

        if cutlass.const_expr(self.use_scale):
            scale_value = mScale[0].to(self.acc_dtype)  # type: ignore[index]

        if cutlass.const_expr(self.use_bias):
            gBias_mnl = cute.local_tile(
                mBias_mnl, (self.cta_m, self.cta_n), (None, None, None)
            )
            tCgBias = thr_mma.partition_C(gBias_mnl)
            tDgBias = thr_copy_t2r.partition_D(
                tCgBias[((None, None), 0, 0, None, None, None)]
            )
            tDgBias = tDgBias[
                (
                    None,
                    None,
                    None,
                    work_tile_info.M_idx,
                    work_tile_info.N_idx,
                    work_tile_info.L_idx,
                )
            ]
            tDrBias = cute.make_rmem_tensor(epi_frag_shape, self.bias_dtype)
            tDrBiasAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)
            cute.arch.mbarrier_wait(tma_epilog_full_bar, 0)
            cute.basic_copy_if(tDpredC, tDgBias, tDrBias)
            tDrBiasAcc.store(tDrBias.load().to(self.acc_dtype))

        mma_epilog_state = make_pipeline_state(PipelineUserType.Consumer, 1)
        cute.arch.mbarrier_wait(
            mma_epilog_full_bar + mma_epilog_state.index, mma_epilog_state.phase
        )
        # PUBLISH: the MMA consumed operand B, which the B-load warp fetched
        # only after the programmatic grid wait, so the previous consumer's
        # generation advance is visible. Loaded here to overlap the reduction.
        stage = _load_volatile_u32_at(Int64(self.stage_state + NEXT_STAGE * 4))

        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
        cute.arch.fence_view_async_tmem_load()

        if cutlass.const_expr(self.split_k > 1):  # noqa: SIM102
            if k_tile_count == 0:
                tDrAcc.fill(0.0)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        elems_per_thread = self._mailbox_elems_per_thread
        if self.split_k == 1:
            if cutlass.const_expr(self.use_scale):
                tDrAcc.store(tDrAcc.load() * scale_value)
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            # PUBLISH: every row of the tile.
            for i in cutlass.range_constexpr(elems_per_thread):
                coord = tDcC[i]
                self.publish_value(
                    f32_to_bf16_bits(tDrC[i].to(Float32)),
                    coord[0],
                    coord[1],
                    tDpredC[i],
                    stage,
                    column_lanes,
                )
            cute.basic_copy_if(tDpredC, tDrC, tDgC)
        else:
            shard_ept = self._shard_elems_per_thread

            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mailbox_mbar, self._mailbox_tx_total
                )

            cute.arch.barrier(barrier_id=15, number_of_threads=128)

            for peer in range(self.split_k):
                if peer != split_rank:
                    shard_start = peer * shard_ept
                    sender_idx = split_rank - (
                        1 if split_rank > cutlass.Int32(peer) else 0
                    )
                    mailbox_base = sender_idx * 128 * shard_ept
                    for i in range(shard_ept):
                        cute.arch.store_async_dsmem(
                            dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid,
                            tDrAcc[shard_start + i].bitcast(cutlass.Int32),
                            mailbox_mbar,
                            cutlass.Int32(peer),
                        )

            cute.arch.mbarrier_wait(mailbox_mbar, 0)

            my_shard_start = split_rank * shard_ept
            for s in range(self.split_k - 1):
                mailbox_base = s * 128 * shard_ept
                for i in range(shard_ept):
                    tDrAcc[my_shard_start + i] = (
                        tDrAcc[my_shard_start + i]
                        + (dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid).load()
                    )

            if cutlass.const_expr(self.use_scale):
                tDrAcc.store(tDrAcc.load() * scale_value)
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())

            for i in range(elems_per_thread):
                if i < my_shard_start or i >= my_shard_start + shard_ept:
                    tDpredC[i] = cutlass.Boolean(False)

            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            # PUBLISH: this CTA's shard of rows (the condition is CTA-uniform,
            # so whole warps shuffle together).
            for i in cutlass.range_constexpr(elems_per_thread):
                if i >= my_shard_start and i < my_shard_start + shard_ept:
                    coord = tDcC[i]
                    self.publish_value(
                        f32_to_bf16_bits(tDrC[i].to(Float32)),
                        coord[0],
                        coord[1],
                        tDpredC[i],
                        stage,
                        column_lanes,
                    )
            cute.basic_copy_if(tDpredC, tDrC, tDgC)

        # PUBLISH: hand the generation to the consumer, once per grid.
        bidx, bidy, bidz = cute.arch.block_idx()
        if bidx == 0 and bidy == 0 and bidz == 0 and epi_tid == 0:
            _store_u32_at(Int64(self.stage_state + ACTIVE_STAGE * 4), stage)


def _gmem_ptr(dtype, address: int, align: int):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


class PublishingMxfp8Linear:
    """Run a low-latency MXFP8 linear whose output a ``BoundaryAllReduce``
    reduces next, publishing the output from the GEMM epilogue.

    Call it in place of the linear; it falls back to the linear itself (and
    publishes nothing) for any input the low-latency backend does not take.
    """

    def __init__(self, all_reduce: BoundaryAllReduce) -> None:
        self.all_reduce = all_reduce
        self._compiled: dict[tuple[Tactic, int], object] = {}

    def supports(self, linear: torch.nn.Module, num_tokens: int) -> bool:
        """Whether ``linear`` on ``num_tokens`` rows runs the publishing GEMM."""
        return (
            mxfp8_eligible(linear)
            and 0 < num_tokens <= min(LOW_BATCH_MAX_ROWS, self.all_reduce.capacity)
            and linear.weight.shape[1] == self.all_reduce.hidden_size
            and getattr(linear, "bias", None) is None
        )

    def __call__(
        self, linear: torch.nn.Module, x: torch.Tensor | QuantizedActivation
    ) -> torch.Tensor:
        """``linear(x)``, published into ``all_reduce``'s mailbox when ``x`` is
        an MXFP8 activation of the low-latency backend's layout."""
        qa = (
            as_quantized_activation(x, linear.quant_method.kernel.input_quant_key())
            if isinstance(x, QuantizedActivation)
            else None
        )
        if (
            qa is None
            or qa.orig_dtype != torch.bfloat16
            or qa.scale.ndim != 1
            or not qa.data.is_contiguous()
            or not self.supports(linear, qa.data.shape[0])
        ):
            return linear(x)
        a, a_scale = qa.data, qa.scale
        weight, weight_scale = linear.weight, linear.weight_scale
        k, n = weight.shape
        m = a.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        tactic = _selected_tactic(a, weight, a_scale, weight_scale, out)
        compiled = self._compile(tactic, a.device)
        compiled(
            _gmem_ptr(cutlass.Float8E4M3FN, weight.data_ptr(), 16),
            _gmem_ptr(cutlass.Float8E8M0FNU, weight_scale.data_ptr(), 32),
            _gmem_ptr(cutlass.Float8E4M3FN, a.data_ptr(), 16),
            _gmem_ptr(cutlass.Float8E8M0FNU, a_scale.data_ptr(), 16),
            _gmem_ptr(cutlass.BFloat16, out.data_ptr(), 16),
            None,
            None,
            (Int32(n), Int32(m), Int32(k), Int32(1)),
            current_cu_stream(),
        )
        out = out.view(*qa.orig_shape[:-1], n)
        self.all_reduce.mark_published(out)
        return out

    def precompile(self, linear: torch.nn.Module, num_tokens: int) -> None:
        """Compile the kernel ``num_tokens`` rows select, launching nothing: a
        publish without its consumer would desynchronize the mailbox."""
        if not self.supports(linear, num_tokens):
            return
        weight, weight_scale = linear.weight, linear.weight_scale
        k, n = weight.shape
        device = weight.device
        a = torch.zeros((num_tokens, k), dtype=torch.float8_e4m3fn, device=device)
        a_scale = torch.zeros(
            mxfp8_scale_bytes(num_tokens, k), dtype=torch.uint8, device=device
        )
        out = torch.empty((num_tokens, n), dtype=torch.bfloat16, device=device)
        self._compile(_selected_tactic(a, weight, a_scale, weight_scale, out), device)

    def _compile(self, tactic: Tactic, device: torch.device):
        """The publishing kernel of ``tactic``, compiled once per device."""
        key = (tactic, device.index)
        compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled
        cta_k, ab_stages, sfb_tmem_stages, split_k = tactic
        all_reduce = self.all_reduce
        multicast, stage_state = all_reduce.publish_target()
        gemm = PublishingLowLatencyBlockscaledGemmKernel(
            mailbox_multicast=multicast,
            stage_state=stage_state,
            tp=all_reduce.tp,
            rank=all_reduce.rank,
            capacity_m=all_reduce.capacity,
            hidden=all_reduce.hidden_size,
            acc_dtype=cutlass.Float32,
            mma_tiler_mnk=(128, 8, cta_k),
            num_ab_stage=ab_stages,
            num_sfb_tmem_stage=sfb_tmem_stages,
            sf_vec_size=32,
            use_pdl=True,
            split_k=split_k,
        )
        with torch.accelerator.device_index(device.index):
            compiled = cute.compile(
                gemm,
                _gmem_ptr(cutlass.Float8E4M3FN, 16, 16),
                _gmem_ptr(cutlass.Float8E8M0FNU, 32, 32),
                _gmem_ptr(cutlass.Float8E4M3FN, 16, 16),
                _gmem_ptr(cutlass.Float8E8M0FNU, 16, 16),
                _gmem_ptr(cutlass.BFloat16, 16, 16),
                None,
                None,
                tuple(Int32(v) for v in (128, 8, cta_k, 1)),
                make_fake_stream(),
            )
        self._compiled[key] = compiled
        return compiled


def _selected_tactic(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> Tactic:
    """The tactic ``mm_mxfp8(backend="cutedsl_low_latency")`` runs for these
    operands: the autotuner's cached choice, else the runner's default (the
    first valid tactic), resolved as ``mm_mxfp8`` resolves it."""
    # mm_mxfp8's private runner, tuning config and workspace, so the lookup
    # hits the same autotuner cache entry.
    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm import gemm_base

    major, minor = torch.cuda.get_device_capability(a.device)
    runner = gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(
        major * 10 + minor, True
    )
    workspace = gemm_base._get_cache_buf(
        "mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, a.device
    )
    inputs = [a, weight, a_scale, weight_scale, out.dtype, out, workspace]
    _, tactic = AutoTuner.get().choose_one(
        "mxfp8_gemm",
        [runner],
        gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG,
        inputs,
    )
    if tactic is None or tactic == -1:
        tactics = runner.get_valid_tactics(inputs, None)
        if not tactics:
            raise ValueError("the low-latency GEMM cannot run these operands")
        tactic = tactics[0]
    return tuple(tactic)
