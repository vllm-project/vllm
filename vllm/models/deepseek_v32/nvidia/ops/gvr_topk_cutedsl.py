# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA/TensorRT-LLM at 3c68ae6ac79c48c6bad5816adcfcefc7d9897d55.
# Keep this upstream kernel mechanically comparable to its source.
# ruff: noqa

"""GVR (Guess-Verify-Refine) Top-K kernel for Blackwell sm_100.

Unified single- and multi-CTA-per-row implementation. Each row is processed by
``cluster_size`` CTAs cooperating via a thread-block cluster; CTA ``r`` scans
``row[r*N/cs : (r+1)*N/cs]`` (vec_w-aligned split, last CTA absorbs remainder).
Per-iter cand_count is aggregated via DSMEM (``mapa.shared::cluster`` +
``ld.shared::cluster``) — no GMEM atomics. ``cluster_size=1`` degenerates to a
plain single-CTA path with all cluster code paths compiled out via
``const_expr``.

Supported (dtype, K): fp32 / bf16 / fp16 x 512 / 1024 / 2048.
cluster_size: 1 (default), 2, 4 (B200 GPC limit caps at ~16).
"""

import math
from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

from .gvr_block_scan import warp_scan

TRTLLM_ENABLE_PDL = True


# ---------------------------------------------------------------------------
# DSMEM primitives (inline PTX)
# Adapted from single_pass_multi_cta_radix_topk_cluster.py.
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None):
    """Map a local SMEM pointer to peer CTA's SMEM in cluster address space.

    PTX: ``mapa.shared::cluster.u32 %dst, %src, %peer_rank;``

    The returned i32 address can be passed to ``ld.shared::cluster.*`` and
    ``st.shared::cluster.*`` to read/write the peer's identically-laid-out
    SMEM allocation. CuTe DSL's high-level SMEM tensor ops do NOT lower to
    cluster-space loads, so DSMEM access must go through inline PTX.
    """
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def mapa_shared_cluster(smem_ptr, peer_rank):
    return _mapa_shared_cluster(smem_ptr, peer_rank)


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr, *, loc=None, ip=None):
    """Load an int32 from a peer CTA's SMEM via cluster mapped address."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def ld_shared_cluster_i32(mapped_addr):
    return _ld_shared_cluster_i32(mapped_addr)


@dsl_user_op
def _ld_shared_cluster_f32(mapped_addr, *, loc=None, ip=None):
    """Load an fp32 from a peer CTA's SMEM via cluster mapped address."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def ld_shared_cluster_f32(mapped_addr):
    return _ld_shared_cluster_f32(mapped_addr)


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern (cuTe DSL bit-cast)."""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


def float_as_int32(float_val):
    """Interpret FP32 value as int32 bit pattern (cuTe DSL bit-cast)."""
    return cutlass.Int32(llvm.bitcast(cutlass.Int32.mlir_type, float_val.ir_value()))


def f32_order_key(float_val):
    """Order-preserving fp32 -> int32 key (unsigned-monotonic bit pattern).

    ``s ^ ((s >> 31) | 0x80000000)``: positive floats map to
    ``bits | 0x80000000``, negative floats to ``~bits`` — the standard radix
    transform whose UNSIGNED order equals fp32 order (NaN-free inputs). The
    returned Int32 must only be consumed digit-wise (``(k >> s) & 0xFF``) or
    via equality / prefix-equality; for a full ordered compare, flip the top
    bit first (``k ^ 0x80000000`` is signed-monotonic).
    """
    s = float_as_int32(float_val)
    return s ^ ((s >> cutlass.Int32(31)) | cutlass.Int32(-2147483648))


def _fmin_f32_inline(a, b):
    """Single PTX ``min.f32`` → one SASS FMNMX.

    cuTe DSL exposes ``cute.arch.fmax`` but not ``fmin``; the canonical
    ``-fmax(-a, -b)`` workaround lowers to 4 SASS insts and was worth
    ~8-10 µs of the prod-GVR gap at fp32 K=2048 BS=1.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [a.ir_value(), b.ir_value()],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# GvrParams<T, K> — parameters for different (dtype, K, compress_ratio) combinations.
# =============================================================================


@dataclass(frozen=True)
class GvrParams:
    kFTarget: int
    kC: int  # candidate buffer cap
    kNumBins: int  # histogram bin count

    @staticmethod
    def get(dtype_name: str, top_k: int, compress_ratio: int = 1) -> "GvrParams":
        """Per-(dtype, K, cr) tuning constants, mirroring CUDA's
        ``GvrParams<T, K>`` template specialization. For K ∈ {512, 1024}
        cr=1 (DSv3.2) and cr=4 (DSv4, PR #14413) use different kFTarget —
        V4 aligns kFTarget with kK to avoid upper-clamp saturation on
        tight-sigma layers (1.5-2.2x fewer P2 iters on swe-bench). K=2048 is
        identical across cr (V4 doesn't natively use it).
        """
        TABLE = {
            # --- cr = 1 (DSv3.2): tuned on V3.2 swe-bench data ---
            ("float32", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=1024),
            ("float32", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float32", 2048, 1): GvrParams(kFTarget=3072, kC=6144, kNumBins=2048),
            ("bfloat16", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("bfloat16", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=512),
            ("bfloat16", 2048, 1): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            ("float16", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("float16", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float16", 2048, 1): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            # --- cr = 4 (DSv4): tuned on V4 Flash/Pro swe-bench data ---
            ("float32", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=1024),
            ("float32", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=1024),
            ("float32", 2048, 4): GvrParams(kFTarget=3072, kC=6144, kNumBins=2048),
            ("bfloat16", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=512),
            ("bfloat16", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=512),
            ("bfloat16", 2048, 4): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            ("float16", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=512),
            ("float16", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=1024),
            ("float16", 2048, 4): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
        }
        key = (dtype_name, top_k, compress_ratio)
        if key not in TABLE:
            raise ValueError(
                f"Unsupported GvrParams<{dtype_name}, {top_k}, cr={compress_ratio}>"
            )
        return TABLE[key]


class GvrTopKKernel:
    """GVR (Guess-Verify-Refine) heuristic top-K kernel using cuTe DSL.

    One CTA processes one row.
    Block size = 512/1024, as specified by num_threads.
    Smem region sized to GvrParams<dtype, top_k>.

    Algorithm phases:
      P1: preIdx Min/Max/Mean → initial threshold
      P1b: 256-bin histogram over prev-topK gathered values → M rung
           thresholds (enable_r0 only)
      P2: threshold admission — default (enable_r0=True) is a single-pass
          multi-threshold rung-ladder; enable_r0=False keeps the classic
          secant threshold search loop (count-only), also the R0-miss fallback
      P3: Ballot-free candidate collect into smem keys[]/vals[]
      P4: rank-and-scatter (enable_r0) / histogram snap → exact top-K + writeback

    For different compress_ratio:
      cr = 1: preIdxOffset = (row_idx % next_n) + 1. V3.2 decode +1 temporal shift.
      cr = 4: preIdxOffset = 0. V4 decode no temporal shift.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        top_k: int,
        next_n: int = 1,
        num_threads: int = 512,
        enable_unroll_4: Optional[bool] = None,
        enable_phase3_unroll: Optional[bool] = None,
        use_constant_hint: bool = False,
        min_blocks_per_mp: int = 3,
        use_256bit_load: bool = False,
        enable_warp_parallel_reduce: Optional[bool] = None,
        compress_ratio: int = 1,
        return_output_values: bool = True,
        cluster_size: int = 1,
        enable_smem_cache: bool = False,
        smem_cache_elems: int = 32768,
        seqlen_sorted: bool = False,
        kc_diet: Optional[bool] = None,
        enable_r0: bool = True,
        r0_qfracs: Optional[tuple] = None,
        mt_unroll: int = 4,
        p1b_cache: Optional[bool] = None,
        fb_fix: bool = True,
        fb_alpha: float = 0.2,
        r0_vseed: Optional[bool] = None,
        adaptive_rungs: bool = True,
        enable_p4_rank_scatter: Optional[bool] = None,
        enable_p4_rank_scatter_exact: Optional[bool] = None,
        p4_exact_tail: Optional[bool] = None,
        p4_tail_fast: Optional[bool] = None,  # [p4tt]
        p4_warp_redundant: bool = True,
        p2_warp_redundant: bool = True,
        fuse_state_store: bool = False,
        fuse_hint_prepare: bool = False,
    ):
        self.fuse_state_store = fuse_state_store
        self.fuse_hint_prepare = fuse_hint_prepare
        # Redundant-warp sync reduction: every warp replays the block
        # reduce + decision from the same staged SMEM partials in the
        # same fp32 order, so results are bit-identical across warps and
        # the publish barrier + leader serialization disappear.
        #   p4_warp_redundant: P4 k-th bin search + snap loop (1 barrier/iter).
        #   p2_warp_redundant: P2 secant cadence (cluster_size == 1 only).
        # Both default ON; OFF restores the leader-based paths (A/B).
        self.p4_warp_redundant = p4_warp_redundant
        self.p2_warp_redundant = p2_warp_redundant
        # cluster_size: number of CTAs cooperating per row. 1 = single-CTA
        # path; 2/4 = thread-block cluster with DSMEM aggregation. Capped at
        # 16 by B200's per-GPC SM count.
        if cluster_size < 1 or cluster_size > 16:
            raise ValueError(
                f"cluster_size must be in [1, 16] (B200 GPC limit); got {cluster_size}"
            )
        self.cluster_size = cluster_size
        # When True, the kernel resolves the owning row per CTA via a
        # caller-provided ``order_row`` indirection — an LJF host-side
        # dispatch order so longer rows hit earlier waves. ``order_row``
        # is REQUEST-level: int32[batch_size = num_rows / next_n],
        # typically a descending argsort of seq_lens. The kernel
        # expands to row level as ``order_row[req] * next_n + nn`` so a
        # request's ``next_n`` rows stay contiguous. Compatible with
        # cluster_size > 1: all cs CTAs in a cluster see the same
        # cluster_id (= bidx // cluster_size), hence the same row.
        self.seqlen_sorted = seqlen_sorted
        # SMEM slice cache (optional): pre-stage each CTA's slice into SMEM
        # once between Phase 1 and Phase 2, so Phase 2/3's GE-count scans
        # read LDS instead of re-streaming GMEM. Caller is responsible for
        # ensuring slice_len <= smem_cache_elems; ``smem_cache_elems`` sets
        # the JIT-time alloc size (see TODO at _compile for host-side assert).
        if enable_smem_cache and smem_cache_elems <= 0:
            raise ValueError("smem_cache_elems must be > 0 when enable_smem_cache")
        self.enable_smem_cache = enable_smem_cache
        self.smem_cache_elems = smem_cache_elems
        # e.g., dtype = cutlass.Float32 / cutlass.BFloat16 / cutlass.Float16
        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        # KV compression ratio:
        #   1 → DSv3.2; preIdxOffset = (row % next_n) + 1 to land prev-step
        #       indices in this step's KV space (with MTP windowing).
        #   4 → DSv4; logits/preIdx live in compressed-token-index space.
        #       New entries are appended at the end so prev indices stay
        #       valid → preIdxOffset = 0.
        assert compress_ratio in (1, 4), (
            f"compress_ratio must be 1 (V3.2) or 4 (V4); got {compress_ratio}"
        )
        self.compress_ratio = compress_ratio

        self.WARP_SIZE = 32
        self.num_threads = num_threads
        self.num_warps = num_threads // self.WARP_SIZE
        # __launch_bounds__(num_threads, min_blocks_per_mp) ptxas hint.
        # On B200 (65536 regs/SM, BS=512), max regs/thread is 128 at mb=1,
        # 64 at mb=2, 42 at mb=3. Pick low mb when num_rows ≤ #SMs so
        # ptxas can spend more regs covering LDG latency.
        self.min_blocks_per_mp = min_blocks_per_mp
        # Vector-load width for Phase 2/3 scans:
        #   False (default): 128-bit LDG  (fp32: 4 / bf16/fp16: 8 elems)
        #   True:            256-bit LDG  (fp32: 8 / bf16/fp16: 16 elems)
        # 256-bit halves the LDG count but needs 32B-aligned addresses
        # (we set assumed_align=32) and doubles fragment reg footprint.
        self.use_256bit_load = use_256bit_load
        self.vec_bits = 256 if use_256bit_load else 128
        self.vec_align_bytes = self.vec_bits // 8  # 32 for 256-bit, 16 for 128-bit
        # Vec-loop unroll switches.
        #   enable_unroll_4:        4-way fast path in block_count_ge.
        #   enable_phase3_unroll:   4-way fast path in phase3_collect.
        #     Independent of enable_unroll_4: Phase 3 has thread-local wc
        #     state + smem writes, so its fast-path trade-off differs.
        #   use_constant_hint:      True → CopyG2ROp(invariant=True) emits
        #     LDG.E.*.CONSTANT (read-only cache, == CUDA __ldg). False →
        #     plain CopyUniversalOp / LDG.E.*.
        if enable_unroll_4 is None:
            enable_unroll_4 = True
        if enable_phase3_unroll is None:
            enable_phase3_unroll = True
        self.enable_unroll_4 = enable_unroll_4
        self.enable_phase3_unroll = enable_phase3_unroll
        self.use_constant_hint = use_constant_hint
        # Replace tid==0 serial block-reduces with warp-parallel reduces
        # in warp 0. Auto-policy: on iff num_threads == 1024 (32 warps),
        # where the serial cost is meaningful; at 512 threads (16 warps)
        # the warp-parallel path regressed ~2pp on synth.
        if enable_warp_parallel_reduce is None:
            enable_warp_parallel_reduce = num_threads == 1024
        self.enable_warp_parallel_reduce = enable_warp_parallel_reduce

        # When False, skip all STG writes to ``output_values`` and accept
        # None at launch — saves LSU bandwidth + reg pressure for callers
        # that only consume top-K indices (e.g. the DSA indexer). When
        # True (default), values are written for bench / standalone use.
        self.return_output_values = return_output_values

        # Map cutlass dtype → GvrParams lookup name
        if dtype == cutlass.Float32:
            self._dtype_name = "float32"
        elif dtype == cutlass.BFloat16:
            self._dtype_name = "bfloat16"
        elif dtype == cutlass.Float16:
            self._dtype_name = "float16"
        else:
            raise ValueError(f"Unsupported dtype for GvrTopKKernel: {dtype}")

        params = GvrParams.get(self._dtype_name, top_k, self.compress_ratio)
        self.kC = params.kC
        self.kNumBins = params.kNumBins
        self.kFTarget = params.kFTarget

        # Kernel-wide constants.
        # self.MAX_REFINE_ITERS: Phase-2 secant refine iteration cap.
        # self.FLT_MAX / self.NEG_FLT_MAX: fp32 IEEE-754 max / negative-max
        # sentinels used as reduction identities and pad values.
        self.MAX_REFINE_ITERS = 15
        self.FLT_MAX = 3.4028235e38
        self.NEG_FLT_MAX = -self.FLT_MAX

        # --- R0 histogram-ladder admission (default ON) ---
        # enable_r0: replace the Phase-2 secant search with a single-pass
        #   multi-threshold "rung ladder" admission seeded by a 256-bin
        #   histogram over the prev-topK gathered values (P1b).
        #   DEFAULT True: validated on real DSv4/V3.2 decode-capture
        #   workloads (25-cell seq-len scan) where R0 wins 24/25 vs the
        #   secant baseline, geomean 1.33x (pro 128k 2.10x). Correctness is
        #   value-set-exact vs torch.topk (186/186 across dtype/K/N/BS/cluster
        #   + tie plateaus).
        #   THE SECANT PATH IS NOT DEAD CODE AND MUST NOT BE DELETED. It has
        #   two distinct roles:
        #     (a) EXACT FALLBACK, live at the default enable_r0=True: when the
        #         rung ladder admits no candidate (R0 miss) the row falls
        #         through to phase2_secant_search, so this code runs in
        #         production on every hint-unrepresentative row;
        #     (b) DIFFERENTIAL ORACLE, via enable_r0=False: it is the classic
        #         baseline the R0 admission is checked against
        #         (test_..._r0_equivalence), and the direct-drive entry used to
        #         bisect admission-vs-baseline regressions.
        #   All R0 fields are const-foldable, so an enable_r0=False kernel is
        #   byte-identical to the pre-R0 upstream base and the disabled branch
        #   costs nothing at runtime.
        # r0_qfracs: descending h-space quantile fractions defining the M
        #   candidate rungs (ascending threshold values); None => no rungs.
        # r0_vseed: park P1's pmean (the secant init probe) as one extra
        #   "virtual seed" rung column in the M-ary count pass (no extra
        #   memory traffic or sync; the column reuses the secant per-thread
        #   count buffer, so SMEM does not grow). Adapts the admission
        #   ladder to the row's value distribution: fixes the fat-admission
        #   regime (a coarse quantile rung admitting ~kC candidates where
        #   pmean admits ~K) and donates a measured interior bracket point
        #   to the fallback refine on a full miss. None => enable_r0.
        # mt_unroll: 4-way unroll factor for block_count_ge_multi.
        # p1b_cache: stash the K gathered preIdx values in SMEM so P1b skips
        #   a second GMEM random gather (dtype-gated in a later commit).
        # fb_fix: R0-miss fallback re-measures the rung bracket ends before
        #   refining (excludes the R2-class unmeasured-seed failure mode).
        self.enable_r0 = bool(enable_r0)
        self.mt_unroll = int(mt_unroll)
        self.fb_fix = bool(fb_fix)
        # C7 dispatch (host policy folded into the ctor; all gated on
        # enable_r0 so an OFF kernel is byte-identical to the base):
        #  - qfracs default = M2D (0.85, 0.35): the shipped dispatch uses M2D for
        #    every (dtype, K, N); the M=2 pass is ~free and the R1 falsi shot
        #    covers the 3-7% bracket misses. uh4 (M=4) was silicon-falsified
        #    (mc geomean 0.956 — admission != latency).
        #  - p1b_cache default is cs-aware:
        #      * cs>1 (cluster): ON for ALL dtypes. The SMEM gather-cache win
        #        holds and the fp32 occupancy regression that hurts the
        #        single-CTA path does NOT reproduce in the cluster kernel
        #        (latency-bound, different SMEM budget). nsys cs=4: K1024
        #        ~1.01x / K2048 ~1.02x / K512 wash, 0 losses, exact. Matches
        #        the multi-CTA dispatch (unconditional ON).
        #      * cs=1 (single-CTA): (dtype != fp32). The gather-cache wins
        #        +0.8-2.8% on 16-bit (random half-prec gather is the cost) but
        #        is flat/negative on fp32 (occupancy at kC=6144), so OFF there.
        #  - kC-diet: K512 single-CTA -> kC=3072 (saves 16KB SMEM; 16-bit win,
        #    fp32 neutral). kC>=2560 is the K512 16-bit tie-safety contract so
        #    3072 is safe; the cluster port and K1024/K2048 stay stock.
        self.adaptive_rungs = bool(
            adaptive_rungs
            and enable_r0
            and dtype == cutlass.Float32
            and top_k == 2048
            and compress_ratio == 1
            and r0_qfracs is None
            and r0_vseed is None
        )
        if self.adaptive_rungs:
            r0_qfracs = (0.6, 0.35, 0.01)
            r0_vseed = False
        if r0_vseed is None:
            r0_vseed = enable_r0
        if enable_r0 and r0_qfracs is None:
            # Per-K default (full-envelope audit, 2772
            # cells): with the virtual seed rung on, pmean covers q.35's
            # admission region for K512/K1024 (2 count columns = zero
            # column tax); K2048 keeps q.35 (kC/K = 2.5 makes a fat admit
            # costlier than a slim 2-pass miss). Without vseed, q.35 must
            # stay for all K (it is the only slim rung).
            # K2048 low rung 0.85 -> 0.6 (real-content rung recalibration,
            # paired cold-L2 A/B): the shipped
            # 0.85 rung's admission straddles [K, kC] on real V3.2 decode
            # captures (bracket on 86% of steps -> one extra falsi pass);
            # 0.6 lands the first pass. Measured: real V3.2 geomean
            # +2.2-2.8% across fp32/bf16/fp16 and the full BS grid (8K
            # rung +10-13% at every BS, no loser cell), favorable
            # synthetic +9-11%, adverse synthetic wash, exact everywhere.
            # K512/K1024 unchanged: moving or widening their ladder
            # measured wash-to-loss (the extra count column costs 3-7%).
            if top_k == 2048:
                r0_qfracs = (0.6, 0.35) if r0_vseed else (0.85, 0.35)
            else:
                r0_qfracs = (0.85,) if r0_vseed else (0.85, 0.35)
        if enable_r0 and p1b_cache is None:
            if cluster_size > 1:
                p1b_cache = True
            else:
                p1b_cache = dtype != cutlass.Float32
        self.p1b_cache = bool(p1b_cache)
        # kc_diet: None → diet iff single-CTA (tuned default). The LB hybrid
        # kernel passes False for BOTH member instances so their SMEM layouts
        # stay byte-identical (the DSL sizes the launch from the last-traced
        # SmemAllocator only; see GvrTopKLBKernel).
        if kc_diet is None:
            kc_diet = cluster_size == 1
        if enable_r0 and top_k == 512 and kc_diet and self.kC > 3072:
            self.kC = 3072
        # K2048 R0 Phase-4 histogram diet: 2048 -> 512 bins (paired
        # cold-L2 A/B, all cells exact). The P4 zero /
        # atomic build / serial scan all shrink 4x; the deeper boundary-bin
        # recursion costs less than the saved passes at kC=6144 candidates.
        # Measured vs this head: real V3.2 decode captures geomean +6.1%
        # (fp32) / +10.9% (bf16) / +6.3% (fp16); favorable synthetic
        # +5.2-11.0%, adverse synthetic +5.1-10.6%; no losing cell
        # (fp32 min 0.994, bf16 min 1.035, fp16 min 0.999). Gated on
        # enable_r0 so the retained secant path (which shares GvrParams
        # and its own P4 histogram) stays byte-identical. P1b reuses this
        # buffer and needs >= 256 bins, so 512 is safe. K512/K1024
        # measured as a wash under the same protocol and stay stock.
        if enable_r0 and top_k == 2048 and self.kNumBins > 512:
            self.kNumBins = 512
        self.r0_qfracs = tuple(float(q) for q in r0_qfracs) if r0_qfracs else ()
        if self.r0_qfracs:
            assert all(0.0 < q < 1.0 for q in self.r0_qfracs), self.r0_qfracs
            assert list(self.r0_qfracs) == sorted(self.r0_qfracs, reverse=True), (
                "r0_qfracs must be descending h (ascending threshold value)"
            )
        self.M_thr = len(self.r0_qfracs)
        # --- vseed: fold P1's pmean (the secant init
        # probe) into the M-ary R0 count pass as one extra "virtual rung".
        # Fixes the flash-1M fat-admission regression (the coarse q.85 rung
        # admits ~4400 candidates where pmean admits ~630 -> 7x P3/P4 cand
        # cost) and, on a true miss, donates a measured interior bracket
        # point to the fallback refine. Const-folded: r0_vseed=False kernels
        # are byte-identical to before. M_qf = rungs P1b places from qneeds;
        # M_thr = total columns counted/admitted (M_qf + 1 when vseed).
        self.r0_vseed = bool(r0_vseed) and bool(enable_r0) and self.M_thr > 0
        self.M_qf = self.M_thr
        if self.r0_vseed:
            self.M_thr = self.M_qf + 1
        # need[m] = ceil(q_m * K) prev-topK values >= rung m.
        self.qneeds = tuple(
            max(1, int(math.ceil(q * self.top_k))) for q in self.r0_qfracs
        )
        self.adaptive_qneeds = tuple(
            max(1, int(math.ceil(q * self.top_k))) for q in (0.35, 0.05, 0.01)
        )
        # R1 inline shot aim in log2-count space: geometric center of the
        # [K, kC] acceptance window.
        self.log2_r1aim = (
            math.log2(math.sqrt(self.top_k * self.kC)) if self.r0_qfracs else 0.0
        )
        # fb_fix interior aim (HLS grid optimum): log2(K * (kC/K)**fb_alpha).
        self.log2_mstar = (
            math.log2(self.top_k * (self.kC / self.top_k) ** float(fb_alpha))
            if self.r0_qfracs
            else 0.0
        )

        # --- P4 fused rank-and-scatter (inert until enable_p4_rank_scatter) ---
        # Replaces phase4_histogram_snap's k-th-bin search + 2-pass writeback
        # with a single rank-and-scatter pass (PR#15709), cutting Phase-4
        # barriers ~14 -> ~7. On a latency-bound kernel that is a whole-kernel
        # win (~1.078x, HW-invariant). enable_p4_rank_scatter_exact adds ONE
        # fine-histogram recursion on the straddling coarse bin so the result is
        # bit-exact vs torch.topk (adds a few barriers back but still < snap).
        # Default ON with R0: measured over the 4k-1M BS=1 best/worst envelope
        # gives geomean ~1.09x (K1024 1.12 / K2048 1.12 / K512 1.05) with NO
        # cell regressing >2%. Resolves to OFF when enable_r0 is False, so the
        # base kernel stays byte-identical to upstream.
        if enable_p4_rank_scatter is None:
            enable_p4_rank_scatter = bool(enable_r0)
        if enable_p4_rank_scatter_exact is None:
            enable_p4_rank_scatter_exact = bool(enable_p4_rank_scatter)
        self.enable_p4_rank_scatter = bool(enable_p4_rank_scatter)
        self.enable_p4_rank_scatter_exact = bool(enable_p4_rank_scatter_exact)
        # p4_exact_tail: ambiguity-gated exact tie-resolution for the fine
        # straddling bin (see phase4_rank_scatter). The fine recursion
        # resolves values to range/(kNumBins*256) — WINDOW-RELATIVE, so ANY
        # dtype (fp32 or upconverted 16-bit) can leave distinct values in
        # one fine bin whenever the Phase-2 bracket is wide relative to the
        # boundary-local ULP (e.g. fp16 1.0 vs 1.25 under a [0, 65504]
        # bracket); values straddling the kK boundary inside one fine bin
        # were previously picked in arrival order (observed as |miss|=1
        # with |dv| ~ 3e-6 on real fp32 captures). The tail radix re-ranks
        # on the full fp32 order key — candidate keys are ALWAYS fp32
        # (16-bit inputs are upcast injectively at collect), so the repair
        # is exact for every supported dtype WHEN ENABLED. Default ON for
        # fp32 only: on fp32 the gate fires rarely and the fix is ~free,
        # but 16-bit quantization puts value plateaus at the boundary on
        # virtually every input, so the gate fires constantly — measured
        # B200 envelope cost (bf16, K512/K1024 x 16k-262k x BS 1-256,
        # same-process paired) is gm 1.29-1.36x, worst 2.27x, while typical
        # bf16 inputs (randn and quantized-tie, 48 paired runs) are already
        # value-exact without it: 16-bit misses need an adversarially wide
        # Phase-2 bracket (distinct 16-bit values inside one fine bin).
        # 16-bit callers that need the guarantee opt in via the knob (see
        # test_cute_dsl_gvr_topk_decode_p4_exact_tail_16bit).
        if p4_exact_tail is None:
            p4_exact_tail = (
                self.enable_p4_rank_scatter_exact and dtype == cutlass.Float32
            )
        self.p4_exact_tail = bool(p4_exact_tail) and self.enable_p4_rank_scatter_exact
        # [p4tt] p4_tail_fast: tiny-tie COLLECT+SELECT fast path inside the
        # exact-tail fire branch. When the (b*, sb*) tie class holds <= 128
        # entries (the real firing cells have 2), ONE candidate pass collects
        # (value_bits, cand_idx) pairs into SMEM and thread0 selects the
        # top-need exactly, replacing the 4 unconditional radix passes
        # (~5.3us -> ~1 pass on pro/512k). Larger tie classes fall through to
        # the existing radix select. Pure optimization (the radix backstop
        # keeps exactness identical either way); False compiles the original
        # text (byte-identical PTX modulo kernel name) for A/B.
        # Default gate = p4_exact_tail AND top_k >= 1024: the non-firing
        # codegen tax concentrates at K512 cs=1 mid-N (flash 64k/128k
        # -6.6/-9.1%, cross-GPU reproducible) while the
        # fire census (pro/512k bench + 9 per-layer fixture cells) contains
        # NO K512 cell — so K512 keeps the original byte-identical kernel.
        if p4_tail_fast is None:  # [p4tt]
            p4_tail_fast = self.p4_exact_tail and top_k >= 1024
        self.p4_tail_fast = bool(p4_tail_fast) and self.p4_exact_tail  # [p4tt]

    # ------------------------------------------------------------------
    # SMEM slice cache loader. Streams this CTA's slice GMEM → SMEM via
    # LDG → STS so Phase 2/3 can read LDS instead of re-streaming GMEM.
    # Iteration pattern mirrors block_count_ge's scan so the SMEM layout
    # is naturally aligned for subsequent LDS reads
    # (smem_input[i] == input_row[slice_start + i]).
    # ------------------------------------------------------------------
    @cute.jit
    def load_slice_to_smem(
        self,
        input_row,
        slice_start,
        slice_end,
        smem_input,
        tidx,
    ):
        num_threads = cutlass.const_expr(self.num_threads)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        step_elem = cutlass.const_expr(num_threads * vec_w)

        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        smem_addr = smem_input.iterator.toint()

        slice_len = slice_end - slice_start
        i_local = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)
        n_aligned_local = (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)

        # Vectorized GMEM→SMEM load via 4-way LDG.E.* unroll, mirroring
        # block_count_ge's fast path. ic_local indexes both GMEM
        # (input_row[slice_start + ic_local]) and SMEM (smem_input[ic_local]).
        if self.enable_unroll_4:
            rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if slice_len > i_local + cutlass.Int32(vec_w - 1):
                big_iters = (
                    slice_len - i_local - cutlass.Int32(vec_w)
                ) // cutlass.Int32(step_elem) + cutlass.Int32(1)
            for k in cutlass.range(big_iters, unroll=4):
                ic_local = i_local + k * cutlass.Int32(step_elem)
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr
                    + cutlass.Int64(slice_start + ic_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src, rng_frag)
                dst_ptr = cute.make_ptr(
                    self.dtype,
                    smem_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=vec_align,
                )
                dst = cute.make_tensor(dst_ptr, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, rng_frag, dst)
            i_local = i_local + big_iters * cutlass.Int32(step_elem)

        # 1-way tail vec loop (slice_len mod step_elem residual).
        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while i_local + cutlass.Int32(vec_w - 1) < slice_len:
            src_ptr = cute.make_ptr(
                self.dtype,
                row_addr
                + cutlass.Int64(slice_start + i_local) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem,
                assumed_align=vec_align,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            dst_ptr = cute.make_ptr(
                self.dtype,
                smem_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.smem,
                assumed_align=vec_align,
            )
            dst = cute.make_tensor(dst_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, tail_frag, dst)
            i_local = i_local + step

        # Scalar tail (slice_len % vec_w). Each thread strides by num_threads.
        it_local = n_aligned_local + tidx
        while it_local < slice_len:
            smem_input[it_local] = input_row[slice_start + it_local]
            it_local = it_local + cutlass.Int32(num_threads)

        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Build a vectorized copy atom for the input scan loops. With
    # use_constant_hint=True we use CopyG2ROp+invariant to get
    # xxx.E.*.CONSTANT (read-only cache, matches CUDA __ldg). Defined as
    # a plain Python method (not @cute.jit) so the if-else branches both
    # bind copy_atom in the same trace scope.
    # ------------------------------------------------------------------
    def _make_load_copy_atom(self):
        # num_bits_per_copy matches self.vec_bits (128 default; 256 when
        # use_256bit_load=True).
        if self.use_constant_hint:
            return cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                self.dtype,
                num_bits_per_copy=self.vec_bits,
                invariant=True,
            )
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.vec_bits,
        )

    # ------------------------------------------------------------------
    # Input load helper — casts to fp32 regardless of self.dtype.
    # ------------------------------------------------------------------
    @cute.jit
    def _load_fp32(self, ptr_view, idx):
        # TODO: instructions?
        v = ptr_view[idx]
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            return v
        else:
            return cutlass.Float32(v)

    # ------------------------------------------------------------------
    # Warp-level reductions
    #
    # ------------------------------------------------------------------
    @cute.jit
    def warp_reduce_sum_i32(self, val):
        # REDUX.SYNC.ADD.S32 (sm_80+)
        return cute.arch.warp_redux_sync(val, "add")

    @cute.jit
    def warp_reduce_sum_f32(self, val):
        # PTX redux.sync has no fadd.
        # will lower to SHFL.BFLY 5-step tree.
        return cute.arch.warp_reduction_sum(val)

    @cute.jit
    def warp_reduce_min_f32(self, val):
        # PTX redux.sync.fmin.f32 (sm_100).
        return cute.arch.warp_redux_sync(val, "fmin")

    @cute.jit
    def warp_reduce_max_f32(self, val):
        # PTX redux.sync.fmax.f32 (sm_100).
        return cute.arch.warp_redux_sync(val, "fmax")

    # ------------------------------------------------------------------
    # Raw-address SMEM scalar access through a pre-hoisted window base.
    #
    # Tensor-indexed SMEM access (smem_keys[i]) makes the compiler
    # re-derive the cluster SMEM window per access (S2R SR_CgaCtaId +
    # LEA<<24) — ncu shows this as the top single-instruction stall in
    # the P3 stream-write and P4 snap loops. Hoisting the base once via
    # iterator.toint() (one S2R per call site) turns every subsequent
    # access into plain integer addressing — the same pattern the P2
    # scan loops already use for smem_input, whose SASS regions show no
    # S2R at all.
    # ------------------------------------------------------------------
    @cute.jit
    def _smem_ref(self, dtype: cutlass.Constexpr, base_addr, idx):
        elem_bytes = cutlass.const_expr(dtype.width // 8)
        p = cute.make_ptr(
            dtype,
            base_addr + cutlass.Int64(idx) * cutlass.Int64(elem_bytes),
            cute.AddressSpace.smem,
            assumed_align=4,
        )
        return cute.make_tensor(p, cute.make_layout((1,)))

    @cute.jit
    def _smem_ld(self, dtype: cutlass.Constexpr, base_addr, idx):
        return self._smem_ref(dtype, base_addr, idx)[0]

    @cute.jit
    def _smem_st(self, dtype: cutlass.Constexpr, base_addr, idx, val):
        t = self._smem_ref(dtype, base_addr, idx)
        t[0] = val

    # ------------------------------------------------------------------
    # Phase 1: preIdx Min/Max/Mean -> initial threshold
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_preidx_stats(
        self,
        input_row,  # cute.Tensor [N] fp32 (post-cast for half-prec)
        N,  # length of input_row
        pre_idx_row,  # cute.Tensor [M] int32
        pre_idx_count,
        pre_idx_offset,
        smem_wmin_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wmax_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wsum_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wcnt_i32,  # cute.Tensor [NUM_WARPS] int32
        s_thr,  # cute.Tensor [3] float32: [threshold, val_lo, val_hi]
        s_iscalars,  # cute.Tensor [6] int32: [cand_count, done, cnt_lo, cnt_hi, out_count, local_cand_count]
        tidx,
        warp_id,
        lane,
        smem_gath=None,  # cute.Tensor [top_k] f32 or None (p1b_cache): stash
        # the gathered value per preIdx slot so P1b skips a 2nd GMEM gather.
        s_mt_thr=None,  # r0_vseed: P1 also parks pmean in the last rung
        # column (visibility via P1's own trailing barrier -> zero extra sync).
        use_cold_hints=False,
        cold_prior_len=0,
    ):
        """preIdx scan + warp reduce + block aggregate + initial threshold.

        Smem layout split: floats kept in fp32 buffers, ints kept in int32
        buffers (no bit-cast tricks needed — avoids ArithValue/ir_value
        coupling and keeps types clean for the MLIR codegen).
        """
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        local_sum = cutlass.Float32(0.0)
        local_cnt = cutlass.Int32(0)

        # Stride loop over preIdx with pre_idx_offset shift. pre_idx_count
        # is compile-time (= top_k). Two cases:
        #   K >= num_threads: every thread loads ≥1 preIdx; fully unrolled
        #     over n_iters = K // num_threads.
        #   K <  num_threads: only the first K threads load (guard below);
        #     remaining threads contribute identity values.
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                raw = pre_idx_row[i]
                if use_cold_hints:
                    raw = i * cold_prior_len // cutlass.Int32(self.top_k)
                idx = raw + pre_idx_offset
                if cutlass.const_expr(smem_gath is not None):
                    smem_gath[i] = cutlass.Float32(self.NEG_FLT_MAX)
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
                    if cutlass.const_expr(smem_gath is not None):
                        smem_gath[i] = v
                    local_max = cute.arch.fmax(local_max, v)
                    local_min = _fmin_f32_inline(local_min, v)
                    local_sum = local_sum + v
                    local_cnt = local_cnt + 1
        else:
            # K < num_threads — only first K threads load a preIdx.
            # cute DSL requires variables to exist before dynamic `if` blocks,
            # so predeclare `idx` with an out-of-range sentinel and update
            # it conditionally; the downstream `if idx >= 0 and idx < N`
            # gate handles the sentinel naturally.
            idx = cutlass.Int32(-1)
            if tidx < cutlass.Int32(pre_idx_count):
                raw = pre_idx_row[tidx]
                if use_cold_hints:
                    raw = tidx * cold_prior_len // cutlass.Int32(self.top_k)
                idx = raw + pre_idx_offset
                if cutlass.const_expr(smem_gath is not None):
                    smem_gath[tidx] = cutlass.Float32(self.NEG_FLT_MAX)
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                if cutlass.const_expr(smem_gath is not None):
                    smem_gath[tidx] = v
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1

        # Warp-level reductions + smem write. When K < num_threads only
        # the first ``active_preidx_warps`` warps have real data — skip
        # the rest to save ~30 cy/warp. K ∈ {512, 1024, 2048} divides
        # evenly by WARP_SIZE, so the clamp to num_warps just handles
        # K > num_threads (avoids OOB into smem[num_warps]).
        active_preidx_warps = cutlass.const_expr(
            min(pre_idx_count // self.WARP_SIZE, self.num_warps)
        )
        if cutlass.const_expr(active_preidx_warps < self.num_warps):
            if warp_id < cutlass.Int32(active_preidx_warps):
                wmin = self.warp_reduce_min_f32(local_min)
                wmax = self.warp_reduce_max_f32(local_max)
                wsum = self.warp_reduce_sum_f32(local_sum)
                wcnt = self.warp_reduce_sum_i32(local_cnt)
                if lane == 0:
                    smem_wmin_f32[warp_id] = wmin
                    smem_wmax_f32[warp_id] = wmax
                    smem_wsum_f32[warp_id] = wsum
                    smem_wcnt_i32[warp_id] = wcnt
        else:
            wmin = self.warp_reduce_min_f32(local_min)
            wmax = self.warp_reduce_max_f32(local_max)
            wsum = self.warp_reduce_sum_f32(local_sum)
            wcnt = self.warp_reduce_sum_i32(local_cnt)
            if lane == 0:
                smem_wmin_f32[warp_id] = wmin
                smem_wmax_f32[warp_id] = wmax
                smem_wsum_f32[warp_id] = wsum
                smem_wcnt_i32[warp_id] = wcnt
        cute.arch.barrier()

        # Block aggregate: 4 reductions across num_warps slots. Warp-parallel
        # path is gated by enable_warp_parallel_reduce (auto-on at 32 warps,
        # off at 16 warps — see __init__).
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # Warp-parallel 4-way reduce in warp 0. Only the first
            # `active_preidx_warps` slots are read (dummy warps skipped).
            if warp_id == cutlass.Int32(0):
                v_min = cutlass.Float32(self.FLT_MAX)
                v_max = cutlass.Float32(self.NEG_FLT_MAX)
                v_sum = cutlass.Float32(0.0)
                v_cnt = cutlass.Int32(0)
                if lane < cutlass.Int32(active_preidx_warps):
                    v_min = smem_wmin_f32[lane]
                    v_max = smem_wmax_f32[lane]
                    v_sum = smem_wsum_f32[lane]
                    v_cnt = smem_wcnt_i32[lane]
                pmin = self.warp_reduce_min_f32(v_min)
                pmax = self.warp_reduce_max_f32(v_max)
                psum = self.warp_reduce_sum_f32(v_sum)
                pcnt = self.warp_reduce_sum_i32(v_cnt)
                if lane == cutlass.Int32(0):
                    pmean = cutlass.Float32(0.0)
                    if pcnt > 0:
                        pmean = psum / cutlass.Float32(pcnt)
                    else:
                        pmean = (pmin + pmax) * cutlass.Float32(0.5)
                    cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
                    s_thr[0] = pmean
                    if cutlass.const_expr(self.r0_vseed):
                        s_mt_thr[self.M_thr - 1] = pmean
                    s_thr[1] = pmin
                    s_thr[2] = pmax
                    s_iscalars[0] = cutlass.Int32(0)  # cand_count
                    s_iscalars[1] = cutlass.Int32(0)  # done
                    s_iscalars[2] = cutlass.Int32(cnt_lo_seed)  # cnt_lo
                    s_iscalars[3] = cutlass.Int32(1)  # cnt_hi
                    s_iscalars[4] = cutlass.Int32(0)  # out_count
        else:
            # tid==0 serial loop.
            if tidx == 0:
                pmin = cutlass.Float32(self.FLT_MAX)
                pmax = cutlass.Float32(self.NEG_FLT_MAX)
                psum = cutlass.Float32(0.0)
                pcnt = cutlass.Int32(0)
                # Iterate over active_preidx_warps (= num_warps when K >=
                # num_threads; smaller when K < num_threads since dummy warps
                # above no longer write smem).
                for w in cutlass.range_constexpr(active_preidx_warps):
                    v_min = smem_wmin_f32[w]
                    v_max = smem_wmax_f32[w]
                    v_sum = smem_wsum_f32[w]
                    v_cnt = smem_wcnt_i32[w]
                    pmax = cute.arch.fmax(pmax, v_max)
                    pmin = _fmin_f32_inline(pmin, v_min)
                    psum = psum + v_sum
                    pcnt = pcnt + v_cnt

                pmean = cutlass.Float32(0.0)
                if pcnt > 0:
                    pmean = psum / cutlass.Float32(pcnt)
                else:
                    pmean = (pmin + pmax) * cutlass.Float32(0.5)

                cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
                s_thr[0] = pmean
                if cutlass.const_expr(self.r0_vseed):
                    s_mt_thr[self.M_thr - 1] = pmean
                s_thr[1] = pmin
                s_thr[2] = pmax
                s_iscalars[0] = cutlass.Int32(0)
                s_iscalars[1] = cutlass.Int32(0)
                s_iscalars[2] = cutlass.Int32(cnt_lo_seed)
                s_iscalars[3] = cutlass.Int32(1)
                s_iscalars[4] = cutlass.Int32(0)
        cute.arch.barrier()

    @cute.jit
    def phase1_full_row_bounds(
        self,
        input_row,
        N,
        smem_wmin_f32,
        smem_wmax_f32,
        s_thr,
        s_iscalars,
        tidx,
        warp_id,
        lane,
    ):
        """Recover a sound threshold bracket by scanning the complete row."""
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        i = tidx
        while i < N:
            value = self._load_fp32(input_row, i)
            local_min = _fmin_f32_inline(local_min, value)
            local_max = cute.arch.fmax(local_max, value)
            i = i + cutlass.Int32(self.num_threads)

        warp_min = self.warp_reduce_min_f32(local_min)
        warp_max = self.warp_reduce_max_f32(local_max)
        if lane == cutlass.Int32(0):
            smem_wmin_f32[warp_id] = warp_min
            smem_wmax_f32[warp_id] = warp_max
        cute.arch.barrier()

        if tidx == cutlass.Int32(0):
            row_min = cutlass.Float32(self.FLT_MAX)
            row_max = cutlass.Float32(self.NEG_FLT_MAX)
            for w in cutlass.range_constexpr(self.num_warps):
                row_min = _fmin_f32_inline(row_min, smem_wmin_f32[w])
                row_max = cute.arch.fmax(row_max, smem_wmax_f32[w])
            s_thr[0] = row_min * cutlass.Float32(0.5) + row_max * cutlass.Float32(0.5)
            s_thr[1] = row_min
            s_thr[2] = row_max
            s_iscalars[0] = cutlass.Int32(0)
            s_iscalars[1] = cutlass.Int32(0)
            s_iscalars[2] = N
            s_iscalars[3] = cutlass.Int32(1)
            s_iscalars[4] = cutlass.Int32(0)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1b — 256-bin SMEM histogram over the prev-topK gathered values
    # (band [v_lo, v_hi] = P1's pmin/pmax = s_thr[1]/s_thr[2]), then M
    # h-space quantile rungs into s_mt_thr (ascending value order). Reuses
    # the Phase-4 smem_hist buffer (kNumBins >= 512 >= 256 in every spec;
    # Phase 4 re-zeroes it later). Provides the R0 admission placement; it
    # is only invoked from the enable_r0 path (added in a follow-up commit),
    # so the base kernel is unaffected.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs(
        self,
        input_row,
        N,
        pre_idx_row,
        pre_idx_count,
        pre_idx_offset,
        smem_hist,
        s_thr,
        s_mt_thr,
        tidx,
        warp_id,
        lane,
        use_cold_hints=False,
        cold_prior_len=0,
    ):
        M = cutlass.const_expr(self.M_qf)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)  # NB / WARP_SIZE bins per lane
        num_threads = cutlass.const_expr(self.num_threads)
        use_adaptive_rungs = (
            N >= cutlass.Int32(32768) and N < cutlass.Int32(65536)
        ) or N >= cutlass.Int32(131072)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)  # caller guards v_hi > v_lo
        inv_w = cutlass.Float32(1.0) / width

        ig = tidx
        while ig < cutlass.Int32(pre_idx_count):
            raw = pre_idx_row[ig]
            if use_cold_hints:
                raw = ig * cold_prior_len // cutlass.Int32(self.top_k)
            idx = raw + pre_idx_offset
            if idx >= cutlass.Int32(0) and idx < N:
                v = cutlass.Float32(input_row[idx])
                bf = (v - v_lo) * inv_w
                b = cutlass.Int32(bf)
                if b < cutlass.Int32(0):
                    b = cutlass.Int32(0)
                if b > cutlass.Int32(NB - 1):
                    b = cutlass.Int32(NB - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
            ig = ig + cutlass.Int32(num_threads)
        cute.arch.barrier()

        # Warp-0-parallel rung extraction (a tid0 256-bin serial walk is a
        # ~10-15us per-CTA dependency chain). Lane l owns the SEG consecutive
        # bins descending from bin NB-1-l*SEG; segment sums -> 5-step shfl_up
        # inclusive scan gives each lane the cumulative count of all
        # higher-value bins; each lane then walks its SEG bins once and fires
        # rung m at the unique crossing bin (cum_before < qneeds[m] <=
        # cum_at). qfracs descending in h => thresholds ascending in m.
        if warp_id == cutlass.Int32(0):
            top = cutlass.Int32(NB - 1) - lane * cutlass.Int32(SEG)
            seg_frag = cute.make_rmem_tensor((SEG,), cutlass.Int32)
            part = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                v8 = smem_hist[top - cutlass.Int32(j)]
                seg_frag[j] = v8
                part = part + v8
            tp = part
            for off_i in cutlass.range_constexpr(5):
                off_v = cutlass.const_expr(1 << off_i)
                other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
                if lane >= cutlass.Int32(off_v):
                    tp = tp + other
            excl = tp - part  # cum of all bins above my segment
            total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
            run = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                run = run + seg_frag[j]
                cum_at = excl + run
                cum_before = cum_at - seg_frag[j]
                for m in cutlass.range_constexpr(M):
                    need = cutlass.Int32(self.qneeds[m])
                    if cutlass.const_expr(self.adaptive_rungs):
                        if use_adaptive_rungs:
                            need = cutlass.Int32(self.adaptive_qneeds[m])
                    if cum_at >= need and cum_before < need:
                        s_mt_thr[m] = (
                            v_lo + cutlass.Float32(top - cutlass.Int32(j)) * width
                        )
            # unfired rungs (heavy invalid-preIdx rows: total < need): v_lo
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    need = cutlass.Int32(self.qneeds[m])
                    if cutlass.const_expr(self.adaptive_rungs):
                        if use_adaptive_rungs:
                            need = cutlass.Int32(self.adaptive_qneeds[m])
                    if total < need:
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1b (p1b_cache variant) — build the rung histogram from the SMEM
    # gathered values that P1 stashed (smem_gath), skipping P1b's second
    # GMEM random gather. Sentinel NEG_FLT_MAX marks invalid/out-of-range
    # preIdx slots. Rung extraction is identical to phase1b_hspace_rungs.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs_cached(
        self,
        N,
        pre_idx_count,
        smem_gath,
        smem_hist,
        s_thr,
        s_mt_thr,
        tidx,
        warp_id,
        lane,
    ):
        M = cutlass.const_expr(self.M_qf)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)
        num_threads = cutlass.const_expr(self.num_threads)
        use_adaptive_rungs = (
            N >= cutlass.Int32(32768) and N < cutlass.Int32(65536)
        ) or N >= cutlass.Int32(131072)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)
        inv_w = cutlass.Float32(1.0) / width

        ig = tidx
        while ig < cutlass.Int32(pre_idx_count):
            v = smem_gath[ig]
            if v > cutlass.Float32(self.NEG_FLT_MAX):
                bf = (v - v_lo) * inv_w
                b = cutlass.Int32(bf)
                if b < cutlass.Int32(0):
                    b = cutlass.Int32(0)
                if b > cutlass.Int32(NB - 1):
                    b = cutlass.Int32(NB - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
            ig = ig + cutlass.Int32(num_threads)
        cute.arch.barrier()

        if warp_id == cutlass.Int32(0):
            top = cutlass.Int32(NB - 1) - lane * cutlass.Int32(SEG)
            seg_frag = cute.make_rmem_tensor((SEG,), cutlass.Int32)
            part = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                v8 = smem_hist[top - cutlass.Int32(j)]
                seg_frag[j] = v8
                part = part + v8
            tp = part
            for off_i in cutlass.range_constexpr(5):
                off_v = cutlass.const_expr(1 << off_i)
                other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
                if lane >= cutlass.Int32(off_v):
                    tp = tp + other
            excl = tp - part
            total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
            run = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                run = run + seg_frag[j]
                cum_at = excl + run
                cum_before = cum_at - seg_frag[j]
                for m in cutlass.range_constexpr(M):
                    need = cutlass.Int32(self.qneeds[m])
                    if cutlass.const_expr(self.adaptive_rungs):
                        if use_adaptive_rungs:
                            need = cutlass.Int32(self.adaptive_qneeds[m])
                    if cum_at >= need and cum_before < need:
                        s_mt_thr[m] = (
                            v_lo + cutlass.Float32(top - cutlass.Int32(j)) * width
                        )
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    need = cutlass.Int32(self.qneeds[m])
                    if cutlass.const_expr(self.adaptive_rungs):
                        if use_adaptive_rungs:
                            need = cutlass.Int32(self.adaptive_qneeds[m])
                    if total < need:
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_count_ge — GE-count of input vs threshold (shared by P2/P3).
    # Per-thread strided accumulate → smem_ptcnt[tid] (for P3 prefix sum)
    # → warp reduce → block reduce → s_iscalars[0] = cand_count.
    # Optionally DSMEM-aggregates across the cluster.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge(
        self,
        input_row,  # cute.Tensor [N] fp32 (full row; this CTA only scans its slice)
        slice_start,  # int32: index in input_row where this CTA's slice starts
        slice_end,  # int32: index in input_row where this CTA's slice ends
        threshold,  # cutlass.Float32 scalar
        smem_ptcnt,  # cute.Tensor [BLOCK_SIZE] int32 (P3 cache)
        smem_wcnt,  # cute.Tensor [NUM_WARPS] int32 (block reduce scratch)
        s_iscalars,  # cute.Tensor [6] int32 (writes [0] = cand_count)
        s_cluster_partial,  # cute.Tensor [1] int32 (per-CTA partial scratch for DSMEM)
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = skip DSMEM aggregation (cs=1 / short-row degrade)
        smem_input=None,  # optional SMEM-cached slice (smem_input[i] == input_row[slice_start+i])
        redundant=False,  # trace-time: every-warp reduce, return the total
        wcnt_off=None,  # int32 staging bank offset into smem_wcnt (parity)
    ):
        """Count input[i] >= threshold across this CTA's row slice, then
        DSMEM-aggregate across the cluster.

        ``redundant=True`` (p2_warp_redundant, cluster_size == 1 only):
        after the staging barrier EVERY warp reduces the warp counts
        lane-parallel and the block total RETURNS in a register —
        bit-identical across warps — instead of a leader writing
        s_iscalars[0] for a barrier-published broadcast. ``wcnt_off``
        parity-banks the smem_wcnt staging so a warp that has moved on
        to the next Phase-2 round cannot clobber a slot a slower warp is
        still reading (the per-round staging barrier bounds the drift to
        one round).

        Vectorized scan: each thread loads vec_w elements per iter (128 or
        256 bits) over ``input_row[slice_start : slice_end)``; scalar tail
        handles the remainder.

        Cluster aggregation (cluster_size > 1): every CTA stages its
        slice-local count into ``s_cluster_partial[call & 1]`` (parity
        double-buffer; slot 2 is the tid0-private call counter), syncs the
        cluster, then DSMEM-reads every peer's slot and sums into
        ``s_iscalars[0]``.
        After this every CTA's ``s_iscalars[0]`` holds the same
        cluster-wide cand_count, so Phase 2's secant update stays a
        leader-only scalar op on a value all CTAs agree on.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        cluster_size = cutlass.const_expr(self.cluster_size)
        c = cutlass.Int32(0)
        copy_atom = self._make_load_copy_atom()

        step_elem = cutlass.const_expr(num_threads * vec_w)

        row_addr = input_row.iterator.toint()
        slice_len = slice_end - slice_start
        # smem-cache path uses slice-LOCAL indices (smem_input[0] ==
        # input_row[slice_start]); GMEM path uses global indices. Set up
        # both upfront so the const_expr branches below stay flat.
        if cutlass.const_expr(smem_input is not None):
            smem_addr = smem_input.iterator.toint()
            n_aligned = (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
            N = slice_len  # upper bound is slice-local
            i = tidx * cutlass.Int32(vec_w)
        else:
            n_aligned = slice_start + (
                slice_len // cutlass.Int32(vec_w)
            ) * cutlass.Int32(vec_w)
            N = slice_end  # global upper bound
            i = slice_start + tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Fast path: 4-way unroll for LSU-pipelining ILP.
        # Each iter loads 1 vec_w chunk; LLVM unrolls 4 iters at IR level
        # so 4 LDG.E.* stay in flight.
        if self.enable_unroll_4:
            rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
            # Number of complete vec_w-aligned loads this thread can do:
            #   need: i + k*step_elem + (vec_w - 1) < N
            #   max k: floor((N - i - vec_w) / step_elem)
            #   N_iters = max_k + 1
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(
                    step_elem
                ) + cutlass.Int32(1)

            for k in cutlass.range(big_iters, unroll=4):
                i_local = i + k * cutlass.Int32(step_elem)
                if cutlass.const_expr(smem_input is not None):
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        smem_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.smem,
                        assumed_align=vec_align,
                    )
                else:
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=vec_align,
                    )
                src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src_k, rng_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = rng_frag[j]
                    else:
                        vj = cutlass.Float32(rng_frag[j])
                    if vj >= threshold:
                        c = c + cutlass.Int32(1)
            # Advance i past all consumed vec_w-aligned positions so the
            # medium/tail loops below correctly skip (they check i + ... < N).
            i = i + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step (= remaining 1
        # full vec_w-stride or less). i is always vec_w-aligned here (it
        # advanced by multiples of step_elem = num_threads*vec_w), so the
        # same vec_align bytes hold.
        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
            if cutlass.const_expr(smem_input is not None):
                src_ptr = cute.make_ptr(
                    self.dtype,
                    smem_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=vec_align,
                )
            else:
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= threshold:
                    c = c + cutlass.Int32(1)
            i = i + step

        # Tail scalar loop. SMEM path uses slice-local indexing
        # (smem_input[it]); GMEM path uses global indices (input_row[it]).
        it = n_aligned + tidx
        while it < N:
            if cutlass.const_expr(smem_input is not None):
                v = smem_input[it]
                if cutlass.const_expr(self.dtype != cutlass.Float32):
                    v = cutlass.Float32(v)
            else:
                v = self._load_fp32(input_row, it)
            if v >= threshold:
                c = c + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)

        # Cache per-thread count for P3 retry-shrink reuse.
        smem_ptcnt[tidx] = c

        # Warp reduce + lane-0 write
        wc = self.warp_reduce_sum_i32(c)
        stage_base = cutlass.Int32(0)
        if cutlass.const_expr(wcnt_off is not None):
            stage_base = wcnt_off
        if lane == 0:
            smem_wcnt[stage_base + warp_id] = wc
        cute.arch.barrier()

        if cutlass.const_expr(redundant):
            # Every warp reduces the staged counts itself; no leader, no
            # publish barrier, no s_iscalars[0] round-trip.
            v_r = cutlass.Int32(0)
            if lane < cutlass.Int32(self.num_warps):
                v_r = smem_wcnt[stage_base + lane]
            total_r = self.warp_reduce_sum_i32(v_r)
            return total_r

        # Block aggregate (sum reduce over num_warps slots). No trailing
        # barrier: caller is expected to insert its own __syncthreads after
        # its post-processing of cand_count.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel sum reduce in warp 0.
            if warp_id == cutlass.Int32(0):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(self.num_warps):
                    v = smem_wcnt[lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == cutlass.Int32(0):
                    s_iscalars[0] = total
        else:
            # tid==0 serial sum.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    total = total + smem_wcnt[w]
                s_iscalars[0] = total

        # Snapshot local cand_count into s_iscalars[5] before the cluster
        # all-reduce overwrites s_iscalars[0]. Only needed when
        # do_cluster_sync=True: the DSMEM gather in Phase 4 reads peer
        # s_iscalars[5] values; skipped in short-row degrade (do_cluster_sync=False)
        # where s_iscalars[0] is never overwritten and the gather never fires.
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                if tidx == cutlass.Int32(0):
                    s_iscalars[5] = s_iscalars[0]
                cute.arch.barrier()

        # Cluster all-reduce of cand_count. Skipped at cluster_size==1.
        # Also skipped at runtime when do_cluster_sync=False (short-row
        # degrade): CTA 0 is the only live CTA in the cluster and its
        # local count IS the total, so s_iscalars[0] already holds the
        # correct value with no DSMEM read needed.
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                cute.arch.barrier()  # publish s_iscalars[0] to all threads of this CTA
                # Parity double-buffer: with a single slot, a straggler's
                # post-wait DSMEM read races the peer's next-call overwrite
                # (PTX-model data race). Writing call k into slot k&1 orders
                # the call-(k+2) overwrite after my call-k reads via the
                # call-(k+1) rendezvous. Slot 2 = tid0-private call counter
                # (zeroed per row); do_cluster_sync is row-uniform, so CTAs
                # step the counter in lockstep and parity stays aligned.
                par = cutlass.Int32(0)
                if tidx == cutlass.Int32(0):
                    par = s_cluster_partial[2]
                    s_cluster_partial[par & cutlass.Int32(1)] = s_iscalars[0]
                    s_cluster_partial[2] = par + cutlass.Int32(1)
                # Non-relaxed arrive: pairs with the peer cluster_wait acquire
                # to release s_cluster_partial writes so the DSMEM ld below
                # observes them. cluster_arrive_relaxed would skip the release
                # fence and risk stale peer reads on hardware that doesn't
                # eagerly publish shared writes.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if tidx == cutlass.Int32(0):
                    total = cutlass.Int32(0)
                    local_ptr = s_cluster_partial.iterator + (par & cutlass.Int32(1))
                    for peer in cutlass.range_constexpr(cluster_size):
                        peer_addr = mapa_shared_cluster(local_ptr, cutlass.Int32(peer))
                        total = total + ld_shared_cluster_i32(peer_addr)
                    s_iscalars[0] = total
                cute.arch.barrier()  # broadcast cluster total within this CTA

        return cutlass.Int32(0)

    # ------------------------------------------------------------------
    # block_count_ge_multi<M> — GE-count of the input row against M
    # thresholds in ONE vectorized scan, reusing block_count_ge's memory
    # path (same vec_w / 4-way-unroll / tail loops) with M static register
    # counters. Caches all M per-thread count columns in smem_ptcnt_multi so
    # the accepted rung's column seeds Phase 3 with zero rescan. This is the
    # R0 admission primitive (multi-threshold lineage); it is only invoked
    # from the enable_r0 path added in a later commit, so the base kernel is
    # unaffected. Slice + cluster form: each CTA scans [slice_start,
    # slice_end) and the M per-CTA totals are DSMEM all-reduced across the
    # cluster (cluster_size>1, do_cluster_sync) with a release cluster_arrive
    # mirroring block_count_ge; at cs==1 (or short-row degrade) the local
    # totals are the answer. smem_ptcnt_multi holds slice-local per-thread
    # columns (the accepted rung's column seeds Phase 3 per CTA).
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge_multi(
        self,
        input_row,
        slice_start,
        slice_end,
        s_mt_thr,
        smem_ptcnt_multi,
        smem_wcnt_multi,
        s_mt_cnt,
        s_cluster_partial_m,
        do_cluster_sync,
        tidx,
        warp_id,
        lane,
        smem_ptcnt=None,  # vseed: last column's per-thread counts land here
    ):
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        cluster_size = cutlass.const_expr(self.cluster_size)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        thr_frag = cute.make_rmem_tensor((M,), cutlass.Float32)
        cnt_frag = cute.make_rmem_tensor((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)

        row_addr = input_row.iterator.toint()
        slice_len = slice_end - slice_start
        n_aligned = slice_start + (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(
            vec_w
        )
        i = slice_start + tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if slice_end > i + cutlass.Int32(vec_w - 1):
                big_iters = (slice_end - i - cutlass.Int32(vec_w)) // cutlass.Int32(
                    step_elem
                ) + cutlass.Int32(1)
            for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                i_local = i + k * cutlass.Int32(step_elem)
                src_ptr_k = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src_k, rng_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = rng_frag[j]
                    else:
                        vj = cutlass.Float32(rng_frag[j])
                    for m in cutlass.range_constexpr(M):
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < slice_end:
            src_ptr = cute.make_ptr(
                self.dtype,
                row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem,
                assumed_align=vec_align,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < slice_end:
            v = self._load_fp32(input_row, it)
            for m in cutlass.range_constexpr(M):
                cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
            it = it + cutlass.Int32(num_threads)

        for m in cutlass.range_constexpr(M):
            if cutlass.const_expr(self.r0_vseed and m == self.M_qf):
                smem_ptcnt[tidx] = cnt_frag[m]
            else:
                smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]

        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        # Block-reduce the M warp counts to this CTA's slice totals. Stage
        # into DSMEM scratch at cs>1 (for the cluster merge below), else
        # write straight to s_mt_cnt.
        if warp_id == cutlass.Int32(0):
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(num_warps):
                    v = smem_wcnt_multi[m * num_warps + lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == cutlass.Int32(0):
                    if cutlass.const_expr(cluster_size > 1):
                        s_cluster_partial_m[m] = total
                    else:
                        s_mt_cnt[m] = total
        cute.arch.barrier()
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                # Release arrive (NOT relaxed): pairs with the peer
                # cluster_wait acquire so the staged M totals are visible
                # before any CTA reads them over DSMEM.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if tidx == cutlass.Int32(0):
                    local_ptr = s_cluster_partial_m.iterator
                    for m in cutlass.range_constexpr(M):
                        total = cutlass.Int32(0)
                        for peer in cutlass.range_constexpr(cluster_size):
                            peer_addr = mapa_shared_cluster(
                                local_ptr + cutlass.Int32(m), cutlass.Int32(peer)
                            )
                            total = total + ld_shared_cluster_i32(peer_addr)
                        s_mt_cnt[m] = total
                cute.arch.barrier()
            else:
                # short-row degrade: this CTA's local totals are the answer.
                if tidx == cutlass.Int32(0):
                    for m in cutlass.range_constexpr(M):
                        s_mt_cnt[m] = s_cluster_partial_m[m]
                cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase 2: Secant-interpolation threshold search
    # Refines threshold to bring cand_count into [kK, kCC] using secant
    # interpolation on (val_lo, cnt_lo) / (val_hi, cnt_hi). At most
    # self.MAX_REFINE_ITERS iterations.
    # ------------------------------------------------------------------
    @cute.jit
    def phase2_secant_search(
        self,
        input_row,
        N,
        slice_start,
        slice_end,
        smem_ptcnt,
        smem_wcnt,
        s_thr,  # [threshold, val_lo, val_hi]
        s_iscalars,  # [cand_count, done, cnt_lo, cnt_hi, out_count]
        s_cluster_partial,  # [3] int32 cluster scratch (parity slots + counter)
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = cs=1 / short-row degrade (skip cluster sync)
        smem_input=None,  # optional SMEM-cached slice
    ):
        """Refine s_thr[0] until cand_count lands in [kK, kCC].

        Each iter calls block_count_ge at the candidate threshold and
        updates the bracket (val_lo, val_hi, cnt_lo, cnt_hi). Sets
        s_iscalars[1] (done) = 1 on convergence, 2 on bracket exhaustion.
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)

        if cutlass.const_expr(self.p2_warp_redundant and self.cluster_size == 1):
            # ---- Redundant-warp cadence: ONE barrier per round ----
            # The whole secant state (threshold, bracket, counts, done)
            # lives in registers; every warp reduces the staged warp
            # counts itself (block_count_ge redundant mode) and replays
            # the identical classify + secant update, so the per-round
            # publish barriers and every s_thr/s_iscalars SMEM round-trip
            # (with its per-access cluster-window S2R recompute)
            # disappear. Canonical exit state is written once for P3.
            nwp2 = cutlass.const_expr(self.num_warps)
            thr_r = s_thr[0]
            vlo_r = s_thr[1]
            vhi_r = s_thr[2]
            clo_r = s_iscalars[2]
            chi_r = s_iscalars[3]
            done_r = cutlass.Int32(0)
            par_r = cutlass.Int32(0)
            cnt_r = self.block_count_ge(
                input_row,
                slice_start,
                slice_end,
                thr_r,
                smem_ptcnt,
                smem_wcnt,
                s_iscalars,
                s_cluster_partial,
                tidx,
                warp_id,
                lane,
                cutlass.Boolean(False),  # do_cluster_sync (cs==1 gate)
                smem_input=smem_input,
                redundant=True,
                wcnt_off=par_r * cutlass.Int32(nwp2),
            )
            if cnt_r >= cutlass.Int32(kK) and cnt_r <= cutlass.Int32(kCC):
                done_r = cutlass.Int32(1)
            elif cnt_r > cutlass.Int32(kCC):
                vlo_r = thr_r
                clo_r = cnt_r
            else:
                vhi_r = thr_r
                chi_r = cnt_r
            it = cutlass.Int32(0)
            while it < cutlass.Int32(self.MAX_REFINE_ITERS) and done_r == cutlass.Int32(
                0
            ):
                rng = vhi_r - vlo_r
                nv = cutlass.Float32(0.0)
                if clo_r > chi_r and rng > cutlass.Float32(1e-10):
                    f = cutlass.Float32(
                        clo_r - cutlass.Int32(kFTarget)
                    ) / cutlass.Float32(clo_r - chi_r)
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    if it == cutlass.Int32(0):
                        f = _fmin_f32_inline(f, cutlass.Float32(0.5))
                    nv = vlo_r + rng * f
                else:
                    nv = (vlo_r + vhi_r) * cutlass.Float32(0.5)
                if nv <= vlo_r:
                    nv = vlo_r + rng * cutlass.Float32(0.05)
                if nv >= vhi_r:
                    nv = vhi_r - rng * cutlass.Float32(0.05)
                if nv == vlo_r or nv == vhi_r:
                    nv = (vlo_r + vhi_r) * cutlass.Float32(0.5)
                    if nv == vlo_r or nv == vhi_r:
                        # ADJACENT-FLOAT bracket, same terminal as the
                        # leader path: a low side over the candidate
                        # buffer plus a high side under K means the
                        # boundary sits inside a bitwise-equal plateau
                        # wider than kC. Keep the sure-winner threshold
                        # and let Phase 4's plateau fill finish the row.
                        if clo_r > cutlass.Int32(kCC) and chi_r < cutlass.Int32(kK):
                            thr_r = vhi_r
                            done_r = cutlass.Int32(3)
                        else:
                            thr_r = vlo_r
                            done_r = cutlass.Int32(2)
                if done_r == cutlass.Int32(0):
                    thr_r = nv
                    par_r = par_r ^ cutlass.Int32(1)
                    cnt_r = self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        thr_r,
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        cutlass.Boolean(False),  # do_cluster_sync (cs==1 gate)
                        smem_input=smem_input,
                        redundant=True,
                        wcnt_off=par_r * cutlass.Int32(nwp2),
                    )
                    if cnt_r >= cutlass.Int32(kK) and cnt_r <= cutlass.Int32(kCC):
                        done_r = cutlass.Int32(1)
                    elif cnt_r > cutlass.Int32(kCC):
                        vlo_r = thr_r
                        clo_r = cnt_r
                    else:
                        vhi_r = thr_r
                        chi_r = cnt_r
                it = it + cutlass.Int32(1)
            # ---- Budget-exhausted plateau collapse (mirrors the leader
            # path): the refine budget can run out while the bracket is
            # still wide because a tie plateau wider than kC admits no
            # threshold. On exactly that signature, bisect to adjacent
            # floats so the plateau terminal is exact. Every thread
            # replays this from identical registers, so the branch stays
            # warp-uniform and block_count_ge keeps its barrier cadence.
            if (
                done_r == cutlass.Int32(0)
                and clo_r > cutlass.Int32(kCC)
                and chi_r >= cutlass.Int32(0)
                and chi_r < cutlass.Int32(kK)
            ):
                itc = cutlass.Int32(0)
                while itc < cutlass.Int32(64) and done_r == cutlass.Int32(0):
                    mid_c = (vlo_r + vhi_r) * cutlass.Float32(0.5)
                    if mid_c == vlo_r or mid_c == vhi_r:
                        thr_r = vhi_r
                        done_r = cutlass.Int32(3)
                    else:
                        thr_r = mid_c
                        par_r = par_r ^ cutlass.Int32(1)
                        cnt_r = self.block_count_ge(
                            input_row,
                            slice_start,
                            slice_end,
                            thr_r,
                            smem_ptcnt,
                            smem_wcnt,
                            s_iscalars,
                            s_cluster_partial,
                            tidx,
                            warp_id,
                            lane,
                            cutlass.Boolean(False),  # do_cluster_sync (cs==1)
                            smem_input=smem_input,
                            redundant=True,
                            wcnt_off=par_r * cutlass.Int32(nwp2),
                        )
                        if cnt_r >= cutlass.Int32(kK) and cnt_r <= cutlass.Int32(kCC):
                            done_r = cutlass.Int32(1)
                        elif cnt_r > cutlass.Int32(kCC):
                            vlo_r = thr_r
                            clo_r = cnt_r
                        else:
                            vhi_r = thr_r
                            chi_r = cnt_r
                    itc = itc + cutlass.Int32(1)
                if done_r == cutlass.Int32(3):
                    # recount at the terminal threshold so Phase 3 sees
                    # per-thread counts for the sure-winner set.
                    par_r = par_r ^ cutlass.Int32(1)
                    cnt_r = self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        thr_r,
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        cutlass.Boolean(False),  # do_cluster_sync (cs==1)
                        smem_input=smem_input,
                        redundant=True,
                        wcnt_off=par_r * cutlass.Int32(nwp2),
                    )
            if done_r == cutlass.Int32(0):
                if clo_r <= cutlass.Int32(kCC * 2):
                    thr_r = vlo_r
                else:
                    thr_r = vhi_r
                done_r = cutlass.Int32(2)
            # Canonical exit state for Phase 3/4 (byte-compatible with the
            # leader path), published once.
            if tidx == 0:
                s_thr[0] = thr_r
                s_thr[1] = vlo_r
                s_thr[2] = vhi_r
                s_iscalars[0] = cnt_r
                s_iscalars[1] = done_r
                s_iscalars[2] = clo_r
                s_iscalars[3] = chi_r
            cute.arch.barrier()
            return

        # ---- Initial count with the Phase-1 mean as threshold ----
        # TODO: smem_ptcnt is not always needed? only for the last block_count_ge.
        # Do we have methods to reduce its write?
        thr_init = s_thr[0]
        self.block_count_ge(
            input_row,
            slice_start,
            slice_end,
            thr_init,
            smem_ptcnt,
            smem_wcnt,
            s_iscalars,
            s_cluster_partial,
            tidx,
            warp_id,
            lane,
            smem_input=smem_input,
            do_cluster_sync=do_cluster_sync,
        )

        # tid==0 classifies the initial count.
        if tidx == 0:
            c0 = s_iscalars[0]
            t0 = s_thr[0]
            if c0 >= cutlass.Int32(kK) and c0 <= cutlass.Int32(kCC):
                s_iscalars[1] = cutlass.Int32(1)  # done = 1 (converged)
            elif c0 > cutlass.Int32(kCC):
                # too many → threshold is the new lower bound (search HIGHER)
                s_thr[1] = t0
                s_iscalars[2] = c0
            else:
                # too few → threshold is the new upper bound (search LOWER)
                s_thr[2] = t0
                s_iscalars[3] = c0
        cute.arch.barrier()

        # ---- Secant refinement loop ----
        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and s_iscalars[
            1
        ] == cutlass.Int32(0):
            # tid==0 computes new threshold via secant interpolation.
            if tidx == 0:
                vlo = s_thr[1]
                vhi = s_thr[2]
                clo = s_iscalars[2]
                chi = s_iscalars[3]
                rng = vhi - vlo
                nv = cutlass.Float32(0.0)
                if clo > chi and rng > cutlass.Float32(1e-10):
                    f = cutlass.Float32(
                        clo - cutlass.Int32(kFTarget)
                    ) / cutlass.Float32(clo - chi)
                    # clamp f to [0.05, 0.95]
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    if it == cutlass.Int32(0):
                        # iter 0: f = min(f, 0.5)  — runtime compare (matches CUDA)
                        f = _fmin_f32_inline(f, cutlass.Float32(0.5))
                    nv = vlo + rng * f
                else:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)

                # clamp nv into (vlo, vhi) range
                if nv <= vlo:
                    nv = vlo + rng * cutlass.Float32(0.05)
                if nv >= vhi:
                    nv = vhi - rng * cutlass.Float32(0.05)

                if nv == vlo or nv == vhi:
                    # Bracket exhausted — try midpoint, else terminal.
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
                        # ADJACENT-FLOAT bracket: every value in
                        # [vlo, vhi) is bitwise-equal to vlo. Low side
                        # overflowing the candidate buffer AND high side
                        # undershooting K means the boundary sits inside
                        # a bitwise-equal plateau wider than kC — record
                        # the plateau terminal (done = 3) and keep the
                        # sure-winner threshold vhi.
                        if clo > cutlass.Int32(kCC) and chi < cutlass.Int32(kK):
                            s_thr[0] = vhi
                            s_iscalars[1] = cutlass.Int32(3)  # done = 3 (plateau)
                        else:
                            s_thr[0] = vlo
                            s_iscalars[1] = cutlass.Int32(2)  # done = 2 (give up)
                    else:
                        s_thr[0] = nv
                else:
                    s_thr[0] = nv
            cute.arch.barrier()

            # Re-check done (tid==0 may have set it to 2)
            if s_iscalars[1] == cutlass.Int32(0):
                new_thr = s_thr[0]
                self.block_count_ge(
                    input_row,
                    slice_start,
                    slice_end,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    smem_input=smem_input,
                    do_cluster_sync=do_cluster_sync,
                )
                # tid==0 classifies the new count.
                if tidx == 0:
                    c_new = s_iscalars[0]
                    t_new = s_thr[0]
                    if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)
                    elif c_new > cutlass.Int32(kCC):
                        s_thr[1] = t_new
                        s_iscalars[2] = c_new
                    else:
                        s_thr[2] = t_new
                        s_iscalars[3] = c_new
                cute.arch.barrier()
            it = it + cutlass.Int32(1)

        # ---- Budget-exhausted plateau collapse ----
        # The refine budget can run out while the bracket is still wide: the
        # secant step keeps making progress (the bracket shrinks every
        # iteration) but a tie plateau wider than kC admits no threshold, so
        # the count never lands in [kK, kCC]. In exactly that signature -
        # count(>= v_lo) > kCC AND count(>= v_hi) < kK, both counts current -
        # collapse the bracket by pure bisection until the ends are ADJACENT
        # floats; every value in [v_lo, v_hi) is then bitwise-equal, so the
        # plateau terminal (done = 3) is exact and Phase 4 completes the row
        # from that tie class. A count landing in [kK, kCC] mid-collapse
        # converges normally. Anything else keeps the legacy give-up below.
        if (
            s_iscalars[1] == cutlass.Int32(0)
            and s_iscalars[2] > cutlass.Int32(kCC)
            and s_iscalars[3] >= cutlass.Int32(0)
            and s_iscalars[3] < cutlass.Int32(kK)
        ):
            itc = cutlass.Int32(0)
            while itc < cutlass.Int32(64) and s_iscalars[1] == cutlass.Int32(0):
                if tidx == 0:
                    vlo_c = s_thr[1]
                    vhi_c = s_thr[2]
                    mid_c = (vlo_c + vhi_c) * cutlass.Float32(0.5)
                    if mid_c == vlo_c or mid_c == vhi_c:
                        s_thr[0] = vhi_c
                        s_iscalars[1] = cutlass.Int32(3)  # plateau terminal
                    else:
                        s_thr[0] = mid_c
                cute.arch.barrier()
                if s_iscalars[1] == cutlass.Int32(0):
                    self.block_count_ge(
                        input_row,
                        slice_start,
                        slice_end,
                        s_thr[0],
                        smem_ptcnt,
                        smem_wcnt,
                        s_iscalars,
                        s_cluster_partial,
                        tidx,
                        warp_id,
                        lane,
                        smem_input=smem_input,
                        do_cluster_sync=do_cluster_sync,
                    )
                    if tidx == 0:
                        c_c = s_iscalars[0]
                        t_c = s_thr[0]
                        if c_c >= cutlass.Int32(kK) and c_c <= cutlass.Int32(kCC):
                            s_iscalars[1] = cutlass.Int32(1)
                        elif c_c > cutlass.Int32(kCC):
                            s_thr[1] = t_c
                            s_iscalars[2] = c_c
                        else:
                            s_thr[2] = t_c
                            s_iscalars[3] = c_c
                    cute.arch.barrier()
                itc = itc + cutlass.Int32(1)
            if s_iscalars[1] == cutlass.Int32(3):
                # recount at the terminal threshold so Phase 3's cached
                # per-thread counts describe the sure-winner set.
                self.block_count_ge(
                    input_row,
                    slice_start,
                    slice_end,
                    s_thr[0],
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    smem_input=smem_input,
                    do_cluster_sync=do_cluster_sync,
                )
                cute.arch.barrier()

        # ---- Post-loop fallback: if still not done, force threshold ----
        if tidx == 0:
            if s_iscalars[1] == cutlass.Int32(0):
                if s_iscalars[2] <= cutlass.Int32(kCC * 2):
                    s_thr[0] = s_thr[1]  # threshold = val_lo
                else:
                    s_thr[0] = s_thr[2]  # threshold = val_hi
                s_iscalars[1] = cutlass.Int32(2)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase 3: Ballot-free candidate collect
    # If P2 ended with done=2 (bracket exhausted), first run a retry-shrink
    # loop (≤10 iters) to bring cand_count <= kCC.
    # Then reuse cached smem_ptcnt → warp prefix sum → block prefix sum
    # → stream-write keys[]/vals[] for v >= threshold.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_candidates(
        self,
        input_row,
        N,
        slice_start,
        slice_end,
        smem_keys,
        smem_vals,
        smem_ptcnt,
        smem_wcnt,
        s_thr,
        s_iscalars,
        s_cluster_partial,
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = cs=1 / short-row degrade (skip cluster sync)
        smem_input=None,  # optional SMEM-cached slice
    ):
        """Retry-shrink (when P2 didn't converge) + prefix sum + stream-write.

        On exit, smem_keys[0 : cand_count] / smem_vals[0 : cand_count]
        hold every (value, index) pair with value >= threshold, in the
        scan order each thread produces them. Uses smem_ptcnt cached by
        the last block_count_ge in Phase 2 (or by the retry-shrink below).
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        num_threads = cutlass.const_expr(self.num_threads)

        # ---- Retry-shrink loop (only if P2 didn't converge cleanly) ----
        # Phase 3 runs cluster-parallel — every CTA shrinks against its own
        # slice but must agree on the threshold update. block_count_ge
        # always aggregates across the cluster, so every CTA sees the same
        # cluster-wide cand_count; cs=1 makes the aggregation a no-op.
        if s_iscalars[1] != cutlass.Int32(1):
            # Re-count with current threshold (may already have stale cand_count)
            cur_thr = s_thr[0]
            self.block_count_ge(
                input_row,
                slice_start,
                slice_end,
                cur_thr,
                smem_ptcnt,
                smem_wcnt,
                s_iscalars,
                s_cluster_partial,
                tidx,
                warp_id,
                lane,
                smem_input=smem_input,
                do_cluster_sync=do_cluster_sync,
            )
            if tidx == 0:
                if s_iscalars[0] > cutlass.Int32(kCC):
                    s_thr[1] = s_thr[0]  # val_lo = threshold
            cute.arch.barrier()

            # 10-iter retry-shrink. Runtime while with `cand_count > kCC` in the
            # loop condition.
            rs = cutlass.Int32(0)
            while rs < cutlass.Int32(10) and s_iscalars[0] > cutlass.Int32(kCC):
                if tidx == 0:
                    lo = s_thr[1]
                    hi = s_thr[2]
                    mid = (lo + hi) * cutlass.Float32(0.5)
                    if mid == lo:
                        mid = hi
                    s_thr[0] = mid
                cute.arch.barrier()
                new_thr = s_thr[0]
                self.block_count_ge(
                    input_row,
                    slice_start,
                    slice_end,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    smem_input=smem_input,
                    do_cluster_sync=do_cluster_sync,
                )
                if tidx == 0:
                    c_rs = s_iscalars[0]
                    if c_rs > cutlass.Int32(kCC):
                        s_thr[1] = s_thr[0]
                    elif c_rs < cutlass.Int32(kK):
                        s_thr[2] = s_thr[0]
                cute.arch.barrier()
                rs = rs + cutlass.Int32(1)

        # ---- Warp prefix sum over smem_ptcnt ----
        # my_total_qual = per-thread count cached by last block_count_ge.
        my_total_qual = smem_ptcnt[tidx]
        tp = my_total_qual

        # 5-level shfl_up_sync inclusive scan within warp.
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp = tp + other

        my_excl_offset = tp - my_total_qual
        # Warp total = lane 31's tp; broadcast via shfl_sync_bfly (or
        # cross-lane read: shuffle_sync_op with lane=31).
        warp_total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))

        if lane == 0:
            smem_wcnt[warp_id] = warp_total
        cute.arch.barrier()

        # Exclusive prefix sum over num_warps warp totals.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel via block_scan.warp_scan (Hillis-Steele
            # inclusive scan, log2(num_warps) shfl_up steps). Exclusive
            # prefix = inclusive - val. Total = inclusive at last lane.
            if warp_id == cutlass.Int32(0):
                if lane < cutlass.Int32(self.num_warps):
                    val = smem_wcnt[lane]
                    inclusive = warp_scan(
                        val, tidx, lane, num_threads_per_warp=self.num_warps
                    )
                    smem_wcnt[lane] = inclusive - val  # exclusive prefix
                    if lane == cutlass.Int32(self.num_warps - 1):
                        s_iscalars[0] = inclusive  # cand_count (total)
        else:
            # tid==0 serial exclusive prefix.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    cnt = smem_wcnt[w]
                    smem_wcnt[w] = total
                    total = total + cnt
                s_iscalars[0] = total
        cute.arch.barrier()

        # Each thread's write base = warp-prefix + intra-warp exclusive offset.
        my_base = smem_wcnt[warp_id]
        my_write_pos = my_base + my_excl_offset

        # ---- Stream-write loop ----
        # Scan bound is this CTA's slice [slice_start, slice_end), not the
        # full row. Phase 2's last block_count_ge populated smem_ptcnt with
        # slice-local counts, so the prefix sum above already reflects
        # "candidates this thread will write" within the slice. After this
        # function returns, smem_keys[0 .. local_cand_count) holds this
        # CTA's slice candidates; the kernel-level handoff DSMEM-gathers
        # peers' chunks into the leader's smem_keys before Phase 4. At
        # cluster_size==1 the slice is [0, N) and behavior is identical to
        # the single-CTA path.
        thr_final = s_thr[0]
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        step_elem = cutlass.const_expr(num_threads * vec_w)
        # Hoisted SMEM window bases (one S2R here vs one per emitted
        # candidate below — this loop is the kernel's biggest instruction
        # region at production shapes).
        keys_base = smem_keys.iterator.toint()
        vals_base = smem_vals.iterator.toint()

        slice_len = slice_end - slice_start
        # When reading from the cached slice, scan indices are slice-LOCAL;
        # the GMEM path uses global indices. smem_vals always stores the
        # GLOBAL position so Phase 4 / writeback stays consistent.
        if cutlass.const_expr(smem_input is not None):
            smem_addr = smem_input.iterator.toint()
            n_aligned = (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
            N_local = slice_len
            ic = tidx * cutlass.Int32(vec_w)
        else:
            n_aligned = slice_start + (
                slice_len // cutlass.Int32(vec_w)
            ) * cutlass.Int32(vec_w)
            N_local = slice_end
            ic = slice_start + tidx * cutlass.Int32(vec_w)
        wc = my_write_pos
        step = cutlass.Int32(step_elem)

        # Phase3 unrolling: master gated by self.enable_phase3_unroll.
        # When OFF, only the tail 1-way loop runs (matches the pre-unroll
        # state of phase3_collect). When ON, the inner enable_unroll_4
        # controls the 4-way fast path.
        if self.enable_phase3_unroll:
            # Fast path: 4-way unrolled vec loop (4 loading instructions in flight).
            if self.enable_unroll_4:
                rng_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
                big_iters = cutlass.Int32(0)
                if N_local > ic + cutlass.Int32(vec_w - 1):
                    big_iters = (N_local - ic - cutlass.Int32(vec_w)) // cutlass.Int32(
                        step_elem
                    ) + cutlass.Int32(1)

                for k in cutlass.range(big_iters, unroll=4):
                    ic_local = ic + k * cutlass.Int32(step_elem)
                    if cutlass.const_expr(smem_input is not None):
                        src_ptr_k = cute.make_ptr(
                            self.dtype,
                            smem_addr
                            + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=vec_align,
                        )
                        global_base = slice_start + ic_local
                    else:
                        src_ptr_k = cute.make_ptr(
                            self.dtype,
                            row_addr
                            + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_align,
                        )
                        global_base = ic_local
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        if vj >= thr_final and wc < cutlass.Int32(kCC):
                            self._smem_st(cutlass.Float32, keys_base, wc, vj)
                            self._smem_st(
                                cutlass.Int32,
                                vals_base,
                                wc,
                                global_base + cutlass.Int32(j),
                            )
                            wc = wc + cutlass.Int32(1)
                # Advance ic past all consumed vec_w-aligned positions.
                ic = ic + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step.
        tail_frag = cute.make_rmem_tensor((vec_w,), self.dtype)
        while ic + cutlass.Int32(vec_w - 1) < N_local:
            if cutlass.const_expr(smem_input is not None):
                src_ptr = cute.make_ptr(
                    self.dtype,
                    smem_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=vec_align,
                )
                global_base_t = slice_start + ic
            else:
                src_ptr = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                global_base_t = ic
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr_final and wc < cutlass.Int32(kCC):
                    self._smem_st(cutlass.Float32, keys_base, wc, vj)
                    self._smem_st(
                        cutlass.Int32, vals_base, wc, global_base_t + cutlass.Int32(j)
                    )
                    wc = wc + cutlass.Int32(1)
            ic = ic + step

        # Tail scalar loop (slice_len % vec_w)
        it = n_aligned + tidx
        while it < N_local:
            if cutlass.const_expr(smem_input is not None):
                v = smem_input[it]
                if cutlass.const_expr(self.dtype != cutlass.Float32):
                    v = cutlass.Float32(v)
                pos_global = slice_start + it
            else:
                v = self._load_fp32(input_row, it)
                pos_global = it
            if v >= thr_final and wc < cutlass.Int32(kCC):
                self._smem_st(cutlass.Float32, keys_base, wc, v)
                self._smem_st(cutlass.Int32, vals_base, wc, pos_global)
                wc = wc + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # block_fused_snap_iter — P4 snap convergence inner step
    # ------------------------------------------------------------------
    @cute.jit
    def block_fused_snap_iter(
        self,
        keys_base,  # hoisted SMEM window base of smem_keys (iterator.toint())
        smem_wcnt,
        smem_hist,  # reused as scratch for s_up/s_down warp aggregates
        s_thr,
        s_iscalars,
        count,
        tidx,
        warp_id,
        lane,
    ):
        """One iteration of histogram snap. Updates s_iscalars[2]=cnt_lo (cge),
        s_iscalars[3]=cnt_hi (cgt), and s_thr[0]=threshold (moves toward
        the cnt-in-(kK_GT, kK_GE) bracket).
        """
        kK = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        thr = s_thr[0]

        lge = cutlass.Int32(0)
        lgt = cutlass.Int32(0)
        s_up = cutlass.Float32(self.FLT_MAX)
        s_down = cutlass.Float32(self.NEG_FLT_MAX)

        isi = tidx
        while isi < count:
            v = self._smem_ld(cutlass.Float32, keys_base, isi)
            if v >= thr:
                lge = lge + cutlass.Int32(1)
            if v > thr:
                lgt = lgt + cutlass.Int32(1)
                # s_up = min(s_up, v) — hot path in block_fused_snap_iter (~10us)
                s_up = _fmin_f32_inline(s_up, v)
            if v < thr:
                s_down = cute.arch.fmax(s_down, v)
            isi = isi + cutlass.Int32(num_threads)

        # Pack lge/lgt into one int32 so the warp reduce sums both counts
        # in a single shuffle. Safe as long as each per-warp count
        # stays < 2^16; lge/lgt are bounded by cand_count ≤ kC ≤ 6144
        # (GvrParams), so we're well clear. Bumping kC past 65535 would
        # silently corrupt this packing.
        packed = (lge << cutlass.Int32(16)) | lgt
        packed = self.warp_reduce_sum_i32(packed)
        s_up = self.warp_reduce_min_f32(s_up)
        s_down = self.warp_reduce_max_f32(s_down)

        # Lane 0 stages results into warp slots (smem_hist[0..NW-1] = s_up,
        # smem_hist[NW..2*NW-1] = s_down stored as int32 bit-cast).
        if lane == 0:
            smem_wcnt[warp_id] = packed
            smem_hist[warp_id] = float_as_uint32(s_up)
            smem_hist[self.num_warps + warp_id] = float_as_uint32(s_down)
        cute.arch.barrier()

        # 3-way block reduce + threshold bound update.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # Warp-parallel 3-way reduce in warp 0.
            if warp_id == cutlass.Int32(0):
                v_tp = cutlass.Int32(0)
                v_up = cutlass.Float32(self.FLT_MAX)
                v_dn = cutlass.Float32(self.NEG_FLT_MAX)
                if lane < cutlass.Int32(self.num_warps):
                    v_tp = smem_wcnt[lane]
                    vu_bits = smem_hist[lane]
                    vd_bits = smem_hist[self.num_warps + lane]
                    v_up = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vu_bits.ir_value())
                    )
                    v_dn = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vd_bits.ir_value())
                    )
                tp = self.warp_reduce_sum_i32(v_tp)
                total_up = self.warp_reduce_min_f32(v_up)
                total_down = self.warp_reduce_max_f32(v_dn)
                if lane == cutlass.Int32(0):
                    cge = tp >> cutlass.Int32(16)
                    cgt = tp & cutlass.Int32(0xFFFF)
                    s_iscalars[2] = cge
                    s_iscalars[3] = cgt
                    if cgt >= cutlass.Int32(kK):
                        if total_up < cutlass.Float32(self.FLT_MAX):
                            s_thr[0] = total_up
                    elif cge < cutlass.Int32(kK):
                        if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                            s_thr[0] = total_down
        else:
            # tid==0 serial 3-way reduce.
            if tidx == 0:
                tp = cutlass.Int32(0)
                total_up = cutlass.Float32(self.FLT_MAX)
                total_down = cutlass.Float32(self.NEG_FLT_MAX)
                for w in cutlass.range_constexpr(self.num_warps):
                    tp = tp + smem_wcnt[w]
                    vu = llvm.bitcast(
                        cutlass.Float32.mlir_type, smem_hist[w].ir_value()
                    )
                    vd = llvm.bitcast(
                        cutlass.Float32.mlir_type,
                        smem_hist[self.num_warps + w].ir_value(),
                    )
                    vu_w = cutlass.Float32(vu)
                    vd_w = cutlass.Float32(vd)
                    total_up = _fmin_f32_inline(total_up, vu_w)
                    total_down = cute.arch.fmax(total_down, vd_w)

                cge = tp >> cutlass.Int32(16)
                cgt = tp & cutlass.Int32(0xFFFF)
                s_iscalars[2] = cge
                s_iscalars[3] = cgt
                if cgt >= cutlass.Int32(kK):
                    if total_up < cutlass.Float32(self.FLT_MAX):
                        s_thr[0] = total_up
                elif cge < cutlass.Int32(kK):
                    if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                        s_thr[0] = total_down
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P4 helpers: histogram build + parallel k-th bin search. Factored
    # out so the level-2 refinement can rerun both over a narrowed window.
    # ------------------------------------------------------------------
    @cute.jit
    def _hist_build(self, keys_base, smem_hist, cand_count, lo, inv, tidx):
        """Zero smem_hist[0:kBins], then histogram keys[0:cand_count] with
        bin = clamp(int((v - lo) * inv), 0, kBins-1). Out-of-window values
        clamp into the edge bins, which keeps cumulative counts from the
        top exact for the k-th search (everything above the window lands
        in the top bin). Barrier after the zero pass and after the build."""
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        i6 = tidx
        while i6 < cutlass.Int32(kBins):
            smem_hist[i6] = cutlass.Int32(0)
            i6 = i6 + cutlass.Int32(num_threads)
        cute.arch.barrier()
        i7 = tidx
        while i7 < cand_count:
            vk = self._smem_ld(cutlass.Float32, keys_base, i7)
            bin_f = (vk - lo) * inv
            # Clamp in the FLOAT domain before the int cast: fptosi is
            # undefined for out-of-range/NaN inputs at the IR level (PTX
            # cvt.rzi saturates, but LLVM may optimize on the poison).
            # fmax first canonicalizes NaN to 0; the pair keeps the
            # edge-bin clamping semantics bit-identical for in-range
            # values.
            bin_f = cute.arch.fmax(bin_f, cutlass.Float32(0.0))
            bin_f = _fmin_f32_inline(bin_f, cutlass.Float32(kBins - 1))
            bin_i = cutlass.Int32(bin_f)
            atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
            i7 = i7 + cutlass.Int32(num_threads)
        cute.arch.barrier()

    @cute.jit
    def _kth_bin_search(
        self, smem_hist, smem_wcnt, s_thr, s_iscalars, lo, binw, tidx, warp_id, lane
    ):
        """Parallel k-th bin search (3-step, high→low). Writes
        s_thr[0] = lower edge of the selected bin (lo + bidx*binw) and
        s_iscalars[4] = selected bin's count (gates the level-2 histogram
        refinement). Clobbers s_iscalars[2]/[3] as staging (both are
        rewritten by the snap loop before anyone else reads them).
        Trailing barrier."""
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        # Step 1: each warp sums BINS_PER_WARP bins (high→low slice).
        # Lane-parallel when the slice divides evenly across the warp:
        # each lane sums bins_per_warp/32 bins + one warp reduce, instead
        # of every lane redundantly walking a bins_per_warp-deep serial
        # LDS+IADD dependency chain (~7% of stall samples at N=8K).
        warp_bin_sum = cutlass.Int32(0)
        if cutlass.const_expr(bins_per_warp % self.WARP_SIZE == 0):
            for jm in cutlass.range_constexpr(bins_per_warp // self.WARP_SIZE):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - (lane + cutlass.Int32(jm * self.WARP_SIZE))
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            warp_bin_sum = self.warp_reduce_sum_i32(warp_bin_sum)
        else:
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
        if lane == 0:
            smem_wcnt[warp_id] = warp_bin_sum
        cute.arch.barrier()

        # Step 2: tid==0 finds target warp; stores prefix-count + warp index
        # into s_iscalars[2] (=cnt_lo: prefix before target warp)
        # and s_iscalars[3] (=cnt_hi: target warp index)
        if tidx == 0:
            cum = cutlass.Int32(0)
            tw = cutlass.Int32(self.num_warps - 1)
            found = cutlass.Int32(0)
            for w2 in cutlass.range_constexpr(self.num_warps):
                cum = cum + smem_wcnt[w2]
                if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                    tw = cutlass.Int32(w2)
                    found = cutlass.Int32(1)
            # Recompute prefix BEFORE target warp
            cum2 = cutlass.Int32(0)
            for w3 in cutlass.range_constexpr(self.num_warps):
                if cutlass.Int32(w3) < tw:
                    cum2 = cum2 + smem_wcnt[w3]
            s_iscalars[2] = cum2  # prefix
            s_iscalars[3] = tw  # target warp index
        cute.arch.barrier()

        # Step 3: target warp's lane 0 scans BINS_PER_WARP bins →
        # threshold. Single-thread serial; the unrolled
        # range_constexpr beats a runtime `for+break` (tried it: -544
        # SASS insts but -7pp fp32 / -14pp bf16, since the
        # branch/counter overhead in a single thread dominates the
        # static math).
        target_warp = s_iscalars[3]
        if warp_id == target_warp and lane == cutlass.Int32(0):
            base_cum = s_iscalars[2]
            thr_local = lo
            sel_cnt = cutlass.Int32(0)
            set_done = cutlass.Int32(0)
            for jb2 in cutlass.range_constexpr(bins_per_warp):
                bidx2 = (
                    cutlass.Int32(kBins - 1)
                    - target_warp * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb2)
                )
                cnt_here = smem_hist[bidx2]
                base_cum = base_cum + cnt_here
                if base_cum >= cutlass.Int32(kK) and set_done == cutlass.Int32(0):
                    thr_local = lo + cutlass.Float32(bidx2) * binw
                    sel_cnt = cnt_here
                    set_done = cutlass.Int32(1)
            s_thr[0] = thr_local
            s_iscalars[4] = sel_cnt
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # _kth_bin_search_rw — redundant-warp variant (p4_warp_redundant).
    # Step 1 stages per-warp bin-slice sums exactly like _kth_bin_search
    # (the ONE barrier). Then EVERY warp redundantly (a) walks the
    # num_warps slot sums with broadcast SMEM reads + predicated adds to
    # locate the target warp, and (b) lane-parallel walks the target
    # slice — each lane owns a contiguous descending sub-range, a
    # shuffle-up prefix + the unique sub-range crossing test find the
    # k-th bin in O(bins_per_warp/32) LDS instead of a 64-deep serial
    # LDS+IADD chain in one thread. Same inputs in the same order on
    # every warp -> bit-identical results, so there is no leader, no
    # publish barrier, and no s_thr/s_iscalars staging; the selected
    # (threshold, bin count) return in registers.
    # ------------------------------------------------------------------
    @cute.jit
    def _kth_bin_search_rw(self, smem_hist, smem_wcnt, lo, binw, tidx, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        # Step 1: identical staging to _kth_bin_search.
        warp_bin_sum = cutlass.Int32(0)
        if cutlass.const_expr(bins_per_warp % self.WARP_SIZE == 0):
            for jm in cutlass.range_constexpr(bins_per_warp // self.WARP_SIZE):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - (lane + cutlass.Int32(jm * self.WARP_SIZE))
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            warp_bin_sum = self.warp_reduce_sum_i32(warp_bin_sum)
        else:
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
        if lane == 0:
            smem_wcnt[warp_id] = warp_bin_sum
        cute.arch.barrier()

        # Step 2 (every warp, lane-parallel): lane w holds slot w; an
        # inclusive idx-shuffle scan + ballot locate the target warp.
        # (shuffle_sync with a computed source lane is the working shfl
        # idiom; shuffle_sync_up ignores its offset — probed.)
        v_s = cutlass.Int32(0)
        if lane < cutlass.Int32(self.num_warps):
            v_s = smem_wcnt[lane]
        run2 = v_s
        for d2 in cutlass.range_constexpr(5):
            off2 = cutlass.const_expr(1 << d2)
            src2 = lane - cutlass.Int32(off2)
            if src2 < cutlass.Int32(0):
                src2 = cutlass.Int32(0)
            up2 = cute.arch.shuffle_sync(run2, src2)
            if lane >= cutlass.Int32(off2):
                run2 = run2 + up2
        m2 = cute.arch.vote_ballot_sync(run2 >= cutlass.Int32(kK))
        tw = cutlass.Int32(self.num_warps - 1)
        if m2 != cutlass.Uint32(0):
            low2 = m2 & (cutlass.Uint32(0) - m2)
            tw = cutlass.Int32(cute.arch.popc(low2 - cutlass.Uint32(1)))
        incl_tw = cute.arch.shuffle_sync(run2, tw)
        slot_tw = cute.arch.shuffle_sync(v_s, tw)
        prefix = incl_tw - slot_tw

        # Step 3 (every warp, lane-parallel): lane l owns the contiguous
        # descending positions [l*ppl, (l+1)*ppl) of the target slice.
        ppl = cutlass.const_expr((bins_per_warp + self.WARP_SIZE - 1) // self.WARP_SIZE)
        cnt_frag = cute.make_rmem_tensor((ppl,), cutlass.Int32)
        my_sum = cutlass.Int32(0)
        for j3 in cutlass.range_constexpr(ppl):
            pos = lane * cutlass.Int32(ppl) + cutlass.Int32(j3)
            cnt_j = cutlass.Int32(0)
            if pos < cutlass.Int32(bins_per_warp):
                bidx3 = (
                    cutlass.Int32(kBins - 1) - tw * cutlass.Int32(bins_per_warp) - pos
                )
                cnt_j = smem_hist[bidx3]
            cnt_frag[j3] = cnt_j
            my_sum = my_sum + cnt_j
        # Exclusive cross-lane prefix of the lane partial sums via the
        # idx-shuffle scan (5 log-steps; shuffle_sync_up ignores its
        # offset — probed — so the scan uses computed source lanes).
        run3 = my_sum
        for d3 in cutlass.range_constexpr(5):
            off3 = cutlass.const_expr(1 << d3)
            src3 = lane - cutlass.Int32(off3)
            if src3 < cutlass.Int32(0):
                src3 = cutlass.Int32(0)
            up3 = cute.arch.shuffle_sync(run3, src3)
            if lane >= cutlass.Int32(off3):
                run3 = run3 + up3
        base3 = prefix + (run3 - my_sum)

        # Unique crossing: the lane where the running count passes kK.
        thr_loc = lo
        sel_loc = cutlass.Int32(0)
        hit = cutlass.Int32(0)
        r3 = base3
        for j4 in cutlass.range_constexpr(ppl):
            pos4 = lane * cutlass.Int32(ppl) + cutlass.Int32(j4)
            cnt4 = cnt_frag[j4]
            if (
                pos4 < cutlass.Int32(bins_per_warp)
                and r3 < cutlass.Int32(kK)
                and r3 + cnt4 >= cutlass.Int32(kK)
                and hit == cutlass.Int32(0)
            ):
                bidx4 = (
                    cutlass.Int32(kBins - 1) - tw * cutlass.Int32(bins_per_warp) - pos4
                )
                thr_loc = lo + cutlass.Float32(bidx4) * binw
                sel_loc = cnt4
                hit = cutlass.Int32(1)
            r3 = r3 + cnt4
        # Broadcast from the (at most one) hitting lane; no hit keeps
        # (lo, 0) — same fallback as _kth_bin_search's set_done guard.
        mask3 = cute.arch.vote_ballot_sync(hit != cutlass.Int32(0))
        thr_out = lo
        sel_out = cutlass.Int32(0)
        if mask3 != cutlass.Uint32(0):
            low = mask3 & (cutlass.Uint32(0) - mask3)
            src = cutlass.Int32(cute.arch.popc(low - cutlass.Uint32(1)))
            thr_out = cute.arch.shuffle_sync(thr_loc, src)
            sel_out = cute.arch.shuffle_sync(sel_loc, src)
        return thr_out, sel_out

    # ------------------------------------------------------------------
    # Phase 4 (alt): fused rank-and-scatter (enable_p4_rank_scatter).
    # Ported verbatim from p4_recursive_digit/gvr_topk_decode_p4.py.
    # ------------------------------------------------------------------
    @cute.jit
    def _p4_exact_tail_radix_select(
        self,
        kK: cutlass.Constexpr,
        kBins: cutlass.Constexpr,
        num_threads: cutlass.Constexpr,
        num_warps: cutlass.Constexpr,
        need0,
        cand_count,
        rank_above_fine,
        b_star,
        sb_star,
        bmin_r,
        f_lo,
        finv,
        fbins,
        inv1,
        tidx,
        lane,
        warp_id,
        smem_hist,
        smem_keys,
        smem_vals,
        smem_wcnt,
        s_iscalars,
        output_indices_row,
        output_values_row,
    ):
        """MSB-first 4x8-bit exact radix select over the straddling
        fine-bin tie set (``p4_exact_tail``) — single source shared by
        the tiny-tie fast path's large-class fallback and the plain
        exact-tail path (previously two verbatim copies; ``@cute.jit``
        helpers inline, so codegen is unchanged)."""
        # Persistent scalars live above the 256 digit bins
        # (kNumBins >= 512 always): [256] key prefix (chosen
        # digits, remaining bits 0), [257] slots still to fill
        # inside the current equal-prefix set, [258] ties
        # strictly above the prefix (their slots precede it).
        if tidx == cutlass.Int32(0):
            smem_hist[256] = cutlass.Int32(0)
            smem_hist[257] = need0
            smem_hist[258] = cutlass.Int32(0)
        cute.arch.barrier()
        for lvl in cutlass.range_constexpr(4):
            shift = cutlass.const_expr(24 - 8 * lvl)
            zero_idx = tidx
            while zero_idx < cutlass.Int32(256):
                smem_hist[zero_idx] = cutlass.Int32(0)
                zero_idx = zero_idx + cutlass.Int32(num_threads)
            cute.arch.barrier()
            uthr_cur = smem_hist[256]
            it2 = tidx
            while it2 < cand_count:
                vt = smem_keys[it2]
                bt = cutlass.Int32((vt - bmin_r) * inv1)
                if bt < cutlass.Int32(0):
                    bt = cutlass.Int32(0)
                if bt > cutlass.Int32(kBins - 1):
                    bt = cutlass.Int32(kBins - 1)
                if bt == b_star:
                    st2 = cutlass.Int32((vt - f_lo) * finv)
                    if st2 < cutlass.Int32(0):
                        st2 = cutlass.Int32(0)
                    if st2 > cutlass.Int32(fbins - 1):
                        st2 = cutlass.Int32(fbins - 1)
                    if st2 == sb_star:
                        uk = f32_order_key(vt)
                        pmatch = cutlass.Int32(1)
                        if cutlass.const_expr(lvl > 0):
                            if (uk >> cutlass.Int32(shift + 8)) != (
                                uthr_cur >> cutlass.Int32(shift + 8)
                            ):
                                pmatch = cutlass.Int32(0)
                        if pmatch == cutlass.Int32(1):
                            dg = (uk >> cutlass.Int32(shift)) & cutlass.Int32(0xFF)
                            atomicAdd(smem_hist.iterator + dg, cutlass.Int32(1))
                it2 = it2 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            # Two-stage descending digit scan (mirrors the
            # fine 3-step search): per-warp partial sums,
            # thread0 picks the target warp, its lane0 walks
            # the warp's digit range — 2*num_warps serial
            # steps instead of 256.
            fdw = cutlass.const_expr(256 // self.num_warps)
            wsum2 = cutlass.Int32(0)
            for jd in cutlass.range_constexpr(fdw):
                dix = (
                    cutlass.Int32(255)
                    - warp_id * cutlass.Int32(fdw)
                    - cutlass.Int32(jd)
                )
                wsum2 = wsum2 + smem_hist[dix]
            if lane == cutlass.Int32(0):
                smem_wcnt[warp_id] = wsum2
            cute.arch.barrier()
            if tidx == cutlass.Int32(0):
                needl = smem_hist[257]
                cw = cutlass.Int32(0)
                tw3 = cutlass.Int32(num_warps - 1)
                f3 = cutlass.Int32(0)
                for w4 in cutlass.range_constexpr(self.num_warps):
                    cw = cw + smem_wcnt[w4]
                    if cw >= needl and f3 == cutlass.Int32(0):
                        tw3 = cutlass.Int32(w4)
                        f3 = cutlass.Int32(1)
                pre3 = cutlass.Int32(0)
                for w5 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w5) < tw3:
                        pre3 = pre3 + smem_wcnt[w5]
                s_iscalars[4] = pre3  # prefix above target warp
                s_iscalars[0] = tw3  # target warp
            cute.arch.barrier()
            pre4 = s_iscalars[4]
            tw4 = s_iscalars[0]
            if warp_id == tw4 and lane == cutlass.Int32(0):
                needl2 = smem_hist[257]
                base4 = pre4
                dstar = cutlass.Int32(0)
                above_d = pre4
                sd4 = cutlass.Int32(0)
                for jd2 in cutlass.range_constexpr(fdw):
                    dix2 = (
                        cutlass.Int32(255)
                        - tw4 * cutlass.Int32(fdw)
                        - cutlass.Int32(jd2)
                    )
                    ra4 = base4
                    base4 = base4 + smem_hist[dix2]
                    if base4 >= needl2 and sd4 == cutlass.Int32(0):
                        dstar = dix2
                        above_d = ra4
                        sd4 = cutlass.Int32(1)
                smem_hist[256] = uthr_cur | (dstar << cutlass.Int32(shift))
                smem_hist[257] = needl2 - above_d
                smem_hist[258] = smem_hist[258] + above_d
            cute.arch.barrier()
        # Rewrite the tie slot range: ties with key > u_thr
        # first (there are exactly cnt_ab of them), then the
        # first need_eq bitwise-equal-to-u_thr ties in arrival
        # order (value-exact by construction). Signed compare
        # needs the top bit flipped (unsigned-monotonic key).
        u_thr = smem_hist[256]
        cnt_ab = smem_hist[258]
        need_eq = smem_hist[257]
        ks_thr = u_thr ^ cutlass.Int32(-2147483648)
        if tidx == cutlass.Int32(0):
            s_iscalars[4] = cutlass.Int32(0)  # above-writer ctr
            s_iscalars[0] = cutlass.Int32(0)  # equal-writer ctr
        cute.arch.barrier()
        ir2 = tidx
        while ir2 < cand_count:
            vr = smem_keys[ir2]
            br = cutlass.Int32((vr - bmin_r) * inv1)
            if br < cutlass.Int32(0):
                br = cutlass.Int32(0)
            if br > cutlass.Int32(kBins - 1):
                br = cutlass.Int32(kBins - 1)
            if br == b_star:
                sr = cutlass.Int32((vr - f_lo) * finv)
                if sr < cutlass.Int32(0):
                    sr = cutlass.Int32(0)
                if sr > cutlass.Int32(fbins - 1):
                    sr = cutlass.Int32(fbins - 1)
                if sr == sb_star:
                    uk2 = f32_order_key(vr)
                    ks2 = uk2 ^ cutlass.Int32(-2147483648)
                    if ks2 > ks_thr:
                        o2 = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(4),
                            cutlass.Int32(1),
                        )
                        pos = rank_above_fine + o2
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(vr)
                            output_indices_row[pos] = smem_vals[ir2]
                    elif ks2 == ks_thr:
                        q2 = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(0),
                            cutlass.Int32(1),
                        )
                        if q2 < need_eq:
                            pos = rank_above_fine + cnt_ab + q2
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(vr)
                                output_indices_row[pos] = smem_vals[ir2]
            ir2 = ir2 + cutlass.Int32(num_threads)
        cute.arch.barrier()

    @cute.jit
    def phase4_rank_scatter(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            # ---- block min/max over candidates ----
            local_cmin = cutlass.Float32(self.FLT_MAX)
            local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
            i5 = tidx
            while i5 < cand_count:
                v = smem_keys[i5]
                local_cmin = _fmin_f32_inline(local_cmin, v)
                local_cmax = cute.arch.fmax(local_cmax, v)
                i5 = i5 + cutlass.Int32(num_threads)
            cmin = self.warp_reduce_min_f32(local_cmin)
            cmax = self.warp_reduce_max_f32(local_cmax)
            if lane == cutlass.Int32(0):
                smem_wcnt[warp_id] = float_as_uint32(cmin)
                smem_hist[warp_id] = float_as_uint32(cmax)
            cute.arch.barrier()
            bmin_r = cutlass.Float32(self.FLT_MAX)
            bmax_r = cutlass.Float32(self.NEG_FLT_MAX)
            for w in cutlass.range_constexpr(self.num_warps):
                vmin = cutlass.Float32(
                    llvm.bitcast(cutlass.Float32.mlir_type, smem_wcnt[w].ir_value())
                )
                vmax = cutlass.Float32(
                    llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[w].ir_value())
                )
                bmin_r = _fmin_f32_inline(bmin_r, vmin)
                bmax_r = cute.arch.fmax(bmax_r, vmax)
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            cute.arch.barrier()
            # ---- zero + build histogram ----
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < cand_count:
                vk = smem_keys[i7]
                bin_i = cutlass.Int32((vk - bmin_r) * inv1)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            # ---- 3-step high→low bin search → straddling bin b* + rank_above ----
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == cutlass.Int32(0):
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()
            if tidx == cutlass.Int32(0):
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2  # prefix-count before target warp
                s_iscalars[3] = tw
            cute.arch.barrier()
            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                b_star = cutlass.Int32(kBins - 1)
                rank_above = base_cum
                set_d = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (
                        cutlass.Int32(kBins - 1)
                        - target_warp * cutlass.Int32(bins_per_warp)
                        - cutlass.Int32(jb2)
                    )
                    ra_before = base_cum
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= cutlass.Int32(kK) and set_d == cutlass.Int32(0):
                        b_star = bidx2
                        rank_above = ra_before  # count in bins strictly above b*
                        set_d = cutlass.Int32(1)
                s_iscalars[2] = rank_above
                s_iscalars[3] = b_star
                s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                s_iscalars[1] = cutlass.Int32(0)  # cnt_straddle
            cute.arch.barrier()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]

            # ---- EXACT: one fine-histogram recursion on the straddling bin b* ----
            if cutlass.const_expr(self.enable_p4_rank_scatter_exact):
                # FIXED small fine-bin count (independent of kNumBins) — cuts the
                # re-zero + 3-step cost (esp. K=2048 where kNumBins=2048); 256
                # sub-bins over bin b* gives kNumBins×256 effective resolution,
                # enough to resolve the straddling bin to ≤1 distinct value.
                fbins = cutlass.const_expr(256)
                fbpw = cutlass.const_expr(256 // self.num_warps)
                # bin b* value range under the inv1 binning: [f_lo, f_lo + 1/inv1)
                f_lo = bmin_r + cutlass.Float32(b_star) / inv1
                finv = (cutlass.Float32(fbins - 1) + cutlass.Float32(0.99)) * inv1
                # re-zero (only fbins slots) + build fine sub-hist of bin-b* cands
                zero_idx = tidx
                while zero_idx < cutlass.Int32(fbins):
                    smem_hist[zero_idx] = cutlass.Int32(0)
                    zero_idx = zero_idx + cutlass.Int32(num_threads)
                cute.arch.barrier()
                ifb = tidx
                while ifb < cand_count:
                    vf = smem_keys[ifb]
                    cb = cutlass.Int32((vf - bmin_r) * inv1)
                    if cb < cutlass.Int32(0):
                        cb = cutlass.Int32(0)
                    if cb > cutlass.Int32(kBins - 1):
                        cb = cutlass.Int32(kBins - 1)
                    if cb == b_star:
                        sb = cutlass.Int32((vf - f_lo) * finv)
                        if sb < cutlass.Int32(0):
                            sb = cutlass.Int32(0)
                        if sb > cutlass.Int32(fbins - 1):
                            sb = cutlass.Int32(fbins - 1)
                        atomicAdd(smem_hist.iterator + sb, cutlass.Int32(1))
                    ifb = ifb + cutlass.Int32(num_threads)
                cute.arch.barrier()
                # fine 3-step search seeded at rank_above (over fbins bins)
                fws = cutlass.Int32(0)
                for jbf in cutlass.range_constexpr(fbpw):
                    bif = (
                        cutlass.Int32(fbins - 1)
                        - warp_id * cutlass.Int32(fbpw)
                        - cutlass.Int32(jbf)
                    )
                    fws = fws + smem_hist[bif]
                if lane == cutlass.Int32(0):
                    smem_wcnt[warp_id] = fws
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    cumf = rank_above
                    twf = cutlass.Int32(num_warps - 1)
                    found = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(self.num_warps):
                        cumf = cumf + smem_wcnt[w2]
                        if cumf >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                            twf = cutlass.Int32(w2)
                            found = cutlass.Int32(1)
                    pre = rank_above
                    for w3 in cutlass.range_constexpr(self.num_warps):
                        if cutlass.Int32(w3) < twf:
                            pre = pre + smem_wcnt[w3]
                    # Stage prefix/target-warp metadata in spare s_iscalars
                    # slots, NOT smem_hist[0]/[1]: the last fine warp's reverse
                    # scan below walks fine bins down to 0/1, so reusing those
                    # histogram bins as scratch would corrupt sb_star/ra_fine
                    # when twf2 == num_warps-1. Slots [4]/[1] are dead here
                    # (re-zeroed at the cnt_above/cnt_strad reset below).
                    s_iscalars[4] = pre  # prefix into target fine warp
                    s_iscalars[1] = twf  # target fine warp
                cute.arch.barrier()
                pre_f = s_iscalars[4]
                twf2 = s_iscalars[1]
                if warp_id == twf2 and lane == cutlass.Int32(0):
                    base_f = pre_f
                    sb_star = cutlass.Int32(fbins - 1)
                    ra_fine = base_f
                    sd = cutlass.Int32(0)
                    for jb3 in cutlass.range_constexpr(fbpw):
                        sbi = (
                            cutlass.Int32(fbins - 1)
                            - twf2 * cutlass.Int32(fbpw)
                            - cutlass.Int32(jb3)
                        )
                        ra_b = base_f
                        base_f = base_f + smem_hist[sbi]
                        if base_f >= cutlass.Int32(kK) and sd == cutlass.Int32(0):
                            sb_star = sbi
                            ra_fine = ra_b
                            sd = cutlass.Int32(1)
                    smem_hist[2] = sb_star
                    smem_hist[3] = ra_fine
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[0] = cutlass.Int32(0)  # cnt_mid (b*, sub>sb*)
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_strad (b*, sub==sb*)
                cute.arch.barrier()
                sb_star = smem_hist[2]
                rank_above_fine = smem_hist[3]
                isc = tidx
                while isc < cand_count:
                    v = smem_keys[isc]
                    bin_i = cutlass.Int32((v - bmin_r) * inv1)
                    if bin_i < cutlass.Int32(0):
                        bin_i = cutlass.Int32(0)
                    if bin_i > cutlass.Int32(kBins - 1):
                        bin_i = cutlass.Int32(kBins - 1)
                    if bin_i > b_star:
                        pos = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1)
                        )
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif bin_i == b_star:
                        sb = cutlass.Int32((v - f_lo) * finv)
                        if sb < cutlass.Int32(0):
                            sb = cutlass.Int32(0)
                        if sb > cutlass.Int32(fbins - 1):
                            sb = cutlass.Int32(fbins - 1)
                        if sb > sb_star:
                            o = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1)
                            )
                            pos = rank_above + o
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif sb == sb_star:
                            o = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1)
                            )
                            pos = rank_above_fine + o
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                    isc = isc + cutlass.Int32(num_threads)
                cute.arch.barrier()
                cnt_strad = s_iscalars[1]
                filled = rank_above_fine + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)

                # ---- EXACT-TAIL repair (p4_exact_tail, fp32): the fine bin
                # resolves values to range/(kBins*fbins); two candidates
                # closer than that can straddle the kK boundary inside ONE
                # fine bin, and the arrival-order fill above then keeps an
                # arbitrary subset. Gated on the ONLY case where that is
                # ambiguous — the tie set overfills the remaining slots — this
                # re-ranks the (b*, sb*) tie set exactly via an MSB-first
                # 8-bit-digit radix select over the order-preserving integer
                # keys (4 levels = bit-exact for fp32) and rewrites the tie
                # slot range [rank_above_fine, kK). Unambiguous rows (the
                # overwhelming majority) pay two scalar compares; the counters
                # and the fine histogram are reused, so SMEM does not grow.
                # [p4tt] tiny-tie fast path: when the exact-tail gate fires
                # with a small (b*, sb*) tie class (cnt_strad <= 128 — the
                # real firing cells hold 2), ONE candidate pass collects the
                # class and thread0 selects the top-need exactly, replacing
                # the 4 unconditional radix passes. Larger classes take the
                # UNMODIFIED radix select below (verbatim copy).
                if cutlass.const_expr(
                    self.p4_exact_tail and self.p4_tail_fast
                ):  # [p4tt]
                    need0 = cutlass.Int32(kK) - rank_above_fine
                    if cnt_strad > need0 and need0 > cutlass.Int32(0):
                        if cnt_strad <= cutlass.Int32(128):
                            # [p4tt] SMEM: (value_bits, cand_idx) pairs at
                            # smem_hist[2*o]/[2*o+1], o < 128 (slots 0..255).
                            # The 256 digit bins are dead here (the fast path
                            # replaces the radix levels that used them); the
                            # sb_star/ra staging in slots 2/3 was read by
                            # every thread before the pre-scatter barrier.
                            # Persistent radix scalars [256..258] untouched.
                            # Collect counter = s_iscalars[0] (dead after the
                            # scatter; same reuse as the radix rewrite pass).
                            if tidx == cutlass.Int32(0):
                                s_iscalars[0] = cutlass.Int32(0)
                            cute.arch.barrier()
                            itc = tidx
                            while itc < cand_count:
                                tv = smem_keys[itc]
                                tb = cutlass.Int32((tv - bmin_r) * inv1)
                                if tb < cutlass.Int32(0):
                                    tb = cutlass.Int32(0)
                                if tb > cutlass.Int32(kBins - 1):
                                    tb = cutlass.Int32(kBins - 1)
                                if tb == b_star:
                                    ts = cutlass.Int32((tv - f_lo) * finv)
                                    if ts < cutlass.Int32(0):
                                        ts = cutlass.Int32(0)
                                    if ts > cutlass.Int32(fbins - 1):
                                        ts = cutlass.Int32(fbins - 1)
                                    if ts == sb_star:
                                        to = atomicAdd(
                                            s_iscalars.iterator + cutlass.Int32(0),
                                            cutlass.Int32(1),
                                        )
                                        if to < cutlass.Int32(128):
                                            smem_hist[to + to] = float_as_int32(tv)
                                            smem_hist[to + to + cutlass.Int32(1)] = (
                                                smem_vals[itc]
                                            )
                                itc = itc + cutlass.Int32(num_threads)
                            cute.arch.barrier()
                            # [p4tt] thread0 exact top-need0 select rewriting
                            # positions [rank_above_fine, kK). Consumed flag =
                            # the cand_idx slot set to -1 (indices are always
                            # >= 0), so a genuine -FLT_MAX value in the class
                            # remains selectable (no value sentinel). Ties
                            # (bit-equal values) pick arbitrarily: value-set
                            # exact.
                            if tidx == cutlass.Int32(0):
                                tj = cutlass.Int32(0)
                                while tj < need0:
                                    tbv = cutlass.Float32(self.NEG_FLT_MAX)
                                    tbi = cutlass.Int32(-1)
                                    ti = cutlass.Int32(0)
                                    while ti < cnt_strad:
                                        tvi = smem_hist[ti + ti + cutlass.Int32(1)]
                                        if tvi >= cutlass.Int32(0):
                                            tvb = smem_hist[ti + ti]
                                            tvv = cutlass.Float32(
                                                llvm.bitcast(
                                                    cutlass.Float32.mlir_type,
                                                    tvb.ir_value(),
                                                )
                                            )
                                            take = cutlass.Int32(0)
                                            if tbi < cutlass.Int32(0):
                                                take = cutlass.Int32(1)
                                            elif tvv > tbv:
                                                take = cutlass.Int32(1)
                                            if take == cutlass.Int32(1):
                                                tbv = tvv
                                                tbi = ti
                                        ti = ti + cutlass.Int32(1)
                                    pos = rank_above_fine + tj
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pos] = self.dtype(tbv)
                                    output_indices_row[pos] = smem_hist[
                                        tbi + tbi + cutlass.Int32(1)
                                    ]
                                    smem_hist[tbi + tbi + cutlass.Int32(1)] = (
                                        cutlass.Int32(-1)
                                    )
                                    tj = tj + cutlass.Int32(1)
                            cute.arch.barrier()
                        else:
                            self._p4_exact_tail_radix_select(
                                kK,
                                kBins,
                                num_threads,
                                num_warps,
                                need0,
                                cand_count,
                                rank_above_fine,
                                b_star,
                                sb_star,
                                bmin_r,
                                f_lo,
                                finv,
                                fbins,
                                inv1,
                                tidx,
                                lane,
                                warp_id,
                                smem_hist,
                                smem_keys,
                                smem_vals,
                                smem_wcnt,
                                s_iscalars,
                                output_indices_row,
                                output_values_row,
                            )
                elif cutlass.const_expr(self.p4_exact_tail):  # [p4tt] if->elif only
                    need0 = cutlass.Int32(kK) - rank_above_fine
                    if cnt_strad > need0 and need0 > cutlass.Int32(0):
                        self._p4_exact_tail_radix_select(
                            kK,
                            kBins,
                            num_threads,
                            num_warps,
                            need0,
                            cand_count,
                            rank_above_fine,
                            b_star,
                            sb_star,
                            bmin_r,
                            f_lo,
                            finv,
                            fbins,
                            inv1,
                            tidx,
                            lane,
                            warp_id,
                            smem_hist,
                            smem_keys,
                            smem_vals,
                            smem_wcnt,
                            s_iscalars,
                            output_indices_row,
                            output_values_row,
                        )
            else:
                # ---- APPROX rank-and-scatter (single pass), arbitrary straddling order ----
                isc = tidx
                while isc < cand_count:
                    v = smem_keys[isc]
                    bin_i = cutlass.Int32((v - bmin_r) * inv1)
                    if bin_i < cutlass.Int32(0):
                        bin_i = cutlass.Int32(0)
                    if bin_i > cutlass.Int32(kBins - 1):
                        bin_i = cutlass.Int32(kBins - 1)
                    if bin_i > b_star:
                        pos = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1)
                        )
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif bin_i == b_star:
                        off = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1)
                        )
                        pos = rank_above + off
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    isc = isc + cutlass.Int32(num_threads)
                cute.arch.barrier()
                cnt_strad = s_iscalars[1]
                filled = rank_above + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)
        else:
            i10 = tidx
            while i10 < cand_count:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i10] = self.dtype(smem_keys[i10])
                output_indices_row[i10] = smem_vals[i10]
                i10 = i10 + cutlass.Int32(num_threads)
            if s_iscalars[6] == cutlass.Int32(0):  # plateau fill completes done=3
                i11 = cand_count + tidx
                while i11 < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[i11] = cutlass.Int32(-1)
                    i11 = i11 + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Phase 4: Histogram-based k-th selection + two-pass writeback
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_histogram_snap(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
    ):
        """Three branches by cand_count vs kK:
        == kK: direct emit (fast path)
        >  kK: histogram k-th bin search → snap → 2-pass writeback
        <  kK: emit cand_count + pad with -FLT_MAX
        """
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        # Hoisted SMEM window bases: every keys/vals element access below
        # goes through raw integer addressing (see _smem_ref rationale).
        keys_base = smem_keys.iterator.toint()
        vals_base = smem_vals.iterator.toint()
        # Scalars base for the snap-loop convergence check (read by ALL
        # threads once per snap iteration — a measured per-iteration
        # LDS hotspot).
        isc_base = s_iscalars.iterator.toint()

        # ----- Branch A: cand_count == kK (fast path) -----
        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(
                        self._smem_ld(cutlass.Float32, keys_base, i4)
                    )
                output_indices_row[i4] = self._smem_ld(cutlass.Int32, vals_base, i4)
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            # ----- Branch B: cand_count > kK → histogram snap -----

            # ---- Histogram window ----
            # Fast path: reuse the P2 exit bracket [vlo, vhi) instead of
            # scanning candidates for min/max. P3 collected v >= s_thr[0]
            # and P2's exit sets s_thr[0] = vlo (= s_thr[1]), so vlo
            # lower-bounds every candidate; the bracket invariant
            # cnt(>= vhi) < kK puts the k-th value inside [vlo, vhi).
            # Out-of-window candidates (row max etc.) clamp into the edge
            # bins — cumulative counts from the top stay exact — and the
            # bracket is P2's acceptance band, far narrower than
            # [cand_min, cand_max], so level-1 bin resolution IMPROVES.
            # Any path that leaves the bracket stale (degenerate-bracket
            # fallback, probe variants) fails the guard and takes the
            # original min/max scan; a plausible-but-wrong bracket can
            # only cost extra snap/refinement steps, never exactness.
            # Uniform branch: SMEM scalars read after the P3-exit barrier.
            w_lo = s_thr[1]
            w_hi = s_thr[2]
            bmin_r = cutlass.Float32(0.0)
            bmax_r = cutlass.Float32(1e-6)
            if (
                s_thr[0] == w_lo
                and w_hi > w_lo
                and w_hi < cutlass.Float32(self.FLT_MAX)
            ):
                bmin_r = w_lo
                bmax_r = w_hi
            else:
                # Block min/max over keys[0:cand_count]
                local_cmin = cutlass.Float32(self.FLT_MAX)
                local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
                i5 = tidx
                while i5 < cand_count:
                    v = self._smem_ld(cutlass.Float32, keys_base, i5)
                    local_cmin = _fmin_f32_inline(local_cmin, v)
                    local_cmax = cute.arch.fmax(local_cmax, v)
                    i5 = i5 + cutlass.Int32(num_threads)
                cmin = self.warp_reduce_min_f32(local_cmin)
                cmax = self.warp_reduce_max_f32(local_cmax)
                # Stage warp results into smem_wcnt[w] (cmin) and smem_hist[w] (cmax)
                # as bit-cast int32. cmax stored at smem_hist[0..NW-1].
                if lane == 0:
                    smem_wcnt[warp_id] = float_as_uint32(cmin)
                    smem_hist[warp_id] = float_as_uint32(cmax)
                cute.arch.barrier()

                # Every thread independently recomputes block_min/block_max
                # from the warp-staged smem slots (CUDA heuristic_topk.cuh:891-898
                # pattern). No tid==0 → s_thr broadcast → saves a block barrier.
                bmin_r = cutlass.Float32(self.FLT_MAX)
                bmax_r = cutlass.Float32(self.NEG_FLT_MAX)
                # Unrolled num_warps times (16 or 32 — fixed at compile time).
                for w in cutlass.range_constexpr(self.num_warps):
                    vmin_bits = smem_wcnt[w]
                    vmax_bits = smem_hist[w]
                    vmin = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vmin_bits.ir_value())
                    )
                    vmax = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vmax_bits.ir_value())
                    )
                    bmin_r = _fmin_f32_inline(bmin_r, vmin)
                    bmax_r = cute.arch.fmax(bmax_r, vmax)
                if bmax_r <= bmin_r:
                    bmax_r = bmin_r + cutlass.Float32(1e-6)
                # Barrier required: smem_hist[0..NW-1] above doubles as cmax
                # scratch and below as the histogram. Without this sync the
                # zeroing pass below can clobber a cmax slot a later warp is
                # still reading → wrong bmax_r → all candidates squashed into
                # bin 0 (hit-rate-dependent race).
                cute.arch.barrier()

            range1 = bmax_r - bmin_r
            # Overflow hardening (pre-existing):
            # a candidate span > FLT_MAX (needs |v| ~ 1.7e38; fuzz-only for
            # real logits) overflows range1 to +inf → inv1 = +0 → every
            # candidate lands in bin 0 → thr = lo + 0*inf = NaN → all snap
            # comparisons false, the walk never moves, and the whole row
            # writes as padding. Clamp to FLT_MAX: the start threshold
            # stays ORDERED (±inf is fine — snap's monotone walk rescues
            # any ordered start; only NaN breaks it).
            if range1 > cutlass.Float32(self.FLT_MAX):
                range1 = cutlass.Float32(self.FLT_MAX)
            # inv1 = (kBins - 1 + 0.99) / range1  (range1 > 0 guaranteed by 1e-6 patch)
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            binw1 = range1 / cutlass.Float32(kBins)

            # Predeclared register state for the redundant-warp path
            # (threshold / counts / staging parity live in registers; the
            # leader path below keeps them in s_thr/s_iscalars instead).
            thr_reg = bmin_r
            selc_reg = cutlass.Int32(0)
            thr_s = bmin_r
            cge_r = cutlass.Int32(0)
            cgt_r = cutlass.Int32(0)
            win_par = cutlass.Int32(0)

            # Level-1: histogram over [bmin, bmax] + k-th bin search.
            self._hist_build(keys_base, smem_hist, cand_count, bmin_r, inv1, tidx)
            if cutlass.const_expr(self.p4_warp_redundant):
                thr_reg, selc_reg = self._kth_bin_search_rw(
                    smem_hist, smem_wcnt, bmin_r, binw1, tidx, warp_id, lane
                )
            else:
                self._kth_bin_search(
                    smem_hist,
                    smem_wcnt,
                    s_thr,
                    s_iscalars,
                    bmin_r,
                    binw1,
                    tidx,
                    warp_id,
                    lane,
                )

            # ---- Level-2 histogram refinement ----
            # The snap loop below steps ONE distinct value per iteration
            # (~0.45us each: full candidate re-scan + 2 barriers), and real
            # logits concentrate count mass right at the k-th boundary, so
            # the selected level-1 bin often holds tens of values → snap
            # stragglers of 10+ us set the wall clock at N<=32K. When the
            # selected bin is dense, re-histogram just that bin (bin width
            # shrinks kBins x) for ~1us of extra scan, leaving the snap
            # loop 0-2 steps. The snap loop converges monotonically from
            # any starting threshold, so this only moves the start point —
            # exactness is untouched (a level-2 edge-rounding error at
            # worst costs one extra snap step). Uniform branch: everyone
            # reads the same post-barrier SMEM scalar.
            # Level 2 fires when a snap walk would cost more than one
            # rebuild (~2 snap steps break even); level 3 only when level 2
            # failed to split the bin (>8: heavy ties or a sub-ulp-wide
            # window — both rare on real logits, where ties at the k-th
            # are ~1 and the acceptance band spans >>1 ulp).
            binw_cur = binw1
            for _lvl in cutlass.range_constexpr(2):
                if cutlass.const_expr(self.p4_warp_redundant):
                    sel_cnt_l = selc_reg
                else:
                    sel_cnt_l = s_iscalars[4]
                gate_l = cutlass.const_expr(2 if _lvl == 0 else 8)
                if sel_cnt_l > cutlass.Int32(gate_l):
                    if cutlass.const_expr(self.p4_warp_redundant):
                        thr_el = thr_reg
                        # _kth_bin_search_rw has no trailing barrier; the
                        # zero pass of the rebuild below must not clobber
                        # smem_hist under a warp still in its step 3.
                        cute.arch.barrier()
                    else:
                        thr_el = s_thr[0]
                    # 2% slop each side absorbs the inv-vs-binw rounding
                    # difference in the previous level's edge estimate.
                    lo_l = thr_el - cutlass.Float32(0.02) * binw_cur
                    range_l = cutlass.Float32(1.04) * binw_cur
                    inv_l = (
                        cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)
                    ) / range_l
                    binw_next = range_l / cutlass.Float32(kBins)
                    self._hist_build(
                        keys_base, smem_hist, cand_count, lo_l, inv_l, tidx
                    )
                    if cutlass.const_expr(self.p4_warp_redundant):
                        thr_l2, selc_l2 = self._kth_bin_search_rw(
                            smem_hist, smem_wcnt, lo_l, binw_next, tidx, warp_id, lane
                        )
                        thr_reg = thr_l2
                        selc_reg = selc_l2
                    else:
                        self._kth_bin_search(
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            lo_l,
                            binw_next,
                            tidx,
                            warp_id,
                            lane,
                        )
                    binw_cur = binw_next

            # ---- Snap convergence loop ----
            # Upper bound = cand_count (matches CUDA heuristic_topk.cuh:985).
            # Common path converges in 1-3 iters; the loose ceiling only
            # matters for adversarial cells where a tighter bound would
            # accept a non-converged threshold (~0.09% of distributions).
            snap_limit = cand_count

            # Runtime break via a guard flag — no `break` in cute.range.
            si = cutlass.Int32(0)
            done_snap = cutlass.Int32(0)
            if cutlass.const_expr(self.p4_warp_redundant):
                # Redundant-warp snap: threshold + convergence state live
                # in registers (every warp reduces the staged partials
                # itself, bit-identically), so each iteration needs ONE
                # barrier (staging visibility) instead of two. Staging is
                # parity double-buffered in smem_hist[par*3NW ..] so a
                # warp one iteration ahead writes the other bank while a
                # slow warp still reads the old one; the staging barrier
                # bounds the drift to a single iteration.
                cute.arch.barrier()  # rw-search step-3 readers vs staging
                nwc = cutlass.const_expr(self.num_warps)
                thr_s = thr_reg
                par4 = cutlass.Int32(0)
                while si < snap_limit and done_snap == cutlass.Int32(0):
                    lge4 = cutlass.Int32(0)
                    lgt4 = cutlass.Int32(0)
                    up4 = cutlass.Float32(self.FLT_MAX)
                    dn4 = cutlass.Float32(self.NEG_FLT_MAX)
                    isi4 = tidx
                    while isi4 < cand_count:
                        v4 = self._smem_ld(cutlass.Float32, keys_base, isi4)
                        if v4 >= thr_s:
                            lge4 = lge4 + cutlass.Int32(1)
                        if v4 > thr_s:
                            lgt4 = lgt4 + cutlass.Int32(1)
                            up4 = _fmin_f32_inline(up4, v4)
                        if v4 < thr_s:
                            dn4 = cute.arch.fmax(dn4, v4)
                        isi4 = isi4 + cutlass.Int32(num_threads)
                    packed4 = (lge4 << cutlass.Int32(16)) | lgt4
                    packed4 = self.warp_reduce_sum_i32(packed4)
                    up4 = self.warp_reduce_min_f32(up4)
                    dn4 = self.warp_reduce_max_f32(dn4)
                    off4 = par4 * cutlass.Int32(3 * nwc)
                    if lane == 0:
                        smem_hist[off4 + warp_id] = packed4
                        smem_hist[off4 + cutlass.Int32(nwc) + warp_id] = (
                            float_as_uint32(up4)
                        )
                        smem_hist[off4 + cutlass.Int32(2 * nwc) + warp_id] = (
                            float_as_uint32(dn4)
                        )
                    cute.arch.barrier()
                    v_tp = cutlass.Int32(0)
                    v_up = cutlass.Float32(self.FLT_MAX)
                    v_dn = cutlass.Float32(self.NEG_FLT_MAX)
                    if lane < cutlass.Int32(nwc):
                        v_tp = smem_hist[off4 + lane]
                        vu_b = smem_hist[off4 + cutlass.Int32(nwc) + lane]
                        vd_b = smem_hist[off4 + cutlass.Int32(2 * nwc) + lane]
                        v_up = cutlass.Float32(
                            llvm.bitcast(cutlass.Float32.mlir_type, vu_b.ir_value())
                        )
                        v_dn = cutlass.Float32(
                            llvm.bitcast(cutlass.Float32.mlir_type, vd_b.ir_value())
                        )
                    tp4 = self.warp_reduce_sum_i32(v_tp)
                    tup4 = self.warp_reduce_min_f32(v_up)
                    tdn4 = self.warp_reduce_max_f32(v_dn)
                    cge_r = tp4 >> cutlass.Int32(16)
                    cgt_r = tp4 & cutlass.Int32(0xFFFF)
                    win_par = par4
                    if cgt_r >= cutlass.Int32(kK):
                        if tup4 < cutlass.Float32(self.FLT_MAX):
                            thr_s = tup4
                    elif cge_r < cutlass.Int32(kK):
                        if tdn4 > cutlass.Float32(self.NEG_FLT_MAX):
                            thr_s = tdn4
                    if cgt_r < cutlass.Int32(kK) and cge_r >= cutlass.Int32(kK):
                        done_snap = cutlass.Int32(1)
                    par4 = par4 ^ cutlass.Int32(1)
                    si = si + cutlass.Int32(1)
            else:
                while si < snap_limit and done_snap == cutlass.Int32(0):
                    self.block_fused_snap_iter(
                        keys_base,
                        smem_wcnt,
                        smem_hist,
                        s_thr,
                        s_iscalars,
                        cand_count,
                        tidx,
                        warp_id,
                        lane,
                    )
                    # After block_fused_snap_iter, s_iscalars[2]=cge, s_iscalars[3]=cgt.
                    cgt_c = self._smem_ld(cutlass.Int32, isc_base, cutlass.Int32(3))
                    cge_c = self._smem_ld(cutlass.Int32, isc_base, cutlass.Int32(2))
                    if cgt_c < cutlass.Int32(kK) and cge_c >= cutlass.Int32(kK):
                        done_snap = cutlass.Int32(1)
                    si = si + cutlass.Int32(1)

            # ---- Writeback (ballot + popc) ----
            # Converged snap (the overwhelmingly common case): SINGLE pass.
            # The converged iteration's cgt (s_iscalars[3]) is the exact
            # strictly-greater count at sel_thr (block_fused_snap_iter does
            # not move the threshold when cgt < kK <= cge), so gt entries
            # can pack into [0, cgt) via counter s_iscalars[4] while
            # tie(==) entries start at offset cgt via counter s_iscalars[5]
            # — same [gt | eq | pad] output partition as the two-pass
            # original, one candidate sweep and one barrier fewer. The
            # non-converged fallback keeps the original two-pass (its cgt
            # would be stale: the last iter may have moved the threshold
            # after counting).
            if cutlass.const_expr(self.p4_warp_redundant):
                sel_thr = thr_s
            else:
                sel_thr = s_thr[0]
            if tidx == 0:
                s_iscalars[4] = cutlass.Int32(0)  # gt out_count
                # s_iscalars[5] (cluster-local scratch, consumed before P4)
                # is reused as the eq counter for the single-pass path.
                s_iscalars[5] = cutlass.Int32(0)
            cute.arch.barrier()

            if done_snap == cutlass.Int32(1):
                # Zero-atomic single pass. The converged snap iteration
                # staged each warp's packed(ge<<16|gt) counts AT sel_thr in
                # smem_wcnt[w] (nothing touches smem_wcnt between the snap
                # exit and here), and the snap scan's tidx-strided
                # partition covers exactly the same element set per warp
                # as this warp-chunk scan. So every warp derives its
                # deterministic output bases from a prefix over
                # smem_wcnt — the ~2*cand/32 serialized SMEM atomics of
                # the claim-based scheme (a top stall region in ncu at
                # N=8K) disappear. Output order within the [gt | eq]
                # segments changes (deterministic instead of claim order),
                # which the contract allows.
                if cutlass.const_expr(self.p4_warp_redundant):
                    cgt_base = cgt_r
                else:
                    cgt_base = s_iscalars[3]
                gt_run = cutlass.Int32(0)
                eq_run = cutlass.Int32(0)
                for wpre in cutlass.range_constexpr(self.num_warps):
                    if cutlass.const_expr(self.p4_warp_redundant):
                        # Converged iteration's packed counts live in the
                        # winning parity bank of smem_hist, not smem_wcnt.
                        pk_w = smem_hist[
                            win_par * cutlass.Int32(3 * self.num_warps) + wpre
                        ]
                    else:
                        pk_w = smem_wcnt[wpre]
                    if cutlass.Int32(wpre) < warp_id:
                        wge_w = pk_w >> cutlass.Int32(16)
                        wgt_w = pk_w & cutlass.Int32(0xFFFF)
                        gt_run = gt_run + wgt_w
                        eq_run = eq_run + (wge_w - wgt_w)
                eq_run = cgt_base + eq_run
                base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
                while base_w < cand_count:
                    ix1 = base_w + lane
                    emit_gt = cutlass.Int32(0)
                    emit_eq = cutlass.Int32(0)
                    v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                    if ix1 < cand_count:
                        v_p1 = self._smem_ld(cutlass.Float32, keys_base, ix1)
                        if v_p1 > sel_thr:
                            emit_gt = cutlass.Int32(1)
                        if v_p1 == sel_thr:
                            emit_eq = cutlass.Int32(1)
                    mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                    lane_mask = (
                        cutlass.Uint32(1) << cutlass.Uint32(lane)
                    ) - cutlass.Uint32(1)
                    if mask_gt != cutlass.Uint32(0):
                        moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask))
                        wpos_p1 = gt_run + moff_gt
                        if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p1] = self.dtype(v_p1)
                            output_indices_row[wpos_p1] = self._smem_ld(
                                cutlass.Int32, vals_base, ix1
                            )
                        gt_run = gt_run + cutlass.Int32(cute.arch.popc(mask_gt))
                    mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                    if mask_eq != cutlass.Uint32(0):
                        moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask))
                        wpos_p2 = eq_run + moff_eq
                        if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p2] = self.dtype(v_p1)
                            output_indices_row[wpos_p2] = self._smem_ld(
                                cutlass.Int32, vals_base, ix1
                            )
                        eq_run = eq_run + cutlass.Int32(cute.arch.popc(mask_eq))
                    base_w = base_w + cutlass.Int32(num_threads)
                cute.arch.barrier()
            else:
                # Pass 1: v > sel_thr, strided over (warp_id * WARP_SIZE, ...).
                base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
                while base_w < cand_count:
                    ix1 = base_w + lane
                    emit_gt = cutlass.Int32(0)
                    v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                    if ix1 < cand_count:
                        v_p1 = self._smem_ld(cutlass.Float32, keys_base, ix1)
                        if v_p1 > sel_thr:
                            emit_gt = cutlass.Int32(1)
                    mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                    if mask_gt != cutlass.Uint32(0):
                        cnt_gt = cutlass.Int32(cute.arch.popc(mask_gt))
                        lane_mask_gt = (
                            cutlass.Uint32(1) << cutlass.Uint32(lane)
                        ) - cutlass.Uint32(1)
                        moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask_gt))
                        bp_gt = cutlass.Int32(0)
                        if lane == cutlass.Int32(0):
                            bp_gt = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(4),
                                cnt_gt,
                            )
                        bp_gt = cute.arch.shuffle_sync(bp_gt, cutlass.Int32(0))
                        wpos_p1 = bp_gt + moff_gt
                        if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p1] = self.dtype(v_p1)
                            output_indices_row[wpos_p1] = self._smem_ld(
                                cutlass.Int32, vals_base, ix1
                            )
                    base_w = base_w + cutlass.Int32(num_threads)
                cute.arch.barrier()

                # Pass 2: v == sel_thr (same pattern + guard as Pass 1).
                base_w2 = warp_id * cutlass.Int32(self.WARP_SIZE)
                while base_w2 < cand_count:
                    ix2 = base_w2 + lane
                    emit_eq = cutlass.Int32(0)
                    v_p2 = cutlass.Float32(self.NEG_FLT_MAX)
                    if ix2 < cand_count:
                        v_p2 = self._smem_ld(cutlass.Float32, keys_base, ix2)
                        if v_p2 == sel_thr:
                            emit_eq = cutlass.Int32(1)
                    mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                    if mask_eq != cutlass.Uint32(0):
                        cnt_eq = cutlass.Int32(cute.arch.popc(mask_eq))
                        lane_mask_eq = (
                            cutlass.Uint32(1) << cutlass.Uint32(lane)
                        ) - cutlass.Uint32(1)
                        moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask_eq))
                        bp_eq = cutlass.Int32(0)
                        if lane == cutlass.Int32(0):
                            bp_eq = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(4),
                                cnt_eq,
                            )
                        bp_eq = cute.arch.shuffle_sync(bp_eq, cutlass.Int32(0))
                        wpos_p2 = bp_eq + moff_eq
                        if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[wpos_p2] = self.dtype(v_p2)
                            output_indices_row[wpos_p2] = self._smem_ld(
                                cutlass.Int32, vals_base, ix2
                            )
                    base_w2 = base_w2 + cutlass.Int32(num_threads)
                cute.arch.barrier()

            # Pad remainder with -self.FLT_MAX / -1. Single-pass filled =
            # cge (= cgt + total ties at sel_thr, from the converged snap
            # iteration; the zero-atomic path leaves counters untouched);
            # two-pass filled = counter [4] (gt + eq accumulated).
            filled_par = cutlass.Int32(0)
            if done_snap == cutlass.Int32(1):
                if cutlass.const_expr(self.p4_warp_redundant):
                    filled_par = cge_r
                else:
                    filled_par = s_iscalars[2]
            else:
                filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[ipad] = cutlass.Int32(-1)
                ipad = ipad + cutlass.Int32(num_threads)

        else:
            # ----- Branch C: cand_count < kK -----
            # Emit cand_count + pad
            i10 = tidx
            while i10 < cand_count:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i10] = self.dtype(
                        self._smem_ld(cutlass.Float32, keys_base, i10)
                    )
                output_indices_row[i10] = self._smem_ld(cutlass.Int32, vals_base, i10)
                i10 = i10 + cutlass.Int32(num_threads)
            if s_iscalars[6] == cutlass.Int32(0):  # plateau fill completes done=3
                i11 = cand_count + tidx
                while i11 < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[i11] = cutlass.Int32(-1)
                    i11 = i11 + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Main kernel — one CTA per row
    # CUDA source: heuristicTopKDecode.cu:49-93 (heuristicTopKMultiRowKernel)
    # ------------------------------------------------------------------
    @cute.kernel
    def gvr_topk_kernel(
        self,
        input_data: cute.Tensor,  # [numRows, stride0] dtype
        pre_idx: cute.Tensor,  # [numRows / next_n, pre_idx_stride] int32
        seq_lens: cute.Tensor,  # [numRows / next_n] int32
        output_values: cute.Tensor,  # [numRows, top_k] dtype
        output_indices: cute.Tensor,  # [numRows, top_k] int32
        order_row: cute.Tensor,  # [batch_size] int32 (or None when seqlen_sorted=False)
        previous_topk: cute.Tensor,  # optional [max_num_reqs, top_k] int32
        state_valid: cute.Tensor,  # optional [max_num_reqs] int32
        request_indices: cute.Tensor,  # optional [numRows] int32
    ):
        """Thin entry: bidx → row_idx → run_one_row.

        grid = (num_rows * cluster_size,) where num_rows = batch_size *
        next_n. cluster_id = bidx // cluster_size, cta_in_cluster ∈
        [0, cluster_size). CTA r scans row[r * N / cs : (r+1) * N / cs]
        in Phase 2, so the per-row GE-count scales as 1 / cs. At
        cluster_size == 1 this collapses to one CTA per row scanning
        the whole row.

        When ``self.seqlen_sorted`` is True, the LJF dispatch order
        operates at REQUEST granularity (``order_row`` has length
        batch_size = num_rows / next_n). The owning row is resolved as
        ``order_row[cluster_id // next_n] * next_n + cluster_id % next_n``
        so the ``next_n`` rows of one request stay contiguous in
        dispatch order. All ``cluster_size`` CTAs within a cluster see
        the same ``cluster_id`` and therefore the same row, preserving
        cluster-sync semantics.

        Body is extracted into :meth:`run_one_row` so other entries (e.g.
        the LB load-balance variant) can resolve ``row_idx`` differently
        from the mappings used here.
        """
        bidx, _, _ = cute.arch.block_idx()
        cluster_size = cutlass.const_expr(self.cluster_size)
        seqlen_sorted = cutlass.const_expr(self.seqlen_sorted)
        next_n = cutlass.const_expr(self.next_n)
        if cutlass.const_expr(cluster_size > 1):
            cluster_id = bidx // cluster_size
        else:
            cluster_id = bidx
        if cutlass.const_expr(seqlen_sorted):
            # order_row is request-level (batch_size); expand to row-level
            # via req_id * next_n + nn so a request's next_n rows stay
            # contiguous in dispatch order (mirrors the LB main entry).
            if cutlass.const_expr(next_n == 1):
                row_idx = order_row[cluster_id]
            else:
                req_offset = cluster_id // cutlass.Int32(next_n)
                nn = cluster_id % cutlass.Int32(next_n)
                req_id = order_row[req_offset]
                row_idx = req_id * cutlass.Int32(next_n) + nn
        else:
            row_idx = cluster_id
        self.run_one_row(
            row_idx,
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            previous_topk,
            state_valid,
            request_indices,
        )

    @cute.jit
    def run_one_row(
        self,
        row_idx,  # int32, owning row in [0, num_rows)
        input_data: cute.Tensor,  # [numRows, stride0] dtype
        pre_idx: cute.Tensor,  # [numRows / next_n, pre_idx_stride] int32
        seq_lens: cute.Tensor,  # [numRows / next_n] int32
        output_values: cute.Tensor,  # [numRows, top_k] dtype, optional
        output_indices: cute.Tensor,  # [numRows, top_k] int32
        previous_topk: cute.Tensor,  # optional [max_num_reqs, top_k] int32
        state_valid: cute.Tensor,  # optional [max_num_reqs] int32
        request_indices: cute.Tensor,  # optional [numRows] int32
    ):
        """Dispatch: compute per-row slice + cluster sync mode, call _run_phases.

        ``run_one_row`` only handles row resolution, SMEM allocation, and
        the per-row long-vs-short decision. Phase 1-4 are in
        :meth:`_run_phases`.

        Short-row degrade: when the actual row workload fits within ONE
        CTA's design slice (``ceil(max_seq_len / cluster_size)``), CTA 0
        solo-scans the row (do_cluster_sync=False, no cluster sync) and
        the other cluster CTAs fall through ``run_one_row`` without
        calling ``_run_phases``. CuTe DSL doesn't support runtime
        ``return``, so non-leader CTAs naturally reach
        ``griddepcontrol_launch_dependents`` at the end.
        """
        tidx, _, _ = cute.arch.thread_idx()

        next_n = cutlass.const_expr(self.next_n)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kC = cutlass.const_expr(self.kC)
        kNumBins = cutlass.const_expr(self.kNumBins)
        cluster_size = cutlass.const_expr(self.cluster_size)

        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        if cutlass.const_expr(cluster_size > 1):
            cta_in_cluster = cute.arch.block_idx_in_cluster()
        else:
            cta_in_cluster = cutlass.Int32(0)
        pre_idx_row_idx = row_idx // next_n
        # Temporal-shift offset, mirroring heuristicTopKDecode.cu PR #14219:
        #   cr == 1 (V3.2): (row % next_n) + 1 maps prev-step indices into this
        #     step's KV space (+1 for the newly appended token).
        #   cr  > 1 (V4):   0 — in compressed-index space, new entries are
        #     appended at the end so prev indices remain valid as-is.
        if cutlass.const_expr(self.compress_ratio == 1):
            pre_idx_offset = cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        else:
            pre_idx_offset = cutlass.Int32(0)

        # Per-row length. seq_lens is in uncompressed-token space; logits/preIdx
        # live in compressed-token-index space when cr > 1 → divide by cr.
        seq_len = seq_lens[pre_idx_row_idx]
        actual_kv_len = (
            seq_len
            - cutlass.Int32(next_n)
            + cutlass.Int32(row_idx % next_n)
            + cutlass.Int32(1)
        )
        if cutlass.const_expr(self.compress_ratio == 1):
            N = actual_kv_len
        else:
            N = actual_kv_len // cutlass.Int32(self.compress_ratio)

        # Slice per-row views.
        input_row = input_data[row_idx, None]
        pre_idx_row = pre_idx[pre_idx_row_idx, None]
        # When return_output_values=False, ``output_values`` is None at
        # launch and the gated writes below are compiled out; slicing into
        # None would crash so we keep the view None as well.
        if cutlass.const_expr(self.return_output_values):
            output_values_row = output_values[row_idx, None]
        else:
            output_values_row = None
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]

        cute.arch.griddepcontrol_wait()

        # ---- Shared memory allocation ----
        smem = SmemAllocator()
        # keys[kC] fp32 (P3 candidate values; smem keys always fp32 even for half-prec)
        # Use fp32 even for half-prec to make secant search algorithm keep the accuracy/precision and converge faster.
        smem_keys = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((kC,), order=(0,)),
            byte_alignment=128,
        )
        # vals[kC] int32 (P3 candidate indices)
        smem_vals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((kC,), order=(0,)),
            byte_alignment=128,
        )
        # histogram[kNumBins] int32 (P4 only)
        smem_hist = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((kNumBins,), order=(0,)),
            byte_alignment=128,
        )
        # per_thread_counts[BLOCK_SIZE] int32 (P2/P3 cached counts)
        smem_ptcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_threads,), order=(0,)),
            byte_alignment=128,
        )
        # warp_counts[NUM_WARPS] int32 (P3 prefix-sum scratch)
        # p2_warp_redundant parity-banks the Phase-2 staging (a warp one
        # round ahead writes the other half) — costs num_warps*4 bytes.
        smem_wcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (
                    2 * num_warps
                    if cutlass.const_expr(self.p2_warp_redundant)
                    else num_warps,
                ),
                order=(0,),
            ),
            byte_alignment=128,
        )
        # Phase-1 warp aggregates (fp32 + int32; ~256 bytes total)
        smem_wmin = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        smem_wmax = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        smem_wsum = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        smem_wcnt_p1 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=64,
        )
        # Float scalars: threshold, val_lo, val_hi
        s_thr = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((3,), order=(0,)),
            byte_alignment=16,
        )
        # Int scalars:
        #   [0] cand_count   (cluster-aggregated total at cs>1; local total at cs=1)
        #   [1] done
        #   [2] cnt_lo
        #   [3] cnt_hi
        #   [4] out_count
        #   [5] local cand_count  (per-CTA snapshot before cluster all-reduce;
        #                          consumed by the kernel-level cluster handoff)
        #   [6] plateau terminal flag, captured from [1] BEFORE Phase 4
        #       (Phase 4 REUSES [1] as radix scratch, so the terminal must
        #       never be re-read from it afterwards)
        #   [7] plateau fill ticket (done == 3 only)
        s_iscalars = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16,
        )
        # Per-CTA DSMEM scratch for the cluster all-reduce of cand_count:
        # slots 0/1 = parity double-buffered count exchange (call k writes
        # slot k&1 — closes the straggler-read-vs-next-write DSMEM race),
        # slot 2 = tid0-private call counter. mapa.shared::cluster relies
        # on every CTA holding this block at the SAME SMEM offset, so it's
        # allocated once here. Only USED at cs>1 (uses are gated by
        # const_expr(cs>1)), but ALLOCATED unconditionally: the LB hybrid
        # kernel inlines a cs>1 and a cs=1 instance into one launch, and the
        # DSL sizes the launch SMEM from the last-traced SmemAllocator only —
        # the layouts must stay byte-identical across cluster_size (16B cost
        # at cs=1).
        s_cluster_partial = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((3,), order=(0,)),
            byte_alignment=16,
        )
        if cutlass.const_expr(cluster_size > 1):
            # Zero the call counter before any block_count_ge call. tid0-
            # private (same thread reads/increments it), so program order
            # suffices — but parity must start at 0 on EVERY CTA of the
            # cluster for lockstep alignment.
            if tidx == cutlass.Int32(0):
                s_cluster_partial[2] = cutlass.Int32(0)

        # SMEM slice cache (optional). Sized in ``self.dtype`` so the same
        # vec_w-wide LDG→STS→LDS pipeline works for fp32/bf16/fp16.
        # enable_smem_cache=False by default; caller ensures slice_len <=
        # smem_cache_elems before enabling (no runtime guard in kernel).
        if cutlass.const_expr(self.enable_smem_cache):
            smem_input = smem.allocate_tensor(
                element_type=self.dtype,
                layout=cute.make_ordered_layout((self.smem_cache_elems,), order=(0,)),
                byte_alignment=128,
            )
        else:
            smem_input = None

        # R0 admission scratch (single-CTA fast path). Allocated only
        # when enable_r0; None otherwise so the base SMEM layout is byte-for-
        # byte unchanged and these propagate harmlessly through _run_phases'
        # const_expr(enable_r0)-gated branch (same idiom as s_cluster_partial
        # / smem_input above). smem_ptcnt_multi caches M per-thread count
        # columns; s_r0col carries the accepted rung index tid0 -> all.
        if cutlass.const_expr(self.enable_r0):
            M_r0 = cutlass.const_expr(self.M_thr)
            # vseed (v3): the pmean column's per-thread counts reuse the
            # existing single-column smem_ptcnt buffer, so the BIG multi
            # buffer only holds the M_qf rung columns -> zero smem growth
            # (the round-1 +2-4KB column pushed 16-bit mb3/T1024 configs over
            # an occupancy cliff: K2048 fp16 BS1024 -26%).
            M_r0_pt = cutlass.const_expr(self.M_qf)
            s_mt_thr = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_ordered_layout((M_r0,), order=(0,)),
                byte_alignment=16,
            )
            smem_ptcnt_multi = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0_pt * num_threads,), order=(0,)),
                byte_alignment=128,
            )
            smem_wcnt_multi = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0 * num_warps,), order=(0,)),
                byte_alignment=64,
            )
            s_mt_cnt = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0,), order=(0,)),
                byte_alignment=16,
            )
            s_r0col = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=16,
            )
            # DSMEM scratch for the M-way cluster all-reduce of the R0 rung
            # counts (mapa.shared::cluster needs the same offset on every
            # CTA). Only USED at cs>1; allocated unconditionally so the
            # cs=1 / cs>1 SMEM layouts stay byte-identical for the LB
            # hybrid kernel (see s_cluster_partial above).
            s_cluster_partial_m = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((M_r0,), order=(0,)),
                byte_alignment=16,
            )
            # p1b_cache: P1 stashes the K gathered preIdx values here so P1b
            # skips a second GMEM random gather (dtype-gated: 16-bit only).
            if cutlass.const_expr(self.p1b_cache):
                smem_gath = smem.allocate_tensor(
                    element_type=cutlass.Float32,
                    layout=cute.make_ordered_layout((self.top_k,), order=(0,)),
                    byte_alignment=128,
                )
            else:
                smem_gath = None
        else:
            s_mt_thr = None
            smem_ptcnt_multi = None
            smem_wcnt_multi = None
            s_mt_cnt = None
            s_r0col = None
            s_cluster_partial_m = None
            smem_gath = None

        prepared_pre_idx_row = pre_idx_row
        use_cold_hints = cutlass.Boolean(False)
        cold_prior_len = cutlass.Int32(0)
        if cutlass.const_expr(self.fuse_hint_prepare):
            hint_request_idx = request_indices[row_idx]
            safe_request_idx = cutlass.Int64(0)
            has_state = cutlass.Boolean(False)
            if hint_request_idx >= 0:
                safe_request_idx = cutlass.Int64(hint_request_idx)
                if seq_len > cutlass.Int32(1):
                    has_state = state_valid[safe_request_idx]
            prepared_pre_idx_row = previous_topk[safe_request_idx, None]
            use_cold_hints = ~has_state
            cold_prior_len = seq_len - cutlass.Int32(1)
            if cold_prior_len < cutlass.Int32(1):
                cold_prior_len = cutlass.Int32(1)

        # ---- Per-row dispatch ----
        # Three branches:
        #   1. Degenerate (N <= top_k): no GVR work, leader emits identity.
        #   2. cs>1 long row:           all cluster CTAs cooperate.
        #   3. cs>1 short row OR cs=1:  leader/single CTA runs solo.
        # Non-leader CTAs in (1)/(3) fall through to the function end (CuTe
        # DSL doesn't support runtime ``return``).
        top_k = cutlass.const_expr(self.top_k)
        if N <= cutlass.Int32(top_k):
            # Degenerate: no GVR, just emit [0..N-1] + (-1) padding.
            # Leader-only write (was an idempotent race across cluster CTAs).
            if cta_in_cluster == cutlass.Int32(0):
                jd = tidx
                while jd < N:
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[jd] = input_row[jd]
                    output_indices_row[jd] = cutlass.Int32(jd)
                    jd = jd + cutlass.Int32(num_threads)
                jp = N + cutlass.Int32(tidx)
                while jp < cutlass.Int32(top_k):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[jp] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[jp] = cutlass.Int32(-1)
                    jp = jp + cutlass.Int32(num_threads)
        else:
            # Normal GVR. Long vs short row decision threshold =
            # ceil(max_seq_len / cluster_size) = one CTA's design
            # workload. When actual seq_len fits within that, cluster
            # cooperation overhead exceeds the work saved → degrade to
            # CTA 0 solo.
            if cutlass.const_expr(cluster_size > 1):
                # max_slice_len: per-CTA slice upper bound when the row is
                # long enough to warrant cluster cooperation.
                max_slice_len = (
                    input_data.shape[1] + cutlass.Int32(cluster_size - 1)
                ) // cutlass.Int32(cluster_size)
                if N > max_slice_len:
                    # Long row: cluster cooperation, all cs CTAs scan
                    # N/cs. slice_base rounded DOWN to vec_w so each
                    # CTA's slice_start stays vec_w-aligned; the last
                    # CTA absorbs the N mod cs remainder.
                    vec_w_const = cutlass.const_expr(self.vec_bits // self.dtype.width)
                    raw_base = N // cutlass.Int32(cluster_size)
                    slice_base = (
                        raw_base // cutlass.Int32(vec_w_const)
                    ) * cutlass.Int32(vec_w_const)
                    slice_start = cta_in_cluster * slice_base
                    slice_is_last = cta_in_cluster == cutlass.Int32(cluster_size - 1)
                    slice_end = N if slice_is_last else (slice_start + slice_base)
                    self._run_phases(
                        input_row,
                        prepared_pre_idx_row,
                        output_values_row,
                        output_indices_row,
                        N,
                        pre_idx_offset,
                        pre_idx_count,
                        use_cold_hints,
                        cold_prior_len,
                        slice_start,
                        slice_end,
                        cutlass.Boolean(True),
                        cta_in_cluster,
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_ptcnt,
                        smem_wcnt,
                        smem_wmin,
                        smem_wmax,
                        smem_wsum,
                        smem_wcnt_p1,
                        s_thr,
                        s_iscalars,
                        s_cluster_partial,
                        smem_input,
                        s_mt_thr,
                        smem_ptcnt_multi,
                        smem_wcnt_multi,
                        s_mt_cnt,
                        s_r0col,
                        s_cluster_partial_m,
                        smem_gath,
                        tidx,
                        warp_id,
                        lane,
                    )
                else:
                    # Short row: only CTA 0 scans the full row; the other
                    # (cluster_size - 1) CTAs fall through without entering
                    # _run_phases and naturally reach the function end.
                    if cta_in_cluster == cutlass.Int32(0):
                        self._run_phases(
                            input_row,
                            prepared_pre_idx_row,
                            output_values_row,
                            output_indices_row,
                            N,
                            pre_idx_offset,
                            pre_idx_count,
                            use_cold_hints,
                            cold_prior_len,
                            cutlass.Int32(0),
                            N,
                            cutlass.Boolean(False),
                            cta_in_cluster,
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_ptcnt,
                            smem_wcnt,
                            smem_wmin,
                            smem_wmax,
                            smem_wsum,
                            smem_wcnt_p1,
                            s_thr,
                            s_iscalars,
                            s_cluster_partial,
                            smem_input,
                            s_mt_thr,
                            smem_ptcnt_multi,
                            smem_wcnt_multi,
                            s_mt_cnt,
                            s_r0col,
                            s_cluster_partial_m,
                            smem_gath,
                            tidx,
                            warp_id,
                            lane,
                        )
            else:
                # cs=1: one CTA per row, no cluster sync.
                self._run_phases(
                    input_row,
                    prepared_pre_idx_row,
                    output_values_row,
                    output_indices_row,
                    N,
                    pre_idx_offset,
                    pre_idx_count,
                    use_cold_hints,
                    cold_prior_len,
                    cutlass.Int32(0),
                    N,
                    cutlass.Boolean(False),
                    cta_in_cluster,
                    smem_keys,
                    smem_vals,
                    smem_hist,
                    smem_ptcnt,
                    smem_wcnt,
                    smem_wmin,
                    smem_wmax,
                    smem_wsum,
                    smem_wcnt_p1,
                    s_thr,
                    s_iscalars,
                    s_cluster_partial,
                    smem_input,
                    s_mt_thr,
                    smem_ptcnt_multi,
                    smem_wcnt_multi,
                    s_mt_cnt,
                    s_r0col,
                    s_cluster_partial_m,
                    smem_gath,
                    tidx,
                    warp_id,
                    lane,
                )

        if cutlass.const_expr(self.fuse_state_store):
            cute.arch.barrier()
            if cta_in_cluster == cutlass.Int32(0):
                state_request_idx = request_indices[row_idx]
                if state_request_idx >= cutlass.Int32(0):
                    state_output_row = previous_topk[state_request_idx, None]
                    state_vec = cutlass.const_expr(4)
                    state_step = cutlass.const_expr(num_threads * state_vec)
                    state_src_addr = output_indices_row.iterator.toint()
                    state_dst_addr = state_output_row.iterator.toint()
                    state_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Int32,
                        num_bits_per_copy=128,
                    )
                    state_frag = cute.make_rmem_tensor((state_vec,), cutlass.Int32)
                    isv = tidx * cutlass.Int32(state_vec)
                    while isv < cutlass.Int32(top_k):
                        state_src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                state_src_addr + cutlass.Int64(isv) * cutlass.Int64(4),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((state_vec,)),
                        )
                        state_dst = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                state_dst_addr + cutlass.Int64(isv) * cutlass.Int64(4),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((state_vec,)),
                        )
                        cute.copy(state_copy, state_src, state_frag)
                        cute.copy(state_copy, state_frag, state_dst)
                        isv = isv + cutlass.Int32(state_step)
                    if tidx == cutlass.Int32(0):
                        state_valid[state_request_idx] = cutlass.Boolean(True)

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _run_phases(
        self,
        input_row,
        pre_idx_row,
        output_values_row,
        output_indices_row,
        N,
        pre_idx_offset,
        pre_idx_count,
        use_cold_hints,
        cold_prior_len,
        slice_start,
        slice_end,
        do_cluster_sync,
        cta_in_cluster,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_ptcnt,
        smem_wcnt,
        smem_wmin,
        smem_wmax,
        smem_wsum,
        smem_wcnt_p1,
        s_thr,
        s_iscalars,
        s_cluster_partial,
        smem_input,
        s_mt_thr,
        smem_ptcnt_multi,
        smem_wcnt_multi,
        s_mt_cnt,
        s_r0col,
        s_cluster_partial_m,
        smem_gath,
        tidx,
        warp_id,
        lane,
    ):
        """Run Phase 1-4 + final cluster barrier on a given row slice.

        Caller (``run_one_row``) decides slice + do_cluster_sync per row:
          - cs=1                 → slice=[0,N), do_cluster_sync=False
          - cs>1, long row       → slice=N/cs per CTA, do_cluster_sync=True
          - cs>1, short row      → slice=[0,N), do_cluster_sync=False, CTA 0 only

        Non-leader CTAs in short-row mode never call this helper.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        cluster_size = cutlass.const_expr(self.cluster_size)
        is_leader = cta_in_cluster == cutlass.Int32(0)

        # ---- Phase 1: preIdx Min/Max/Mean ----
        self.phase1_preidx_stats(
            input_row,
            N,
            pre_idx_row,
            pre_idx_count,
            pre_idx_offset,
            smem_wmin,
            smem_wmax,
            smem_wsum,
            smem_wcnt_p1,
            s_thr,
            s_iscalars,
            tidx,
            warp_id,
            lane,
            smem_gath=smem_gath,  # p1b_cache: stash gathered values (None-op OFF)
            s_mt_thr=s_mt_thr,  # r0_vseed: park pmean in the last rung column
            use_cold_hints=use_cold_hints,
            cold_prior_len=cold_prior_len,
        )

        # A stale or low-diversity temporal hint can gather one repeated value.
        # Retry with row-spanning cold hints before paying for a complete scan.
        v_lo = s_thr[1]
        v_hi = s_thr[2]
        if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
            self.phase1_preidx_stats(
                input_row,
                N,
                pre_idx_row,
                pre_idx_count,
                pre_idx_offset,
                smem_wmin,
                smem_wmax,
                smem_wsum,
                smem_wcnt_p1,
                s_thr,
                s_iscalars,
                tidx,
                warp_id,
                lane,
                smem_gath=smem_gath,
                s_mt_thr=s_mt_thr,
                use_cold_hints=cutlass.Boolean(True),
                cold_prior_len=N - cutlass.Int32(1),
            )
        v_lo = s_thr[1]
        v_hi = s_thr[2]
        if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
            self.phase1_full_row_bounds(
                input_row,
                N,
                smem_wmin,
                smem_wmax,
                s_thr,
                s_iscalars,
                tidx,
                warp_id,
                lane,
            )
        v_lo = s_thr[1]
        v_hi = s_thr[2]
        # Identity is valid only after the full-row scan proves every value is
        # identical (or the complete row is -inf).
        if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
            if cutlass.const_expr(cluster_size == 1):
                if tidx == 0:
                    top_k = cutlass.const_expr(self.top_k)
                    # Emit identity output (first min(top_k, N) indices)
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
            else:
                # cs>1: all cluster CTAs enter _run_phases; only leader writes.
                if is_leader & (tidx == cutlass.Int32(0)):
                    top_k = cutlass.const_expr(self.top_k)
                    # Emit identity output (first min(top_k, N) indices)
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
        else:
            # Stage this CTA's slice into SMEM once before Phase 2's
            # 6-10 secant iters re-scan it. Phase 1 (preIdx) uses
            # scatter-loads OUTSIDE this slice, so it stays on GMEM.
            if cutlass.const_expr(self.enable_smem_cache):
                self.load_slice_to_smem(
                    input_row,
                    slice_start,
                    slice_end,
                    smem_input,
                    tidx,
                )

            # ---- Phase 2: R0 histogram-ladder admission (single-CTA fast
            # path) or the secant threshold search ----
            # R0 covers every cluster size: at cs>1 each CTA scans its own
            # slice and block_count_ge_multi cluster-merges the rung counts
            # (the P1b rungs are per-CTA identical because the preIdx stats are
            # full-row). The secant search below is the exact fallback taken
            # when the ladder admits nothing, plus the enable_r0=False
            # differential-oracle entry.
            if cutlass.const_expr(self.enable_r0):
                # P1b rung placement -> ONE M-ary R0 count pass -> accept the
                # tightest rung with count in [K, kC]. On a miss, fall back to
                # the inline log-falsi R1 shot / fb_fix refine. At cs>1 each
                # CTA scans its slice and block_count_ge_multi cluster-merges
                # the rung counts (phase1b rungs are per-CTA identical since
                # preIdx stats are full-row).
                if cutlass.const_expr(self.p1b_cache):
                    # rungs from the SMEM gather-cache P1 stashed (no 2nd
                    # GMEM gather); 16-bit only.
                    self.phase1b_hspace_rungs_cached(
                        N,
                        pre_idx_count,
                        smem_gath,
                        smem_hist,
                        s_thr,
                        s_mt_thr,
                        tidx,
                        warp_id,
                        lane,
                    )
                else:
                    self.phase1b_hspace_rungs(
                        input_row,
                        N,
                        pre_idx_row,
                        pre_idx_count,
                        pre_idx_offset,
                        smem_hist,
                        s_thr,
                        s_mt_thr,
                        tidx,
                        warp_id,
                        lane,
                        use_cold_hints=use_cold_hints,
                        cold_prior_len=cold_prior_len,
                    )
                if cutlass.const_expr(self.adaptive_rungs):
                    use_default_rungs = N < cutlass.Int32(32768) or (
                        N >= cutlass.Int32(65536) and N < cutlass.Int32(131072)
                    )
                    if use_default_rungs:
                        if tidx == 0:
                            s_mt_thr[2] = s_thr[0]
                        cute.arch.barrier()
                self.block_count_ge_multi(
                    input_row,
                    slice_start,
                    slice_end,
                    s_mt_thr,
                    smem_ptcnt_multi,
                    smem_wcnt_multi,
                    s_mt_cnt,
                    s_cluster_partial_m,
                    do_cluster_sync,
                    tidx,
                    warp_id,
                    lane,
                    smem_ptcnt=smem_ptcnt,
                )
                cute.arch.barrier()
                if tidx == 0:
                    # tightest admissible rung = SMALLEST count in [K, kC].
                    # (Explicit argmin: with r0_vseed the pmean column is not
                    # sorted into the rung order; for sorted rungs this is
                    # equivalent to the old "last m in window" rule.)
                    best_m = cutlass.Int32(-1)
                    best_c = cutlass.Int32(2147483647)
                    for m in cutlass.range_constexpr(cutlass.const_expr(self.M_thr)):
                        cm = s_mt_cnt[m]
                        if (
                            cm >= cutlass.Int32(self.top_k)
                            and cm <= cutlass.Int32(self.kC)
                            and cm < best_c
                        ):
                            best_m = cutlass.Int32(m)
                            best_c = cm
                    s_r0col[0] = best_m
                    if best_m >= cutlass.Int32(0):
                        s_thr[0] = s_mt_thr[best_m]
                        s_iscalars[0] = s_mt_cnt[best_m]
                        # done=1: the threshold is admitted, so Phase 3 must
                        # SKIP its retry-shrink and honor s_thr[0]. (block_count
                        # _ge / secant leave done via their own path; the R0
                        # admission must set it explicitly or Phase 3 re-searches
                        # and the cluster collect diverges -> wrong output.)
                        s_iscalars[1] = cutlass.Int32(1)
                        # Snapshot this CTA's LOCAL slice count for the chosen
                        # rung into s_iscalars[5] — the per-CTA cand_count that
                        # Phase 3/4's cluster gather consumes (block_count_ge
                        # sets it too; the R0 admission must match). Without it
                        # the cluster collect under-counts -> wrong output.
                        if cutlass.const_expr(cluster_size > 1):
                            s_iscalars[5] = s_cluster_partial_m[best_m]
                cute.arch.barrier()
                bc = s_r0col[0]
                if bc >= cutlass.Int32(0) and bc < cutlass.Int32(self.M_qf):
                    # accepted rung column: copy its cached per-thread counts
                    # into the secant hand-off buffer (zero rescan). The vseed
                    # column (bc == M_qf) is ALREADY in smem_ptcnt (v3 reuse).
                    smem_ptcnt[tidx] = smem_ptcnt_multi[
                        bc * cutlass.Int32(num_threads) + tidx
                    ]
                cute.arch.barrier()
                # ---- R0 miss: SEEDED bounded log-falsi refine ----
                # At large N the M2D rungs straddle [K, kC]; the refine must
                # find a threshold with count in [K, kC] between the measured
                # rungs. SEED the loop with the rung bracket AND its known
                # counts (clo/chi) so it does log-count regula-falsi from
                # iter 0 with no re-measure and no separate R1 shot -> ~2-3
                # count passes instead of ~6. done=1 on
                # accept so Phase 3 skips its retry-shrink.
                if bc < cutlass.Int32(0):
                    if cutlass.const_expr(self.fb_fix):
                        if tidx == cutlass.Int32(0):
                            M = cutlass.const_expr(self.M_thr)
                            blo = v_lo
                            bhi = v_hi
                            clo = cutlass.Int32(-1)
                            chi = cutlass.Int32(-1)
                            for m in cutlass.range_constexpr(M):
                                cm = s_mt_cnt[m]
                                tm = s_mt_thr[m]
                                if cm > cutlass.Int32(self.kC) and (
                                    clo < cutlass.Int32(0) or tm > blo
                                ):
                                    blo = tm
                                    clo = cm
                                if cm < cutlass.Int32(self.top_k) and (
                                    chi < cutlass.Int32(0) or tm < bhi
                                ):
                                    bhi = tm
                                    chi = cm
                            s_thr[1] = blo
                            s_thr[2] = bhi
                            s_iscalars[2] = clo  # SEED known rung counts
                            s_iscalars[3] = chi
                            s_iscalars[1] = cutlass.Int32(0)  # done=0
                            cand = (blo + bhi) * cutlass.Float32(0.5)
                            if clo > cutlass.Int32(0) and chi >= cutlass.Int32(0):
                                chic = chi
                                if chic < cutlass.Int32(1):
                                    chic = cutlass.Int32(1)
                                l_lo = cmath.log2(cutlass.Float32(clo), fastmath=True)
                                l_hi = cmath.log2(cutlass.Float32(chic), fastmath=True)
                                den = l_lo - l_hi
                                if den > cutlass.Float32(0.0):
                                    t3 = (cutlass.Float32(self.log2_mstar) - l_hi) / den
                                    cnd3 = bhi + t3 * (blo - bhi)
                                    if cnd3 > blo and cnd3 < bhi:
                                        cand = cnd3
                            elif chi < cutlass.Int32(0):
                                cand = bhi
                            elif clo < cutlass.Int32(0):
                                cand = blo
                            s_thr[0] = cand
                        cute.arch.barrier()
                        rs = cutlass.Int32(0)
                        while rs < cutlass.Int32(8) and s_iscalars[1] == cutlass.Int32(
                            0
                        ):
                            if rs > cutlass.Int32(0):
                                if tidx == cutlass.Int32(0):
                                    lo3 = s_thr[1]
                                    hi3 = s_thr[2]
                                    clo3 = s_iscalars[2]
                                    chi3 = s_iscalars[3]
                                    cand = (lo3 + hi3) * cutlass.Float32(0.5)
                                    if chi3 < cutlass.Int32(0):
                                        cand = hi3
                                    elif clo3 < cutlass.Int32(0):
                                        cand = lo3
                                    else:
                                        chic = chi3
                                        if chic < cutlass.Int32(1):
                                            chic = cutlass.Int32(1)
                                        l_lo = cmath.log2(
                                            cutlass.Float32(clo3), fastmath=True
                                        )
                                        l_hi = cmath.log2(
                                            cutlass.Float32(chic), fastmath=True
                                        )
                                        den3 = l_lo - l_hi
                                        if den3 > cutlass.Float32(0.0):
                                            t3 = (
                                                cutlass.Float32(self.log2_mstar) - l_hi
                                            ) / den3
                                            cnd3 = hi3 + t3 * (lo3 - hi3)
                                            if cnd3 > lo3 and cnd3 < hi3:
                                                cand = cnd3
                                    s_thr[0] = cand
                                cute.arch.barrier()
                            self.block_count_ge(
                                input_row,
                                slice_start,
                                slice_end,
                                s_thr[0],
                                smem_ptcnt,
                                smem_wcnt,
                                s_iscalars,
                                s_cluster_partial,
                                tidx,
                                warp_id,
                                lane,
                                do_cluster_sync=do_cluster_sync,
                                smem_input=smem_input,
                            )
                            cute.arch.barrier()
                            if tidx == cutlass.Int32(0):
                                c3 = s_iscalars[0]
                                t3v = s_thr[0]
                                if c3 >= cutlass.Int32(
                                    self.top_k
                                ) and c3 <= cutlass.Int32(self.kC):
                                    s_iscalars[1] = cutlass.Int32(1)  # accept
                                elif c3 > cutlass.Int32(self.kC):
                                    s_thr[1] = t3v
                                    s_iscalars[2] = c3
                                    if t3v >= s_thr[2]:
                                        rng3 = s_thr[2] - s_thr[1]
                                        if rng3 < cutlass.Float32(1.0):
                                            rng3 = cutlass.Float32(1.0)
                                        s_thr[2] = s_thr[2] + rng3 * cutlass.Float32(
                                            8.0
                                        )
                                        s_iscalars[3] = cutlass.Int32(-1)
                                else:
                                    s_thr[2] = t3v
                                    s_iscalars[3] = c3
                                    if t3v <= s_thr[1]:
                                        rng3 = s_thr[2] - s_thr[1]
                                        if rng3 < cutlass.Float32(1.0):
                                            rng3 = cutlass.Float32(1.0)
                                        s_thr[1] = s_thr[1] - rng3 * cutlass.Float32(
                                            8.0
                                        )
                                        s_iscalars[2] = cutlass.Int32(-1)
                            cute.arch.barrier()
                            rs = rs + cutlass.Int32(1)
                        if s_iscalars[1] != cutlass.Int32(1):
                            # The retry budget could not land in [K, kC].
                            # ONLY the coherent undershoot-overflow corner
                            # (count(>= lo) > kC AND 0 <= count(>= hi) < K,
                            # both counts CURRENT — the retry's bracket
                            # widening marks a side stale with -1 and thus
                            # fails this guard) collapses the bracket by
                            # pure bisection to ADJACENT floats, where the
                            # plateau terminal (done = 3, threshold = hi)
                            # is exact: Phase 4 emits the sure winners and
                            # the plateau fill completes the row from the
                            # tie class. A mid-collapse count landing in
                            # [K, kC] converges normally; anything else
                            # (incl. an exhausted collapse budget) falls
                            # through to the fail-soft terminal below.
                            it4 = cutlass.Int32(0)
                            if (
                                s_iscalars[2] <= cutlass.Int32(self.kC)
                                or s_iscalars[3] < cutlass.Int32(0)
                                or s_iscalars[3] >= cutlass.Int32(self.top_k)
                            ):
                                it4 = cutlass.Int32(40)  # guard: skip collapse
                            while it4 < cutlass.Int32(40) and s_iscalars[
                                1
                            ] == cutlass.Int32(0):
                                if tidx == cutlass.Int32(0):
                                    lo4 = s_thr[1]
                                    hi4 = s_thr[2]
                                    mid4 = (lo4 + hi4) * cutlass.Float32(0.5)
                                    if mid4 == lo4 or mid4 == hi4:
                                        s_thr[0] = hi4
                                        s_iscalars[1] = cutlass.Int32(3)
                                    else:
                                        s_thr[0] = mid4
                                cute.arch.barrier()
                                if s_iscalars[1] == cutlass.Int32(0):
                                    self.block_count_ge(
                                        input_row,
                                        slice_start,
                                        slice_end,
                                        s_thr[0],
                                        smem_ptcnt,
                                        smem_wcnt,
                                        s_iscalars,
                                        s_cluster_partial,
                                        tidx,
                                        warp_id,
                                        lane,
                                        do_cluster_sync=do_cluster_sync,
                                        smem_input=smem_input,
                                    )
                                    cute.arch.barrier()
                                    if tidx == cutlass.Int32(0):
                                        c4 = s_iscalars[0]
                                        t4 = s_thr[0]
                                        if c4 >= cutlass.Int32(
                                            self.top_k
                                        ) and c4 <= cutlass.Int32(self.kC):
                                            s_iscalars[1] = cutlass.Int32(1)
                                        elif c4 > cutlass.Int32(self.kC):
                                            s_thr[1] = t4
                                            s_iscalars[2] = c4
                                        else:
                                            s_thr[2] = t4
                                            s_iscalars[3] = c4
                                    cute.arch.barrier()
                                it4 = it4 + cutlass.Int32(1)
                            if s_iscalars[1] == cutlass.Int32(3):
                                # recount at the terminal threshold so P3's
                                # cached per-thread counts describe the
                                # sure-winner set the fill completes.
                                self.block_count_ge(
                                    input_row,
                                    slice_start,
                                    slice_end,
                                    s_thr[0],
                                    smem_ptcnt,
                                    smem_wcnt,
                                    s_iscalars,
                                    s_cluster_partial,
                                    tidx,
                                    warp_id,
                                    lane,
                                    do_cluster_sync=do_cluster_sync,
                                    smem_input=smem_input,
                                )
                                cute.arch.barrier()
                            elif s_iscalars[1] != cutlass.Int32(1):
                                # fail-soft (non-plateau): land on the
                                # measured undershoot side (count <= kC =>
                                # no overflow; -1 pad stays the documented
                                # non-convergence encoding).
                                self.block_count_ge(
                                    input_row,
                                    slice_start,
                                    slice_end,
                                    s_thr[2],
                                    smem_ptcnt,
                                    smem_wcnt,
                                    s_iscalars,
                                    s_cluster_partial,
                                    tidx,
                                    warp_id,
                                    lane,
                                    do_cluster_sync=do_cluster_sync,
                                    smem_input=smem_input,
                                )
                                cute.arch.barrier()
                                if tidx == cutlass.Int32(0):
                                    s_thr[0] = s_thr[2]
                                    s_iscalars[1] = cutlass.Int32(1)
                                cute.arch.barrier()
                    else:
                        self.phase2_secant_search(
                            input_row,
                            N,
                            slice_start,
                            slice_end,
                            smem_ptcnt,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            s_cluster_partial,
                            tidx,
                            warp_id,
                            lane,
                            do_cluster_sync=do_cluster_sync,
                            smem_input=smem_input,
                        )
            else:
                self.phase2_secant_search(
                    input_row,
                    N,
                    slice_start,
                    slice_end,
                    smem_ptcnt,
                    smem_wcnt,
                    s_thr,
                    s_iscalars,
                    s_cluster_partial,
                    tidx,
                    warp_id,
                    lane,
                    do_cluster_sync=do_cluster_sync,
                    smem_input=smem_input,
                )

            # Cluster handoff #1 (end of Phase 2). Skipped when
            # do_cluster_sync is False (cs=1 or short-row degrade).
            if cutlass.const_expr(cluster_size > 1):
                if do_cluster_sync:
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()

            # ---- Phase 3: cluster-parallel candidate collect ----
            self.phase3_collect_candidates(
                input_row,
                N,
                slice_start,
                slice_end,
                smem_keys,
                smem_vals,
                smem_ptcnt,
                smem_wcnt,
                s_thr,
                s_iscalars,
                s_cluster_partial,
                tidx,
                warp_id,
                lane,
                do_cluster_sync=do_cluster_sync,
                smem_input=smem_input,
            )

            # Cluster handoff #2: leader's DSMEM gather of peer
            # smem_keys/smem_vals. Skipped at do_cluster_sync=False.
            if cutlass.const_expr(cluster_size > 1):
                if do_cluster_sync:
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

            # Phase 4 runs on the leader only. const_expr (compile-
            # time eliminated) split from runtime so cs=1 gets a flat
            # code path with no leader/sync checks.
            # Pre-init cand_count_p4 so CuTe DSL sees a stable Int32 type
            # across the runtime ``if is_leader:`` branch in cs>1 mode
            # (DSL forbids first-assigning a variable inside a dynamic if).
            cand_count_p4 = cutlass.Int32(0)
            if cutlass.const_expr(cluster_size == 1):
                # cs=1: the single CTA per row IS the leader.
                # Capture the P2 terminal BEFORE Phase 4: P4 reuses
                # s_iscalars[1] as radix scratch.
                if tidx == cutlass.Int32(0):
                    s_iscalars[6] = cutlass.Int32(0)
                    if s_iscalars[1] == cutlass.Int32(3):
                        s_iscalars[6] = cutlass.Int32(1)
                cute.arch.barrier()
                cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                if cutlass.const_expr(self.enable_p4_rank_scatter):
                    self.phase4_rank_scatter(
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_wcnt,
                        s_thr,
                        s_iscalars,
                        output_values_row,
                        output_indices_row,
                        cand_count_p4,
                        tidx,
                        warp_id,
                        lane,
                    )
                else:
                    self.phase4_histogram_snap(
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_wcnt,
                        s_thr,
                        s_iscalars,
                        output_values_row,
                        output_indices_row,
                        cand_count_p4,
                        tidx,
                        warp_id,
                        lane,
                    )
                # ---- plateau fill (done == 3): complete the row from the
                # bitwise-equal plateau class. The terminal is only set on an
                # ADJACENT-FLOAT bracket, so every value in [s_thr[1], s_thr[0])
                # is bitwise-equal; Phase 4 has already emitted the
                # cnt(>= s_thr[0]) sure winners, and ANY (K - count)-subset of
                # the tie class is a valid tie-aware completion. Ticket counter
                # lives in the DEDICATED s_iscalars[7].
                if s_iscalars[6] == cutlass.Int32(1):
                    pv_lo = s_thr[1]
                    pv_hi = s_thr[0]
                    if tidx == cutlass.Int32(0):
                        # cand_count_p4 was captured BEFORE Phase 4; s_iscalars[0]
                        # is radix scratch by now (same hazard as the flag).
                        s_iscalars[7] = cand_count_p4
                    cute.arch.barrier()
                    ifp = tidx
                    while ifp < N:
                        vfp = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vfp = input_row[ifp]
                        else:
                            vfp = cutlass.Float32(input_row[ifp])
                        if vfp >= pv_lo and vfp < pv_hi:
                            pfill = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(7), cutlass.Int32(1)
                            )
                            if pfill < cutlass.Int32(self.top_k):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pfill] = self.dtype(vfp)
                                output_indices_row[pfill] = ifp
                        ifp = ifp + cutlass.Int32(self.num_threads)
                    cute.arch.barrier()
            else:
                # cs>1: only the leader (CTA 0 in cluster) runs Phase 4.
                if is_leader:
                    if do_cluster_sync:
                        # DSMEM-gather peer candidates into the leader's
                        # smem_keys/smem_vals. Layout: leader's chunk goes
                        # to [0 .. leader_local_cnt); each peer r's chunk
                        # appends the next peer_r_local_cnt entries.
                        local_cnt_self = s_iscalars[5]
                        local_iscalars_ptr = s_iscalars.iterator + cutlass.Int32(5)
                        smem_keys_iter = smem_keys.iterator
                        smem_vals_iter = smem_vals.iterator
                        base_offset = local_cnt_self
                        for peer in cutlass.range_constexpr(1, cluster_size):
                            peer_iscalars_addr = mapa_shared_cluster(
                                local_iscalars_ptr, cutlass.Int32(peer)
                            )
                            peer_cnt = ld_shared_cluster_i32(peer_iscalars_addr)
                            # Cap to kC (defense-in-depth vs. the
                            # done==2 bracket-exhaustion path).
                            peer_cnt = min(peer_cnt, cutlass.Int32(self.kC))
                            i_gather = tidx
                            while i_gather < peer_cnt:
                                peer_key_addr = mapa_shared_cluster(
                                    smem_keys_iter + i_gather, cutlass.Int32(peer)
                                )
                                peer_val_addr = mapa_shared_cluster(
                                    smem_vals_iter + i_gather, cutlass.Int32(peer)
                                )
                                k_val = ld_shared_cluster_f32(peer_key_addr)
                                v_val = ld_shared_cluster_i32(peer_val_addr)
                                dst = base_offset + i_gather
                                if dst < cutlass.Int32(self.kC):
                                    smem_keys[dst] = k_val
                                    smem_vals[dst] = v_val
                                i_gather = i_gather + cutlass.Int32(num_threads)
                            base_offset = base_offset + peer_cnt
                        # Reset s_iscalars[0] to cluster-wide cand_count.
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = base_offset
                        cute.arch.barrier()
                    # else: short-row degrade — leader (CTA 0) already
                    # holds the full row's candidates in its own
                    # smem_keys/smem_vals (no peers to gather from).

                    # ---- Phase 4: histogram snap + writeback ----
                    # Capture the P2 terminal BEFORE Phase 4: P4
                    # reuses s_iscalars[1] as radix scratch.
                    if tidx == cutlass.Int32(0):
                        s_iscalars[6] = cutlass.Int32(0)
                        if s_iscalars[1] == cutlass.Int32(3):
                            s_iscalars[6] = cutlass.Int32(1)
                    cute.arch.barrier()
                    cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                    if cutlass.const_expr(self.enable_p4_rank_scatter):
                        self.phase4_rank_scatter(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                        )
                    else:
                        self.phase4_histogram_snap(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                        )

                    # ---- plateau fill (done == 3): complete the row from the
                    # bitwise-equal plateau class. The terminal is only set on an
                    # ADJACENT-FLOAT bracket, so every value in [s_thr[1], s_thr[0])
                    # is bitwise-equal; Phase 4 has already emitted the
                    # cnt(>= s_thr[0]) sure winners, and ANY (K - count)-subset of
                    # the tie class is a valid tie-aware completion. Ticket counter
                    # lives in the DEDICATED s_iscalars[7].
                    if s_iscalars[6] == cutlass.Int32(1):
                        pv_lo = s_thr[1]
                        pv_hi = s_thr[0]
                        if tidx == cutlass.Int32(0):
                            # cand_count_p4 was captured BEFORE Phase 4; s_iscalars[0]
                            # is radix scratch by now (same hazard as the flag).
                            s_iscalars[7] = cand_count_p4
                        cute.arch.barrier()
                        ifp = tidx
                        while ifp < N:
                            vfp = cutlass.Float32(0.0)
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                vfp = input_row[ifp]
                            else:
                                vfp = cutlass.Float32(input_row[ifp])
                            if vfp >= pv_lo and vfp < pv_hi:
                                pfill = atomicAdd(
                                    s_iscalars.iterator + cutlass.Int32(7),
                                    cutlass.Int32(1),
                                )
                                if pfill < cutlass.Int32(self.top_k):
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pfill] = self.dtype(vfp)
                                    output_indices_row[pfill] = ifp
                            ifp = ifp + cutlass.Int32(self.num_threads)
                        cute.arch.barrier()

        # Final cluster barrier: keep peer CTAs (and their SMEM) alive
        # until the leader's gather + Phase 4 finish. Skipped at
        # do_cluster_sync=False (no peers; short-row degrade non-leaders
        # already fell through ``run_one_row``).
        if cutlass.const_expr(cluster_size > 1):
            if do_cluster_sync:
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # Host-side launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        output_values: cute.Tensor,  # or None.
        output_indices: cute.Tensor,
        order_row: cute.Tensor,  # or None when seqlen_sorted=False
        previous_topk: cute.Tensor,  # or None when fuse_state_store=False
        state_valid: cute.Tensor,  # or None when fuse_state_store=False
        request_indices: cute.Tensor,  # or None when fuse_state_store=False
        stream,
    ):
        num_rows = input_data.shape[0]
        cluster_size = cutlass.const_expr(self.cluster_size)
        # TODO: n_cols (= input_data.shape[1] = max_seq_len) is sym_int here
        # because the wrapper compiles with cute.sym_int() for n_cols. In
        # practice max_seq_len is static (from model config), so adding n_cols
        # to the wrapper cache key would allow a concrete-int fake tensor and
        # enable a real enable_smem_cache size assertion in _compile().

        # Grid = num_rows * cluster_size. Adjacent bidx in
        # [cluster_id*cs, (cluster_id+1)*cs) form one thread-block cluster
        # that owns row[cluster_id]. ``cluster=None`` at cs=1 keeps the
        # launch identical to a plain single-CTA-per-row kernel.
        total_ctas = num_rows * cluster_size
        self.gvr_topk_kernel(
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            order_row,
            previous_topk,
            state_valid,
            request_indices,
        ).launch(
            grid=(total_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(cluster_size, 1, 1)
            if cutlass.const_expr(cluster_size > 1)
            else None,
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    # ------------------------------------------------------------------ #
    #  Host-side launch-shape policy + self-contained launcher            #
    # ------------------------------------------------------------------ #
    # cluster_size / num_threads / min_blocks_per_mp / use_256bit_load are
    # compile-time ctor knobs: a compiled kernel cannot change its own grid
    # or cluster shape, so batch-size adaptation MUST happen at launch time
    # by picking a different compiled variant. ``pick_config`` is that
    # policy as a pure function colocated with the kernel (single source of
    # truth), and ``launch`` is a thin variant-cache wrapper so direct-drive
    # users (tests, benchmarks) get the same shapes production would pick.
    # The production custom op delegates here (``pick_cluster_size`` /
    # ``pick_tuning``) — one policy, two shells. Intentional shell
    # divergence: on a 32B-misaligned logits pointer the production runner
    # ASSERTS (contract violation), while ``launch`` silently downgrades to
    # 128-bit loads (dev convenience for ad-hoc tensors).

    _NUM_SMS: Optional[int] = None
    _LAUNCH_CACHE: dict = {}

    @staticmethod
    def _device_num_sms() -> int:
        if GvrTopKKernel._NUM_SMS is None:
            import torch  # local: keep the module importable without torch
            from vllm.utils.platform_utils import num_compute_units

            device_index = torch.accelerator.current_device_index()
            GvrTopKKernel._NUM_SMS = num_compute_units(device_index)
        return GvrTopKKernel._NUM_SMS

    @staticmethod
    def pick_cluster_size(num_rows: int, n_row: int, num_sms: int) -> int:
        """Cluster-size policy: N < 64K -> 1 (sync unrecouped); tiny grid
        at large N -> 8; single-wave -> 4/2; multi-wave -> 1 (row
        parallelism already saturates the SMs; per-row splitting is pure
        overhead past one wave)."""
        if n_row < 65536:
            return 1
        if num_rows <= 4 and n_row >= 131072:
            return 8
        if num_rows * 4 <= num_sms:
            return 4
        if num_rows * 2 <= num_sms:
            return 2
        return 1

    @staticmethod
    def pick_tuning(
        torch_dtype,
        num_rows: int,
        n_per_cta: int,
        num_sms: int,
        graph_capture: bool,
    ) -> dict:
        """T / V / min_blocks_per_mp / warp-reduce policy at a given
        per-CTA row width (cluster split already applied).

        ``graph_capture``: raise the half-prec T=1024 bar so a small
        capture-N does not pin T=1024 onto small-N replays.
        Returns ``num_threads``, ``use_256bit_load``,
        ``min_blocks_per_mp``, ``enable_warp_parallel_reduce``.
        """
        import torch  # local: keep the module importable without torch

        is_fp32 = torch_dtype == torch.float32
        # T=1024 needs a 1 CTA/SM grid AND enough per-CTA vec work.
        n_thresh_t = 131072 if (graph_capture and not is_fp32) else 65536
        num_threads = 1024 if (num_rows <= num_sms and n_per_cta >= n_thresh_t) else 512
        # V=256-bit only helps fp32 at large N; half-prec cvt doubles reg
        # pressure. Requires a 32B-aligned contiguous tensor (see the
        # shell-divergence note above).
        use_256bit_load = is_fp32 and n_per_cta >= 16384
        enable_warp_parallel_reduce = num_threads == 1024

        # min_blocks_per_mp: reg-vs-occupancy tiers (fp32 wants ~70 regs
        # for 4-LDG ILP -> mb<=2; half-prec fits 40 regs -> mb=3 packs
        # 3 CTA/SM when rows oversubscribe the device).
        vec_bits = 256 if use_256bit_load else 128
        vec_w = vec_bits // (32 if is_fp32 else 16)
        n_vec_iters = max(1, n_per_cta // (num_threads * vec_w))
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and n_per_cta <= 32768:
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

        return dict(
            num_threads=num_threads,
            use_256bit_load=use_256bit_load,
            min_blocks_per_mp=min_blocks_per_mp,
            enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        )

    @staticmethod
    def pick_config(
        torch_dtype,
        num_rows: int,
        num_candidates: int,
        max_seq_len: Optional[int] = None,
        num_sms: Optional[int] = None,
    ) -> dict:
        """Launch-shape ctor kwargs for ``(dtype, BS, N)`` — the single
        source of truth shared by the production runner
        (``CuteDSLGvrTopKDecodeRunner``) and direct-drive users (tests,
        benchmarks): composition of :meth:`pick_cluster_size` and
        :meth:`pick_tuning`.

        ``max_seq_len``: pass the peak runtime N under CUDA-graph capture
        so the variant is picked for the replay shape, not the capture
        shape.
        """
        if num_sms is None:
            num_sms = GvrTopKKernel._device_num_sms()
        n_row = max_seq_len if max_seq_len is not None else num_candidates
        cluster_size = GvrTopKKernel.pick_cluster_size(num_rows, n_row, num_sms)
        cfg = GvrTopKKernel.pick_tuning(
            torch_dtype,
            num_rows,
            n_row // cluster_size,
            num_sms,
            graph_capture=max_seq_len is not None,
        )
        cfg["cluster_size"] = cluster_size
        return cfg

    @classmethod
    def launch(
        cls,
        logits,
        pre_idx,
        seq_lens,
        output_indices,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        num_sms: Optional[int] = None,
        previous_topk=None,
        state_valid=None,
        request_indices=None,
        fuse_hint_prepare: bool = False,
        **kernel_overrides,
    ) -> None:
        """Compile-and-launch with ``pick_config`` shapes (indices-only path).

        Owns a class-level compiled-variant cache keyed by every ctor knob,
        so repeated calls at any (BS, N, dtype) reuse the right variant.
        ``kernel_overrides`` (e.g. ``enable_r0=False``, ``cluster_size=8``)
        override the picked config and participate in the cache key.
        Mirrors the custom op's compile contract: sym_int shapes, tvm-ffi
        env stream (launches on the ambient torch stream), fixed
        ``return_output_values=False`` / ``seqlen_sorted=False``.
        """
        import torch  # local: keep the module importable without torch
        from cutlass.cute import runtime as _crt

        _cute_dt = {
            torch.float32: cutlass.Float32,
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
        }
        num_rows, num_candidates = logits.shape
        cfg = cls.pick_config(
            logits.dtype,
            num_rows,
            num_candidates,
            max_seq_len=max_seq_len,
            num_sms=num_sms,
        )
        state_args = (previous_topk, state_valid, request_indices)
        fuse_state_store = any(arg is not None for arg in state_args)
        if fuse_state_store and not all(arg is not None for arg in state_args):
            raise ValueError(
                "previous_topk, state_valid, and request_indices must be provided together"
            )
        if fuse_hint_prepare and not fuse_state_store:
            raise ValueError("fuse_hint_prepare requires the decode state tensors")
        cfg.update(kernel_overrides)
        cfg["fuse_state_store"] = fuse_state_store
        cfg["fuse_hint_prepare"] = fuse_hint_prepare
        if cfg["cluster_size"] > 1:
            try:
                from .single_pass_multi_cta_radix_topk_cluster import (
                    _query_max_cluster_size,
                )

                cfg["cluster_size"] = min(
                    cfg["cluster_size"], _query_max_cluster_size()
                )
            except ImportError:
                pass  # standalone snapshot: trust the [1, 16] ctor bound
        if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
            cfg["use_256bit_load"] = False  # 256-bit vec loads need 32B alignment

        request_dtype = request_indices.dtype if fuse_state_store else None
        key = (
            logits.dtype,
            top_k,
            next_n,
            compress_ratio,
            request_dtype,
        ) + tuple(sorted(cfg.items()))
        compiled = cls._LAUNCH_CACHE.get(key)
        if compiled is None:
            kernel = cls(
                dtype=_cute_dt[logits.dtype],
                top_k=top_k,
                next_n=next_n,
                compress_ratio=compress_ratio,
                return_output_values=False,
                **cfg,
            )
            n_rows, n_cols, n_batch, n_state = (
                cute.sym_int(),
                cute.sym_int(),
                cute.sym_int(),
                cute.sym_int(),
            )
            in_align = 32 if cfg["use_256bit_load"] else 16
            input_fake = _crt.make_fake_compact_tensor(
                kernel.dtype,
                (n_rows, n_cols),
                stride_order=(1, 0),
                assumed_align=in_align,
            )
            pre_idx_fake = _crt.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, top_k), stride_order=(1, 0), assumed_align=16
            )
            seq_lens_fake = _crt.make_fake_compact_tensor(
                cutlass.Int32, (n_batch,), stride_order=(0,)
            )
            out_indices_fake = _crt.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k), stride_order=(1, 0), assumed_align=16
            )
            if fuse_state_store:
                previous_topk_fake = _crt.make_fake_compact_tensor(
                    cutlass.Int32,
                    (n_state, top_k),
                    stride_order=(1, 0),
                    assumed_align=16,
                )
                state_valid_fake = _crt.make_fake_compact_tensor(
                    cutlass.Boolean, (n_state,), stride_order=(0,)
                )
                request_cutlass_dtype = {
                    torch.int32: cutlass.Int32,
                    torch.int64: cutlass.Int64,
                }.get(request_dtype)
                if request_cutlass_dtype is None:
                    raise ValueError(
                        "request_indices must have dtype torch.int32 or torch.int64"
                    )
                request_indices_fake = _crt.make_fake_compact_tensor(
                    request_cutlass_dtype, (n_rows,), stride_order=(0,)
                )
            else:
                previous_topk_fake = None
                state_valid_fake = None
                request_indices_fake = None
            fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
            compiled = cute.compile(
                kernel,
                input_fake,
                pre_idx_fake,
                seq_lens_fake,
                None,
                out_indices_fake,
                None,
                previous_topk_fake,
                state_valid_fake,
                request_indices_fake,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )
            cls._LAUNCH_CACHE[key] = compiled
        compiled(
            logits,
            pre_idx,
            seq_lens,
            None,
            output_indices,
            None,
            previous_topk,
            state_valid,
            request_indices,
        )


__all__ = ["GvrTopKKernel", "GvrParams"]
