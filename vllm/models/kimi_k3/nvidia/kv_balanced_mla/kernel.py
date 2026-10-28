# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

# Vendored from TokenSpeed MLA 2.0.3 (tokenspeed_mla/mla_decode_fp8.py, MIT, see
# below) for the Kimi K3 KV-balanced MLA decode. Changes from 2.0.3:
# - ``work_list``: one wave of persistent clusters walks a device work list of
#   (query tile, request, split) entries instead of launching the split grid, and a
#   request with a single split stores its final output, skipped by the reduction;
# - ``p_prescale_log2``: P is stored as E4M3 scaled by 2^p_prescale_log2 so small
#   probabilities do not flush to zero, and the LSE removes the factor;
# - ``reg_split``: the per-warpgroup register split can be set;
# - helpers come from the vendored ``helpers.py``; the benchmark harness is removed.
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from types import SimpleNamespace
from typing import Optional, Tuple, Type, cast

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05

# Compat shim: get_max_tmem_alloc_cols added in cutlass-dsl 4.4;
# older versions don't have it, so we provide a fallback implementation.
_TMEM_MAX_ALLOC_COLUMNS_MAP = {"sm_100": 512, "sm_103": 512, "sm_107": 576}


def _get_max_tmem_alloc_cols(compute_capability: str) -> int:
    if hasattr(cute.arch, "get_max_tmem_alloc_cols"):
        return cute.arch.get_max_tmem_alloc_cols(compute_capability)
    if compute_capability not in _TMEM_MAX_ALLOC_COLUMNS_MAP:
        raise ValueError(f"Unsupported compute capability: {compute_capability}")
    return _TMEM_MAX_ALLOC_COLUMNS_MAP[compute_capability]


import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import builtin, llvm, nvvm
from cutlass.cute.typing import Boolean, Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .helpers import (
    LOG2_E,
    MAX_SPLITS,
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    MLAWorkListTileScheduler,
    ceil_div,
    compute_q_tile_layout,
    create_mla_static_tile_scheduler,
    create_mla_static_tile_scheduler_params,
    get_mla_decode_arch,
)

# Please refer to https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp#L412
# For how to construct the Instruction Descriptor value.
# Note: this is a WAR for WS mode MMA, need to be replaced when CuTe DSL supports.
_IDESC_F8E4M3_F32_M64_N128_KMAJ_1CTA = 0x04200010
_IDESC_F8E4M3_F32_M64_N128_AK_BMN_1CTA = _IDESC_F8E4M3_F32_M64_N128_KMAJ_1CTA | (
    1 << 16
)
_IDESC_F8E4M3_F32_M64_N256_KMAJ_1CTA = 0x04400010
_IDESC_F8E4M3_F32_M64_N256_AK_BMN_1CTA = _IDESC_F8E4M3_F32_M64_N256_KMAJ_1CTA | (
    1 << 16
)

_TMEM_ROW_STRIDE = 1 << 16
_P_COR_METADATA_FIELDS = 4


@cute.jit
def tcgen05_mma_ws_f8f6f4_one(
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtC: cute.Tensor,
    idesc_val: Int32,
    accumulate: Boolean,
):
    """Issue one K-block tcgen05.mma.ws.kind::f8f6f4 instruction.
    Should be replaced when CuTe DSL supports."""
    d = builtin.unrealized_conversion_cast([Int32.mlir_type], [tCtC.iterator.value])
    d = llvm.inttoptr(llvm.PointerType.get(6), d)

    a = tcgen05.smem_descriptor_to_int(tCrA.iterator).ir_value()
    b = tcgen05.smem_descriptor_to_int(tCrB.iterator).ir_value()
    idesc = idesc_val.ir_value()
    enable_input_d = accumulate.ir_value()

    with cute.arch.elect_one():
        nvvm.tcgen05_mma_ws(
            nvvm.Tcgen05MMAKind.F8F6F4,
            d,
            a,
            b,
            idesc,
            enable_input_d,
        )


"""
A Multi-Head Latent Attention (MLA) example with FP8 inputs for SM100, SM103 and SM107 using CuTe DSL.

This example implements multi-head latent attention using a TMA + TensorCore
warp-specialized persistent kernel. The implementation integrates the (Qc + Qr)*(Kc + Kr)^T
matrix multiplication, softmax normalization, and softmax((Qc + Qr)*(Kc + Kr)^T)*Vc into a single kernel.
The kernel provides support for page table storage and variable-length KV cache sequences. It implements KV splitting
functionality to minimize latency when processing long KV sequences.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Two sub-kernels (split KV kernel and reduction kernel) that enable split KV processing

Constraints for this example:
* Data type requirements:
  - Input/output: Float8E4M3FN
  - Accumulation and LSE: Float32
* Fixed architecture parameters:
  - Number of attention heads: up to 128
  - Latent dimension: 512
  - RoPE dimension: 64
* Input query modes should be (NumHeads, LatentDim/RopeDim, SeqLenQ, BatchSize)
* Input kv latent/rope modes should be (SeqLenK, LatentDim/RopeDim, BatchSize)
* Query sequence length must be positive
* Only supports 2-CTA instructions
* Variable sequence length requires page table storage enabled
"""


class BlackwellMultiHeadLatentAttentionForwardFP8:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size: int,
        skip_correction_threshold: float,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        fold_sq_factor: int = 1,
        is_causal: bool = False,
        num_heads: int = 128,
        seq_len_q: int = 1,
        window_left: int = -1,
        cp_world: int = 1,  # DCP world size; >1 enables strided global-coord causal masking
        cp_rank: int = 0,
        cp_interleave_size: int = 1,
        reducer_d_tiles: int = 1,
        pack_q: bool = False,
        *,
        compute_capability: tuple[int, int],
        reducer_max_splits: int,
        partial_fp16: bool,
        work_list: bool = False,
        reg_split: Optional[Tuple[int, int, int]] = None,
        p_prescale_log2: float = 0.0,
    ):
        """Initializes the shared SM100/SM103/SM107 MLA kernel configuration.

        :param compute_capability: Target GPU capability: (10, 0), (10, 3) or (10, 7)
        :type compute_capability: tuple[int, int]
        :param acc_dtype: Data type for accumulation S and O
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: Data type for output LSE
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_s_tiler: The (H, K) tile shape of the MMA instruction for S
        :type mma_s_tiler: Tuple[int, int]
        :param mma_p_tiler: The (H, D) tile shape of the MMA instruction for P
        :type mma_p_tiler: Tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: int
        :param page_size: The page size
        :type page_size: int
        :param skip_correction_threshold: Threshold to skip correction
        :type skip_correction_threshold: float
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split KV
        :type is_var_split_kv: bool
        :param window_left: Sliding-window span, or -1 for full history. When
            non-negative, query row ``q_tok`` of the ``seq_len_q`` block sees
            keys ``[max(0, K - seq_len_q - window_left + q_tok), k_bound)``:
            the whole block plus ``window_left`` tokens of history, which is
            the mask a block drafter's ``sliding_attention`` layers declare.
        :type window_left: int
        :param reducer_d_tiles: Number of disjoint D512 output bands (1, 2 or 4)
        :type reducer_d_tiles: int
        :param reducer_max_splits: Static reducer capacity; must cover every split
        :type reducer_max_splits: int
        :param pack_q: Pack consecutive query/head rows into M128 tiles. The
            caller must size split-KV workspace for (128, ceil(H*Sq/128)).
            False preserves the folded layout for existing direct callers.
        :type pack_q: bool
        :param partial_fp16: Store split-KV partials as fp16 instead of acc_dtype
        :type partial_fp16: bool
        """

        self.arch = get_mla_decode_arch(compute_capability)
        # One wave of persistent clusters walks a device work list of
        # (q tile, batch, split) entries instead of the split grid: no empty split
        # cluster is launched, however short or unequal the KV lengths.
        self.use_work_list = work_list
        self.is_sm107 = compute_capability == (10, 7)
        self.latent_dim = 512
        if reducer_d_tiles not in (1, 2, 4):
            raise ValueError("reducer_d_tiles must be 1, 2 or 4")
        if not 1 <= reducer_max_splits <= MAX_SPLITS:
            raise ValueError(f"reducer_max_splits must be in [1, {MAX_SPLITS}]")
        self.reducer_d_tiles = reducer_d_tiles
        self.reducer_d_tile = self.latent_dim // reducer_d_tiles
        # the reducer keeps a row's splits in registers: capacity is a multiple of 4,
        # loaded in sets of up to 64
        self.reducer_accumulate_group = 4
        self.reducer_max_splits = (
            ceil_div(reducer_max_splits, self.reducer_accumulate_group)
            * self.reducer_accumulate_group
        )
        self.reducer_batch_splits = min(self.reducer_max_splits, 64)
        self.rope_dim = 64
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        # split-KV partials are normalized (|v| <= 448), so fp16 is enough
        self.partial_dtype = cutlass.Float16 if partial_fp16 else acc_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.max_active_clusters = max_active_clusters
        self.skip_correction_threshold = skip_correction_threshold
        # P is stored as E4M3 scaled by 2^p_prescale_log2 (its row max is at most
        # 2^(skip_correction_threshold + p_prescale_log2) <= 448), keeping small
        # probabilities above the E4M3 underflow; the LSE removes the factor.
        assert skip_correction_threshold + p_prescale_log2 <= math.log2(448.0)
        self.p_prescale_log2 = p_prescale_log2
        self.is_persistent = is_persistent
        self.page_size = page_size
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        self.fold_sq_factor = fold_sq_factor
        self.pack_q = pack_q
        if pack_q and (mma_qk_tiler_mn[0] != 128 or fold_sq_factor != 1):
            raise ValueError("Packed Q requires M128 and fold_sq_factor=1")
        if pack_q:
            self.total_q_rows, self.num_q_tiles, self.tail_q_rows = (
                compute_q_tile_layout(num_heads, seq_len_q, mma_qk_tiler_mn[0])
            )
        self.is_causal = is_causal
        # Original (pre-fold) num_heads and seq_len_q used for per-row causal
        # q_token_index computation. When fold_sq_factor > 1, each S_q group folds
        # fold_sq_factor tokens into M: M_fold = H * fold_sq_factor.
        # row r → (q_tok_in_group = r // num_heads, head = r % num_heads).
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        self.window_left = window_left
        self.cp_world = cp_world
        self.cp_rank = cp_rank
        self.cp_interleave_size = cp_interleave_size
        if mma_qk_tiler_mn[0] == 128:
            self.cluster_shape_mnk = (2, 1, 1)
            self.use_2cta_instrs = True
        elif mma_qk_tiler_mn[0] == 64:
            self.cluster_shape_mnk = (1, 1, 1)
            self.use_2cta_instrs = False
        else:
            raise ValueError(f"Unsupported mma_qk_tiler_mn[0]: {mma_qk_tiler_mn[0]}")
        if self.arch != "sm_100" and not self.use_2cta_instrs:
            raise ValueError("M64 MLA decode is only supported on SM100")
        self.use_m64_ws = not self.use_2cta_instrs
        # Warps 0-1 handle first N half; warps 2-3 handle second N half.
        self.warps_in_n = 2
        self.num_compute_warps = 4
        self.threads_per_warp = 32
        mma_qk_tiler_k = self.rope_dim * 2
        self.mma_qk_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            mma_qk_tiler_k,
        )
        # The M64 WS split-N epilogue helper models a 2-CTA logical C tile,
        # so its C partition spans two M64 slices in M.
        self.ws_splitn_epi_tiler_mn = (
            (self.mma_qk_tiler[0] * 2, self.mma_qk_tiler[1])
            if self.use_m64_ws
            else self.mma_qk_tiler[:2]
        )
        self.ws_p_cor_rows = self.mma_qk_tiler[0] // self.cluster_shape_mnk[0]
        self.ws_p_cor_n_groups = self.warps_in_n
        self.mma_qk_rope_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            self.rope_dim,
        )
        self.mma_pv_tiler = (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            self.mma_qk_tiler[1] * self.mma_qk_tiler[2] // self.mma_pv_tiler_mn[1],
        )
        self.iterations_qk_latent = self.latent_dim // self.mma_qk_tiler[2]
        self.iterations_qk_rope = 1
        self.iterations_qk = self.iterations_qk_latent + self.iterations_qk_rope
        self.iterations_pv_k = self.mma_qk_tiler[1] // self.mma_pv_tiler[2]
        self.iterations_pv_n = self.latent_dim // self.mma_pv_tiler[1]

        # Set specialized warp ids
        self.compute_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_tma_k_warp_id = 9
        self.load_tma_v_warp_id = 10
        self.empty_warp_ids = (11,)
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                self.load_tma_k_warp_id,
                self.load_tma_v_warp_id,
                *self.compute_warp_ids,
                *self.correction_warp_ids,
                *self.empty_warp_ids,
            )
        )

        # register settings
        self.softmax_reg_num = 208
        self.correction_reg_num = 240 if self.use_m64_ws else 256
        self.other_reg_num = 56 if self.use_m64_ws else 40
        if reg_split is not None:
            # setmaxnreg values (softmax, correction, MMA / load warpgroups): each in
            # [24, 256] and a multiple of 8, with the same total budget as the
            # defaults (208 + 256 + 40).
            assert sum(reg_split) <= 504
            assert all(24 <= r <= 256 and r % 8 == 0 for r in reg_split)
            self.softmax_reg_num, self.correction_reg_num, self.other_reg_num = (
                reg_split
            )
        # Named barriers
        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=(
                self.threads_per_warp
                + self.threads_per_warp * self.num_compute_warps * 2
            ),
        )
        # Split exchange barriers: warp0↔warp2 and warp1↔warp3 are independent
        # exchange pairs. Using separate barriers avoids cross-pair waiting.
        self.softmax_exchange_sync_bar_pair02 = pipeline.NamedBarrier(
            barrier_id=2, num_threads=(self.threads_per_warp * 2)
        )
        self.softmax_exchange_sync_bar_pair13 = pipeline.NamedBarrier(
            barrier_id=3, num_threads=(self.threads_per_warp * 2)
        )
        self.epilogue_exchange_sync_bar_pair02 = pipeline.NamedBarrier(
            barrier_id=4, num_threads=(self.threads_per_warp * 2)
        )
        self.epilogue_exchange_sync_bar_pair13 = pipeline.NamedBarrier(
            barrier_id=5, num_threads=(self.threads_per_warp * 2)
        )

        self.load_q_stage = 1
        if self.use_2cta_instrs:
            self.load_k_stage = 4 if self.is_sm107 else 3
            self.load_v_stage = 4 if self.is_sm107 else 2
        else:
            # No enough smem for WS mode
            self.load_k_stage = 1
            self.load_v_stage = 1
        self.mma_s_stage = 2
        self.p_mma_stage = 2
        self.p_cor_stage = 2
        self.mma_o_stage = 2
        self.load_k_pipeline_stage = (
            self.iterations_qk_latent * self.load_k_stage
            if self.use_m64_ws
            else self.load_k_stage
        )
        self.load_k_rope_pipeline_stage = self.load_k_stage if self.use_m64_ws else 0
        self.load_v_pipeline_stage = (
            self.iterations_pv_n * self.iterations_pv_k * self.load_v_stage
            if self.use_m64_ws
            else self.load_v_stage
        )

        self.tmem_o_offset = self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n
        self.correction_factor_offset = (
            self.tmem_o_offset + self.latent_dim // self.warps_in_n
        )
        self.pv_acc_cols = self.mma_pv_tiler[1] // self.warps_in_n

    @cute.jit
    def _get_dcp_local_bound(self, global_bound: Int32) -> Int32:
        """Map global causal bound to this rank's local bound."""
        if cutlass.const_expr(self.cp_interleave_size == 1):
            return (global_bound - self.cp_rank + self.cp_world - 1) // self.cp_world

        cycle_width = self.cp_world * self.cp_interleave_size
        full_cycles = global_bound // cycle_width
        cycle_offset = global_bound - full_cycles * cycle_width
        local_tail = cycle_offset - self.cp_rank * self.cp_interleave_size
        if cute.elem_less(local_tail, 0):
            local_tail = Int32(0)
        local_tail = cutlass.min(local_tail, self.cp_interleave_size)
        return full_cycles * self.cp_interleave_size + local_tail

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: Optional[cute.Tensor],
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        causal_seqs: Optional[cute.Tensor],  # per-request global causal bound (DCP)
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
        use_pdl: cutlass.Constexpr = True,
        local_visible_lens: Optional[cute.Tensor] = None,
        work_list: Optional[cute.Tensor] = None,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for split-KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [batch_size, seq_len_q, num_head, latent_dim] (contiguous)
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [batch_size, seq_len_q, num_head, rope_dim] (contiguous)
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [num_pages, page_size, latent_dim] (contiguous)
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [num_pages, page_size, rope_dim] (contiguous)
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [batch_size, page_count] (contiguous)
        :type page_table: cute.Tensor
        :param local_visible_lens: Optional [batch_size, seq_len_q] exclusive
            prefix bounds into the compact local page table.
        :param o: The output tensor with shape [batch_size, seq_len_q, num_head, latent_dim] (contiguous)
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [batch_size, seq_len_q, num_head] (contiguous)
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type
        self.skip_lse = lse is None

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )

        # Reinterpret contiguous [B, S_q, H, D] as [H, D, S_q, B]
        # Input stride: (S_q*H*D, H*D, D, 1) → Target: (D, 1, H*D, S_q*H*D)
        def _reinterpret_4d(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], t.shape[1], t.shape[0]),
                    stride=(t.stride[2], t.stride[3], t.stride[1], t.stride[0]),
                ),
            )

        q_latent = _reinterpret_4d(q_latent)
        q_rope = _reinterpret_4d(q_rope)
        o = _reinterpret_4d(o)

        # Reinterpret contiguous [num_pages, page_size, D] as [page_size, D, num_pages]
        # Input stride: (PS*D, D, 1) → Target: (D, 1, PS*D)
        def _reinterpret_3d_kv(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[1], t.shape[2], t.shape[0]),
                    stride=(t.stride[1], t.stride[2], t.stride[0]),
                ),
            )

        c_latent = _reinterpret_3d_kv(c_latent)
        c_rope = _reinterpret_3d_kv(c_rope)

        # Reinterpret contiguous [B, page_count] as [page_count, B]
        page_table = cute.make_tensor(
            page_table.iterator,
            cute.make_layout(
                (page_table.shape[1], page_table.shape[0]),
                stride=(page_table.stride[1], page_table.stride[0]),
            ),
        )

        # Reinterpret contiguous [B, S_q, H] as [H, S_q, B]
        # Input stride: (S_q*H, H, 1) → Target: (1, H, S_q*H)
        if cutlass.const_expr(not self.skip_lse):
            lse_in = cast(cute.Tensor, lse)
            lse = cute.make_tensor(
                lse_in.iterator,
                cute.make_layout(
                    (lse_in.shape[2], lse_in.shape[1], lse_in.shape[0]),
                    stride=(lse_in.stride[2], lse_in.stride[1], lse_in.stride[0]),
                ),
            )

        o_reduction = o
        lse_reduction = lse
        if cutlass.const_expr(self.pack_q):
            # Bound Q's TMA row extent by the real rows; only the final tile
            # can be partial. Keep a singleton S mode for the existing loader.
            def _flatten_q(t):
                return cute.make_tensor(
                    t.iterator,
                    cute.make_layout(
                        (self.total_q_rows, t.shape[1], 1, t.shape[3]),
                        stride=(t.stride[0], t.stride[1], t.stride[3], t.stride[3]),
                    ),
                )

            q_latent = _flatten_q(q_latent)
            q_rope = _flatten_q(q_rope)
            m_tile = self.mma_qk_tiler_mn[0]
            num_q_tiles = cute.ceil_div(o.shape[0] * o.shape[2], m_tile)
            o = cute.make_tensor(
                o.iterator,
                cute.make_layout(
                    (m_tile, o.shape[1], num_q_tiles, o.shape[3]),
                    stride=(
                        o.stride[0],
                        o.stride[1],
                        m_tile * o.stride[0],
                        o.stride[3],
                    ),
                ),
            )
            if cutlass.const_expr(not self.skip_lse):
                lse_in = cast(cute.Tensor, lse)
                lse = cute.make_tensor(
                    lse_in.iterator,
                    cute.make_layout(
                        (m_tile, num_q_tiles, lse_in.shape[2]),
                        stride=(
                            lse_in.stride[0],
                            m_tile * lse_in.stride[0],
                            lse_in.stride[2],
                        ),
                    ),
                )

        # When num_heads < M tile (128), fold fold_sq_factor query tokens into
        # the head dimension to better fill MMA M dimension per S_q group.
        # E.g., H=64, S_q=4, fold_sq_factor=2 → M_eff=128, S_q_eff=2.
        # This works because MLA shares KV across all heads/queries independently.
        elif cutlass.const_expr(self.fold_sq_factor > 1):

            def _fold_sq_4d(t):
                # [H, D, S_q, B] → [H*F, D, S_q/F, B], where F=fold_sq_factor.
                # Use S_q//F to keep dynamic value type in MLIR.
                fold_groups = t.shape[2] // self.fold_sq_factor
                return cute.make_tensor(
                    t.iterator,
                    cute.make_layout(
                        (
                            t.shape[0] * self.fold_sq_factor,
                            t.shape[1],
                            fold_groups,
                            t.shape[3],
                        ),
                        stride=(
                            t.stride[0],
                            t.stride[1],
                            t.stride[2] * self.fold_sq_factor,
                            t.stride[3],
                        ),
                    ),
                )

            q_latent = _fold_sq_4d(q_latent)
            q_rope = _fold_sq_4d(q_rope)
            o = _fold_sq_4d(o)
            if cutlass.const_expr(not self.skip_lse):
                # [H, S_q, B] → [H*F, S_q/F, B], where F=fold_sq_factor.
                lse_in = cast(cute.Tensor, lse)
                fold_groups = lse_in.shape[1] // self.fold_sq_factor
                lse = cute.make_tensor(
                    lse_in.iterator,
                    cute.make_layout(
                        (
                            lse_in.shape[0] * self.fold_sq_factor,
                            fold_groups,
                            lse_in.shape[2],
                        ),
                        stride=(
                            lse_in.stride[0],
                            lse_in.stride[1] * self.fold_sq_factor,
                            lse_in.stride[2],
                        ),
                    ),
                )

        if cutlass.const_expr(not self.pack_q):
            o_reduction = o
            lse_reduction = lse

        acc_o, acc_lse = self.initialize_workspace(
            o.shape[0],
            o.shape[1],
            o.shape[2],
            o.shape[3],
            split_kv,
            self.acc_dtype,
            workspace,
        )

        c_latent_transpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_transpose_layout
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        cta_group = (
            tcgen05.CtaGroup.TWO
            if cutlass.const_expr(self.use_2cta_instrs)
            else tcgen05.CtaGroup.ONE
        )
        # the intermediate tensor p is from smem & k-major
        p_major_mode = OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )
        ws_tiled_mma_epi = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            (
                (128, self.mma_pv_tiler[1] // self.warps_in_n)
                if cutlass.const_expr(self.use_m64_ws)
                else self.mma_qk_tiler[:2]
            ),
        )
        ws_tiled_mma_splitn_epi = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.TWO if cutlass.const_expr(self.use_m64_ws) else cta_group,
            self.ws_splitn_epi_tiler_mn,
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.mma_pv_tiler[:2]

        q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent * self.load_q_stage),
        )
        q_latent_smem_layout_staged = cute.logical_divide(
            q_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )
        q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.q_dtype,
            self.load_q_stage,
        )

        kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_page_tile_size = min(
            self.page_size, qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape
        )
        kc_latent_smem_layout_staged = cute.logical_divide(
            kc_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_latent_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, self.mma_qk_tiler[2]),
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_latent_smem_layout_for_tma = cute.tiled_divide(
            kc_latent_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_tiler[2])
        )
        kc_latent_smem_layout_for_tma = cute.logical_divide(
            kc_latent_smem_layout_for_tma, (None, None, None, self.iterations_qk_latent)
        )

        kc_rope_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.k_dtype,
            self.load_k_stage,
        )
        kc_rope_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (
                self.mma_qk_rope_tiler[0] // qk_tiled_mma.thr_id.shape,
                self.mma_qk_rope_tiler[2],
            ),
            self.k_dtype,
            (self.iterations_qk_rope * self.load_k_stage),
        )
        kc_rope_smem_layout_for_tma = cute.tiled_divide(
            kc_rope_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_rope_tiler[2])
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_mma_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )

        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_staged = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_staged,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        vc_page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.MN,
            (self.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, self.mma_pv_tiler[2]),
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_for_tma = cute.tiled_divide(
            vc_smem_layout_for_tma,
            (
                pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
                vc_page_tile_size,
            ),
        )
        vc_smem_layout_for_tma = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_for_tma,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        # TMA load for Q latent and rope
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])

        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_rope,
            q_rope_smem_layout,
            self.mma_qk_rope_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent and k rope
        kc_smem_layout = cute.select(kc_latent_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent, tma_tensor_c_latent = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent,
            kc_smem_layout,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )
        kc_rope_smem_layout = cute.select(kc_rope_smem_layout_for_tma, mode=[0])
        tma_atom_c_rope, tma_tensor_c_rope = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_rope,
            kc_rope_smem_layout,
            (self.mma_qk_rope_tiler[1], self.mma_qk_rope_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )

        # TMA load for c latent transpose
        vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_transpose,
                vc_smem_layout,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
            )
        )

        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        q_rope_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_rope_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        kc_latent_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        kc_latent_subtile_copy_size = cute.size_in_bytes(
            self.k_dtype,
            cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
        ) * cute.size(qk_tiled_mma.thr_id.shape)
        kc_rope_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_rope_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        vc_copy_size = (
            cute.size_in_bytes(
                self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
            )
            * cute.size(pv_tiled_mma.thr_id.shape)
            * self.iterations_pv_n
            * self.iterations_pv_k
        )
        vc_subtile_copy_size = cute.size_in_bytes(
            self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
        ) * cute.size(pv_tiled_mma.thr_id.shape)

        self.tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size
        self.tma_copy_kc_bytes = kc_latent_copy_size + kc_rope_copy_size
        self.tma_copy_kc_latent_bytes = (
            kc_latent_subtile_copy_size
            if cutlass.const_expr(self.use_m64_ws)
            else self.tma_copy_kc_bytes
        )
        self.tma_copy_kc_rope_bytes = kc_rope_copy_size
        self.tma_copy_vc_bytes = (
            vc_subtile_copy_size
            if cutlass.const_expr(self.use_m64_ws)
            else vc_copy_size
        )

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.cluster_shape_mnk,
            self.max_active_clusters,
            self.is_persistent,
        )
        if cutlass.const_expr(self.use_work_list):
            grid = (self.max_active_clusters * self.cluster_shape_mnk[0], 1, 1)

        @cute.struct
        class SplitKVKernelSharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_k_pipeline_stage * 2
            ]
            load_k_rope_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_k_rope_pipeline_stage * 2
            ]
            load_v_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_v_pipeline_stage * 2
            ]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_mma_stage * 2]
            p_cor_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]

            # Smem tensors
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            smem_kc_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_latent_smem_layout_staged)
                ],
                1024,
            ]

            smem_kc_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_latent_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_vc: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(vc_smem_layout_staged)],
                1024,
            ]
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]

            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: cutlass.Int64

            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32

        softmax_scale_log2 = softmax_scale * LOG2_E

        self.split_kv_kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            ws_tiled_mma_epi,
            ws_tiled_mma_splitn_epi,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            tma_atom_q_rope,
            tma_tensor_q_rope,
            tma_atom_c_latent,
            tma_tensor_c_latent,
            tma_atom_c_rope,
            tma_tensor_c_rope,
            tma_atom_c_latent_transpose,
            tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse if not self.skip_lse else None,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            causal_seqs,
            local_visible_lens,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            q_latent_smem_layout_staged,
            q_rope_smem_layout_staged,
            kc_latent_smem_layout_staged,
            kc_rope_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            kc_latent_smem_layout_for_tma,
            kc_rope_smem_layout_for_tma,
            vc_smem_layout_for_tma,
            cta_layout_vmnk,
            tile_sched_params,
            work_list,
            SplitKVKernelSharedStorage,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=SplitKVKernelSharedStorage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o_reduction,
                lse_reduction,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
                local_visible_lens is not None,
                output_scale,
            ).launch(
                grid=(
                    o_reduction.shape[0] * self.reducer_d_tiles,
                    o_reduction.shape[2],
                    o_reduction.shape[3],
                ),
                block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                smem=ceil_div(self.reducer_max_splits, self.threads_per_warp)
                * self.threads_per_warp
                * self.acc_dtype.width
                // 8,
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=use_pdl,
            )

    def _make_scheduler(self, tile_sched_params, work_list):
        """The grid's tile scheduler, or the persistent work-list scheduler."""
        if cutlass.const_expr(not self.use_work_list):
            return create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
        bidx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        return MLAWorkListTileScheduler(tile_sched_params, work_list, bidx, num_ctas)

    @cute.jit
    def make_paged_tiled_tma_atom(
        self,
        tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
        gmem: cute.Tensor,
        smem_layout: cute.Layout,
        mma_tiler,
        tiled_mma: cute.TiledMma,
        is_k_load: bool,
    ):
        ident = cute.make_identity_layout(gmem.shape)
        g_tile = cute.composition(ident, mma_tiler)
        cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
        cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
        cta_v_map = cute.select(cta_v_map, mode=[0, 2])
        page_tile_size = (
            min(self.page_size, cta_mn)
            if is_k_load
            else min(self.page_size, mma_tiler[1])
        )
        cta_v_map = cute.zipped_divide(
            cta_v_map,
            (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
        )
        cta_v_map = cute.select(cta_v_map, mode=[0])
        from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
        )
        return (
            cute.CopyAtom(
                tma_load_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
            ),
            res[1],
        )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_ws_epi: cute.TiledMma,
        tiled_mma_ws_splitn_epi: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        causal_seqs: cute.Tensor,  # per-request global causal bound (DCP)
        local_visible_lens: Optional[cute.Tensor],
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: cute.ComposedLayout,
        kc_rope_smem_layout_for_tma: cute.ComposedLayout,
        vc_smem_layout_for_tma: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        work_list: Optional[cute.Tensor],
        SharedStorage: cutlass.Constexpr,
    ):
        """The device split_kv kernel implementation of the Multi-Head Latent Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the MLA computation:
        1. Load warp: Loads Q/C latent/rope data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
        to global memory

        The kernel produces either intermediate or final results of the MLA computation based on the split_kv parameter.
        When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
        that will later be combined by a reduction kernel.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases.

        :param tiled_mma_qk: Tiled MMA for Q*K^T
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: Tiled MMA for P*V
        :type tiled_mma_pv: cute.TiledMma
        :param tma_atom_q_latent: TMA copy atom for query latent tensor
        :type tma_atom_q_latent: cute.CopyAtom
        :param mQL: query latent tensor
        :type mQL: cute.Tensor
        :param tma_atom_q_rope: TMA copy atom for query rope tensor
        :type tma_atom_q_rope: cute.CopyAtom
        :param mKR: Compressed rope tensor
        :type mKR: cute.Tensor
        :param tma_atom_c_latent: TMA copy atom for c latent tensor
        :type tma_atom_c_latent: cute.CopyAtom
        :param mCL: Compressed latent tensor
        :type mCL: cute.Tensor
        :param tma_atom_c_rope: TMA copy atom for c rope tensor
        :type tma_atom_c_rope: cute.CopyAtom
        :param mCLT: Compressed latent transpose tensor
        :type mCLT: cute.Tensor
        :param mPT: Page table tensor
        :type mPT: cute.Tensor
        :param mO: Output tensor
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor
        :type mLSE: cute.Tensor
        :param mAccO: Intermediate accumulator output tensor
        :type mAccO: cute.Tensor
        :param mAccLSE: Intermediate accumulator log-sum-exp tensor
        :type mAccLSE: cute.Tensor
        :param split_kv: The split_kv parameter
        :type split_kv: cutlass.Int32
        :param cache_seqs: The variable sequence length tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param softmax_scale_log2: The log2 scale factor for softmax
        :type softmax_scale_log2: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param q_latent_smem_layout_staged: Shared memory layout for query tensor
        :type q_latent_smem_layout_staged: cute.ComposedLayout
        :param q_rope_smem_layout_staged: Shared memory layout for query rope tensor
        :type q_rope_smem_layout_staged: cute.ComposedLayout
        :param kc_latent_smem_layout_staged: Shared memory layout for key tensor
        :type kc_latent_smem_layout_staged: cute.ComposedLayout
        :param kc_rope_smem_layout_staged: Shared memory layout for key rope tensor
        :type kc_rope_smem_layout_staged: cute.ComposedLayout
        :param p_smem_layout_staged: Shared memory layout for probability matrix
        :type p_smem_layout_staged: cute.ComposedLayout
        :param vc_smem_layout_staged: Shared memory layout for value tensor
        :type vc_smem_layout_staged: cute.ComposedLayout
        :param cta_layout_vmnk: Layout for compute threads
        :type cta_layout_vmnk: cute.Layout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: MLAStaticTileSchedulerParams
        :param SharedStorage: Shared storage for the kernel
        :type SharedStorage: cutlass.Constexpr
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch TMA descriptors from load warps (not MMA warp) so the
        # MMA warp can proceed to TMEM allocation sooner.  The prefetch
        # populates the L2 cache and is not warp-specific.
        if warp_idx == self.load_tma_k_warp_id:
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
        if warp_idx == self.load_tma_v_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        # Alloc
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        load_q_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_q_stage,
            self.tma_copy_q_bytes,
        )
        load_k_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_k_pipeline_stage,
            self.tma_copy_kc_latent_bytes,
        )
        if cutlass.const_expr(self.use_m64_ws):
            load_k_rope_pipeline = self.make_and_init_load_qkv_pipeline(
                storage.load_k_rope_mbar_ptr.data_ptr(),
                cta_layout_vmnk,
                self.load_k_rope_pipeline_stage,
                self.tma_copy_kc_rope_bytes,
            )
        else:
            load_k_rope_pipeline = load_k_pipeline
        load_v_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_v_pipeline_stage,
            self.tma_copy_vc_bytes,
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )

        # Cluster arrive after cross-CTA barrier init.
        # p_cor_pipeline is intra-CTA only, so it can be initialized after
        # arrive to overlap with the cluster sync.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        p_cor_pipeline = self.make_and_init_p_cor_pipeline(
            storage.p_cor_mbar_ptr.data_ptr()
        )

        # Generate smem tensor Q/KC/VC/exchange
        # (MMA, MMA_H, MMA_R, PIPE)
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer, swizzle=q_latent_smem_layout_staged.inner
        )
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer, swizzle=q_rope_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        sKC_rope = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_staged.outer, swizzle=kc_rope_smem_layout_staged.inner
        )
        sKC_for_tma = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma.outer,
            swizzle=kc_latent_smem_layout_for_tma.inner,
        )
        sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_for_tma.outer, swizzle=kc_rope_smem_layout_for_tma.inner
        )
        # (MMA, MMA_D, MMA_K, PIPE)
        sVC = storage.smem_vc.get_tensor(
            vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
        )
        sVC_for_tma = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma.outer, swizzle=vc_smem_layout_for_tma.inner
        )
        # (MMA, MMA_H, MMA_K)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (compute_threads,)
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.empty_warp_ids[0] and warp_idx <= self.empty_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.other_reg_num)

        if warp_idx == self.load_tma_k_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            cute.arch.griddepcontrol_wait()
            load_k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_k_pipeline_stage
            )
            if cutlass.const_expr(self.use_m64_ws):
                load_k_rope_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_k_rope_pipeline_stage
                )
            else:
                load_k_rope_producer_state = load_k_producer_state
            tile_sched = self._make_scheduler(tile_sched_params, work_list)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_k_pipeline=load_k_pipeline,
                        load_k_rope_pipeline=load_k_rope_pipeline,
                        mPT=mPT,
                    )
                    tma_k_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_c_latent=tma_atom_c_latent,
                        tma_atom_c_rope=tma_atom_c_rope,
                        mCL=mCL,
                        mKR=mKR,
                        sKC=sKC_for_tma,
                        sKC_full=sKC,
                        sKC_rope=sKC_rope_for_tma,
                    )
                    # Load K only
                    (
                        load_k_producer_state,
                        load_k_rope_producer_state,
                    ) = self.load_tma_k(
                        tma_common_params,
                        tma_k_params,
                        k_index,
                        k_tile_count,
                        load_k_producer_state,
                        load_k_rope_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            load_k_pipeline.producer_tail(load_k_producer_state)
            if cutlass.const_expr(self.use_m64_ws):
                load_k_rope_pipeline.producer_tail(load_k_rope_producer_state)

        if warp_idx == self.load_tma_v_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            cute.arch.griddepcontrol_wait()
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            load_v_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_v_pipeline_stage
            )
            tile_sched = self._make_scheduler(tile_sched_params, work_list)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    # Load Q first (once per work tile)
                    tma_q_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        load_q_pipeline=load_q_pipeline,
                    )
                    tma_q_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        tma_atom_q_rope=tma_atom_q_rope,
                        mQL=mQL,
                        mQR=mQR,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                    )
                    load_q_producer_state = self.load_tma_q(
                        tma_q_common_params,
                        tma_q_params,
                        load_q_producer_state,
                    )
                    # Then load V
                    tma_v_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_v_pipeline=load_v_pipeline,
                        mPT=mPT,
                    )
                    tma_pv_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                        mCLT=mCLT,
                        sVC=sVC_for_tma,
                    )
                    load_v_producer_state = self.load_tma_v(
                        tma_v_common_params,
                        tma_pv_params,
                        k_index,
                        k_tile_count,
                        load_v_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_q_pipeline.producer_tail(load_q_producer_state)
            load_v_pipeline.producer_tail(load_v_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            # Fire off async TMEM allocation request early, then overlap
            # the allocation latency with pipeline state creation and
            # tile scheduler setup.
            tmem.allocate(_get_max_tmem_alloc_cols(self.arch))

            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_q_stage
            )
            load_k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_k_pipeline_stage
            )
            if cutlass.const_expr(self.use_m64_ws):
                load_k_rope_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_k_rope_pipeline_stage
                )
            else:
                load_k_rope_consumer_state = load_k_consumer_state
            load_v_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_v_pipeline_stage
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_s_stage
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_o_stage
            )
            tile_sched = self._make_scheduler(tile_sched_params, work_list)
            work_tile = tile_sched.initial_work_tile_info()

            # Now wait for TMEM allocation and broadcast pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        K=cache_seqs[blk_coord[2]],
                        k_index=k_index,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_k_pipeline=load_k_pipeline,
                        load_k_rope_pipeline=load_k_rope_pipeline,
                        load_v_pipeline=load_v_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL.shape[1],
                        tiled_mma_ws_epi=tiled_mma_ws_epi,
                        tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                        sKC=sKC,
                        sKC_rope=sKC_rope,
                    )
                    mma_pv_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        load_k_rope_consumer_state,
                        load_v_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma(
                        mma_common_params,
                        mma_qk_params,
                        mma_pv_params,
                        k_tile_count,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        load_k_rope_consumer_state,
                        load_v_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mma_s_pipeline.producer_tail(mma_s_producer_state)
            mma_o_pipeline.producer_tail(mma_o_producer_state)
            cute.arch.griddepcontrol_launch_dependents()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = self._make_scheduler(tile_sched_params, work_list)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        K_causal=causal_seqs[blk_coord[2]],
                        local_visible_lens=local_visible_lens,
                        L=mCL.shape[1],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        tiled_mma_ws_epi=tiled_mma_ws_epi,
                        tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                        p_cor_pipeline=p_cor_pipeline,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tiled_mma_ws_epi=tiled_mma_ws_epi,
                        tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = (
                        self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                        )
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            p_cor_pipeline.producer_tail(p_cor_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.correction_warp_ids[0]
            and warp_idx <= self.correction_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.correction_reg_num)
            p_cor_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = self._make_scheduler(tile_sched_params, work_list)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=epilogue_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        K_causal=causal_seqs[blk_coord[2]],
                        local_visible_lens=local_visible_lens,
                        L=mCL.shape[1],
                        H=(
                            cutlass.min(
                                self.mma_qk_tiler[0],
                                self.total_q_rows - blk_coord[1] * self.mma_qk_tiler[0],
                            )
                            if cutlass.const_expr(self.pack_q)
                            else mQL.shape[0]
                        ),
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        tiled_mma_pv=tiled_mma_pv,
                        tiled_mma_ws_epi=tiled_mma_ws_epi,
                        tiled_mma_ws_splitn_epi=tiled_mma_ws_splitn_epi,
                        p_cor_pipeline=p_cor_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        output_scale=output_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                    )
                    p_cor_consumer_state, mma_o_consumer_state = self.correction(
                        compute_common_params,
                        compute_epilogue_params,
                        k_tile_count=k_tile_count,
                        p_cor_consumer_state=p_cor_consumer_state,
                        mma_o_consumer_state=mma_o_consumer_state,
                    )
                elif cutlass.const_expr(
                    mAccO is None and local_visible_lens is not None
                ):
                    # No producer ran for this zero-K tile.
                    self.write_empty_output(mO, mLSE, blk_coord, tidx)
                elif cutlass.const_expr(self.use_work_list):
                    # An empty request (e.g. a padded row or an empty DCP shard) is
                    # planned as one split, which the reduction skips: write it here.
                    if local_split_kv <= 1:
                        self.write_empty_output(mO, mLSE, blk_coord, tidx)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        return

    @cute.jit
    def write_empty_output(self, mO, mLSE, blk_coord, tidx):
        """Write a zero-K tile without touching TMEM or pipeline states."""
        thread = tidx % (self.num_compute_warps * self.threads_per_warp)
        threads = self.num_compute_warps * self.threads_per_warp
        rows = self.mma_pv_tiler[0] // self.cluster_shape_mnk[0]
        for index in cutlass.range(thread, rows * self.latent_dim, threads):
            row = blk_coord[0] * rows + index // self.latent_dim
            valid = row < mO.shape[0]
            if cutlass.const_expr(self.pack_q):
                valid = (
                    valid
                    and blk_coord[1] * self.mma_qk_tiler[0] + row < self.total_q_rows
                )
            if valid:
                mO[row, index % self.latent_dim, blk_coord[1], blk_coord[2]] = (
                    self.o_dtype(0)
                )
                if cutlass.const_expr(mLSE is not None):
                    if index % self.latent_dim == 0:
                        mLSE[row, blk_coord[1], blk_coord[2]] = -self.lse_dtype.inf

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        has_local_visible_lens: cutlass.Constexpr,
        output_scale: cutlass.Float32,
    ):
        """The reduction kernel for Multi-Head Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        :param output_scale: Scale applied after combining normalized partial outputs
        :type output_scale: cutlass.Float32
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        # Output coordinates retain the original query/head layout for packed Q.
        # Each CTA writes a disjoint D band; only band zero writes the LSE.
        d_tile_idx = bidx % self.reducer_d_tiles
        blk_coord = (bidx // self.reducer_d_tiles, bidy, bidz)
        acc_row = blk_coord[0]
        acc_tile = blk_coord[1]
        if cutlass.const_expr(self.pack_q):
            flat_row = bidy * self.num_heads + blk_coord[0]
            acc_row = flat_row % self.mma_qk_tiler[0]
            acc_tile = flat_row // self.mma_qk_tiler[0]
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.mma_qk_tiler[1])
        # Mirror get_k_tile_count's window-aware split, or this reduction reads
        # partials from CTAs that were given no tiles to write them.
        if cutlass.const_expr(self.window_left >= 0):
            k_tile_lo = (
                max(
                    0,
                    cache_seqs[blk_coord[2]] - self.seq_len_q - self.window_left,
                )
                // self.mma_qk_tiler[1]
            )
        else:
            k_tile_lo = 0
        k_tile_per_cta = cute.ceil_div(k_tile_total - k_tile_lo, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total - k_tile_lo, max(1, k_tile_per_cta))

        # Work list: requests with a single split stored their final output; their
        # CTAs exit before allocating or waiting (the counts and lengths come from
        # the plan and the inputs, complete before the split kernel).
        reduce_row = True
        if cutlass.const_expr(self.use_work_list):
            reduce_row = local_split_kv > 1
        if reduce_row:
            # Alloc shared memory
            smem = cutlass.memory.SmemAllocator()
            # LSE phase: one weight per lane of warp 0
            reducer_scale_count = (
                ceil_div(self.reducer_max_splits, self.threads_per_warp)
                * self.threads_per_warp
            )
            storage = smem.allocate(reducer_scale_count * self.acc_dtype.width // 8, 16)
            lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
            smem_lse_scale = cute.make_tensor(
                lse_scale_ptr, cute.make_layout(reducer_scale_count)
            )

            # let the next split kernel get scheduled early; its own wait still orders the data
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

            # issue the first set of partial loads before the LSE phase; later sets
            # only if the row needs them
            reducer_threads = self.threads_per_warp * self.num_compute_warps
            vec = self.reducer_d_tile // reducer_threads
            gAccO = mAccO[acc_row, None, None, acc_tile, blk_coord[2]]
            band_ptr = cute.make_ptr(
                self.partial_dtype,
                (gAccO.iterator + d_tile_idx * self.reducer_d_tile).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            partial_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.partial_dtype,
                num_bits_per_copy=vec * self.partial_dtype.width,
            )
            partial_thr_copy = cute.make_tiled_copy_tv(
                partial_copy_atom,
                cute.make_ordered_layout((1, reducer_threads), order=(1, 0)),
                cute.make_layout((1, vec)),
            ).get_slice(tidx)
            partial_params = SimpleNamespace(
                band_ptr=band_ptr,
                split_stride=gAccO.stride[0],
                copy_atom=partial_copy_atom,
                thr_copy=partial_thr_copy,
                local_split_kv=local_split_kv,
                has_local_visible_lens=has_local_visible_lens,
                smem_lse_scale=smem_lse_scale,
                vec=vec,
            )
            num_sets = ceil_div(self.reducer_max_splits, self.reducer_batch_splits)
            num_pre_sets = 1
            pre_frags = []
            for s in cutlass.range_constexpr(num_pre_sets):
                pre_frags.append(
                    self._reducer_issue_set(
                        partial_params, s * self.reducer_batch_splits
                    )
                )

            gLSE = mAccLSE[acc_row, None, acc_tile, blk_coord[2]]
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                # calculate the global lse and exp ^ (local_lse - global_lse)
                lse_per_thread = cute.ceil_div(
                    self.reducer_max_splits, self.threads_per_warp
                )

                local_lse = cute.make_rmem_tensor(
                    cute.make_layout(lse_per_thread), self.lse_dtype
                )
                lse_max = -self.lse_dtype.inf
                # find the max lse
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    local_lse[i] = (
                        gLSE[split_kv_idx]
                        if cute.elem_less(split_kv_idx, local_split_kv)
                        else -self.lse_dtype.inf
                    )
                    # reduce the local lse
                    lse_max = cute.arch.fmax(lse_max, local_lse[i])
                lse_max = cute.arch.warp_reduction_max(lse_max)
                lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
                # calculate sum_lse
                sum_lse = 0.0
                for i in cutlass.range_constexpr(lse_per_thread):
                    sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
                sum_lse = cute.arch.warp_reduction_sum(sum_lse)
                # calculate the global_lse
                global_lse = (
                    lse_max + cute.math.log2(sum_lse, fastmath=True)
                    if not sum_lse == self.lse_dtype(0.0) or sum_lse != sum_lse  # noqa: SIM201
                    else -self.lse_dtype.inf
                )
                if tidx == 0 and d_tile_idx == 0:
                    if cutlass.const_expr(not self.skip_lse):
                        cast(cute.Tensor, mLSE)[
                            blk_coord[0], blk_coord[1], blk_coord[2]
                        ] = global_lse
                # store the scale to shared memory, including output_scale (0 past local_split_kv)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    smem_lse_scale[split_kv_idx] = (
                        cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                        * output_scale
                        if not has_local_visible_lens or sum_lse != 0.0
                        else 0.0
                    )

            pipeline.sync(barrier_id=4)

            rAccO = cute.make_rmem_tensor(cute.make_layout(vec), self.acc_dtype)
            rAccO.fill(0.0)
            for s in cutlass.range_constexpr(num_pre_sets):
                self._reducer_accumulate_set(
                    partial_params, pre_frags[s], s * self.reducer_batch_splits, rAccO
                )
            for s in cutlass.range_constexpr(num_pre_sets, num_sets):
                first = s * self.reducer_batch_splits
                if local_split_kv > first:
                    frag = self._reducer_issue_set(partial_params, first)
                    self._reducer_accumulate_set(partial_params, frag, first, rAccO)
            rO = cute.make_rmem_tensor(cute.make_layout(vec), self.o_dtype)
            rO.store(rAccO.load().to(self.o_dtype))
            gO = mO[blk_coord[0], None, blk_coord[1], blk_coord[2]]
            out_ptr = cute.make_ptr(
                self.o_dtype,
                (gO.iterator + d_tile_idx * self.reducer_d_tile + tidx * vec).toint(),
                cute.AddressSpace.gmem,
                assumed_align=vec * self.o_dtype.width // 8,
            )
            cute.autovec_copy(rO, cute.make_tensor(out_ptr, cute.make_layout(vec)))

    @cute.jit
    def _reducer_issue_set(self, params: SimpleNamespace, first: int) -> cute.Tensor:
        """Issue predicated partial loads for splits [first, first + batch)."""
        batch = self.reducer_batch_splits
        gSet = cute.make_tensor(
            params.band_ptr + first * params.split_stride,
            cute.make_layout(
                (batch, self.reducer_d_tile), stride=(params.split_stride, 1)
            ),
        )
        tPgP = params.thr_copy.partition_S(gSet)
        tPrP = cute.make_fragment_like(tPgP)
        tPpP = cute.make_rmem_tensor(cute.make_layout((1, batch, 1)), cutlass.Boolean)
        for k in cutlass.range(batch, unroll_full=True):
            tPpP[0, k, 0] = cute.elem_less(first + k, params.local_split_kv)
        tPrP.fill(0.0)
        cute.copy(params.copy_atom, tPgP, tPrP, pred=tPpP)
        return tPrP

    @cute.jit
    def _reducer_accumulate_set(
        self,
        params: SimpleNamespace,
        tPrP: cute.Tensor,
        first: int,
        rAccO: cute.Tensor,
    ):
        """Accumulate splits [first, first + batch) into rAccO, skipping empty groups."""
        batch = self.reducer_batch_splits
        group = self.reducer_accumulate_group
        for g in cutlass.range_constexpr(batch // group):
            if params.local_split_kv > first + g * group:
                for k in cutlass.range_constexpr(group):
                    split = g * group + k
                    lse_scale = params.smem_lse_scale[first + split]
                    # Empty DCP partials must not propagate NaNs through 0 * NaN.
                    if not params.has_local_visible_lens or lse_scale != 0.0:
                        for v in cutlass.range_constexpr(params.vec):
                            rAccO[v] += tPrP[v, split, 0].to(self.acc_dtype) * lse_scale
        return

    @staticmethod
    def get_split_kv(
        B: int,
        S: int,
        K: int,
        mma_qk_tiler_mn: tuple,
        max_active_blocks: int,
        m_ctas_per_tile: int = 2,
    ) -> int:
        """Get the proper split_kv value for the MLA kernel based on parameters.

        :param B: Batch size
        :type B: int
        :param S: Sequence length
        :type S: int
        :param K: Sequence length
        :type K: int
        :param mma_qk_tiler_mn: MLA tiling parameters
        :type mma_qk_tiler_mn: tuple
        :param max_active_blocks: Maximum number of active blocks
        :type max_active_blocks: int
        :param m_ctas_per_tile: Number of CTAs launched in M for one logical tile
        :type m_ctas_per_tile: int
        :return: Split_kv value
        :rtype: int
        """
        max_splits = ceil_div(K, mma_qk_tiler_mn[1])
        blocks_per_batch = max(1, max_active_blocks // B // (S * m_ctas_per_tile))
        split_heur = min(max_splits, blocks_per_batch)
        k_waves = ceil_div(max_splits, split_heur)
        split_wave_aware = ceil_div(max_splits, k_waves)
        max_split_kv = 64 if m_ctas_per_tile == 1 else 32
        return min(split_wave_aware, max_split_kv)

    @staticmethod
    def get_split_kv_simplified(B: int, S: int, max_active_blocks: int) -> int:
        blocks_per_batch = max(1, max_active_blocks // B // (S * 2))
        max_split_kv = 32
        return min(blocks_per_batch, max_split_kv)

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value for the MLA kernel.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        # A sliding window makes every tile below the earliest visible key dead
        # work, so the split starts there instead of at 0. window_left < 0 is a
        # compile-time zero and leaves the full-history schedule untouched.
        if cutlass.const_expr(self.window_left >= 0):
            k_tile_lo = (
                max(0, K - self.seq_len_q - self.window_left) // self.mma_qk_tiler[1]
            )
        else:
            k_tile_lo = 0
        k_tile_per_cta = cute.ceil_div(k_tile_total - k_tile_lo, split_kv)
        k_index = k_tile_lo + blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @cute.jit
    def load_tma_q(
        self,
        common_params: SimpleNamespace,
        q_params: SimpleNamespace,
        load_q_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load Q tensors (latent + rope) once per work tile.

        :param common_params: The common parameters (blk_coord, load_q_pipeline)
        :type common_params: SimpleNamespace
        :param q_params: The q parameters (tiled_mma_qk, tma atoms, mQL, mQR, sQ, sQ_rope)
        :type q_params: SimpleNamespace
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState

        :return: The updated load q producer state
        :rtype: pipeline.PipelineState
        """
        # Flatten divide and partition global tensors for Q TMA load
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(q_params.mQL, mma_qk_tiler_mk)
        mma_qk_tiler_mk_rope = cute.select(self.mma_qk_rope_tiler, mode=[0, 2])
        gQR = cute.flat_divide(q_params.mQR, mma_qk_tiler_mk_rope)

        thr_mma_qk = q_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(q_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)
        tSgQR = thr_mma_qk.partition_A(gQR)

        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            q_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(q_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        tQsQ_rope, tQRgQR_mkl = cpasync.tma_partition(
            q_params.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(q_params.sQ_rope, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )

        q_group = 0 if cutlass.const_expr(self.pack_q) else common_params.blk_coord[1]
        q_m_tile = common_params.blk_coord[1] if cutlass.const_expr(self.pack_q) else 0
        tQLgQL = tQLgQL_mkl[None, None, None, q_group, common_params.blk_coord[2]]
        tQRgQR = tQRgQR_mkl[None, None, None, q_group, common_params.blk_coord[2]]

        # Issue Q TMA copies
        load_q_pipeline = common_params.load_q_pipeline
        tma_bar_ptr = load_q_pipeline.producer_get_barrier(load_q_producer_state)
        load_q_pipeline.producer_acquire(load_q_producer_state)
        for i in cutlass.range_constexpr(self.iterations_qk_latent):
            cute.copy(
                q_params.tma_atom_q_latent,
                tQLgQL[None, q_m_tile, i],
                tQsQ[None, (i, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )
        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            cute.copy(
                q_params.tma_atom_q_rope,
                tQRgQR[None, q_m_tile, i],
                tQsQ_rope[None, i],
                tma_bar_ptr=tma_bar_ptr,
            )
        load_q_producer_state.advance()
        return load_q_producer_state

    @cute.jit
    def load_tma_k(
        self,
        common_params: SimpleNamespace,
        k_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_k_producer_state: pipeline.PipelineState,
        load_k_rope_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Load K tensors (latent + rope) for all k tiles.

        :param common_params: The common parameters (blk_coord, load_k_pipeline, mPT)
        :type common_params: SimpleNamespace
        :param k_params: The k parameters (tiled_mma_qk, tma atoms, mCL, mKR, sKC, sKC_rope)
        :type k_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_k_producer_state: The load k producer state
        :type load_k_producer_state: pipeline.PipelineState

        :return: The updated load k producer state
        :rtype: pipeline.PipelineState
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        cta_m = min(
            k_params.tiled_mma_qk.op.shape_mnk[0] // k_params.tiled_mma_qk.thr_id.shape,
            self.page_size,
        )
        page_tile_size = min(self.page_size, cta_m)
        gCL = cute.tiled_divide(k_params.mCL, (page_tile_size, self.mma_qk_tiler[2]))
        tSgCL = (
            gCL[
                None,
                common_params.blk_coord[0] % k_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m < self.page_size
            else gCL[None, 0, None, None]
        )
        gKR = cute.tiled_divide(
            k_params.mKR, (page_tile_size, self.mma_qk_rope_tiler[2])
        )
        tSgKR = (
            gKR[
                None,
                common_params.blk_coord[0] % k_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m < self.page_size
            else gKR[None, 0, None, None]
        )

        tKCsKC, tCLgCL = cpasync.tma_partition(
            k_params.tma_atom_c_latent,
            0,
            cute.make_layout(1),
            k_params.sKC,
            tSgCL,
        )

        tKCsKC_rope, tKRgKR = cpasync.tma_partition(
            k_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            k_params.sKC_rope,
            tSgKR,
        )
        # set extra params
        common_params.mPT = mPT
        k_params.tSgCL = tSgCL
        k_params.tSgKR = tSgKR
        k_params.tCLgCL = tCLgCL
        k_params.tKRgKR = tKRgKR
        k_params.tKCsKC = tKCsKC
        k_params.tKCsKC_rope = tKCsKC_rope

        while k_tile_count > 0:
            (
                load_k_producer_state,
                load_k_rope_producer_state,
            ) = self.load_tma_k_one_k_tile(
                common_params,
                k_params,
                k_index,
                load_k_producer_state,
                load_k_rope_producer_state,
            )
            k_index += 1
            k_tile_count -= 1

        return load_k_producer_state, load_k_rope_producer_state

    @cute.jit
    def load_tma_v(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load wrap to load V tensors. Updates the load v producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load v producer state
        :rtype: pipeline.PipelineState
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        # Flatten divide and partition global tensors for V TMA load
        page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        gCLT = cute.flat_divide(v_params.mCLT, (self.mma_pv_tiler[1], page_tile_size))
        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape
        gCLT = cute.logical_divide(gCLT, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT = cute.tiled_divide(gCLT, (cta_n, page_tile_size))
        tOgCLT = tOgCLT[None, 0, 0, None, None, None]
        # tma partition for vc
        # smem: ((atom_v, rest_v), STAGE)
        # gmem: ((atom_v, rest_v), RestM, RestK, RestL)
        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            v_params.sVC,
            tOgCLT,
        )

        # set extra params
        common_params.mPT = mPT
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

        while k_tile_count > 0:
            load_v_producer_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index,
                load_v_producer_state,
            )
            k_index += 1
            k_tile_count -= 1
        return load_v_producer_state

    @cute.jit
    def load_tma_k_one_k_tile(
        self,
        common_params: SimpleNamespace,
        k_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_k_producer_state: pipeline.PipelineState,
        load_k_rope_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Load one k-tile of K latent/rope tensors.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param k_params: The k parameters
        :type k_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_k_producer_state: The load k producer state
        :type load_k_producer_state: pipeline.PipelineState

        :return: The updated load k producer state
        :rtype: pipeline.PipelineState
        """
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size, k_params.tiled_mma_qk.thr_id.shape
        )
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.mPT[k_index]
                if self.mma_qk_tiler[1] // self.page_size == 1
                else common_params.mPT[
                    (
                        k_index * k_params.tiled_mma_qk.thr_id.shape
                        + common_params.blk_coord[0]
                    )
                    * page_per_tile
                    + i
                ]
            )
        if cutlass.const_expr(self.use_m64_ws):
            for i in cutlass.range_constexpr(self.iterations_qk_latent):
                tma_bar_ptr = common_params.load_k_pipeline.producer_get_barrier(
                    load_k_producer_state
                )
                common_params.load_k_pipeline.producer_acquire(load_k_producer_state)
                for k in cutlass.range_constexpr(page_per_tile):
                    if cutlass.const_expr(page_per_tile > 1):
                        sKC_page_base = cute.domain_offset(
                            ((k * self.page_size, 0), 0, 0, (i, 0)),
                            k_params.sKC_full,
                        )
                        sKC_page = cute.make_tensor(
                            sKC_page_base.iterator, k_params.sKC.layout
                        )
                        tKCsKC_page, tCLgCL_page = cpasync.tma_partition(
                            k_params.tma_atom_c_latent,
                            0,
                            cute.make_layout(1),
                            sKC_page,
                            k_params.tSgCL,
                        )
                        cute.copy(
                            k_params.tma_atom_c_latent,
                            tCLgCL_page[None, i, k_idx[k]],
                            tKCsKC_page[None, 0, 0, (0, 0)],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                    else:
                        cute.copy(
                            k_params.tma_atom_c_latent,
                            k_params.tCLgCL[None, i, k_idx[k]],
                            k_params.tKCsKC[None, k, 0, (i, 0)],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                load_k_producer_state.advance()

            tma_bar_ptr = common_params.load_k_rope_pipeline.producer_get_barrier(
                load_k_rope_producer_state
            )
            common_params.load_k_rope_pipeline.producer_acquire(
                load_k_rope_producer_state
            )
            for i in cutlass.range_constexpr(self.iterations_qk_rope):
                for k in cutlass.range_constexpr(page_per_tile):
                    if cutlass.const_expr(page_per_tile > 1):
                        sKC_rope_page = cute.domain_offset(
                            ((k * self.page_size, 0), 0, 0, 0),
                            k_params.sKC_rope,
                        )
                        tKCsKC_rope_page, tKRgKR_page = cpasync.tma_partition(
                            k_params.tma_atom_c_rope,
                            0,
                            cute.make_layout(1),
                            sKC_rope_page,
                            k_params.tSgKR,
                        )
                        cute.copy(
                            k_params.tma_atom_c_rope,
                            tKRgKR_page[None, i, k_idx[k]],
                            tKCsKC_rope_page[None, 0, 0, 0],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                    else:
                        cute.copy(
                            k_params.tma_atom_c_rope,
                            k_params.tKRgKR[None, i, k_idx[k]],
                            k_params.tKCsKC_rope[None, k, 0, 0],
                            tma_bar_ptr=tma_bar_ptr,
                        )
            load_k_rope_producer_state.advance()
        else:
            tma_bar_ptr = common_params.load_k_pipeline.producer_get_barrier(
                load_k_producer_state
            )
            common_params.load_k_pipeline.producer_acquire(load_k_producer_state)
            for i in range(self.iterations_qk_latent):
                for k in range(page_per_tile):
                    cute.copy(
                        k_params.tma_atom_c_latent,
                        k_params.tCLgCL[None, i, k_idx[k]],
                        k_params.tKCsKC[None, k, 0, (i, load_k_producer_state.index)],
                        tma_bar_ptr=tma_bar_ptr,
                    )

            for i in cutlass.range_constexpr(self.iterations_qk_rope):
                for k in cutlass.range_constexpr(page_per_tile):
                    cute.copy(
                        k_params.tma_atom_c_rope,
                        k_params.tKRgKR[None, i, k_idx[k]],
                        k_params.tKCsKC_rope[None, k, 0, load_k_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
            load_k_producer_state.advance()

        return load_k_producer_state, load_k_rope_producer_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load one k-tile of compressed latent transpose tensor(v). Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The load tma v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load qkv producer state
        :rtype: pipeline.PipelineState
        """
        page_per_tile = self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size
        page_per_subtile = ceil_div(page_per_tile, self.iterations_pv_k)
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.mPT[k_index]
                if page_per_tile == 1
                else common_params.mPT[k_index * page_per_tile + i]
            )
        if cutlass.const_expr(self.use_m64_ws):
            for j in cutlass.range_constexpr(self.iterations_pv_n):
                for i in cutlass.range_constexpr(self.iterations_pv_k):
                    tma_bar_ptr = common_params.load_v_pipeline.producer_get_barrier(
                        load_v_producer_state
                    )
                    common_params.load_v_pipeline.producer_acquire(
                        load_v_producer_state
                    )
                    if cutlass.const_expr(page_per_tile > 1):
                        for k in cutlass.range_constexpr(page_per_subtile):
                            k_idx_i = k_idx[k + i * page_per_subtile]
                            cute.copy(
                                v_params.tma_atom_c_latent_transpose,
                                v_params.tCLTgCLT[None, j, 0, k_idx_i],
                                v_params.tVCsVC[None, 0, k, ((j, i), 0)],
                                tma_bar_ptr=tma_bar_ptr,
                            )
                    else:
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose,
                            v_params.tCLTgCLT[None, j, i, k_idx[0]],
                            v_params.tVCsVC[None, 0, 0, ((j, i), 0)],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                    load_v_producer_state.advance()
        else:
            tma_bar_ptr = common_params.load_v_pipeline.producer_get_barrier(
                load_v_producer_state
            )
            common_params.load_v_pipeline.producer_acquire(load_v_producer_state)
            for j in cutlass.range_constexpr(self.iterations_pv_n):
                for i in cutlass.range_constexpr(self.iterations_pv_k):
                    if cutlass.const_expr(page_per_tile > 1):
                        for k in cutlass.range_constexpr(page_per_subtile):
                            k_idx_i = k_idx[k + i * page_per_subtile]
                            cute.copy(
                                v_params.tma_atom_c_latent_transpose,
                                v_params.tCLTgCLT[None, j, 0, k_idx_i],
                                v_params.tVCsVC[
                                    None,
                                    0,
                                    k,
                                    ((j, i), load_v_producer_state.index),
                                ],
                                tma_bar_ptr=tma_bar_ptr,
                            )
                    else:
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose,
                            v_params.tCLTgCLT[None, j, i, k_idx[0]],
                            v_params.tVCsVC[
                                None, 0, 0, ((j, i), load_v_producer_state.index)
                            ],
                            tma_bar_ptr=tma_bar_ptr,
                        )
            load_v_producer_state.advance()
        return load_v_producer_state

    @cute.jit
    def mma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        load_k_rope_consumer_state: pipeline.PipelineState,
        load_v_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """MMA warp to compute the result of Q*K^T and P*V. Updates the tiled mma and pipeline states.

        :param common_params: The common parameters for mma qk and pv
        :type common_params: SimpleNamespace
        :param qk_params: The mma qk parameters
        :type qk_params: SimpleNamespace
        :param pv_params: The mma pv parameters
        :type pv_params: SimpleNamespace
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_k_consumer_state: The load k consumer state
        :type load_k_consumer_state: pipeline.PipelineState
        :param load_v_consumer_state: The load v consumer state
        :type load_v_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The p mma consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The mma o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the tiled mma pv, the load q consumer state, the load k consumer state, the load v consumer state, the mma s producer state, the p mma consumer state, and the mma o producer state
        :rtype: tuple[cute.TiledMma, cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        tSrQ = tiled_mma_qk.make_fragment_A(qk_params.sQ)
        tSrQ_rope = tiled_mma_qk.make_fragment_A(qk_params.sQ_rope)
        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)
        tSrKC_rope = tiled_mma_qk.make_fragment_B(qk_params.sKC_rope)
        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        if cutlass.const_expr(self.use_m64_ws):
            tStS_shape = common_params.tiled_mma_ws_splitn_epi.partition_shape_C(
                self.ws_splitn_epi_tiler_mn
            )
            tStS_staged_fake = common_params.tiled_mma_ws_splitn_epi.make_fragment_C(
                cute.append(tStS_shape, self.mma_s_stage)
            )
        else:
            tStS_shape = tiled_mma_qk.partition_shape_C(
                cute.select(self.mma_qk_tiler, mode=[0, 1])
            )
            tStS_staged_fake = tiled_mma_qk.make_fragment_C(
                cute.append(tStS_shape, self.mma_s_stage)
            )
        # use real tmem ptr for tStS
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tO_tmem_shape = (
            common_params.tiled_mma_ws_epi.partition_shape_C(
                (128, self.mma_pv_tiler[1] // self.warps_in_n)
            )
            if cutlass.const_expr(self.use_m64_ws)
            else tiled_mma_pv.partition_shape_C(
                cute.select(self.mma_pv_tiler, mode=[0, 1])
            )
        )
        # mma O has 1 stage.
        tO_tmem = (
            common_params.tiled_mma_ws_epi.make_fragment_C(tO_tmem_shape)
            if cutlass.const_expr(self.use_m64_ws)
            else tiled_mma_pv.make_fragment_C(tO_tmem_shape)
        )
        tO_tmem_layout = cute.append(
            tO_tmem.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.pv_acc_cols,
            ),
        )
        tO_tmem_staged = cute.make_tensor(
            tStS_staged.iterator + self.tmem_o_offset, tO_tmem_layout
        )

        # set more parameters
        qk_params.tSrQ = tSrQ
        qk_params.tSrQ_rope = tSrQ_rope
        qk_params.tSrKC = tSrKC
        qk_params.tSrKC_rope = tSrKC_rope
        qk_params.tStS_staged = tStS_staged
        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tO_tmem_staged = tO_tmem_staged

        # mma O accumulates on K, so the accumulate flag is set to False once before all K blocks.
        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        pv_k_index = common_params.k_index
        if common_params.is_leader_cta:
            load_q_release_state = load_q_consumer_state.clone()
            (
                tiled_mma_qk,
                load_q_consumer_state,
                load_k_consumer_state,
                load_k_rope_consumer_state,
                mma_s_producer_state,
            ) = self.mma_qk(
                common_params,
                qk_params,
                tiled_mma_qk,
                load_q_consumer_state,
                load_k_consumer_state,
                load_k_rope_consumer_state,
                mma_s_producer_state,
                wait_q=True,
            )
            qk_tile_count = k_tile_count - 1
            pv_tile_count = k_tile_count
            # Warm up the same schedule with a shape-dependent QK lead.
            qk_lead = 2 if self.use_2cta_instrs else 1
            for _ in cutlass.range_constexpr(qk_lead - 1):
                if qk_tile_count > 0:
                    (
                        tiled_mma_qk,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        load_k_rope_consumer_state,
                        mma_s_producer_state,
                    ) = self.mma_qk(
                        common_params,
                        qk_params,
                        tiled_mma_qk,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        load_k_rope_consumer_state,
                        mma_s_producer_state,
                        wait_q=False,
                    )
                    qk_tile_count -= 1

            while qk_tile_count > 0:
                (
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    load_k_rope_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    load_k_rope_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                qk_tile_count -= 1
                (
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    pv_k_index,
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                pv_k_index = pv_k_index + 1
                pv_tile_count -= 1
            # release q consumer states, then drain the remaining PV tiles
            load_q_pipeline.consumer_release(load_q_release_state)
            load_q_release_state.advance()
            while pv_tile_count > 0:
                (
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    pv_k_index,
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                pv_k_index = pv_k_index + 1
                pv_tile_count -= 1

        return (  # type: ignore[return-value]
            tiled_mma_qk,
            tiled_mma_pv,
            load_q_consumer_state,
            load_k_consumer_state,
            load_k_rope_consumer_state,
            load_v_consumer_state,
            mma_s_producer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        load_k_rope_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        wait_q: bool,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for Q*K^T. Updates the tiled MMA QK and pipeline states.

        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_k_consumer_state: The load k consumer state
        :type load_k_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the load q consumer state, the load k consumer state, and the mma s producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        load_k_pipeline = common_params.load_k_pipeline
        if cutlass.const_expr(wait_q):
            load_q_pipeline.consumer_wait(load_q_consumer_state)
        if cutlass.const_expr(self.use_m64_ws):
            for q_stage in cutlass.range_constexpr(self.iterations_qk_latent):
                load_k_pipeline.consumer_wait(load_k_consumer_state)
                for k_block in cutlass.range_constexpr(
                    cute.size(qk_params.tSrQ.shape[2])
                ):
                    tcgen05_mma_ws_f8f6f4_one(
                        qk_params.tSrQ[None, None, k_block, (q_stage, 0)],
                        qk_params.tSrKC[None, None, k_block, (q_stage, 0)],
                        tStS,
                        Int32(_IDESC_F8E4M3_F32_M64_N128_KMAJ_1CTA),
                        Boolean(q_stage != 0 or k_block != 0),
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                load_k_pipeline.consumer_release(load_k_consumer_state)
                load_k_consumer_state.advance()

            load_k_rope_pipeline = common_params.load_k_rope_pipeline
            load_k_rope_pipeline.consumer_wait(load_k_rope_consumer_state)
            for q_stage in cutlass.range_constexpr(self.iterations_qk_rope):
                for k_block in cutlass.range_constexpr(
                    self.rope_dim // tiled_mma_qk.shape_mnk[2]
                ):
                    tcgen05_mma_ws_f8f6f4_one(
                        qk_params.tSrQ_rope[None, None, k_block, q_stage],
                        qk_params.tSrKC_rope[None, None, k_block, 0],
                        tStS,
                        Int32(_IDESC_F8E4M3_F32_M64_N128_KMAJ_1CTA),
                        Boolean(True),
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
            load_k_rope_pipeline.consumer_release(load_k_rope_consumer_state)
            load_k_rope_consumer_state.advance()
        else:
            load_k_pipeline.consumer_wait(load_k_consumer_state)
            for q_stage in range(self.iterations_qk_latent):
                kc_stage = load_k_consumer_state.index
                for k_block in cutlass.range_constexpr(
                    cute.size(qk_params.tSrQ.shape[2])
                ):
                    cute.gemm(
                        tiled_mma_qk,
                        tStS,
                        qk_params.tSrQ[None, None, k_block, (q_stage, 0)],
                        qk_params.tSrKC[None, None, k_block, (q_stage, kc_stage)],
                        tStS,
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)

            for q_stage in range(self.iterations_qk_rope):
                kc_stage = load_k_consumer_state.index
                for k_block in cutlass.range_constexpr(
                    self.rope_dim // tiled_mma_qk.shape_mnk[2]
                ):
                    cute.gemm(
                        tiled_mma_qk,
                        tStS,
                        qk_params.tSrQ_rope[None, None, k_block, q_stage],
                        qk_params.tSrKC_rope[None, None, k_block, kc_stage],
                        tStS,
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
            load_k_pipeline.consumer_release(load_k_consumer_state)
            load_k_consumer_state.advance()
        if cutlass.const_expr(wait_q):
            load_q_consumer_state.advance()

        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        return (
            tiled_mma_qk,
            load_q_consumer_state,
            load_k_consumer_state,
            load_k_rope_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_index: cutlass.Int32,
        tiled_mma_pv: cute.TiledMma,
        load_v_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for P*V. Updates the tiled mma pv and pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param pv_params: The pv parameters
        :type pv_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_v_consumer_state: The load v consumer state
        :type load_v_consumer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The P MMA consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The MMA o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma pv, the load v consumer state, the P MMA consumer state, and the MMA o producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        pv_params.p_mma_pipeline.consumer_wait(p_mma_consumer_state)
        load_v_pipeline = common_params.load_v_pipeline
        accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
        mma_o_pipeline = pv_params.mma_o_pipeline

        if cutlass.const_expr(self.use_m64_ws):
            for acc_stage in cutlass.range_constexpr(self.iterations_pv_n):
                mma_o_pipeline.producer_acquire(mma_o_producer_state)
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                for p_stage in cutlass.range_constexpr(self.iterations_pv_k):
                    load_v_pipeline.consumer_wait(load_v_consumer_state)
                    tO_tmem_m64 = pv_params.tO_tmem_staged[None, None, None, acc_stage]
                    for k_block in cutlass.range_constexpr(pv_params.tOrP.shape[2]):
                        idesc = (
                            _IDESC_F8E4M3_F32_M64_N256_AK_BMN_1CTA
                            if cutlass.const_expr(self.mma_pv_tiler[1] == 256)
                            else _IDESC_F8E4M3_F32_M64_N128_AK_BMN_1CTA
                        )
                        tcgen05_mma_ws_f8f6f4_one(
                            pv_params.tOrP[
                                None,
                                None,
                                k_block,
                                (p_stage, p_mma_consumer_state.index),
                            ],
                            pv_params.tOrVC[
                                None, None, k_block, ((acc_stage, p_stage), 0)
                            ],
                            tO_tmem_m64,
                            Int32(idesc),
                            Boolean(
                                accumulate_flag
                                if p_stage == 0 and k_block == 0
                                else True
                            ),
                        )
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                    load_v_pipeline.consumer_release(load_v_consumer_state)
                    load_v_consumer_state.advance()

                mma_o_pipeline.producer_commit(mma_o_producer_state)
                mma_o_producer_state.advance()
        else:
            load_v_pipeline.consumer_wait(load_v_consumer_state)
            vc_stage = load_v_consumer_state.index
            for acc_stage in range(self.iterations_pv_n):
                mma_o_pipeline.producer_acquire(mma_o_producer_state)
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                for p_stage in range(self.iterations_pv_k):
                    tO_tmem = pv_params.tO_tmem_staged[None, None, None, acc_stage]
                    for k_block in cutlass.range_constexpr(pv_params.tOrP.shape[2]):
                        cute.gemm(
                            tiled_mma_pv,
                            tO_tmem,
                            pv_params.tOrP[
                                None,
                                None,
                                k_block,
                                (p_stage, p_mma_consumer_state.index),
                            ],
                            pv_params.tOrVC[
                                None, None, k_block, ((acc_stage, p_stage), vc_stage)
                            ],
                            tO_tmem,
                        )
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)

                mma_o_pipeline.producer_commit(mma_o_producer_state)
                mma_o_producer_state.advance()
            load_v_pipeline.consumer_release(load_v_consumer_state)
            load_v_consumer_state.advance()
        pv_params.p_mma_pipeline.consumer_release(p_mma_consumer_state)
        p_mma_consumer_state.advance()

        return (
            tiled_mma_pv,
            load_v_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState

        :return: The MMA s consumer state, the P MMA producer state, and the P correction producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        row_max = -self.acc_dtype.inf
        row_sum = self.acc_dtype(0)
        correction_factor = self.acc_dtype(1)
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)

        # Number of tiles from the global-K end that may contain causal-masked
        # positions. For causal, min k_bound = K - (S_q-1), which can span up
        # to ceil((S_q-1)/tile_N)+1 tiles (tile-boundary-crossing case).
        # Non-causal only needs the final K-bound tile.
        tile_n = self.mma_qk_tiler[1]
        if cutlass.const_expr(self.is_causal):
            mask_tile_count = (self.seq_len_q - 1 + tile_n - 1) // tile_n + 1
        else:
            mask_tile_count = 1

        # first_mask_tile_idx is the global index of the first tile that may
        # need masking. Runtime because it depends on K (per-batch in
        # var-seq / split-KV).
        first_mask_tile_idx = k_tile_total - mask_tile_count
        if cutlass.const_expr(common_params.local_visible_lens is not None):
            # Every query can see tiles wholly before the minimum local bound.
            # Do this once per work tile, outside the KV loop.
            min_visible = common_params.K
            for query_index in cutlass.range_constexpr(self.seq_len_q):
                min_visible = min(
                    min_visible,
                    common_params.local_visible_lens[
                        common_params.blk_coord[2], query_index
                    ],
                )
            first_mask_tile_idx = min_visible // tile_n

        # Phase 0: tiles straddling a sliding window's low edge. The edge moves
        # by one key per query row, so it spans the same tile count the causal
        # edge does, starting at the first tile the window reaches.
        if cutlass.const_expr(self.window_left >= 0):
            low_mask_end = (
                max(0, common_params.K - self.seq_len_q - self.window_left) // tile_n
                + (self.seq_len_q - 1 + tile_n - 1) // tile_n
                + 1
            )
            while k_tile_count > 1 and k_index < low_mask_end:
                (
                    mma_s_consumer_state,
                    p_mma_producer_state,
                    p_cor_producer_state,
                    row_max,
                    row_sum,
                    correction_factor,
                ) = self.softmax(
                    common_params,
                    softmax_params,
                    k_index,
                    mma_s_consumer_state,
                    p_mma_producer_state,
                    p_cor_producer_state,
                    row_max,
                    row_sum,
                    correction_factor,
                    True,
                    False,
                )
                k_index = k_index + 1
                k_tile_count = k_tile_count - 1

        # Phase 1: pure unmasked bulk tiles (all columns strictly < min k_bound).
        while k_tile_count > 1 and k_index < first_mask_tile_idx:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                False,
                False,
            )
            k_index = k_index + 1
            k_tile_count = k_tile_count - 1

        # Phase 2: intermediate tiles that overlap the causal/K-bound region
        # but are not this work-split's final tile.
        while k_tile_count > 1:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                True,
                False,
            )
            k_index = k_index + 1
            k_tile_count = k_tile_count - 1

        # Phase 3: this work-split's final tile. A window's low edge can land
        # on any tile, including a split's last one, so windowed splits always
        # mask here rather than testing only the K-bound edge.
        if cutlass.const_expr(self.window_left >= 0):
            final_tile_apply_mask = True
        else:
            final_tile_apply_mask = k_index >= first_mask_tile_idx
        if cutlass.const_expr(common_params.mAccO is not None):
            # Split-KV: only apply mask when this final tile is globally in
            # the mask region (covers both last-split last-tile and straddling
            # splits). Runtime comparison.
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                final_tile_apply_mask,
                True,
            )
        else:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                True,
                True,
            )

        return mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state

    @cute.jit
    def correction(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        p_cor_consumer_state: pipeline.PipelineState,
        mma_o_consumer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState
        :param mma_o_consumer_state: The MMA o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, and the MMA o consumer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_count_init = k_tile_count
        while k_tile_count > 0:
            p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction = (
                self.get_correction_factor(common_params, p_cor_consumer_state)
            )
            if k_tile_count_init != k_tile_count:
                mma_o_consumer_state = self.rescale(
                    common_params,
                    mma_o_consumer_state,
                    correction_factor,
                    no_correction,
                )
            k_tile_count = k_tile_count - 1
            if k_tile_count == 0:
                mma_o_consumer_state = self.epilogue(
                    common_params,
                    epilogue_params,
                    mma_o_consumer_state,
                    row_sum,
                    row_max,
                )
        return p_cor_consumer_state, mma_o_consumer_state

    @cute.jit
    def exchange_p_cor_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        row_sum: cutlass.Float32,
        row_max_new: cutlass.Float32,
        no_correction: cutlass.Int32,
        tAcc: cute.Tensor,
        tidx: cutlass.Int32,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Send the tile's correction metadata to the correction warp."""

        # pad for 4x32b
        if cutlass.const_expr(self.use_2cta_instrs):
            corr_layout = cute.make_layout(
                (
                    tAcc.shape[0],
                    (_P_COR_METADATA_FIELDS, tAcc.shape[1][1]),
                    self.mma_s_stage,
                ),
                stride=(
                    tAcc.stride[0],
                    (1, tAcc.stride[1][1]),
                    _P_COR_METADATA_FIELDS,
                ),
            )
        elif cutlass.const_expr(self.use_m64_ws):
            corr_layout = cute.make_layout(
                (
                    self.ws_p_cor_rows,
                    (_P_COR_METADATA_FIELDS, self.ws_p_cor_n_groups),
                    self.p_cor_stage,
                ),
                stride=(
                    _TMEM_ROW_STRIDE,
                    (1, self.ws_p_cor_rows * _TMEM_ROW_STRIDE),
                    _P_COR_METADATA_FIELDS,
                ),
            )
        else:
            corr_layout = cute.make_layout(
                (tAcc.shape[0], _P_COR_METADATA_FIELDS, self.mma_s_stage),
                stride=(tAcc.stride[0], 1, _P_COR_METADATA_FIELDS),
            )
        tCor = cute.make_tensor(
            common_params.tmem_ptr + self.correction_factor_offset,
            corr_layout,
        )
        cCor = cute.make_identity_tensor(tCor.shape)
        if cutlass.const_expr(self.use_2cta_instrs or self.use_m64_ws):
            corr_tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
        else:
            corr_tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
        corr_tmem_store_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_store_atom, tCor)
        corr_tmem_store_thr_copy = corr_tmem_store_tiled_copy.get_slice(tidx)
        cCor_for_copy = corr_tmem_store_thr_copy.partition_S(cCor)
        tCor_for_copy = corr_tmem_store_thr_copy.partition_D(tCor)
        rCor = cute.make_fragment_like(
            cCor_for_copy[None, None, None, 0], self.acc_dtype
        )
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        rCor[0] = row_sum
        rCor[1] = row_max_new
        rCor[2] = correction_factor
        rCor_int[3] = no_correction

        cute.copy(
            corr_tmem_store_tiled_copy,
            rCor,
            tCor_for_copy[None, None, None, p_cor_producer_state.index],
        )
        # fence between tmem store and correction warp
        cute.arch.fence_view_async_tmem_store()
        common_params.p_cor_pipeline.producer_commit(p_cor_producer_state)
        p_cor_producer_state.advance()
        return p_cor_producer_state

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        apply_mask: bool,
        is_local_last_tile: cutlass.Boolean,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        """Softmax for one k-tile. Updates the related pipeline states and returns the computed results.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState
        :param row_max: The row max
        :type row_max: cutlass.Float32
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param apply_mask: Whether the tile needs K-bound / causal masking (Python bool
            for the unmasked/masked bulk loops; runtime cutlass.Boolean for the
            split-KV final iter where mask only applies on the global last tile).
        :type apply_mask: bool | cutlass.Boolean
        :param is_local_last_tile: Whether the last tile is local
        :type is_local_last_tile: cutlass.Boolean

        :return: The MMA s consumer state, the P MMA producer state, the P correction producer state, the row max, the row sum, and the correction factor
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32]
        """

        mma_s_try_token = softmax_params.mma_s_pipeline.consumer_try_wait(
            mma_s_consumer_state
        )
        softmax_params.mma_s_pipeline.consumer_wait(
            mma_s_consumer_state, try_wait_token=mma_s_try_token
        )

        # load S from tmem
        if cutlass.const_expr(self.use_m64_ws):
            tStS_shape = softmax_params.tiled_mma_ws_splitn_epi.partition_shape_C(
                self.ws_splitn_epi_tiler_mn
            )
            tStS_staged_fake = softmax_params.tiled_mma_ws_splitn_epi.make_fragment_C(
                cute.append(tStS_shape, self.mma_s_stage)
            )
        else:
            tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
                cute.select(self.mma_qk_tiler, mode=[0, 1])
            )
            tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
                cute.append(tStS_shape, self.mma_s_stage)
            )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        if cutlass.const_expr(self.use_2cta_instrs or self.use_m64_ws):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(16)), self.acc_dtype
            )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = row_max
        # Causal mask: emulate single-token-per-CTA causal masking for
        # spec-decoding / MTP generation. Each row represents one (q_token, head)
        # pair; row r's effective K bound is K - (S_q - 1 - q_tok(r)).
        #   fold_sq_factor>1 : q_tok(r) = blk_coord[1] * F + (r_global // num_heads),
        #                      where F=fold_sq_factor and M is laid out as [F, H].
        #   fold_sq_factor=1 : q_tok(r) = blk_coord[1] (single q token per work tile)
        # r_global = row_in_cta + cluster_idx * (M_tile / cluster_m)
        # Keep the legacy finite sentinel. Local prefixes need exact zero
        # probability for empty rows; the softmax below handles their -inf max.
        masked_score = self.acc_dtype(-1.0e6)
        if cutlass.const_expr(common_params.local_visible_lens is not None):
            masked_score = -self.acc_dtype.inf
        cta_m_rows = self.mma_qk_tiler[0] // self.cluster_shape_mnk[0]
        if cutlass.const_expr(self.arch == "sm_100"):
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            # Early S release: fence TMEM load and release S buffer immediately
            # so MMA warp can start the next QK computation without waiting for
            # the entire softmax computation to complete.
            cute.arch.fence_view_async_tmem_load()
            softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
            mma_s_consumer_state.advance()
            for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                if apply_mask:
                    qk_col = tTR_tS[i][1]
                    if cutlass.const_expr(
                        self.is_causal or common_params.local_visible_lens is not None
                    ):
                        if cutlass.const_expr(self.pack_q):
                            q_tok = (
                                common_params.blk_coord[1] * self.mma_qk_tiler[0]
                                + common_params.blk_coord[0] * cta_m_rows
                                + tTR_tS[i][0]
                            ) // self.num_heads
                        elif cutlass.const_expr(self.fold_sq_factor > 1):
                            qk_row = tTR_tS[i][0]
                            q_tok = (
                                common_params.blk_coord[1] * self.fold_sq_factor
                                + (qk_row + common_params.blk_coord[0] * cta_m_rows)
                                // self.num_heads
                            )
                        else:
                            q_tok = common_params.blk_coord[1]
                        if cutlass.const_expr(
                            common_params.local_visible_lens is not None
                        ):
                            # Padded query rows must not read beyond metadata.
                            k_bound = common_params.local_visible_lens[
                                common_params.blk_coord[2],
                                min(q_tok, self.seq_len_q - 1),
                            ]
                        elif cutlass.const_expr(self.cp_world == 1):
                            # Non-DCP: causal bound is the full local context length
                            # (identical to the pre-DCP behavior; folded at compile time).
                            k_bound = common_params.K - (self.seq_len_q - 1) + q_tok
                        else:
                            global_bound = (
                                common_params.K_causal - (self.seq_len_q - 1) + q_tok
                            )
                            k_bound = self._get_dcp_local_bound(global_bound)
                    else:
                        k_bound = common_params.K
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            qk_col + self.mma_qk_tiler[1] * k_index,
                            k_bound,
                        )
                        else masked_score
                    )
                    if cutlass.const_expr(self.window_left >= 0):
                        # Row q_tok's window opens at
                        # K - seq_len_q - window_left + q_tok, so a key strictly
                        # before that is out of view however the upper bound
                        # was computed.
                        if cutlass.const_expr(self.pack_q):
                            win_row = (
                                common_params.blk_coord[1] * self.mma_qk_tiler[0]
                                + common_params.blk_coord[0] * cta_m_rows
                                + tTR_tS[i][0]
                            ) // self.num_heads
                        elif cutlass.const_expr(self.fold_sq_factor > 1):
                            win_row = (
                                common_params.blk_coord[1] * self.fold_sq_factor
                                + (
                                    tTR_tS[i][0]
                                    + common_params.blk_coord[0] * cta_m_rows
                                )
                                // self.num_heads
                            )
                        else:
                            win_row = common_params.blk_coord[1]
                        tTR_rAcc[i] = (
                            tTR_rAcc[i]
                            if cute.elem_less(
                                common_params.K
                                - self.seq_len_q
                                - self.window_left
                                + win_row
                                - 1,
                                qk_col + self.mma_qk_tiler[1] * k_index,
                            )
                            else masked_score
                        )
            # reduction for row_max
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)
        else:  # SM103 and SM107 support TMEM load with a MAX reduction.
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            cute.copy(
                tmem_red_tiled_copy,
                tTR_tAcc_red,
                (tTR_rAcc_red, tTR_rMax),
            )
            # Early S release for the SM103/SM107 reduction path
            cute.arch.fence_view_async_tmem_load()
            softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
            mma_s_consumer_state.advance()
            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if apply_mask:
                for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                    if cutlass.const_expr(
                        self.is_causal or common_params.local_visible_lens is not None
                    ):
                        if cutlass.const_expr(self.pack_q):
                            q_tok = (
                                common_params.blk_coord[1] * self.mma_qk_tiler[0]
                                + common_params.blk_coord[0] * cta_m_rows
                                + tTR_tS[i][0]
                            ) // self.num_heads
                        elif cutlass.const_expr(self.fold_sq_factor > 1):
                            q_tok = (
                                common_params.blk_coord[1] * self.fold_sq_factor
                                + (
                                    tTR_tS[i][0]
                                    + common_params.blk_coord[0] * cta_m_rows
                                )
                                // self.num_heads
                            )
                        else:
                            q_tok = common_params.blk_coord[1]
                        # effective K(row) = K - (S_q - 1) + q_tok; equivalent to
                        # K - (S_q - 1 - q_tok), written to avoid Python int/CuTe
                        # Int32 __rsub__ quirks when self.seq_len_q==1.
                        if cutlass.const_expr(
                            common_params.local_visible_lens is not None
                        ):
                            # Padded query rows must not read beyond metadata.
                            k_bound = common_params.local_visible_lens[
                                common_params.blk_coord[2],
                                min(q_tok, self.seq_len_q - 1),
                            ]
                        elif cutlass.const_expr(self.cp_world == 1):
                            # Non-DCP: causal bound is the full local context length
                            # (identical to the pre-DCP behavior; folded at compile time).
                            k_bound = common_params.K - (self.seq_len_q - 1) + q_tok
                        else:
                            global_bound = (
                                common_params.K_causal - (self.seq_len_q - 1) + q_tok
                            )
                            k_bound = self._get_dcp_local_bound(global_bound)
                    else:
                        k_bound = common_params.K
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            k_bound,
                        )
                        else masked_score
                    )
                    if cutlass.const_expr(self.window_left >= 0):
                        # Row q_tok's window opens at
                        # K - seq_len_q - window_left + q_tok, so a key strictly
                        # before that is out of view however the upper bound
                        # was computed.
                        if cutlass.const_expr(self.pack_q):
                            win_row = (
                                common_params.blk_coord[1] * self.mma_qk_tiler[0]
                                + common_params.blk_coord[0] * cta_m_rows
                                + tTR_tS[i][0]
                            ) // self.num_heads
                        elif cutlass.const_expr(self.fold_sq_factor > 1):
                            win_row = (
                                common_params.blk_coord[1] * self.fold_sq_factor
                                + (
                                    tTR_tS[i][0]
                                    + common_params.blk_coord[0] * cta_m_rows
                                )
                                // self.num_heads
                            )
                        else:
                            win_row = common_params.blk_coord[1]
                        tTR_rAcc[i] = (
                            tTR_rAcc[i]
                            if cute.elem_less(
                                common_params.K
                                - self.seq_len_q
                                - self.window_left
                                + win_row
                                - 1,
                                tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            )
                            else masked_score
                        )
                # reduction for row_max after manual masking
                row_max_new = tTR_rAcc.load().reduce(
                    cute.ReductionOp.MAX, row_max_new, 0
                )
            else:
                # The TMEM load reduction is valid here because
                # tTR_rAcc is unmodified (no mask applied to this tile).
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])

        # if warps in N is 2, reduce row_max across warps (0, 1) and (2, 3)
        # Exchange pairs: {warp0, warp2} and {warp1, warp3} are independent.
        # Use split barriers so each pair only waits for its partner.
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_max_new
            warp_in_group = tidx // self.threads_per_warp
            if warp_in_group % 2 == 0:
                self.softmax_exchange_sync_bar_pair02.arrive_and_wait()
            else:
                self.softmax_exchange_sync_bar_pair13.arrive_and_wait()
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )
        # find correction factor
        max_delta = row_max - row_max_new
        if cutlass.const_expr(common_params.local_visible_lens is not None):
            # Both maxima are -inf until this row sees a local key.
            max_delta = max_delta if row_max_new != -self.acc_dtype.inf else 0.0
        correction_factor = cute.math.exp2(
            max_delta * softmax_params.softmax_scale_log2, fastmath=True
        )
        # skip the rescale if the row max barely grew; the first tile never skips
        skip = (
            row_max_new - row_max
        ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold
        no_correction = cutlass.Int32(1) if skip else cutlass.Int32(0)
        row_max_new = row_max if skip else row_max_new
        correction_factor = cutlass.Float32(1.0) if skip else correction_factor
        # split kv case
        if cutlass.const_expr(not is_local_last_tile):
            p_cor_producer_state = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max_new,
                no_correction,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        # softmax
        fma_b = softmax_params.softmax_scale_log2
        safe_row_max = row_max_new
        if cutlass.const_expr(common_params.local_visible_lens is not None):
            safe_row_max = row_max_new if row_max_new != -self.acc_dtype.inf else 0.0
        fma_c = (
            0.0 - safe_row_max
        ) * softmax_params.softmax_scale_log2 + cutlass.Float32(self.p_prescale_log2)

        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        # Compute row_sum immediately after exp2, before quantize/P-store.
        # This overlaps the FADD reductions with any remaining MUFU pipeline drain
        # and moves row_sum off the critical path between P-commit and p_cor-acquire.
        row_sum = row_sum * correction_factor
        row_sum_vec = (0.0, 0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(
                row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1])
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        # quantize
        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        # Acquire close to STS so softmax work can overlap with downstream MMA.
        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        # change to PISL
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = (
            int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        )
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        universal_copy_bits = 128
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        rP_copy_view = smem_thr_copy.retile(tTR_rS)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)

        # fence between smem store and mma o
        cute.arch.fence_view_async_shared()
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        p_mma_producer_state.advance()

        # Issue non-blocking try_acquire for p_cor pipeline early, so the
        # blocking acquire at the end can overlap with remaining work.
        p_cor_try_token = common_params.p_cor_pipeline.producer_try_acquire(
            p_cor_producer_state
        )

        # split kv case
        if cutlass.const_expr(is_local_last_tile):
            p_cor_producer_state = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max_new,
                no_correction,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        # store correction factor/row_sum/row_max to tmem for correction warp
        common_params.p_cor_pipeline.producer_acquire(
            p_cor_producer_state, try_acquire_token=p_cor_try_token
        )

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            p_cor_producer_state,
            row_max_new,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def _tmem_load_partition(
        self,
        common_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        iter_n: int,
        final: cutlass.Constexpr = False,
    ) -> tuple[
        cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma
    ]:
        """Tensor memory load partition for rescale and epilogue.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param iter_n: The iteration number
        :type iter_n: int

        :return: The tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv
        :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
        """

        if cutlass.const_expr(self.use_m64_ws):
            tO_tmem_shape = common_params.tiled_mma_ws_epi.partition_shape_C(
                (128, self.mma_pv_tiler[1] // self.warps_in_n)
            )
            tO_tmem = common_params.tiled_mma_ws_epi.make_fragment_C(tO_tmem_shape)
        else:
            tO_tmem_shape = tiled_mma_pv.partition_shape_C(
                cute.select(self.mma_pv_tiler, mode=[0, 1])
            )
            tO_tmem = tiled_mma_pv.make_fragment_C(tO_tmem_shape)
        tO_tmem_layout = cute.append(
            tO_tmem.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.pv_acc_cols,
            ),
        )
        tO_tmem = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tO_tmem_layout
        )
        tO_tmem = tO_tmem[None, None, None, iter_n]

        tAcc = tO_tmem[(None, None), 0, 0]

        if cutlass.const_expr(self.use_m64_ws):
            if cutlass.const_expr(self.mma_pv_tiler[1] == 256):
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
                    self.acc_dtype,
                )
            else:
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
                    self.acc_dtype,
                )
        elif cutlass.const_expr(self.use_2cta_instrs):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype
            )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        # Flatten divide and partition global tensors for O
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gO = None
        if cutlass.const_expr(common_params.mAccO is not None and not final):
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[3], None, None, None],
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(
                    common_params.mAccO[
                        None, common_params.blk_coord[3], None, None, None
                    ].shape
                ),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(common_params.mO.shape),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        if cutlass.const_expr(self.use_m64_ws):
            if cutlass.const_expr(self.mma_pv_tiler[1] == 256):
                gO_phys = cute.make_tensor(
                    gO.iterator,
                    cute.make_layout(
                        ((64, 2), (64, 2)),
                        stride=(
                            (
                                gO.stride[0],
                                (self.mma_pv_tiler[1] // self.warps_in_n)
                                * gO.stride[1],
                            ),
                            (gO.stride[1], 64 * gO.stride[1]),
                        ),
                    ),
                )
                cO_phys = cute.make_identity_tensor((128, 128))
            else:
                gO_phys = cute.make_tensor(
                    gO.iterator,
                    cute.make_layout(
                        ((64, 2), 64),
                        stride=((gO.stride[0], 64 * gO.stride[1]), gO.stride[1]),
                    ),
                )
                cO_phys = cute.make_identity_tensor((128, 64))
            tTR_gO = tmem_load_thr_copy.partition_D(gO_phys)
            tTR_cO = tmem_load_thr_copy.partition_D(cO_phys)
        else:
            tTR_gO = tmem_load_thr_copy.partition_D(gO)
            tTR_cO = tmem_load_thr_copy.partition_D(cO)
        tTR_rAcc = cute.make_fragment_like(tTR_gO, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc  # type: ignore[return-value]

    def get_correction_factor(
        self,
        common_params: SimpleNamespace,
        p_cor_consumer_state: pipeline.PipelineState,
    ) -> tuple[
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Int32,
    ]:
        """Get the correction factor from the P correction consumer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, the row_sum, the row_max, and the correction factor
        :rtype: tuple[pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Int32]
        """
        common_params.p_cor_pipeline.consumer_wait(p_cor_consumer_state)
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        # load correction factor
        _, tAcc, _, _, _, _ = self._tmem_load_partition(
            common_params, common_params.tiled_mma_pv, 0
        )
        if cutlass.const_expr(self.use_2cta_instrs):
            corr_layout = cute.make_layout(
                (
                    tAcc.shape[0],
                    (_P_COR_METADATA_FIELDS, tAcc.shape[1][1]),
                    self.p_cor_stage,
                ),
                stride=(
                    tAcc.stride[0],
                    (1, tAcc.stride[1][1]),
                    _P_COR_METADATA_FIELDS,
                ),
            )
        elif cutlass.const_expr(self.use_m64_ws):
            corr_layout = cute.make_layout(
                (
                    self.ws_p_cor_rows,
                    (_P_COR_METADATA_FIELDS, self.ws_p_cor_n_groups),
                    self.p_cor_stage,
                ),
                stride=(
                    _TMEM_ROW_STRIDE,
                    (1, self.ws_p_cor_rows * _TMEM_ROW_STRIDE),
                    _P_COR_METADATA_FIELDS,
                ),
            )
        else:
            corr_layout = cute.make_layout(
                (tAcc.shape[0], _P_COR_METADATA_FIELDS, self.p_cor_stage),
                stride=(tAcc.stride[0], 1, _P_COR_METADATA_FIELDS),
            )
        tCor = cute.make_tensor(
            common_params.tmem_ptr + self.correction_factor_offset, corr_layout
        )
        cCor = cute.make_identity_tensor(tCor.shape)
        if cutlass.const_expr(self.use_2cta_instrs or self.use_m64_ws):
            corr_tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
        else:
            corr_tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
        corr_tmem_load_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_load_atom, tCor)
        corr_tmem_load_thr_copy = corr_tmem_load_tiled_copy.get_slice(tidx)
        tCor_for_copy = corr_tmem_load_thr_copy.partition_S(tCor)
        cCor_for_copy = corr_tmem_load_thr_copy.partition_D(cCor)
        rCor = cute.make_fragment_like(
            cCor_for_copy[None, None, None, 0], self.acc_dtype
        )
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        cute.copy(
            corr_tmem_load_tiled_copy,
            tCor_for_copy[None, None, None, p_cor_consumer_state.index],
            rCor,
        )
        row_sum = rCor[0]
        row_max = rCor[1]
        correction_factor = rCor[2]
        no_correction = rCor_int[3]

        common_params.p_cor_pipeline.consumer_release(p_cor_consumer_state)
        p_cor_consumer_state.advance()
        return p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
    ) -> pipeline.PipelineState:
        """Rescale for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param no_correction: Whether to apply correction factor
        :type no_correction: cutlass.Int32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
            if not skip_correction:
                # tmem load tiled copy and partition results.
                tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                    self._tmem_load_partition(
                        common_params, common_params.tiled_mma_pv, iter_n
                    )
                )

                # tmem store tiled copy
                if cutlass.const_expr(self.use_m64_ws):
                    if cutlass.const_expr(self.mma_pv_tiler[1] == 256):
                        tmem_store_atom = cute.make_copy_atom(
                            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
                            self.acc_dtype,
                        )
                    else:
                        tmem_store_atom = cute.make_copy_atom(
                            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)),
                            self.acc_dtype,
                        )
                elif cutlass.const_expr(self.use_2cta_instrs):
                    tmem_store_atom = cute.make_copy_atom(
                        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                        self.acc_dtype,
                    )
                else:
                    tmem_store_atom = cute.make_copy_atom(
                        tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(16)),
                        self.acc_dtype,
                    )
                tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

                # load o
                cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
                # rescale, using `mul_packed_f32x2` to reduce the number of instructions
                for i in cutlass.range(
                    cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                ):
                    tTR_rAcc[i] = tTR_rAcc[i] * correction_factor

                # store o to tensor memory for next k tile
                cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

            cute.arch.fence_view_async_tmem_store()
            common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
            mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ) -> pipeline.PipelineState:
        """Epilogue for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param row_max: The row max
        :type row_max: cutlass.Float32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        # exchange row_sum between warps (0, 1) and (2, 3)
        # Use split barriers so each pair only waits for its partner.
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            warp_in_group = tidx // self.threads_per_warp
            if warp_in_group % 2 == 0:
                self.epilogue_exchange_sync_bar_pair02.arrive_and_wait()
            else:
                self.epilogue_exchange_sync_bar_pair13.arrive_and_wait()
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
            )
        # Pre-compute the output scale factor once to avoid redundant MUFU.RCP
        # instructions inside the per-element loop (rcp_approx generates MUFU.RCP
        # which has 8-cycle throughput; hoisting it saves up to ~500 cycles).
        safe_row_sum = row_sum
        if cutlass.const_expr(common_params.local_visible_lens is not None):
            safe_row_sum = row_sum if row_sum != 0.0 else 1.0
        # partials skip output_scale; the reduction kernel applies it
        if cutlass.const_expr(common_params.mAccO is None):
            epi_scale = epilogue_params.output_scale * cute.arch.rcp_approx(
                safe_row_sum
            )
            store_dtype = self.o_dtype
        else:
            epi_scale = cute.arch.rcp_approx(safe_row_sum)
            store_dtype = self.partial_dtype
        # Work list: a request with a single split stores its final output (with
        # output_scale) instead of a partial, and the reduction skips it.
        direct_final = False
        if cutlass.const_expr(self.use_work_list and common_params.mAccO is not None):
            direct_final = common_params.local_split_kv == 1
        final_scale = epilogue_params.output_scale * cute.arch.rcp_approx(safe_row_sum)

        # mma_o pipeline consumer wait
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
            # tmem load tiled copy and partition results (full tile).
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, common_params.tiled_mma_pv, iter_n
                )
            )

            # Full TMEM load to registers
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            if cutlass.const_expr(common_params.local_visible_lens is not None):
                if row_sum == 0.0:
                    tTR_rAcc.fill(0.0)

            if cutlass.const_expr(self.use_m64_ws):
                num_epi_sub_tiles = cute.size(tTR_rAcc, mode=[2])
                vec_size = cute.size(cute.select(tTR_rAcc.shape, mode=[0]))
                tR2G_rO_src = cute.make_fragment_like(tTR_gO, store_dtype)

                for d_sub in cutlass.range_constexpr(num_epi_sub_tiles):
                    for i in cutlass.range(vec_size, vectorize=True, unroll_full=True):
                        tTR_rAcc[i, 0, d_sub] = tTR_rAcc[i, 0, d_sub] * epi_scale

                    tR2G_rO_src[None, None, d_sub].store(
                        tTR_rAcc[None, None, d_sub].load().to(store_dtype)
                    )

                    if cute.elem_less(tTR_cO[0][0] % 64, common_params.H):
                        cute.autovec_copy(
                            tR2G_rO_src[None, None, d_sub],
                            tTR_gO[None, None, d_sub],
                            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                        )
            elif cutlass.const_expr(
                self.use_work_list and common_params.mAccO is not None
            ):
                if direct_final:
                    _, _, _, tTR_gF, tTR_cF, _ = self._tmem_load_partition(
                        common_params, common_params.tiled_mma_pv, iter_n, final=True
                    )
                    self._store_rows(
                        tTR_rAcc,
                        tTR_gF,
                        tTR_cF,
                        final_scale,
                        self.o_dtype,
                        common_params.H,
                    )
                else:
                    self._store_rows(
                        tTR_rAcc,
                        tTR_gO,
                        tTR_cO,
                        epi_scale,
                        store_dtype,
                        common_params.H,
                    )
            else:
                self._store_rows(
                    tTR_rAcc, tTR_gO, tTR_cO, epi_scale, store_dtype, common_params.H
                )

            # store the lse to global memory
            cta_pv_tiler = (
                self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
                self.mma_pv_tiler[1],
                self.mma_pv_tiler[2],
            )
            if cutlass.const_expr(epilogue_params.mAccLSE is None):
                if cutlass.const_expr(not self.skip_lse):
                    lse = (
                        cute.math.log2(row_sum, fastmath=True)
                        - cutlass.Float32(self.p_prescale_log2)
                    ) + epilogue_params.softmax_scale_log2 * row_max
                    gLSE = cute.local_tile(
                        epilogue_params.mLSE,
                        (cta_pv_tiler[0], 1, 1),
                        (
                            common_params.blk_coord[0],
                            common_params.blk_coord[1],
                            common_params.blk_coord[2],
                        ),
                        (1, 1, 1),
                    )
                    cLSE = cute.local_tile(
                        cute.make_identity_tensor(epilogue_params.mLSE.shape),
                        (cta_pv_tiler[0], 1, 1),
                        (
                            common_params.blk_coord[0],
                            common_params.blk_coord[1],
                            common_params.blk_coord[2],
                        ),
                        (1, 1, 1),
                    )
                    if cutlass.const_expr(self.use_m64_ws):
                        if cute.elem_less(cLSE[tidx][0], common_params.H):
                            gLSE[tidx] = lse
                    elif cutlass.const_expr(self.warps_in_n == 2):
                        if cute.elem_less(cLSE[tidx][0], common_params.H):
                            gLSE[tidx] = lse
            else:
                if not direct_final:
                    lse = (
                        cute.math.log2(row_sum, fastmath=True)
                        - cutlass.Float32(self.p_prescale_log2)
                    ) + epilogue_params.softmax_scale_log2 * row_max
                    gLSE = cute.local_tile(
                        epilogue_params.mAccLSE[
                            None, common_params.blk_coord[3], None, None
                        ],
                        (cta_pv_tiler[0], 1, 1),
                        (
                            common_params.blk_coord[0],
                            common_params.blk_coord[1],
                            common_params.blk_coord[2],
                        ),
                        (1, 1, 1),
                    )
                    cLSE = cute.local_tile(
                        cute.make_identity_tensor(
                            epilogue_params.mAccLSE[
                                None, common_params.blk_coord[3], None, None
                            ].shape
                        ),
                        (cta_pv_tiler[0], 1, 1),
                        (
                            common_params.blk_coord[0],
                            common_params.blk_coord[1],
                            common_params.blk_coord[2],
                        ),
                        (1, 1, 1),
                    )
                    if cutlass.const_expr(self.use_m64_ws):
                        if cute.elem_less(cLSE[tidx][0], common_params.H):
                            gLSE[tidx] = lse
                    elif cutlass.const_expr(self.warps_in_n == 2):
                        if cute.elem_less(cLSE[tidx][0], common_params.H):
                            gLSE[tidx] = lse
                else:
                    if cutlass.const_expr(not self.skip_lse):
                        # A single-split request's final LSE (the reduction skips it).
                        lse = (
                            cute.math.log2(row_sum, fastmath=True)
                            - cutlass.Float32(self.p_prescale_log2)
                        ) + epilogue_params.softmax_scale_log2 * row_max
                        gLSE = cute.local_tile(
                            epilogue_params.mLSE,
                            (cta_pv_tiler[0], 1, 1),
                            (
                                common_params.blk_coord[0],
                                common_params.blk_coord[1],
                                common_params.blk_coord[2],
                            ),
                            (1, 1, 1),
                        )
                        cLSE = cute.local_tile(
                            cute.make_identity_tensor(epilogue_params.mLSE.shape),
                            (cta_pv_tiler[0], 1, 1),
                            (
                                common_params.blk_coord[0],
                                common_params.blk_coord[1],
                                common_params.blk_coord[2],
                            ),
                            (1, 1, 1),
                        )
                        if cutlass.const_expr(self.use_m64_ws):
                            if cute.elem_less(cLSE[tidx][0], common_params.H):
                                gLSE[tidx] = lse
                        elif cutlass.const_expr(self.warps_in_n == 2):
                            if cute.elem_less(cLSE[tidx][0], common_params.H):
                                gLSE[tidx] = lse

            cute.arch.fence_view_async_tmem_load()
            common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
            mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def _store_rows(self, tTR_rAcc, tTR_gO, tTR_cO, scale, store_dtype, H):
        """Scale the accumulator rows, convert and store them (rows < ``H``).

        Subtiled elementwise + STG: elementwise of subtile N+1 overlaps with STG
        of subtile N via instruction-level parallelism. tTR_rAcc shape:
        ((V, 1), 1, rest_tiles) where rest_tiles = 4, split along mode 2."""
        num_epi_sub_tiles = tTR_rAcc.shape[2]
        vec_size = cute.size(cute.select(tTR_rAcc.shape, mode=[0]))
        tR2G_rO_src = cute.make_fragment_like(tTR_gO, store_dtype)
        for d_sub in cutlass.range_constexpr(num_epi_sub_tiles):
            for i in cutlass.range(vec_size, vectorize=True, unroll_full=True):
                tTR_rAcc[i, 0, d_sub] = tTR_rAcc[i, 0, d_sub] * scale
            # vectorized load/to/store (scalar f32->fp8 cvt is not supported)
            tR2G_rO_src[None, None, d_sub].store(
                tTR_rAcc[None, None, d_sub].load().to(store_dtype)
            )
            # vectorized stores (STG.128)
            if cute.elem_less(tTR_cO[0][0], H):
                cute.autovec_copy(
                    tR2G_rO_src[None, None, d_sub],
                    tTR_gO[None, None, d_sub],
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )

    def make_and_init_load_qkv_pipeline(
        self, load_qkv_mbar_ptr, cta_layout_vmnk, load_stages, tx_count
    ) -> pipeline.PipelineTmaUmma:
        """Create and initialize the tma load qkv pipeline.

        :param load_qkv_mbar_ptr: The load qkv mbar pointer
        :type load_qkv_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]
        :param load_stages: The load stages
        :type load_stages: list[int]
        :param tx_count: The tx count
        :type tx_count: int

        :return: The tma load qkv pipeline
        :rtype: pipeline.PipelineTmaUmma
        """
        load_qkv_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_tma_k_warp_id])
        )
        load_qkv_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_qkv_mbar_ptr,
            num_stages=load_stages,
            producer_group=load_qkv_producer_group,
            consumer_group=load_qkv_consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_s_pipeline(
        self, mma_s_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma s pipeline.

        :param mma_s_mbar_ptr: The mma s mbar pointer
        :type mma_s_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma s pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_s_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_s_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_s_mbar_ptr,
            num_stages=self.mma_s_stage,
            producer_group=mma_s_producer_group,
            consumer_group=mma_s_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_mma_pipeline(
        self, p_mma_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p mma pipeline.

        :param p_mma_mbar_ptr: The p mma mbar pointer
        :type p_mma_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The p mma pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        p_mma_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_mma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=p_mma_mbar_ptr,
            num_stages=self.p_mma_stage,
            producer_group=p_mma_producer_group,
            consumer_group=p_mma_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_cor_pipeline(
        self, p_cor_mbar_ptr
    ) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p correction pipeline.

        :param p_cor_mbar_ptr: The p correction mbar pointer
        :type p_cor_mbar_ptr: cute.Tensor

        :return: The p correction pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = self.threads_per_warp * len(self.compute_warp_ids)
        p_cor_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_cor_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=p_cor_mbar_ptr,
            num_stages=self.p_cor_stage,
            producer_group=p_cor_producer_group,
            consumer_group=p_cor_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_o_pipeline(
        self, mma_o_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma o pipeline.

        :param mma_o_mbar_ptr: The mma o mbar pointer
        :type mma_o_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma o pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_o_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_o_mbar_ptr,
            num_stages=self.mma_o_stage,
            producer_group=mma_o_producer_group,
            consumer_group=mma_o_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[MLAStaticTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[MLAStaticTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        tile_sched_params = create_mla_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[3]),  # batch
            cute.size(o_shape[2]),  # S_q or S_q/fold_sq_factor
            cluster_shape_mnk,
            split_kv,
        )
        grid = MLAStaticTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def get_workspace_size(
        H: int,
        S: int,
        D: int,
        B: int,
        split_kv: int,
        acc_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Get the extra workspace(device memory) size for the MLA kernel when split_kv is not 1.

        :param H: The height of the output tensor C
        :type H: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param D: The depth of the output tensor C
        :type D: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]

        :return: The workspace size for the MLA kernel
        :rtype: int
        """
        if split_kv == 1:
            return 0
        return B * H * S * split_kv * (D + 1) * acc_dtype.width // 8

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the MLA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param S: The sequence length of the output tensor C
        :type S: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 256 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            # partials use partial_dtype, LSE partials stay in acc_dtype
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=self.partial_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            # the partial block is a multiple of 1 KB; say so or LSE loads drop to 16-bit halves
            # Convert bits to bytes before multiplying the Int32 element count.
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator
                + cute.assume(
                    cute.cosize(acc_o_layout) * (self.partial_dtype.width // 8), 32
                ),
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def can_implement(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size: int,
    ) -> bool:
        """Check if the MLA kernel can be implemented.

        :param B: The batch size of the output tensor C
        :type B: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param K: The width of the output tensor KV
        :type K: int
        :param H: The number of heads of the output tensor C
        :type H: int
        :param L: The number of latent dimensions of the tensor KV
        :type L: int
        :param R: The number of rope dimensions of the tensor C_rope
        :type R: int
        :param in_dtype: The data type of the input tensor
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: The data type of the log-sum-exp
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_qk_tiler_mn: The tile shape of the query-key matrix multiplication
        :type mma_qk_tiler_mn: Tuple[int, int]
        :param mma_pv_tiler_mn: The tile shape of the probability-value matrix multiplication
        :type mma_pv_tiler_mn: Tuple[int, int]
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split_kv
        :type is_var_split_kv: bool
        :param page_size: The page size of the page table
        :type page_size: int

        :return: Whether the MLA kernel can be implemented
        :rtype: bool
        """
        if L != 512 or R != 64:
            return False
        if in_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if out_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            return False
        # page size equals 1 is prohibited by tma specification, not 128B aligned.
        if mma_qk_tiler_mn[1] % page_size != 0 or page_size == 1:
            return False
        if mma_qk_tiler_mn[0] != mma_pv_tiler_mn[0] or mma_qk_tiler_mn[0] not in (
            64,
            128,
        ):
            return False
        if mma_qk_tiler_mn[0] == 64 and (
            mma_qk_tiler_mn[1] != 128 or mma_pv_tiler_mn[1] not in (128, 256)
        ):
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if H <= 0 or H > mma_qk_tiler_mn[0]:
            return False
        # When H < M tile, fold S_q into H to fill the MMA M dimension
        # H*S can be < M tile (padding with zeros via TMA OOB); it just can't exceed it
        if (
            mma_qk_tiler_mn[0] == 64
            and H < mma_qk_tiler_mn[0]
            and H * S > mma_qk_tiler_mn[0]
        ):
            return False
        if S < 1:
            return False
        if K <= 0:
            return False
        return True
