# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OAI Triton MXFP4 MoE expert.

Defines the ``OAITritonMxfp4Experts`` modular fused-MoE expert: BF16
activations × MXFP4 W4A16 weights, with activation and topk-sum kept
outside ``matmul_ogs`` so LoRA deltas can be injected between the two
GEMMs.

Module-level setup installs monkey-patches against ``triton_kernels`` so
its routing kernels compile for non-power-of-2 top_k.
"""

import torch

import vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.experts.lora_experts_mixin import (  # noqa: E501
    LoRAExpertsMixin,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.topk_weight_and_reduce import (  # noqa: E501
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.utils import (
    _resize_cache,
    swiglu_limit_func,
)
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)


def _patch_make_bitmatrix_metadata() -> None:
    """Monkey-patch make_bitmatrix_metadata to support non-power-of-2 top_k.

    triton's ``tl.arange`` requires a power-of-2 range. Patch is installed
    via ``SparseMatrix.__post_init__.__globals__`` so callers pick it up
    regardless of how they imported the original.
    """
    import triton as _triton
    import triton.language as _tl

    try:
        from triton_kernels.tensor_details import bitmatrix as _bm
        from triton_kernels.tensor_details.bitmatrix import (
            BitmatrixMetadata,
            _keyed_add,
            cdiv,
        )
        from triton_kernels.tensor_details.bitmatrix_details.sum_bitmatrix_rows import (  # noqa: E501
            sum_bitmatrix_rows,
        )
    except ImportError:
        return

    @_triton.jit
    def _stage2_pow2(
        ColSortedIndx,
        RowSortedIndx,
        NonzeroIndx,
        n_tokens,
        ColPartialSum,
        stride_pm,
        stride_pn,
        ColOffs,
        TOKS_PER_ROW: _tl.constexpr,
        BLOCK_PER_TOK: _tl.constexpr,
        BLOCK_SIZE_PADDED: _tl.constexpr,
    ):
        BLOCK_SIZE: _tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
        _tl.static_assert(BLOCK_SIZE_PADDED <= 32768)
        if isinstance(n_tokens, _tl.tensor) and n_tokens.dtype.is_ptr():
            n_tokens = _tl.load(n_tokens)
        nonzero_indx_size = n_tokens * TOKS_PER_ROW
        pid_m = _tl.program_id(0)
        offs_local = _tl.arange(0, BLOCK_SIZE_PADDED)
        offs_global = pid_m * BLOCK_SIZE + offs_local
        mask = offs_global < nonzero_indx_size
        col_indx = _tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(
            _tl.uint32
        )
        kv_pairs = ((col_indx << 16) | offs_local).to(_tl.uint32)
        kv_pairs = _tl.sort(kv_pairs, 0)
        col_indx = kv_pairs >> 16
        offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xFFFF)
        mask = col_indx != 0xFFFF
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        cols_and_inclusive_run_lengths = _tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xFFFF
        row_sorted_indx = _tl.load(
            ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask
        )
        row_sorted_indx += _tl.load(ColOffs + col_indx, mask=mask)
        row_sorted_indx += exclusive_run_lengths
        _tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
        _tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)

    def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatrix):
        assert nonzero_indx.ndim == 2
        PARTIAL_BLOCK_M = 32
        col_sum, col_partial_sum = sum_bitmatrix_rows(
            bitmatrix, partials_block_size=PARTIAL_BLOCK_M
        )
        device = bitmatrix.device
        n_indx = nonzero_indx.numel()
        n_cols = bitmatrix.shape[1]
        col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
        col_sorted_indx = combined_indx[:n_indx]
        row_sorted_indx = combined_indx[n_indx:]
        MEMSET_BLOCK = 1024
        memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1,)
        _bm._bitmatrix_metadata_compute_stage1[memset_grid](
            combined_indx,
            n_indx * 2,
            -1,
            MEMSET_BLOCK,
            col_sum,
            col_offs,
            col_sum.shape[0],
            col_partial_sum,
            col_partial_sum.shape[0],
            col_partial_sum.stride(0),
            col_partial_sum.stride(1),
            BLOCK_M=512,
            BLOCK_N=512,
        )
        toks_per_row = nonzero_indx.shape[-1]
        block_size = PARTIAL_BLOCK_M * toks_per_row
        block_size_padded = 1 << (max(block_size, 1) - 1).bit_length()
        compute_grid = (cdiv(bitmatrix.shape_max[0], PARTIAL_BLOCK_M),)
        _stage2_pow2[compute_grid](
            col_sorted_indx,
            row_sorted_indx,
            nonzero_indx,
            bitmatrix.shape[0],
            col_partial_sum,
            col_partial_sum.stride(0),
            col_partial_sum.stride(1),
            col_offs,
            TOKS_PER_ROW=toks_per_row,
            BLOCK_PER_TOK=PARTIAL_BLOCK_M,
            BLOCK_SIZE_PADDED=block_size_padded,
        )
        return BitmatrixMetadata(
            col_sum=col_sum,
            col_sorted_indx=col_sorted_indx,
            row_sorted_indx=row_sorted_indx,
        )

    from triton_kernels.tensor import SparseMatrix as _SparseMatrix

    _SparseMatrix.__post_init__.__globals__["make_bitmatrix_metadata"] = (
        _make_bitmatrix_metadata_pow2_safe
    )
    _bm.make_bitmatrix_metadata = _make_bitmatrix_metadata_pow2_safe


def _patch_legacy_routing_for_nonpow2_topk() -> None:
    """Monkey-patch legacy (v3.5.1) routing to support non-pow2 top_k.

    Only needed on the legacy path; the v3.6+ SparseMatrix path is handled
    by ``_patch_make_bitmatrix_metadata``.
    """
    import triton as _triton
    import triton.language as _tl

    try:
        import triton_kernels.routing as _routing
        from triton_kernels.routing_details import _routing_compute as _rc
    except ImportError:
        return

    _keyed_add = _rc._keyed_add
    _expt_data_compute = _rc._expt_data_compute

    @_triton.jit
    def _routing_compute_indx_pow2(
        pid_m,
        GatherIndx,
        ScatterIndx,
        GateScal,
        ExptScal,
        ExptIndx,
        PartialOffs,
        stride_pm,
        stride_pn,
        TokensStart,
        n_tokens,
        BLOCK_M: _tl.constexpr,
        N_EXPTS_ACT: _tl.constexpr,
        BLOCK_SIZE_PADDED: _tl.constexpr,
    ):
        if isinstance(n_tokens, _tl.tensor) and n_tokens.dtype.is_ptr():
            n_tokens = _tl.load(n_tokens)
        n_gates = n_tokens * N_EXPTS_ACT
        BLOCK_SIZE: _tl.constexpr = N_EXPTS_ACT * BLOCK_M
        _tl.static_assert(BLOCK_SIZE_PADDED <= 32768)
        local_offs = _tl.arange(0, BLOCK_SIZE_PADDED)
        offs = pid_m * BLOCK_SIZE + local_offs
        expert = _tl.load(
            ExptIndx + offs,
            mask=(local_offs < BLOCK_SIZE) & (offs < n_gates),
            other=-1,
        ).to(_tl.uint32)
        kv_pairs = ((expert << 16) | local_offs).to(_tl.uint32)
        kv_pairs = _tl.sort(kv_pairs, 0)
        expert = kv_pairs >> 16
        offs = pid_m * BLOCK_SIZE + (kv_pairs & 0xFFFF)
        mask = expert != 0xFFFF
        gate_scal = _tl.load(ExptScal + offs, mask=mask)
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        run_lengths = _tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (run_lengths - 1) & 0xFFFF
        gates = _tl.load(
            PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask
        )
        gates += _tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths
        _tl.store(ScatterIndx + offs, gates, mask=mask)
        _tl.store(GatherIndx + gates, offs, mask=mask)
        _tl.store(GateScal + gates, gate_scal, mask=mask)

    @_triton.jit
    def _combined_routing_compute_pow2(
        GatherIndx,
        ScatterIndx,
        GateScal,
        ExptScal,
        ExptIndx,
        PartialOffs,
        stride_pm,
        stride_pn,
        TokensStart,
        n_tokens,
        BLOCK_M: _tl.constexpr,
        N_EXPTS_ACT: _tl.constexpr,
        Hist,
        MDTileStarts,
        tile_starts_stridem,
        MDTileInfo,
        tile_info_stridem,
        first_tile_dim_log2,
        SIZES: _tl.constexpr,
        BLOCK: _tl.constexpr,
        blocks2a,
        BLOCK_SIZE_PADDED: _tl.constexpr,
    ):
        pid = _tl.program_id(0)
        if pid < blocks2a:
            _expt_data_compute(
                Hist,
                MDTileStarts,
                tile_starts_stridem,
                MDTileInfo,
                tile_info_stridem,
                first_tile_dim_log2,
                SIZES,
                BLOCK,
            )
        else:
            pid -= blocks2a
            _routing_compute_indx_pow2(
                pid,
                GatherIndx,
                ScatterIndx,
                GateScal,
                ExptScal,
                ExptIndx,
                PartialOffs,
                stride_pm,
                stride_pn,
                TokensStart,
                n_tokens,
                BLOCK_M,
                N_EXPTS_ACT,
                BLOCK_SIZE_PADDED,
            )

    def _sort_tokens_pow2(expt_scal, expt_indx, n_expts_tot, bitmatrix):
        HIST_BLOCK_M = 32
        INDX_OFFS_BLOCK_M = 512
        MEMSET_BLOCK = 1024
        cdiv = _triton.cdiv
        device = expt_scal.device
        dtype = expt_scal.dtype
        n_tokens_raw, _ = bitmatrix.shape
        n_tokens_pad, n_expts_act = expt_scal.shape
        n_gates_pad = n_tokens_pad * n_expts_act
        block_size_padded = _triton.next_power_of_2(HIST_BLOCK_M * n_expts_act)

        hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
        hist = hist[:n_expts_tot]
        expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
        topk_indx = combined_indx[:n_gates_pad]
        gate_indx = combined_indx[n_gates_pad:]
        gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)

        (
            token_offs_combined,
            token_offs_raw,
            token_offs_pad,
            block_pid_map,
            blocks1a,
            blocks2a,
            MEMSET_BLOCK_A,
            HIST2_BLOCK_M,
            block_m_log2_start,
            block_m_num,
        ) = _routing._compute_expt_data_internal(hist, n_expts_tot, n_gates_pad)

        blocks1b = cdiv(n_gates_pad * 2, MEMSET_BLOCK) + n_expts_tot + 1
        blocks2b = cdiv(n_tokens_pad, HIST_BLOCK_M)

        _rc._combined_routing_memset[(blocks1a + blocks1b,)](
            combined_indx,
            n_gates_pad * 2,
            -1,
            MEMSET_BLOCK,
            hist,
            expt_offs,
            hist.shape[0],
            n_expts_tot,
            partial_hist,
            partial_hist.shape[0],
            partial_hist.stride(0),
            partial_hist.stride(1),
            token_offs_combined,
            token_offs_combined.stride(0),
            blocks1a,
            block_pid_map,
            block_m_log2_start,
            SIZES=block_m_num,
            BLOCK_A=MEMSET_BLOCK_A,
            BLOCK_N=512,
            BLOCK_M=INDX_OFFS_BLOCK_M,
        )

        indx_offs = partial_hist
        _combined_routing_compute_pow2[(blocks2a + blocks2b,)](
            topk_indx,
            gate_indx,
            gate_scal,
            expt_scal,
            expt_indx,
            indx_offs,
            indx_offs.stride(0),
            indx_offs.stride(1),
            expt_offs,
            n_tokens_raw,
            HIST_BLOCK_M,
            n_expts_act,
            hist,
            token_offs_pad,
            token_offs_pad.stride(0),
            block_pid_map,
            block_pid_map.stride(0),
            block_m_log2_start,
            block_m_num,
            HIST2_BLOCK_M,
            blocks2a,
            block_size_padded,
        )
        return (
            hist,
            topk_indx,
            gate_indx,
            gate_scal,
            token_offs_raw,
            token_offs_pad,
            block_pid_map,
        )

    _routing.sort_tokens = _sort_tokens_pow2


# v3.5.1 uses ``routing()`` / ``Bitmatrix(scratchpad=...)``; v3.6.0+ uses
# ``SparseMatrix`` / ``Bitmatrix(dtype=BIT)``. Detect at import time by
# probing for the v3.6+ ``SparseMatrix`` symbol.
use_legacy_triton_kernels = False

if has_triton_kernels():
    try:
        from triton_kernels.matmul_ogs import (
            GatherIndx,
            RoutingData,
            ScatterIndx,
            matmul_ogs,
        )
        from triton_kernels.tensor import BIT, Bitmatrix

        try:
            from triton_kernels.tensor import (  # noqa: F401
                SparseMatrix,
                make_ragged_tensor_metadata,
            )
        except ImportError:
            use_legacy_triton_kernels = True
        if not use_legacy_triton_kernels:
            _patch_make_bitmatrix_metadata()
        else:
            _patch_legacy_routing_for_nonpow2_topk()
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@triton.jit
def _pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Pack a (n_rows, n_expts_act) topk_ids tensor into a bitmatrix.

    Each row of the bitmatrix has ``bm_cols`` int32 bitpacks; bit ``j``
    of column ``i`` is set when expert ``i * 32 + j`` is selected.
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    valid = indices >= 0
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # ``valid`` guards against negative ``topk_ids`` producing spurious
        # bits: ``-1 // 32`` and ``1 << (-1 % 32)`` are implementation-defined
        # in Triton, and on some backends set bit 31.
        x = tl.where(
            valid[:, :, None] & (div[:, :, None] == offs[None, None, :]),
            (one << rem)[:, :, None],
            0,
        )
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
    """Build ``triton_kernels`` RoutingData + gather/scatter indices."""
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    _pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = (
        Bitmatrix(
            bitmatrix,
            dtype=BIT,
            shape=bitmatrix_shape,
            shape_max=bitmatrix_shape_max,
        )
        if not use_legacy_triton_kernels
        else Bitmatrix(
            bitmatrix,
            shape=bitmatrix_shape,
            shape_max=bitmatrix_shape_max,
            scratchpad=None,
        )
    )

    # ``matmul_ogs`` expects invalid topk_weights to be -1.
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)

    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing_from_bitmatrix

        return routing_from_bitmatrix(
            bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
        )

    from triton_kernels.tensor import SparseMatrix, make_ragged_tensor_metadata

    sparse_logits = SparseMatrix(indx=topk_ids, vals=topk_weights, mask=bitmatrix)
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_indx.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        num_local_experts,
        num_topk,
        ragged_batch_metadata,
    )
    gather_indx = GatherIndx(combine_indx, dispatch_indx)
    scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_indx, scatter_indx


@triton.jit
def _masked_topk_sum_kernel(
    inp_ptr,  # (M, topk, K) contiguous
    topk_ids_ptr,  # (M, topk) int: -1 marks an invalid / non-local slot
    out_ptr,  # (M, K), same dtype as inp
    K,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < K
    base = pid_m * topk
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for j in tl.static_range(topk):
        eid = tl.load(topk_ids_ptr + base + j)
        # Invalid slots are skipped, so the accumulator is NaN-safe.
        if eid >= 0:
            x = tl.load(inp_ptr + (base + j) * K + k, mask=k_mask)
            acc += x.to(tl.float32)
    tl.store(out_ptr + pid_m * K + k, acc.to(out_ptr.dtype.element_ty), mask=k_mask)


def masked_moe_sum(
    intermediate: torch.Tensor,  # (M, topk, K)
    topk_ids: torch.Tensor,  # (M, topk) int, -1 = invalid / non-local slot
    output: torch.Tensor,  # (M, K)
) -> None:
    """Reduce expert outputs over topk while skipping -1 slots."""
    M, topk, K = intermediate.shape
    BLOCK_K = 1024
    grid = (M, triton.cdiv(K, BLOCK_K))
    _masked_topk_sum_kernel[grid](
        intermediate, topk_ids, output, K, topk=topk, BLOCK_K=BLOCK_K
    )


@triton.jit
def _remap_topk_to_local_kernel(
    topk_ids_ptr,  # [n] global expert IDs (-1 = invalid)
    expert_map_ptr,  # [num_experts] global->local (-1 for non-local)
    out_ptr,  # [n] int64 local expert IDs (-1 for invalid/non-local)
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tid = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = tid >= 0
    idx = tl.where(valid, tid, 0)
    local = tl.load(expert_map_ptr + idx, mask=mask, other=-1)
    out = tl.where(valid, local.to(tl.int64), -1)
    tl.store(out_ptr + offs, out, mask=mask)


def remap_topk_to_local(
    topk_ids: torch.Tensor, expert_map: torch.Tensor
) -> torch.Tensor:
    """Fused global->local expert-id mapping over a topk_ids tensor.

    Returns a NEW int64 tensor (callers may keep the original ``topk_ids``
    as ``global_topk_ids``).
    """
    out = torch.empty_like(topk_ids, dtype=torch.int64)
    n = topk_ids.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _remap_topk_to_local_kernel[grid](topk_ids, expert_map, out, n, BLOCK=BLOCK)
    return out


class OAITritonMxfp4Experts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Modular MXFP4 expert built on ``triton_kernels.matmul_ogs``.

    Keeps the activation and topk-sum outside ``matmul_ogs`` so LoRA deltas
    can be added between the w13 GEMM and the activation, and between the
    w2 GEMM and the reduce step.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return has_triton_kernels()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """Problem-size extraction tailored to ``matmul_ogs``.

        ``w1`` stays in ``(E, N, K)`` layout (no transpose) and ``a1`` is
        always 2D ``(M, K)``.
        """
        assert len(w1.shape) == 3 and len(w2.shape) == 3
        E, _, N = w1.shape
        K = a1.size(-1)

        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Weights and the topk reduce are handled inside apply().
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, activation_out_dim)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        quant_config = self.quant_config or FUSED_MOE_UNQUANTIZED_CONFIG
        if (
            activation == MoEActivation.SILU
            and quant_config.gemm1_clamp_limit is not None
        ):
            swiglu_limit_func(output, input, quant_config.gemm1_clamp_limit)
            return
        super().activation(
            activation,
            output,
            input,
            clamp_limit=quant_config.gemm1_clamp_limit,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        quant_config = self.quant_config
        if quant_config is None:
            quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

        global_topk_ids = topk_ids
        if expert_map is not None:
            # Preserve -1 (invalid / non-local slots, e.g. from EP dispatch):
            # ``make_routing_data`` treats -1 as the skip sentinel.
            topk_ids = remap_topk_to_local(topk_ids, expert_map)

        local_num_experts = w1.shape[0]
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)

        assert hidden_states.dtype == torch.bfloat16
        assert (
            quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
        )
        assert (
            quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
        )

        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

        batch_dim = 1
        M, K = hidden_states.shape
        E, _, N = w1.shape

        if global_num_experts == -1:
            global_num_experts = E

        intermediate_cache1 = _resize_cache(workspace2, (batch_dim, M * topk, N))
        intermediate_cache3 = _resize_cache(workspace2, (batch_dim, M * topk, K))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(workspace13, (M * topk, activation_out_dim))

        gammas = routing_data.gate_scal if routing_data else None

        matmul_ogs(
            hidden_states,
            w1,
            quant_config.w1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=None,
            y=intermediate_cache1,
        )

        # Gather into expert-sorted order so apply_w13_lora can add the LoRA
        # delta in-place before activation.
        act_input = intermediate_cache1.view(-1, N)[gather_indx.dst_indx]

        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context
        if lora_context is not None:
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = self.apply_w13_lora(
                lora_context,
                y=act_input,
                x=hidden_states,
                topk_ids=global_topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1,
                w2=w2,
                num_tokens=M,
                top_k_num=topk,
            )

        self.activation(
            activation,
            intermediate_cache2,
            act_input,
        )

        # Unfuse matmul_ogs's grouped topk sum so masked_moe_sum below handles
        # the reduction (skipping invalid slots).
        routing_data.n_expts_act = 1

        matmul_ogs(
            intermediate_cache2[gather_indx.src_indx],
            w2,
            quant_config.w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
            y=intermediate_cache3,
        )

        if lora_context is not None:
            self.apply_w2_lora(
                lora_context,
                y=intermediate_cache3.view(-1, topk, K),
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=M,
                w1=w1,
                w2=w2,
                top_k_num=topk,
            )

        # matmul_ogs leaves -1 slots unwritten; masked sum skips them.
        masked_moe_sum(intermediate_cache3.view(-1, topk, K), topk_ids, output)
