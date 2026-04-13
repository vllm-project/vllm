# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant NVFP4 fused MoE expert implementations."""

from dataclasses import dataclass

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import _compute_pid
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

logger = init_logger(__name__)


def _map_extra_rows(m_total: int, cols: int) -> int:
    """Extra workspace13 rows needed to embed ``a_map`` and ``c_map``.

    Both maps are ``[M_total]`` int32 tensors.  We pack them into the
    trailing rows of the bfloat16 workspace13 buffer so they are managed
    by the workspace-manager and sized correctly for CUDA-graph capture.
    """
    bytes_needed = 2 * m_total * 4  # 2 maps × M_total × sizeof(int32)
    row_bytes = cols * 2  # cols × sizeof(bfloat16)
    return (bytes_needed + row_bytes - 1) // row_bytes


@dataclass
class NvFP4MoEWorkspace:
    """Pre-allocated scratch buffers for batch-invariant NVFP4 MoE.

    Expert-count-dependent tensors are allocated once in
    ``BatchInvariantNvfp4Experts.__init__`` and reused across forward calls
    to avoid hot-path allocations and stay compatible with CUDA graph
    capture.

    The M-dependent ``a_map`` / ``c_map`` buffers are carved out of the
    extra rows appended to ``workspace13`` by ``workspace_shapes`` (see
    ``_map_extra_rows``), so they inherit the workspace-manager's
    pre-allocation and CUDA-graph-safe lifetime.
    """

    expert_offsets: torch.Tensor  # [E+1] int32
    blockscale_offsets: torch.Tensor  # [E+1] int32
    problem_sizes1: torch.Tensor  # [E, 3] int32
    problem_sizes2: torch.Tensor  # [E, 3] int32
    tiles_per_expert: torch.Tensor  # [E] int32
    expert_tile_start: torch.Tensor  # [E+1] int32
    dummy_bias: torch.Tensor  # [0] float32


@triton.jit
def _unswizzle_scale(
    scale_raw,
    TILE_ROWS: tl.constexpr,
    TILE_SCALE_COLS: tl.constexpr,
):
    """Un-swizzle NVFP4 block scales from hardware-interleaved 128x4 layout
    to standard 2D row-major layout expected by tl.dot_scaled.

    The swizzled layout stores a (M, K_s) scale tensor as:
        (M//128, K_s//4, 32, 4, 4)
    The standard layout is:
        (M//128, 4, 32, K_s//4, 4)
    The inverse permutation (0, 3, 2, 1, 4) recovers the standard layout.
    """
    return (
        scale_raw.reshape(TILE_ROWS // 128, TILE_SCALE_COLS // 4, 32, 4, 4)
        .trans(0, 3, 2, 1, 4)
        .reshape(TILE_ROWS, TILE_SCALE_COLS)
    )


def _validate_fused_moe_batch_invariant_nvfp4_inputs(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    a1_gscale: torch.Tensor | None,
    g1_alphas: torch.Tensor | None,
    a2_gscale: torch.Tensor | None,
    g2_alphas: torch.Tensor | None,
    output: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_map: torch.Tensor | None,
    num_experts: int,
) -> None:
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
        f"hidden_states must be float16 or bfloat16, got {hidden_states.dtype}"
    )
    assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
    assert topk_ids.dtype == torch.int32, (
        f"topk_ids must be int32, got {topk_ids.dtype}"
    )
    assert topk_weights.is_contiguous(), "topk_weights must be contiguous"
    assert topk_weights.dtype == torch.float32, (
        f"topk_weights must be float32, got {topk_weights.dtype}"
    )
    assert output.is_contiguous(), "output must be contiguous"
    assert workspace13.is_contiguous(), "workspace13 must be contiguous"
    assert workspace2.is_contiguous(), "workspace2 must be contiguous"
    assert w13_weight.is_contiguous(), "w13_weight must be contiguous"
    assert w13_weight_scale.is_contiguous(), "w13_weight_scale must be contiguous"
    assert w2_weight.is_contiguous(), "w2_weight must be contiguous"
    assert w2_weight_scale.is_contiguous(), "w2_weight_scale must be contiguous"
    assert expert_map is None, (
        "Batch-invariant NVFP4 MoE does not support expert_map. "
        "This kernel requires local expert IDs and ep_size == 1."
    )

    for name, tensor in (
        ("a1_gscale", a1_gscale),
        ("a2_gscale", a2_gscale),
        ("g1_alphas", g1_alphas),
        ("g2_alphas", g2_alphas),
    ):
        if tensor is None:
            raise RuntimeError(f"Missing required NVFP4 MoE tensor: {name}")
        assert tensor.ndim == 1, (
            f"NVFP4 MoE tensor '{name}' must be 1-D, got shape {tuple(tensor.shape)}."
        )
        assert tensor.dtype == torch.float32, (
            f"NVFP4 MoE tensor '{name}' must be float32, got {tensor.dtype}."
        )
        assert tensor.is_contiguous(), f"NVFP4 MoE tensor '{name}' must be contiguous."
        assert tensor.numel() == num_experts, (
            f"NVFP4 MoE tensor '{name}' must have {num_experts} elements, "
            f"got {tensor.numel()}."
        )


# ---------------------------------------------------------------------------
# Packed grouped NVFP4 GEMM kernel and wrapper
# ---------------------------------------------------------------------------


@triton.jit
def _find_expert_bin_search(
    expert_tile_start_ptr,
    global_tile_id,
    num_experts,
):
    """Binary search to find which expert owns ``global_tile_id``.

    ``expert_tile_start_ptr`` points to an ``[E+1]`` prefix-sum array where
    ``expert_tile_start[e]`` is the first global tile belonging to expert
    ``e``.  Returns the expert index ``e`` such that
    ``expert_tile_start[e] <= global_tile_id < expert_tile_start[e+1]``.
    """
    lo: tl.int32 = 0
    hi: tl.int32 = num_experts
    while lo < hi:
        mid = (lo + hi) // 2
        val = tl.load(expert_tile_start_ptr + mid + 1).to(tl.int32)
        if val <= global_tile_id:
            lo = mid + 1
        else:
            hi = mid
    return lo


@triton.jit
def _maybe_widen(val, LARGE: tl.constexpr):
    """Promote *val* to int64 when LARGE (tensor exceeds int32 indexing)."""
    if LARGE:
        return val.to(tl.int64)
    return val


@triton.jit
def _grouped_matmul_fp4_packed_persistent_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    alpha_ptr,
    bias_ptr,
    expert_offsets_ptr,
    a_scale_offsets_ptr,
    problem_sizes_ptr,
    expert_tile_start_ptr,
    num_experts,
    M_total,
    N_total,
    K_total,
    a_scale_rows_total,
    stride_ps0,
    stride_ps1,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bm,
    stride_bk,
    stride_bsk,
    stride_cm,
    stride_cn,
    stride_bias_e,
    a_scale_cols_total,
    b_scale_cols_total,
    b_scale_n_per_expert,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    B_SCALE_GROUP: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Persistent packed grouped NVFP4 GEMM with FP4 activations and weights.

    Both A and B are packed FP4 with per-block float8_e4m3fn scales.
    Launches ``NUM_SMS`` programs that loop over a global tile index space
    built from a prefix-sum of per-expert tile counts.  Each iteration maps
    a global tile id to ``(expert_id, local_tile_id)`` via binary search,
    then executes the GEMM tile.

    B and B-scale tensors are pre-flattened by the wrapper:
    ``b_fp4`` from ``[E, N, K_packed]`` to ``[E*N, K_packed]``, and
    ``b_scale`` from ``[E, N_pad, K_s]`` to ``[E*N_pad, K_s]``.
    ``b_scale_n_per_expert`` = N_pad (may differ from N_total).

    When ``HAS_BIAS=True``, a per-expert bias vector ``[E, N]`` is added
    to the float32 accumulator after the optional alpha multiply and before
    the output cast/store.
    """
    start_pid = tl.program_id(axis=0)
    total_tiles = tl.load(expert_tile_start_ptr + num_experts).to(tl.int32)

    K_BYTES: tl.constexpr = BLOCK_SIZE_K // 2
    SCALE_K_TILE: tl.constexpr = BLOCK_SIZE_K // B_SCALE_GROUP
    SCALE_K_TILES: tl.constexpr = SCALE_K_TILE // 4
    SCALE_M_TILES: tl.constexpr = BLOCK_SIZE_M // 128
    SCALE_N_TILES: tl.constexpr = BLOCK_SIZE_N // 128

    k_bytes_total = K_total // 2

    # N and K are uniform across experts; hoist derived values before the loop.
    k_tiles = tl.cdiv(K_total, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N_total, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Global descriptors created once before the loop.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M_total, k_bytes_total],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, K_BYTES],
        padding_option="zero",
    )
    # B flattened to [E*N, K_packed] by the wrapper.
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[num_experts * N_total, k_bytes_total],
        strides=[stride_bm, stride_bk],
        block_shape=[BLOCK_SIZE_N, K_BYTES],
        padding_option="zero",
    )
    a_scale_desc = tl.make_tensor_descriptor(
        a_scale_ptr,
        shape=[
            1,
            tl.cdiv(a_scale_rows_total, 128),
            a_scale_cols_total // 4,
            2,
            256,
        ],
        strides=[
            tl.cdiv(a_scale_rows_total, 128)
            * (a_scale_cols_total // 4)
            * 512
            * stride_ask,
            (a_scale_cols_total // 4) * 512 * stride_ask,
            512 * stride_ask,
            256 * stride_ask,
            stride_ask,
        ],
        block_shape=[1, SCALE_M_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )
    # B-scale flattened to [E*N_pad, K_s] by the wrapper.
    b_scale_n_tiles = tl.cdiv(num_experts * b_scale_n_per_expert, 128)
    b_scale_desc = tl.make_tensor_descriptor(
        b_scale_ptr,
        shape=[1, b_scale_n_tiles, b_scale_cols_total // 4, 2, 256],
        strides=[
            b_scale_n_tiles * (b_scale_cols_total // 4) * 512 * stride_bsk,
            (b_scale_cols_total // 4) * 512 * stride_bsk,
            512 * stride_bsk,
            256 * stride_bsk,
            stride_bsk,
        ],
        block_shape=[1, SCALE_N_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )

    for global_tid in tl.range(start_pid, total_tiles, NUM_SMS, flatten=True):
        expert_id = _find_expert_bin_search(
            expert_tile_start_ptr, global_tid, num_experts
        )
        local_tile_start = tl.load(expert_tile_start_ptr + expert_id).to(tl.int32)
        local_tile_id = global_tid - local_tile_start

        M = tl.load(problem_sizes_ptr + expert_id * stride_ps0).to(tl.int32)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

        pid_m, pid_n = _compute_pid(
            local_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, 0
        )

        expert_row_offset = tl.load(expert_offsets_ptr + expert_id).to(tl.int32)
        expert_scale_offset = tl.load(a_scale_offsets_ptr + expert_id).to(tl.int32)
        expert_row_offset = _maybe_widen(expert_row_offset, A_LARGE)
        expert_scale_offset = _maybe_widen(expert_scale_offset, A_LARGE)

        if HAS_ALPHA:
            alpha = tl.load(alpha_ptr + expert_id).to(tl.float32)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N_total

        # B offset into the flattened [E*N, K_packed] view.
        b_row_offset = expert_id * N_total
        # B-scale offset into the flattened [E*N_pad, K_s] view.
        bs_row_offset = expert_id * b_scale_n_per_expert
        b_row_offset = _maybe_widen(b_row_offset, B_LARGE)
        bs_row_offset = _maybe_widen(bs_row_offset, B_LARGE)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            k_start_bytes = _maybe_widen(ki * K_BYTES, A_LARGE or B_LARGE)

            k_byte_mask = (ki * K_BYTES + tl.arange(0, K_BYTES)) < k_bytes_total

            a_m_start = _maybe_widen(expert_row_offset + start_m, A_LARGE)
            b_n_start = _maybe_widen(b_row_offset + start_n, B_LARGE)

            a = a_desc.load([a_m_start, k_start_bytes])
            a = tl.where(m_mask[:, None] & k_byte_mask[None, :], a, 0)
            b_raw = b_desc.load([b_n_start, k_start_bytes])
            b = tl.where(n_mask[:, None] & k_byte_mask[None, :], b_raw, 0).T

            scale_tile_k = _maybe_widen(ki * SCALE_K_TILES, A_LARGE or B_LARGE)
            scale_tile_n = _maybe_widen((bs_row_offset + start_n) // 128, B_LARGE)

            b_scale_raw = b_scale_desc.load([0, scale_tile_n, scale_tile_k, 0, 0])
            b_scale = _unswizzle_scale(
                b_scale_raw.reshape(BLOCK_SIZE_N, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_N,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )

            scale_tile_m = _maybe_widen((expert_scale_offset + start_m) // 128, A_LARGE)
            a_scale_raw = a_scale_desc.load([0, scale_tile_m, scale_tile_k, 0, 0])
            a_scale = _unswizzle_scale(
                a_scale_raw.reshape(BLOCK_SIZE_M, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_M,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )

            accumulator = tl.dot_scaled(
                a,
                a_scale,
                "e2m1",
                b,
                b_scale,
                "e2m1",
                accumulator,
            )

        if HAS_ALPHA:
            accumulator *= alpha
        if HAS_BIAS:
            bias_ptrs = bias_ptr + expert_id * stride_bias_e + offs_n
            bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            accumulator += bias_vals[None, :]
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_expert_ptr = c_ptr + expert_row_offset * stride_cm
        offs_m_c = _maybe_widen(offs_m, A_LARGE)
        offs_n_c = _maybe_widen(offs_n, B_LARGE)
        c_ptrs = (
            c_expert_ptr + offs_m_c[:, None] * stride_cm + offs_n_c[None, :] * stride_cn
        )
        tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def _grouped_matmul_nvfp4_packed(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    *,
    output: torch.Tensor,
    tiles_per_expert: torch.Tensor,
    expert_tile_start: torch.Tensor,
    dummy_bias: torch.Tensor,
    a_scale_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed grouped NVFP4 GEMM with expert-local problem metadata.

    TMA descriptors use ``padding_option="zero"`` so boundary tiles that
    overshoot ``M_total`` are zero-filled by hardware -- no guard rows or
    ``F.pad`` copies are needed.

    Args:
        a_fp4: [M_total, K_packed] uint8 packed FP4 activations in
            expert-packed row order.
        b_fp4: [E, N_max, K_packed] uint8 packed FP4 expert weights.
        a_scale: [S_total, K_s] float8_e4m3fn swizzled activation block
            scales.
        b_scale: [E, N_pad, K_s] float8_e4m3fn swizzled weight block scales.
        alpha: [E] float32 per-expert alpha.
        expert_offsets: [E+1] int32 prefix-sum row offsets in ``a_fp4``/output.
        problem_sizes: [E, 3] int tensor containing per-expert (M, N, K).
        a_scale_offsets: Optional [E+1] int32 prefix-sum row offsets in
            ``a_scale``. If not provided, ``expert_offsets`` are reused.
        output: Pre-allocated [>= M_total, >= N_max] tensor for the GEMM output.

    Returns:
        [M_total, N_max] tensor with grouped expert GEMM outputs.
    """
    assert a_fp4.dtype == torch.uint8 and b_fp4.dtype == torch.uint8
    assert a_scale.dtype == torch.float8_e4m3fn and b_scale.dtype == torch.float8_e4m3fn
    assert alpha.dtype == torch.float32

    E = b_fp4.shape[0]
    if E == 0:
        M_logical = a_fp4.shape[0]
        N_out = b_fp4.shape[1]
        return output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if a_scale_offsets is None:
        a_scale_offsets = expert_offsets

    expert_offsets = expert_offsets[:-1]
    a_scale_offsets = a_scale_offsets[:-1]

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 256

    M_logical = a_fp4.shape[0]

    # Upper-bound tile count from static tensor shapes.  The persistent
    # kernel grid is min(NUM_SMS, worst_case_tiles), both derived from
    # fixed shapes / device properties, so the grid is constant across
    # calls and safe for CUDA graph capture.
    max_tiles_m = triton.cdiv(M_logical, BLOCK_SIZE_M)
    max_tiles_n = triton.cdiv(b_fp4.shape[1], BLOCK_SIZE_N)
    max_tiles_per_expert = max_tiles_m * max_tiles_n

    N_out = b_fp4.shape[1]
    c_work = output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if max_tiles_per_expert == 0:
        return c_work

    NUM_SMS = num_compute_units(a_fp4.device.index)
    worst_case_tiles = E * max_tiles_per_expert
    A_LARGE = max(a_fp4.numel(), a_scale.numel(), c_work.numel()) > 2**31
    B_LARGE = max(b_fp4.numel(), b_scale.numel()) > 2**31

    # Pre-compute tile-to-expert mapping (all device-side, no host sync).
    # N and K are uniform across experts; only M varies.
    # Reuse pre-allocated tiles_per_expert and expert_tile_start buffers.
    M_per_expert = problem_sizes[:, 0].to(torch.int32)
    torch.div(
        M_per_expert + BLOCK_SIZE_M - 1,
        BLOCK_SIZE_M,
        rounding_mode="floor",
        out=tiles_per_expert,
    )
    tiles_per_expert.mul_(max_tiles_n)
    tiles_per_expert.mul_((M_per_expert > 0).to(torch.int32))
    torch.cumsum(tiles_per_expert, dim=0, out=expert_tile_start[1:])

    # Flatten B [E, N, K] -> [E*N, K] and B-scale [E, Np, Ks] -> [E*Np, Ks]
    # for a single global TMA descriptor.
    b_fp4_flat = b_fp4.reshape(-1, b_fp4.shape[2])
    b_scale_flat = b_scale.reshape(-1, b_scale.shape[2])

    grid = (min(NUM_SMS, worst_case_tiles),)
    _grouped_matmul_fp4_packed_persistent_kernel[grid](
        a_fp4,
        a_scale,
        b_fp4_flat,
        b_scale_flat,
        c_work,
        alpha,
        dummy_bias,
        expert_offsets,
        a_scale_offsets,
        problem_sizes,
        expert_tile_start,
        E,
        a_fp4.shape[0],
        b_fp4.shape[1],
        a_fp4.shape[1] * 2,
        a_scale.shape[0],
        problem_sizes.stride(0),
        problem_sizes.stride(1),
        a_fp4.stride(0),
        a_fp4.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_fp4_flat.stride(0),
        b_fp4_flat.stride(1),
        b_scale_flat.stride(1),
        c_work.stride(0),
        c_work.stride(1),
        0,  # stride_bias_e (unused)
        a_scale.shape[1],
        b_scale.shape[2],
        b_scale.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
        A_LARGE=A_LARGE,
        B_LARGE=B_LARGE,
        B_SCALE_GROUP=16,
        HAS_ALPHA=True,
        HAS_BIAS=False,
        num_stages=2,
        num_warps=8,
    )
    return c_work


# ---------------------------------------------------------------------------
# Standalone deterministic NVFP4 MoE implementation
# ---------------------------------------------------------------------------


def fused_moe_batch_invariant_nvfp4(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    a1_gscale: torch.Tensor | None,
    g1_alphas: torch.Tensor | None,
    a2_gscale: torch.Tensor | None,
    g2_alphas: torch.Tensor | None,
    activation: MoEActivation,
    *,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    output: torch.Tensor,
    workspace: NvFP4MoEWorkspace,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """
    Deterministic NVFP4 MoE using packed routing metadata and grouped GEMMs.

    The input ``[M, K]`` is expanded and permuted into expert-major packed
    order ``[M * topk, K]`` via ``shuffle_rows`` with CUTLASS metadata
    (``a_map``/``c_map``).  Both GEMMs then run in packed 2D form using
    per-expert offsets/problem sizes.  The epilogue uses ``moe_unpermute``
    to gather, apply router weights and reduce back to ``[M, K]`` while
    safely skipping invalid routes.  The final reduction writes into
    ``output``.

    ``workspace13`` and ``workspace2`` must match
    ``BatchInvariantNvfp4Experts.workspace_shapes`` (large enough for GEMM1 /
    GEMM2 staging and activations).
    """

    assert hidden_states.ndim == 2, (
        f"Expected 2D hidden_states for NVFP4 MoE fallback, got {hidden_states.shape}."
    )
    assert topk_ids.shape == topk_weights.shape, (
        "NVFP4 MoE fallback expects topk_ids and topk_weights to have identical shapes."
    )
    assert topk_ids.ndim == 2, (
        f"Expected 2D top-k routing tensors, got shape {topk_ids.shape}."
    )
    assert not apply_router_weight_on_input or topk_ids.shape[1] == 1, (
        "apply_router_weight_on_input=True is only supported for top_k == 1."
    )
    assert quant_backend == "cutlass", (
        "Packed batch-invariant NVFP4 MoE requires quant_backend='cutlass'."
    )
    activation_kind = activation

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]
    _validate_fused_moe_batch_invariant_nvfp4_inputs(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        output=output,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_map=expert_map,
        num_experts=num_experts,
    )
    M_total = num_tokens * top_k
    if M_total == 0:
        return output

    routed_topk_ids = topk_ids
    # Out-of-range IDs are treated as invalid routes.
    valid_routes = (routed_topk_ids >= 0) & (routed_topk_ids < num_experts)
    routed_topk_ids = torch.where(valid_routes, routed_topk_ids, -1)
    routed_topk_weights = topk_weights

    if apply_router_weight_on_input:
        packed_hidden_states = hidden_states * routed_topk_weights.view(-1, 1).to(
            hidden_states.dtype
        )
    else:
        packed_hidden_states = hidden_states

    w1_output_size = w13_weight.shape[1]
    activation_out_dim = (
        w1_output_size // 2 if activation_kind.is_gated else w1_output_size
    )
    w2_output_size = hidden_dim
    w1_padding_cols = max(0, w13_weight.shape[-1] - hidden_dim // 2)
    w2_padding_cols = max(0, w2_weight.shape[-1] - activation_out_dim // 2)
    min_w13_cols = max(w13_weight.shape[1], hidden_dim)
    if workspace13.shape[0] < M_total or workspace13.shape[1] < min_w13_cols:
        raise RuntimeError(
            "workspace13 is too small for GEMM1 staging. "
            f"Need at least ({M_total}, {min_w13_cols}), "
            f"got {tuple(workspace13.shape)}."
        )
    required_workspace2_cols = max(activation_out_dim, w2_output_size)
    if workspace2.shape[0] < M_total or workspace2.shape[1] < required_workspace2_cols:
        raise RuntimeError(
            "workspace2 is too small for activation/GEMM2 staging. "
            f"Need at least ({M_total}, {required_workspace2_cols}), "
            f"got {tuple(workspace2.shape)}."
        )
    # Reuse pre-allocated workspace buffers for routing/GEMM dispatch.
    expert_offsets = workspace.expert_offsets
    blockscale_offsets = workspace.blockscale_offsets
    problem_sizes1 = workspace.problem_sizes1
    problem_sizes2 = workspace.problem_sizes2
    # a_map and c_map are carved from the extra rows that workspace_shapes
    # appended to workspace13.  This keeps them inside the workspace-manager's
    # pre-allocation so they are CUDA-graph safe.
    map_i32 = workspace13[M_total:].flatten().view(torch.int32)[: 2 * M_total]
    a_map = map_i32[:M_total]
    c_map = map_i32[M_total:]
    a_map.zero_()
    ops.get_cutlass_moe_mm_data(
        routed_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        activation_out_dim,
        hidden_dim,
        blockscale_offsets,
    )

    # get_cutlass_moe_mm_data() assumes gated MLP (`w13` width == 2 * n).
    # For *_no_mul activations, overwrite the logical problem shapes.
    if not activation_kind.is_gated:
        problem_sizes1[:, 1].fill_(w1_output_size)
        problem_sizes2[:, 2].fill_(activation_out_dim)

    packed_hidden_states = ops.shuffle_rows(packed_hidden_states, a_map)

    a1_fp4, a1_scale = ops.scaled_fp4_experts_quant(
        packed_hidden_states,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        top_k,
    )
    a1_fp4 = pad_nvfp4_activation_for_cutlass(a1_fp4, w1_padding_cols)
    gemm1_out = slice_nvfp4_output(
        _grouped_matmul_nvfp4_packed(
            a_fp4=a1_fp4,
            b_fp4=w13_weight,
            a_scale=a1_scale,
            b_scale=w13_weight_scale,
            alpha=g1_alphas,
            expert_offsets=expert_offsets,
            a_scale_offsets=blockscale_offsets,
            problem_sizes=problem_sizes1,
            output=workspace13,
            tiles_per_expert=workspace.tiles_per_expert,
            expert_tile_start=workspace.expert_tile_start,
            dummy_bias=workspace.dummy_bias,
        ),
        w1_output_size,
    )

    if activation_kind == MoEActivation.SILU:
        int_fp4, int_scale = ops.silu_and_mul_scaled_fp4_experts_quant(
            gemm1_out,
            a2_gscale,
            expert_offsets,
            blockscale_offsets,
            top_k,
        )
    else:
        act_out = _resize_cache(workspace2, (M_total, activation_out_dim))
        apply_moe_activation(
            activation=activation_kind,
            output=act_out,
            input=gemm1_out,
        )
        int_fp4, int_scale = ops.scaled_fp4_experts_quant(
            act_out,
            a2_gscale,
            expert_offsets,
            blockscale_offsets,
            top_k,
        )
    int_fp4 = pad_nvfp4_activation_for_cutlass(int_fp4, w2_padding_cols)
    gemm2_out = slice_nvfp4_output(
        _grouped_matmul_nvfp4_packed(
            a_fp4=int_fp4,
            b_fp4=w2_weight,
            a_scale=int_scale,
            b_scale=w2_weight_scale,
            alpha=g2_alphas,
            expert_offsets=expert_offsets,
            a_scale_offsets=blockscale_offsets,
            problem_sizes=problem_sizes2,
            output=workspace2,
            tiles_per_expert=workspace.tiles_per_expert,
            expert_tile_start=workspace.expert_tile_start,
            dummy_bias=workspace.dummy_bias,
        ),
        w2_output_size,
    )

    epilogue_topk_weights = (
        torch.ones_like(routed_topk_weights)
        if apply_router_weight_on_input
        else routed_topk_weights
    )
    inv_permuted_idx = c_map.view(num_tokens, top_k)
    # moe_unpermute reads from gemm2_out and writes to reduced. Source/destination
    # must never alias (particularly in non-chunked mode where workspaces are reused).
    reduced = output
    moe_unpermute(
        out=reduced,
        permuted_hidden_states=gemm2_out,
        topk_weights=epilogue_topk_weights,
        inv_permuted_idx=inv_permuted_idx,
        expert_first_token_offset=expert_offsets.to(torch.int64),
    )
    return reduced


# ---------------------------------------------------------------------------
# Modular-kernel wrapper class
# ---------------------------------------------------------------------------


class BatchInvariantNvfp4Experts(mk.FusedMoEExpertsModular):
    """Batch-invariant NVFP4 (W4A4) MoE experts."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._num_local_experts = moe_config.num_local_experts
        self._cached_scale_vecs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

        E = moe_config.num_local_experts
        device = moe_config.device
        self._workspace = NvFP4MoEWorkspace(
            expert_offsets=torch.empty(E + 1, dtype=torch.int32, device=device),
            blockscale_offsets=torch.empty(E + 1, dtype=torch.int32, device=device),
            problem_sizes1=torch.empty(E, 3, dtype=torch.int32, device=device),
            problem_sizes2=torch.empty(E, 3, dtype=torch.int32, device=device),
            tiles_per_expert=torch.empty(E, dtype=torch.int32, device=device),
            expert_tile_start=torch.zeros(E + 1, dtype=torch.int32, device=device),
            dummy_bias=torch.empty(0, device=device, dtype=torch.float32),
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.has_device_capability(90)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls,
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if (
            not envs.VLLM_BATCH_INVARIANT
            and moe_config.moe_backend != "batch_invariant"
        ):
            return (
                False,
                "NvFP4 batch-invariant MoE is not available unless explicitly enabled "
                "using VLLM_BATCH_INVARIANT=1 or moe_backend='batch_invariant'",
            )
        # NVFP4 + EP: expert_map maps non-local experts to -1. MoE shuffle keeps a
        # fixed (M * topk, K) buffer so CUDA graphs see stable tensor shapes; valid
        # packed rows are only expert_offsets[-1]. Padding rows must not be quantized
        # incorrectly (see nvfp4_experts_quant.cu). Combining EP, expert_map, graphs,
        # and libtorch NVFP4 expert quant is unsupported—require ep_size == 1.
        if (weight_key, activation_key) == (
            kNvfp4Static,
            kNvfp4Dynamic,
        ) and moe_config.moe_parallel_config.ep_size > 1:
            return (
                False,
                "kernel does not support expert parallel for NVFP4 batch-invariant "
                "MoE (expert_map / -1 routes, fixed (M*topk,K) activations for CUDA "
                "graphs vs expert_offsets[-1] packed rows; libtorch "
                "scaled_fp4_experts_quant). Use ep_size==1.",
            )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
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
        act_out_dim = self.adjust_N_for_activation(N, activation)
        m_total = M * topk
        cols13 = max(N, K)
        extra = _map_extra_rows(m_total, cols13)
        workspace13 = (m_total + extra, cols13)
        workspace2 = (m_total, max(act_out_dim, K))
        output_shape = (M, K)
        return (workspace13, workspace2, output_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fuse activation scales into w_scale_2 in-place so that
        # g1/g2_alphas (which reference the same tensor) stay in sync
        # when EPLB rearranges the parameter.
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    def _get_nvfp4_scale_vecs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cached_scale_vecs is not None:
            return self._cached_scale_vecs
        self._cached_scale_vecs = (
            self.a1_gscale,
            self.a2_gscale,
            self.g1_alphas,
            self.g2_alphas,
        )
        return self._cached_scale_vecs

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
        apply_router_weight_on_input: bool | None,
    ) -> None:
        a1_gscale_vec, a2_gscale_vec, g1_alpha_vec, g2_alpha_vec = (
            self._get_nvfp4_scale_vecs()
        )
        fused_moe_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w1,
            w13_weight_scale=self.w1_scale,
            w2_weight=w2,
            w2_weight_scale=self.w2_scale,
            a1_gscale=a1_gscale_vec,
            g1_alphas=g1_alpha_vec,
            a2_gscale=a2_gscale_vec,
            g2_alphas=g2_alpha_vec,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            output=output,
            workspace=self._workspace,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            expert_map=expert_map,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for batch-invariant FP4 MoE")
