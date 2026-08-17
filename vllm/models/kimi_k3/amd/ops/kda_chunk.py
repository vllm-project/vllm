# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry point for the fused Kimi-K3 KDA chunk kernel.

The kernel in ``csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu``
replaces the chunk-state recurrence and the output GEMM of the Triton chunk
path with a single launch that keeps the per-chunk state in registers, so the
``[chunks, H, V, K]`` state tensor and the recomputed values never reach HBM.
"""

from functools import cache

import torch

from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

CHUNK_SIZE = 64
HEAD_DIM = 128
# Chunk-group split policy constants; see _chunk_groups.
_BLOCKS_P1 = 2  # (kV + kK) / kBV
_BLOCKS_P2 = 1  # kV / kBV
_MIN_LEN = 4
_DEEP_LEN = 12
_MAX_GROUPS = 32


@cache
def is_fused_kda_chunk_supported() -> bool:
    from vllm.platforms.rocm import on_gfx950

    if not hasattr(torch.ops._C, "fused_kda_chunk"):
        return False
    # TODO: Verify on other archs, currently only validated on gfx950
    return on_gfx950()


def can_use_fused_kda_chunk(
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
    chunk_size: int,
) -> bool:
    return (
        head_k_dim == HEAD_DIM
        and head_v_dim == HEAD_DIM
        and chunk_size == CHUNK_SIZE
        and dtype == torch.bfloat16
        and is_fused_kda_chunk_supported()
    )


def fused_kda_prologue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: torch.Tensor,
    conv_weight: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    conv_has_initial_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the whole chunk-path prologue in one launch.

    Replaces the two L2 norms, the gate cumsum, both intra-chunk passes and the
    w/u recompute. Returns the operands the fused chunk kernel consumes.

    The kernel can also apply the depthwise conv and its silu, in which case
    ``q``/``k``/``v`` are the three raw bands of the QKV projection and
    ``conv_state`` is updated in place. This path is currently not used.
    """
    # `beta` and `g` reach the layer as last-dim slices of the fused QKVGFAB
    # projection and carry its row stride. The kernel reads both with an
    # explicit per-token stride, so neither is copied here.
    _, t_total, num_heads, _ = q.shape
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE).to(torch.int32)
    num_chunks = chunk_indices.shape[0]
    dev = q.device

    # q may be a strided band view, so the workspaces are sized rather than
    # cloned from it.
    def _like(last: int) -> torch.Tensor:
        return torch.empty(1, t_total, num_heads, last, dtype=q.dtype, device=q.device)

    ws = dict(
        qg=_like(HEAD_DIM),
        w=_like(HEAD_DIM),
        u=_like(HEAD_DIM),
        kg_t=torch.empty(
            num_chunks, num_heads, HEAD_DIM, CHUNK_SIZE, dtype=q.dtype, device=dev
        ),
        aqk=torch.empty(1, t_total, num_heads, CHUNK_SIZE, dtype=q.dtype, device=dev),
        decay=torch.empty(
            num_chunks, num_heads, HEAD_DIM, dtype=torch.float32, device=dev
        ),
    )
    torch.ops._C.fused_kda_prologue(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log.reshape(-1),
        dt_bias.reshape(-1),
        ws["qg"],
        ws["w"],
        ws["u"],
        ws["kg_t"],
        ws["aqk"],
        ws["decay"],
        cu_seqlens.to(torch.int32),
        chunk_indices,
        conv_weight,
        conv_state,
        conv_state_indices,
        conv_has_initial_state,
        scale,
        lower_bound,
    )
    return ws


@cache
def _num_cus(device: int) -> int:
    from vllm.platforms import current_platform

    return current_platform.get_device_properties(device).multi_processor_count


@cache
def _chunk_groups(chunks_per_seq: int, num_seqs: int, num_heads: int) -> int:
    """How many parallel chunk groups to cut each sequence into.

    The scan composes the group operators with the exact transfer
    ``M_g = prod_c (diag(d_c) - w_c^T kg_c)``, so the composition error does
    not grow with group length and the choice is purely a machine fit:

    * ``fill1``/``fill2``: the largest G whose pass-one / pass-two grid still
      fits in one scheduling batch. Rows are launched in blocks of at most
      ``kBV = 128``, so pass one (``kV + kK`` rows) needs two blocks per
      ``(n, h, g)`` and pass two one. Past a pass's fill point its block count
      grows as fast as its depth shrinks and the rung stops paying.
    * ``_DEEP_LEN``: unless the groups are still long, where pass two's halving
      is worth letting pass one spill to a second batch.
    * ``_MIN_LEN``: below this many chunks per group the per-group fixed cost
      exceeds what the shorter walk saves.

    ``G == 2`` is depth-neutral -- two passes of ``nt/2`` -- and only doubles
    staging, so the split is taken only from ``G >= 4``.
    """
    cus = _num_cus(torch.accelerator.current_device_index())
    nh = max(num_seqs * num_heads, 1)
    fill1 = cus // (_BLOCKS_P1 * nh)
    fill2 = cus // (_BLOCKS_P2 * nh)
    cand = fill1
    if fill2 > 0 and chunks_per_seq // fill2 >= _DEEP_LEN:
        cand = fill2
    cand = min(cand, chunks_per_seq // _MIN_LEN, _MAX_GROUPS)
    return cand if cand >= 4 else 1


def _kda_group_workspace(
    groups: int, nh: int, device: torch.device
) -> torch.Tensor | None:
    """One fp32 buffer the kernel carves into bg / sin_ / ag / mgT.

    Kept caller-owned so the memory stays inside the caching allocator and
    vLLM's memory profiling accounts for it; the kernel's carve order must
    match the layout here.
    """
    if groups <= 1:
        return None
    planes = groups * nh
    plane = HEAD_DIM * HEAD_DIM
    floats = 2 * planes * plane + planes * HEAD_DIM  # bg, sin_, ag
    floats += (planes * plane + 1) // 2  # mgT, bf16
    return torch.empty(floats, dtype=torch.float32, device=device)


def fused_kda_chunk(
    qg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    kg_t: torch.Tensor,
    aqk: torch.Tensor,
    decay: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the chunk recurrence and the output projection in one launch.

    Args:
        qg: ``q * exp2(gk_cumsum)``, ``[1, T, H, 128]``.
        kg_t: chunk-major transposed gated keys, ``[chunks, H, 128, 64]``.
        decay: ``exp2`` of each chunk's last gate row, ``[chunks, H, 128]``.
        out: output buffer, ``[1, T, H, 128]``; may alias ``u``'s source.
    """
    num_seqs = cu_seqlens.numel() - 1
    final_state = None
    if output_final_state:
        final_state = torch.empty(
            num_seqs,
            u.shape[2],
            u.shape[3],
            qg.shape[3],
            dtype=torch.float32,
            device=u.device,
        )
    cu_seqlens = cu_seqlens.to(torch.int32)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE).to(torch.int32)

    # The chunk walk is serial in chunks. Splitting each sequence into `groups`
    # stretches that run in parallel costs a second pass and a scan, and only
    # pays once the chain is long enough to amortise both.
    chunks_per_seq = kg_t.shape[0] // max(num_seqs, 1)
    groups = _chunk_groups(chunks_per_seq, num_seqs, u.shape[2])
    group_state = _kda_group_workspace(groups, num_seqs * u.shape[2], u.device)

    torch.ops._C.fused_kda_chunk(
        qg,
        w,
        u,
        kg_t,
        aqk,
        decay,
        initial_state,
        final_state,
        out,
        cu_seqlens,
        chunk_offsets,
        scale,
        group_state,
        groups,
    )
    return out, final_state
