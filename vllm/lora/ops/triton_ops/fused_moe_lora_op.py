# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.mem_utils import get_max_shared_memory_bytes
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import supports_pdl, supports_tma


@triton.jit
def _get_lora_id(
    lora_ids,
    token_lora_mapping_ptr,
    lora_idx,
    pid_m,
    top_k_num,
    naive_block_assignment: tl.constexpr,
):
    """Returns lora_id"""
    if naive_block_assignment:
        token_idx = pid_m // top_k_num
        return tl.load(token_lora_mapping_ptr + token_idx)
    else:
        return tl.load(lora_ids + lora_idx)


@triton.jit
def _get_expert_id(
    expert_ids_ptr,
    lora_id,
    pid_m,
    stride_el,
    max_loras,
    naive_block_assignment: tl.constexpr,
):
    """Returns expert_id"""
    if naive_block_assignment:
        return tl.load(expert_ids_ptr + pid_m)
    else:
        ind = lora_id * stride_el + pid_m
        return tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)


@triton.jit
def _get_token_offs(
    sorted_token_ids_ptr,
    lora_id,
    pid_m,
    offs,
    stride_tl,
    max_loras,
    num_valid_tokens,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Returns token offsets"""
    if naive_block_assignment:
        return tl.where(offs == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        token_ind = stride_tl * lora_id + offs_token_id
        return tl.load(
            sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0
        )


@triton.jit
def _get_c_ptrs(
    cur_c_ptr,
    lora_id,
    pid_m,
    offs,
    offs_token,
    offs_cn,
    stride_cm,
    stride_cn,
    EM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    sort_c: tl.constexpr,
):
    # When sort_c is true, store the output in c_ptr using token order defined
    # in sorted_token_ids_ptr; otherwise, use the original token order from the prompt
    if sort_c:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        c_ptrs = (
            cur_c_ptr
            + lora_id * EM * stride_cm
            + stride_cm * offs_token_id[:, None]
            + stride_cn * offs_cn[None, :]
        )
    else:
        c_ptrs = (
            cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        )
    return c_ptrs


_LORA_PTR_DICT: dict[tuple[int, ...], torch.tensor] = {}


# ---------------------------------------------------------------------------
# Fully-fused MoE-LoRA kernel (one-shot): shrink + expand combined into a single
# launch with the rank-dim intermediate kept in registers. Used by the fast
# path of `_fused_moe_lora` for `fully_sharded=False`. The legacy two-kernel
# path (`_fused_moe_lora_kernel` above) is retained for `fully_sharded=True`
# because that path needs to materialise the intermediate cache for an
# all_reduce / all_gather between shrink and expand.
# ---------------------------------------------------------------------------


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _fused_moe_lora_one_shot_kernel(
    # ---- pointers ----
    x_ptr,
    A_ptrs,
    B_ptrs,
    out_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    lora_ids_ptr,
    adapter_enabled_ptr,
    # ---- dims ----
    N,
    K,
    num_valid_tokens,
    top_k_num,
    max_loras,
    # ---- strides ----
    stride_xm,
    stride_xk,
    stride_A_lora,
    stride_A_expert,
    stride_A_r,
    stride_A_k,
    stride_B_lora,
    stride_B_expert,
    stride_B_n,
    stride_B_r,
    stride_om,
    stride_on,
    stride_tl_,
    stride_el,
    # ---- scalar ----
    slice_n_offset,
    # ---- constexpr (set per call) ----
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    actual_rank: tl.constexpr,
    NPID_FACTOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    pid_full = tl.program_id(axis=0)
    pid_m = pid_full // NPID_FACTOR
    pid_n_outer = pid_full % NPID_FACTOR
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    # Resolve lora_id.
    if naive_block_assignment:
        token_idx_for_lora = pid_m // top_k_num
        lora_id = tl.load(token_lora_mapping_ptr + token_idx_for_lora)
    else:
        lora_id = tl.load(lora_ids_ptr + lora_idx)
    if lora_id < 0:
        return
    if lora_id >= max_loras:
        return
    enabled = tl.load(adapter_enabled_ptr + lora_id)
    if enabled == 0:
        return

    if not naive_block_assignment:
        ntpp = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_M >= ntpp:
            return

    # Resolve expert_id.
    if naive_block_assignment:
        expert_id = tl.load(expert_ids_ptr + pid_m)
    else:
        ind = lora_id * stride_el + pid_m
        expert_id = tl.load(
            expert_ids_ptr + ind, mask=ind < max_loras * stride_el, other=-1
        )
    if expert_id < 0:
        return

    # Compute offs_token (flat token ids).
    offs = tl.arange(0, BLOCK_M).to(tl.int64)
    if naive_block_assignment:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_M + offs
        token_ind = stride_tl_ * lora_id + offs_token_id
        offs_token = tl.load(
            sorted_token_ids_ptr + token_ind,
            mask=token_ind < max_loras * stride_tl_,
            other=num_valid_tokens,
        )
    token_mask = offs_token < num_valid_tokens

    # N range owned by this program. Splitting [0, N) into NPID_FACTOR
    # contiguous outer blocks lets us scale parallelism for small batches.
    n_per_outer = tl.cdiv(N, NPID_FACTOR)
    n_lo = pid_n_outer * n_per_outer
    n_hi = tl.minimum((pid_n_outer + 1) * n_per_outer, N)
    if n_lo >= N:
        return

    # Slice pointers.
    cur_A_ptr = tl.load(A_ptrs + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))
    cur_B_ptr = tl.load(B_ptrs + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))

    A_base = cur_A_ptr + lora_id * stride_A_lora + expert_id * stride_A_expert
    B_base = cur_B_ptr + lora_id * stride_B_lora + expert_id * stride_B_expert

    # SHRINK: tmp[BLOCK_M, BLOCK_R] = x @ A^T, accumulated in fp32 registers.
    offs_r = tl.arange(0, BLOCK_R)
    rank_mask = offs_r < actual_rank
    # Clamp rank offsets so OOB rows of A / B map to address 0; the mask
    # zeros the loaded values. Required when BLOCK_R > actual_rank
    # (e.g. rank=4 padded to 16) -- without clamping, tl.load would address
    # the next expert's memory.
    safe_offs_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BLOCK_K)

    offs_x_row = offs_token // token_mapping_factor
    x_ptrs = x_ptr + offs_x_row[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = A_base + offs_k[:, None] * stride_A_k + safe_offs_r[None, :] * stride_A_r

    tmp = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    if EVEN_K:
        for _ in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=token_mask[:, None], other=0.0)
            a = tl.load(a_ptrs, mask=rank_mask[None, :], other=0.0)
            tmp += tl.dot(x, a)
            x_ptrs += BLOCK_K * stride_xk
            a_ptrs += BLOCK_K * stride_A_k
    else:
        for kb in range(0, K, BLOCK_K):
            k_remain = K - kb
            k_mask = offs_k < k_remain
            x = tl.load(x_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=k_mask[:, None] & rank_mask[None, :], other=0.0)
            tmp += tl.dot(x, a)
            x_ptrs += BLOCK_K * stride_xk
            a_ptrs += BLOCK_K * stride_A_k

    tmp_typed = tmp.to(out_ptr.dtype.element_ty)

    # EXPAND: out[tokens, n] += tmp @ B^T, looped over BLOCK_N tiles within
    # this program's [n_lo, n_hi). The (offs_n < n_hi) mask is required
    # whenever BLOCK_N > n_per_outer to keep adjacent outer blocks from
    # writing into each other's columns.
    if MUL_ROUTED_WEIGHT:
        moe_w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0).to(
            tl.float32
        )

    out_slice_base = out_ptr + slice_id * slice_n_offset

    for n_start in range(n_lo, n_hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = (offs_n < N) & (offs_n < n_hi)

        b_ptrs = (
            B_base + safe_offs_r[:, None] * stride_B_r + offs_n[None, :] * stride_B_n
        )
        b = tl.load(b_ptrs, mask=rank_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(tmp_typed, b)  # (BLOCK_M, BLOCK_N) fp32
        if MUL_ROUTED_WEIGHT:
            acc = acc * moe_w[:, None]

        out_ptrs = (
            out_slice_base
            + offs_token[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
        if ADD_INPUTS:
            prev = tl.load(out_ptrs, mask=out_mask, other=0.0)
            tl.store(out_ptrs, prev + acc.to(out_ptr.dtype.element_ty), mask=out_mask)
        else:
            tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _run_fused_moe_lora_one_shot(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool,
    block_size_m: int,
    add_inputs: bool = True,
) -> None:
    """Fast-path wrapper: launches one fused shrink+expand kernel.

    The shape contract matches `_fused_moe_lora`. `output` has shape
    `(num_tokens, top_k_num, num_slices * N_per_slice)`. When
    `add_inputs=True` (default) the kernel reads-modifies-writes `output`
    in place; when `add_inputs=False` the kernel overwrites `output` with
    the LoRA delta only. The latter is used by the dual-stream path that
    sums LoRA into the base output on a separate stream.
    """
    num_slices = len(lora_a_stacked)
    device = qcurr_hidden_states.device

    A0 = lora_a_stacked[0]
    B0 = lora_b_stacked[0]
    max_loras_w = A0.shape[0]
    rank = A0.shape[2]
    K = A0.shape[3]
    N_per_slice = B0.shape[2]

    # rank padding is to next pow2 with a floor of 16 (tensor-core minimum
    # K-dim). Beyond 128 the (BLOCK_M, BLOCK_R) accumulator outgrows the
    # register file; rank tiling would be needed but is out of scope for
    # this kernel. Tried floor=32 to double MMA density per K-step but it
    # regressed across all M (+8 to +40%): the (64,32) fp32 accumulator +
    # widened B tile pushed register count past spill threshold, lowering
    # occupancy by more than the MMA gain saved.
    assert rank <= 128, (
        f"fused_moe_lora_one_shot supports max_lora_rank<=128; got rank={rank}"
    )
    BLOCK_R = max(triton.next_power_of_2(rank), 16)

    num_experts = A0.shape[1]
    naive = sorted_token_ids is None
    if sorted_token_ids is None:
        EM_grid = topk_weights.numel()
        BLOCK_M = 16
        stride_tl_ = 0
        stride_el = 0
        grid_lora_dim = 1
    else:
        EM_grid = sorted_token_ids.shape[1]
        # BLOCK_M must equal moe_lora_align_block_size's block_size. The
        # caller passes that explicitly; deriving it from tensor shapes is
        # unsafe because sorted_token_ids.shape[1] is the raw padded length
        # (not necessarily a multiple of block_size — e.g. OLMoE prefill
        # produces sorted=139200 with expert_ids=1088 and block_size=128).
        # tl.arange and tl.dot need block_size_m to be a power of 2 and at
        # least 16. The Python-side assertion gives a clearer error than
        # the cryptic Triton compile failure.
        assert block_size_m >= 16 and (block_size_m & (block_size_m - 1)) == 0, (
            f"shrink_block_size_m must be a power of 2 and >=16; got {block_size_m}"
        )
        BLOCK_M = block_size_m
        stride_tl_ = sorted_token_ids.stride(0)
        stride_el = expert_ids.stride(0)
        grid_lora_dim = int(num_active_loras.item())

    # Empty-work guards: the grid would otherwise have a zero dimension,
    # which Triton rejects. None of these is a hot path in production -- a
    # batch with zero tokens, an EM_grid of zero, or zero active LoRAs all
    # mean there's nothing to add to `output`.
    if EM_grid == 0 or grid_lora_dim == 0 or num_slices == 0:
        return

    token_mapping_factor = 1 if mul_routed_weight else top_k_num

    A_ptrs = _get_ptr(lora_a_stacked, device)
    B_ptrs = _get_ptr(lora_b_stacked, device)

    # Flatten (num_tokens, top_k) → flat_token axis. The kernel addresses
    # output via offs_token * stride_om, which is correct iff the dim-0 /
    # dim-1 strides collapse cleanly: stride(0) == top_k * stride(1). All
    # production callers pass contiguous output, so this always holds; the
    # explicit check guards against future regressions where a non-trivial
    # view (e.g. permute) would silently break in-place accumulation.
    assert output.dim() == 3, f"output must be 3-D, got {output.shape}"
    assert output.stride(0) == output.shape[1] * output.stride(1), (
        "fused_moe_lora_one_shot requires output.stride(0) == top_k*stride(1); "
        f"got shape={output.shape} strides={output.stride()}"
    )
    out_view = output.view(-1, output.shape[-1])
    M_blocks = triton.cdiv(EM_grid, BLOCK_M) if not naive else EM_grid

    # NPID_FACTOR heuristic: scale N-axis parallelism when base CTA count is
    # short of saturating the SM array. Cap by the cost of redundant shrink.
    sm_count = current_platform.num_compute_units(device.index)
    base_programs = max(M_blocks * num_slices * grid_lora_dim, 1)
    shrink_ratio = K / max(K + N_per_slice, 1)
    max_npid_by_budget = max(1, int(1.5 / max(shrink_ratio, 1e-3)) + 1)
    target = 2 * sm_count
    if base_programs >= int(1.5 * sm_count):
        npid = 1
    else:
        npid_occ = max(1, min(16, (target + base_programs - 1) // base_programs))
        npid = min(npid_occ, max_npid_by_budget)
    npid = max(1, min(npid, max(1, N_per_slice // 128)))

    # Robust defaults across the prefill regime (H100/H200/B200, bf16/fp16).
    # NPID > 1 is the small-M / under-saturated path -- more warps help
    # amortise the inner-N expand loop. ns=3 instead of 4: GB200 ncu showed
    # the 4-stage pipeline pushed register count to 168/thread and capped
    # achieved occupancy at ~17% (3 blocks/SM, register-bound); ns=3 frees
    # ~30 regs/thread which keeps a 4th block resident on small grids.
    # Tried BLOCK_N=64 for w13 (N=192) to avoid the half-wasted second
    # tile: regressed 11-29% because the "waste" was just masked stores
    # (cheap) and the extra iteration added load + index overhead.
    if npid > 1:
        block_n, nw, ns = 128, 8, 3
    else:
        block_n, nw, ns = 128, 4, 3

    # Devices with max shmem size less than 68KB can't support 3-stage
    # pipeline. Fall back to a 2-stage on such devices
    if current_platform.is_cuda_alike():
        max_shmem_bytes = 68 * 1024
        if get_max_shared_memory_bytes(device.index) < max_shmem_bytes:
            ns = min(ns, 2)

    # BLOCK_K choice: for hidden-sized K (≥256, i.e. the K=hidden_size
    # shrink input on w13) force BLOCK_K=128 -- the wider tile halves the
    # K-loop trip count and removes the scoreboard stalls that dominated
    # M=16-64 on GB200 (kernel time -13% to -37% vs the work_per_expert
    # heuristic which picked 64 for low-tokens-per-expert ratios). For
    # small-K shapes (e.g. w2 with K=192 where the down-proj reads the
    # MoE intermediate) keep the work_per_expert heuristic: BLOCK_K=128
    # would force the EVEN_K=False masked path and add no K-loop savings
    # (K/64=3 vs K/128=2 masked) while inflating per-program startup.
    if K >= 256:
        block_k = 128
    else:
        work_per_expert = topk_weights.numel() / max(num_experts, 1)
        block_k = 128 if work_per_expert >= 16 else 64

    grid = (M_blocks * npid, num_slices, grid_lora_dim)

    _fused_moe_lora_one_shot_kernel[grid](
        qcurr_hidden_states,
        A_ptrs,
        B_ptrs,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        lora_ids,
        adapter_enabled,
        N_per_slice,
        K,
        topk_weights.numel(),
        top_k_num,
        max_loras_w,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        A0.stride(0),
        A0.stride(1),
        A0.stride(2),
        A0.stride(3),
        B0.stride(0),
        B0.stride(1),
        B0.stride(2),
        B0.stride(3),
        out_view.stride(0),
        out_view.stride(1),
        stride_tl_,
        stride_el,
        N_per_slice,
        token_mapping_factor=token_mapping_factor,
        naive_block_assignment=naive,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
        actual_rank=rank,
        NPID_FACTOR=npid,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ADD_INPUTS=add_inputs,
        num_warps=nw,
        num_stages=ns,
    )


# ---------------------------------------------------------------------------
# Small-batch (decode-style) fused MoE-LoRA kernel — sub-path of the
# one_shot fast path.
# ---------------------------------------------------------------------------


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _fused_moe_lora_small_batch_kernel(
    # ---- pointers ----
    x_ptr,
    A_ptrs,
    B_ptrs,
    out_ptr,
    topk_weights_ptr,
    expert_ids_ptr,  # (num_tokens * top_k_num,)
    token_lora_mapping_ptr,  # (num_tokens,)
    adapter_enabled_ptr,
    # ---- dims ----
    N,
    K,
    top_k_num,
    max_loras,
    work_total,  # = pair_slices * n_chunks_per_pair_slice
    pair_slices,  # = num_tokens * top_k_num * NUM_SLICES
    # ---- strides ----
    stride_xm,
    stride_xk,
    stride_A_lora,
    stride_A_expert,
    stride_A_r,
    stride_A_k,
    stride_B_lora,
    stride_B_expert,
    stride_B_n,
    stride_B_r,
    stride_om,
    stride_on,
    # ---- scalar (runtime ints, NOT constexpr) ----
    # n_tiles_per_program / n_chunks_per_pair_slice are deliberately
    # runtime: each distinct value would otherwise trigger a fresh Triton
    # compile -> fresh kernel binary -> fresh CUDA graph instance per
    # batch size. Production traces showed that variant explosion adding
    # ~5.9k graph instantiations on top of legacy. Runtime args mean one
    # shared binary across all chunk sizes.
    slice_n_offset,
    n_tiles_per_program,
    n_chunks_per_pair_slice,
    # ---- constexpr ----
    token_mapping_factor: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    actual_rank: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Persistent fused MoE-LoRA kernel for naive_block_assignment inputs.

    Each program owns one (pair × slice × n_chunk) work item. A "chunk"
    covers `n_tiles_per_program` consecutive output-N tiles, all of which
    share a single shrink — so the rank-vector is computed once per
    program and the A weights for that (lora, expert, slice) are loaded
    once instead of n_tiles_per_program times.

    The wrapper picks `n_tiles_per_program` to keep the grid close to
    2*SM_count: at very small batch (work_total ≤ SM_count) the chunk
    size collapses to 1 and behaviour matches a per-tile GEMV; as batch
    grows the chunk grows so we trade some N-axis parallelism for shrink
    reuse. When `work_total` exceeds the launched grid, the outer stride
    loop drains the leftover work units serially.
    """
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    offs_r = tl.arange(0, BLOCK_R)
    rank_mask = offs_r < actual_rank
    # Clamp OOB rank lanes so they address row 0 of A/B; the mask zeros
    # the loaded values. Required when BLOCK_R > actual_rank (e.g. rank=4
    # padded to 16) -- without clamping, tl.load would address the next
    # expert's memory.
    safe_offs_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BLOCK_K)

    # Persistent stride loop: when grid < work_total each program walks
    # multiple work items. When grid == work_total the loop runs exactly
    # once and the kernel degenerates to the per-tile GEMV.
    for work_id in range(pid, work_total, num_programs):
        n_chunk_idx = work_id % n_chunks_per_pair_slice
        pair_slice_idx = work_id // n_chunks_per_pair_slice
        # NUM_SLICES is constexpr (typ. 1 or 2) so divmod folds.
        pair_idx = pair_slice_idx // NUM_SLICES
        slice_id = pair_slice_idx % NUM_SLICES

        # Resolve lora_id / expert_id; skip the body for inactive lanes.
        # Using a single `valid` flag instead of early `return` keeps the
        # outer stride loop alive — `return` would exit the whole program
        # and skip later work items assigned to this SM.
        token_idx = pair_idx // top_k_num
        lora_id = tl.load(token_lora_mapping_ptr + token_idx)
        valid = (lora_id >= 0) & (lora_id < max_loras)
        enabled = tl.load(adapter_enabled_ptr + tl.where(valid, lora_id, 0))
        valid = valid & (enabled != 0)
        expert_id = tl.load(expert_ids_ptr + pair_idx)
        valid = valid & (expert_id >= 0)

        if valid:
            cur_A_ptr = tl.load(A_ptrs + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )
            cur_B_ptr = tl.load(B_ptrs + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )
            A_base = cur_A_ptr + lora_id * stride_A_lora + expert_id * stride_A_expert
            B_base = cur_B_ptr + lora_id * stride_B_lora + expert_id * stride_B_expert

            x_row = pair_idx // token_mapping_factor
            x_row_ptr = x_ptr + x_row * stride_xm

            # SHRINK GEMV (once per program; reused across n_tiles_per_program
            # expand tiles below). Sum-reduction over BLOCK_K with fp32
            # accumulator — same precision path as the one_shot kernel.
            rank_vec = tl.zeros((BLOCK_R,), dtype=tl.float32)
            if EVEN_K:
                for kb in range(0, K, BLOCK_K):
                    cur_k = kb + offs_k
                    x_tile = tl.load(x_row_ptr + cur_k * stride_xk).to(tl.float32)
                    a_tile = tl.load(
                        A_base
                        + safe_offs_r[:, None] * stride_A_r
                        + cur_k[None, :] * stride_A_k,
                        mask=rank_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    rank_vec += tl.sum(a_tile * x_tile[None, :], axis=1)
            else:
                for kb in range(0, K, BLOCK_K):
                    cur_k = kb + offs_k
                    k_mask = cur_k < K
                    x_tile = tl.load(
                        x_row_ptr + cur_k * stride_xk, mask=k_mask, other=0.0
                    ).to(tl.float32)
                    a_tile = tl.load(
                        A_base
                        + safe_offs_r[:, None] * stride_A_r
                        + cur_k[None, :] * stride_A_k,
                        mask=rank_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    rank_vec += tl.sum(a_tile * x_tile[None, :], axis=1)

            # EXPAND: walk n_tiles_per_program consecutive output-N tiles
            # using the same rank_vec. The loop is a runtime range (not
            # tl.static_range) so a single compiled kernel handles every
            # chunk size — see the note on the kernel signature.
            n_tile_start = n_chunk_idx * n_tiles_per_program
            out_row_ptr = out_ptr + slice_id * slice_n_offset + pair_idx * stride_om

            if MUL_ROUTED_WEIGHT:
                moe_w = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)

            for nt in range(n_tiles_per_program):
                n_lo = (n_tile_start + nt) * BLOCK_N
                if n_lo < N:
                    offs_n = n_lo + tl.arange(0, BLOCK_N)
                    n_mask = offs_n < N
                    b_tile = tl.load(
                        B_base
                        + offs_n[:, None] * stride_B_n
                        + safe_offs_r[None, :] * stride_B_r,
                        mask=n_mask[:, None] & rank_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    out_tile = tl.sum(b_tile * rank_vec[None, :], axis=1)

                    if MUL_ROUTED_WEIGHT:
                        out_tile = out_tile * moe_w

                    out_ptrs = out_row_ptr + offs_n * stride_on
                    if ADD_INPUTS:
                        prev = tl.load(out_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                        tl.store(
                            out_ptrs,
                            (prev + out_tile).to(out_ptr.dtype.element_ty),
                            mask=n_mask,
                        )
                    else:
                        tl.store(
                            out_ptrs,
                            out_tile.to(out_ptr.dtype.element_ty),
                            mask=n_mask,
                        )


def _pick_small_batch_chunk(pair_slices: int, N_tiles: int, sm_count: int) -> int:
    """Pick `n_tiles_per_program` so the launched grid stays near
    2*SM_count.

    Sizes for occupancy first (more programs in flight → better latency
    hiding for the K-loop A/x loads). Once the per-tile grid already
    exceeds 2*SM_count we increase the chunk size to amortise the shrink
    cost — at that point the GPU is saturated by per-program work and
    packing more tiles per program lets the rank_vec be reused.
    """
    target_grid = max(1, 2 * sm_count)
    total_work = pair_slices * N_tiles
    if total_work <= target_grid:
        return 1
    ntpp = (total_work + target_grid - 1) // target_grid
    return min(ntpp, N_tiles)


def _run_fused_moe_lora_small_batch(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids_flat: torch.Tensor,  # (num_tokens * top_k_num,)
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool,
    add_inputs: bool = True,
) -> None:
    """Small-batch GEMV-style wrapper. Naive-block-assignment inputs only.

    Shape contract matches `_run_fused_moe_lora_one_shot`: `output` is
    `(num_tokens, top_k_num, num_slices * N_per_slice)` with
    contiguous-style strides, `expert_ids_flat` is the flattened
    `topk_ids` of shape `(num_tokens * top_k_num,)`, and the
    rank-padded LoRA weights live in `lora_a_stacked` /
    `lora_b_stacked`.

    The kernel is persistent over (pair × slice × n_chunk) work items —
    each program does one shrink and reuses the rank vector across
    `n_tiles_per_program` expand tiles. The chunk size scales with the
    pair-slice count so very small batches keep per-tile parallelism
    while medium batches cut redundant shrinks.
    """
    num_slices = len(lora_a_stacked)
    device = qcurr_hidden_states.device

    A0 = lora_a_stacked[0]
    B0 = lora_b_stacked[0]
    max_loras_w = A0.shape[0]
    rank = A0.shape[2]
    K = A0.shape[3]
    N_per_slice = B0.shape[2]

    # Rank padding: floor 16 (tensor-core min K), ceil to next pow2. The
    # ≤64 cap is set conservatively for the prototype: at rank 64 the
    # per-program register footprint is rank_vec(64 fp32) + b_tile(BLOCK_N
    # × 64 fp32) = e.g. 128*64*4 = 32 KiB, comfortably within the 64 KiB
    # register file even with num_warps=8. Doubling to 128 would push us
    # against the limit and require shared-memory staging.
    assert rank <= 64, f"fused_moe_lora_small_batch supports rank<=64; got rank={rank}"
    BLOCK_R = max(triton.next_power_of_2(rank), 16)

    num_tokens = topk_weights.shape[0]
    M_grid = num_tokens * top_k_num
    if M_grid == 0 or num_slices == 0:
        return

    token_mapping_factor = 1 if mul_routed_weight else top_k_num

    A_ptrs = _get_ptr(lora_a_stacked, device)
    B_ptrs = _get_ptr(lora_b_stacked, device)

    assert output.dim() == 3, f"output must be 3-D, got {output.shape}"
    assert output.stride(0) == output.shape[1] * output.stride(1), (
        "fused_moe_lora_small_batch requires output.stride(0) == "
        f"top_k*stride(1); got shape={output.shape} strides={output.stride()}"
    )
    out_view = output.view(-1, output.shape[-1])

    # Block sizes. BLOCK_N=128 matches the one_shot's expand tile and gives
    # 6-24 N tiles for typical N ∈ [768, 3072], enough to saturate the SM
    # array once M_grid * num_slices reaches ~SM_count. BLOCK_K=128 halves
    # the K-loop trip count vs 64 and pays for itself once K ≥ 1024 (the
    # only regime we care about — hidden sizes are always large here).
    BLOCK_N = 128
    BLOCK_K = 128
    nw = 4
    ns = 3

    N_tiles = triton.cdiv(N_per_slice, BLOCK_N)
    pair_slices = M_grid * num_slices

    sm_count = current_platform.num_compute_units(device.index)
    n_tiles_per_program = _pick_small_batch_chunk(pair_slices, N_tiles, sm_count)
    n_chunks = triton.cdiv(N_tiles, n_tiles_per_program)
    work_total = pair_slices * n_chunks

    # Grid sizing: keep parallelism uncapped when work_total is small (so
    # very small batches still spread across SMs); cap at 2*SM_count once
    # we have plenty of work, letting the in-kernel stride loop drain the
    # remainder.
    grid_size = min(work_total, max(1, 2 * sm_count))
    grid = (grid_size,)

    _fused_moe_lora_small_batch_kernel[grid](
        qcurr_hidden_states,
        A_ptrs,
        B_ptrs,
        out_view,
        topk_weights,
        expert_ids_flat,
        token_lora_mapping,
        adapter_enabled,
        N_per_slice,
        K,
        top_k_num,
        max_loras_w,
        work_total,
        pair_slices,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        A0.stride(0),
        A0.stride(1),
        A0.stride(2),
        A0.stride(3),
        B0.stride(0),
        B0.stride(1),
        B0.stride(2),
        B0.stride(3),
        out_view.stride(0),
        out_view.stride(1),
        N_per_slice,
        n_tiles_per_program,
        n_chunks,
        token_mapping_factor=token_mapping_factor,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=add_inputs,
        BLOCK_R=BLOCK_R,
        actual_rank=rank,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SLICES=num_slices,
        num_warps=nw,
        num_stages=ns,
    )


def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)

    if (ptr_tensor := _LORA_PTR_DICT.get(key)) is not None:
        return ptr_tensor

    tensor_ptrs = []
    for lora_weight in lora_weights:
        tensor_ptrs.append(lora_weight.data_ptr())
    ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)

    _LORA_PTR_DICT[key] = ptr_tensor
    return _LORA_PTR_DICT.get(key)


def _adjust_kernel_inputs(
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
):
    """
    helper function to adjust kernel inputs when sorted_token_ids is None
    """
    if sorted_token_ids is None:
        stride_tl = 0
        stride_el = 0
        grid_lora_dim = 1
    else:
        stride_tl = sorted_token_ids.stride(0)
        stride_el = expert_ids.stride(0)
        grid_lora_dim = num_active_loras.item()
    return grid_lora_dim, stride_tl, stride_el


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
    ]
)
def _fused_moe_lora_kernel(
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids,
    adapter_enabled,
    max_loras,  # <<< PR2: rename, used for masks when grid axis-2 != max_loras
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    slice_a_size,
    slice_c_size,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    # top_k_num or 1 depending on input token
    # is expanded by top_k or not
    token_mapping_factor: tl.constexpr,
    # whether use naive block assignment
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,  # new, enable .ca load for B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    USE_TMA: tl.constexpr,
    # sort_c determines whether tokens are stored in C in the order determined
    # by sorted_token_ids to enable later TMA loads from this tensor.
    #
    # When USE_TMA is enabled, the parameter combinations are:
    #   a_desc  | b_desc  | sort_c | Use Case
    #   --------|---------|--------|-----------------------------
    #   yes     | yes     | False  | expand kernel (num_slices=1)
    #   no      | yes     | True   | shrink kernel (num_slices=1)
    #   yes     | no      | False  | expand kernel (num_slices>1)
    #   no      | no      | True   | shrink kernel (num_slices>1)
    sort_c: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
    lora_idx = tl.program_id(axis=2)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # Get lora_id
    lora_id = _get_lora_id(
        lora_ids,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    # Non-naive only: check num_tokens_post_padded
    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    # Get expert_id
    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    # Get token offsets
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    if USE_TMA and a_desc is not None:
        # Expand path - with TMA enabled, load from A using TMA descriptor
        offs_am = (
            slice_id * max_loras * EM
            + lora_id * EM
            + pid_m * BLOCK_SIZE_M // token_mapping_factor
        )
        offs_ak = pid_sk * BLOCK_SIZE_K
    else:
        # Shrink path - load hidden states based on order defined in
        # 'sorted_token_ids_ptr' then store them in c_ptr in this same sorted order
        tl.static_assert(a_desc is None, "a_desc must be none")
        a_ptrs = cur_a_ptr + (
            offs_token[:, None] // token_mapping_factor * stride_am
            + offs_k[None, :] * stride_ak
        )

    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = pid_sk * BLOCK_SIZE_K
        if b_desc is None:
            # Note(@gnovack) - Allocation of TMA descriptors on-device
            # can cause conflicts when running in parallel via PDL
            if USE_GDC and not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()

            b_desc = tl.make_tensor_descriptor(
                cur_b_ptr,
                shape=[max_loras, num_experts, N, K],
                strides=[stride_bl, stride_be, stride_bn, stride_bk],
                block_shape=[1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        b_ptrs = (
            cur_b_ptr
            + lora_id * stride_bl
            + expert_id * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if USE_GDC and IS_PRIMARY:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        tl.extra.cuda.gdc_launch_dependents()

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    for k in range(0, grid_k):
        cur_k_offset = k * (BLOCK_SIZE_K * SPLIT_K)
        k_remaining = K - cur_k_offset
        # pre-fetch lora weight
        if b_desc is not None:
            b = (
                b_desc.load([lora_id, expert_id, offs_bn, offs_bk + cur_k_offset])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        else:
            # add (offs_bn < N) mask; optional .ca for B
            b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
            if USE_B_L2_CACHE:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
            else:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

        if a_desc is not None:
            a = a_desc.load([offs_am, offs_ak + cur_k_offset])
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

        # Cast operands to matching dtype for tl.dot. On ROCm, Triton's
        # compiler may infer different types for a and b when merging
        # if/else branches (TMA desc path returns fp32, tl.load returns
        # the pointer's element type).
        accumulator += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = _get_c_ptrs(
        cur_c_ptr,
        lora_id,
        pid_m,
        offs,
        offs_token,
        offs_cn,
        stride_cm,
        stride_cn,
        EM,
        BLOCK_SIZE_M,
        sort_c,
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
            tl.store(c_ptrs, prev + accumulator, mask=c_mask)
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    # (num_slices, num_tokens, top_k_num, max_lora_rank)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor | None,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,) or (num_tokens * top_k,)
    num_tokens_post_padded: torch.Tensor | None,  # (max_loras, )
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    w1_lora_a_stacked = lora_a_stacked[0]
    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
        "USE_TMA": use_tma,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )
    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        grid_lora_dim,
    )

    a_desc = None
    b_desc = None
    if use_tma and num_slices == 1:
        b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            lora_a_stacked[0],
            [1, 1, shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"]],
        )

    _fused_moe_lora_kernel[grid](
        qcurr_hidden_states,
        a_desc,
        b_ptr,
        b_desc,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_a_stacked[0].shape[0],
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(-2),
        a_intermediate_cache1.stride(-1),
        stride_tl,
        stride_el,
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        token_mapping_factor=1 if mul_routed_weight else top_k_num,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=False,
        ADD_INPUTS=False,
        USE_B_L2_CACHE=True,
        sort_c=use_tma and sorted_token_ids is not None,
        IS_PRIMARY=True,
        **shrink_config,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor | None,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,) or (num_tokens * top_k,)
    num_tokens_post_padded: torch.Tensor | None,  # (max_loras, )
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[-1]
    )

    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": 1,  # Set split_k = 1 for expand calls
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
        "USE_TMA": use_tma,
    }

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        grid_lora_dim,
    )

    # Fast path: directly accumulate into the corresponding slice interval of output.
    out_view = output[:, :, offset : offset + num_slices * N]
    slice_c_size = N * out_view.stride(2)
    a_desc = None
    b_desc = None
    if use_tma:
        if sorted_token_ids is not None:
            a_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                a_intermediate_cache1,
                [expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_K"]],
            )
        if num_slices == 1:
            b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_b_stacked[0],
                [1, 1, expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"]],
            )
    else:
        b_desc = None

    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        a_desc,
        b_ptr,
        b_desc,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_b_stacked[0].shape[0],
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        stride_tl,
        stride_el,
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        token_mapping_factor=1,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=True,
        USE_B_L2_CACHE=True,
        sort_c=False,
        IS_PRIMARY=False,
        **expand_config,
    )


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor | None,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,) or (num_tokens * top_k,)
    num_tokens_post_padded: torch.Tensor | None,  # (max_loras, )
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    add_inputs: bool = True,
) -> None:
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2
    if sorted_token_ids is None:
        assert expert_ids.dim() == 1
    else:
        assert sorted_token_ids is not None
        assert num_tokens_post_padded is not None
        assert (
            sorted_token_ids.dim()
            == expert_ids.dim()
            == topk_weights.dim()
            == qcurr_hidden_states.dim()
            == 2
        )
        assert (
            sorted_token_ids.shape[0]
            == expert_ids.shape[0]
            == num_tokens_post_padded.shape[0]
        )
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]

    # Fast path: single fused kernel
    if not fully_sharded:
        M_pairs = topk_weights.numel()
        if (
            sorted_token_ids is None
            and max_lora_rank <= 64
            and M_pairs * max_lora_rank <= 1024
        ):
            _run_fused_moe_lora_small_batch(
                output,
                qcurr_hidden_states,
                lora_a_stacked,
                lora_b_stacked,
                topk_weights,
                expert_ids,
                token_lora_mapping,
                top_k_num,
                adapter_enabled,
                mul_routed_weight,
                add_inputs=add_inputs,
            )
            return
        # shrink/expand BLOCK_SIZE_M must match the block_size that
        # moe_lora_align_block_size used; both shrink and expand pass the
        # same value (asserted by `shrink_block_size_m == expand_block_size_m`
        # below).
        _run_fused_moe_lora_one_shot(
            output,
            qcurr_hidden_states,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mapping,
            max_lora_rank,
            top_k_num,
            lora_ids,
            num_active_loras,
            adapter_enabled,
            mul_routed_weight,
            shrink_block_size_m,
            add_inputs=add_inputs,
        )
        return

    assert add_inputs, (
        "fused_moe_lora(add_inputs=False) is only supported on the "
        "fully_sharded=False fast path"
    )

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]
    assert shrink_block_size_m == expand_block_size_m
    EM = (
        sorted_token_ids.shape[1]
        if sorted_token_ids is not None
        else num_tokens * shrink_block_size_m
    )

    # TMA is not currently compatiple with fully_sharded due to the non-determinism
    # of token id sorting across ranks.
    use_tma = supports_tma(device) and not fully_sharded

    intermediate_cache_shape = (
        num_slices,
        M,
        top_k_num,
        max_lora_rank,
    )
    if use_tma:
        if num_slices > 1:
            # if num_slices > 1, we construct TMA descriptors for LoRA
            # weights within the kernel, which requires us to first set an allocator
            set_triton_allocator(device)

        # When storing intermediate data in sorted order for TMA, we
        # need an extra 'num_active_loras' dim in the cache to avoid conflicts
        if sorted_token_ids is not None:
            intermediate_cache_shape = (
                num_slices,
                sorted_token_ids.shape[0],
                EM,
                max_lora_rank,
            )

    a_intermediate_cache1 = torch.zeros(
        intermediate_cache_shape,
        dtype=output.dtype,
        device=device,
    )

    use_gdc = supports_pdl(device) and not fully_sharded
    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        shrink_block_size_m,
        shrink_block_size_n,
        shrink_block_size_k,
        shrink_group_size_m,
        shrink_num_warps,
        shrink_num_stages,
        shrink_split_k,
        num_active_loras,
        mul_routed_weight,
        use_gdc=use_gdc,
        use_tma=use_tma,
    )

    if fully_sharded:
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                a_intermediate_cache1
            )
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(
                a_intermediate_cache1
            )

            # reset max_lora_rank to the full rank after allgather
            max_lora_rank = a_intermediate_cache1.shape[-1]

    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        max_lora_rank,
        w1_output_dim_size,
        expand_block_size_m,
        expand_block_size_n,
        expand_block_size_k,
        expand_group_size_m,
        expand_num_warps,
        expand_num_stages,
        expand_split_k,
        num_active_loras,
        mul_routed_weight,
        offset,
        use_gdc=use_gdc,
        use_tma=use_tma,
    )


def _fused_moe_lora_fake(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    add_inputs: bool = True,
) -> None:
    return


def _fused_moe_lora_shrink_fake(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    return


def _fused_moe_lora_expand_fake(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="fused_moe_lora",
        op_func=_fused_moe_lora,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_shrink",
        op_func=_fused_moe_lora_shrink,
        mutates_args=["a_intermediate_cache1"],
        fake_impl=_fused_moe_lora_shrink_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_expand",
        op_func=_fused_moe_lora_expand,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_expand_fake,
    )

    fused_moe_lora = torch.ops.vllm.fused_moe_lora
    fused_moe_lora_shrink = torch.ops.vllm.fused_moe_lora_shrink
    fused_moe_lora_expand = torch.ops.vllm.fused_moe_lora_expand

except AttributeError:
    fused_moe_lora = _fused_moe_lora
    fused_moe_lora_shrink = _fused_moe_lora_shrink
    fused_moe_lora_expand = _fused_moe_lora_expand
