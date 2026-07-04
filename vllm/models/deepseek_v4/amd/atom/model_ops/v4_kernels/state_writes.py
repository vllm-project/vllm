# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State-write Triton kernels for V4 attention backend.

Replaces the per-seq Python state writes in `deepseek_v4.py` (PR-A Phase 1).
Inputs are flat batched tensors; per-token slot/position lookups happen
inside the kernel — no `.item()` syncs.

Currently implemented:
- `swa_write`: writes the LAST `min(tok_n_b, write_per_batch)` tokens of
  every seq `b ∈ [0, bs)` into `swa_kv[state_slot_per_seq[b],
  positions[src] % cache_size, :] = kv[src, :]`. `src_id` is derived inside
  the kernel from `cu_seqlens_q + row_in_batch` — no shared per-token
  `write_indices` GPU buffer (which had a DMA-tear race when the next fwd's
  CPU rewrite landed mid-H2D). `cache_size = window_size + max_spec_steps`
  — for non-MTP this reduces to `window_size`; for MTP-k draft tokens get
  their own ring slots separate from the verified token's slot.
- `update_compressor_states`: unified in-place update of Compressor's
  per-request `kv_state` + `score_state` ring buffers, covering both prefill
  (B-side overlap context + tail) and decode (every token at `pos % STATE_SIZE`
  in a single ring). Layout follows paper §3.6.1 (per-request fixed-size state
  cache) but indexes the buffer as ONE ring of size `STATE_SIZE = 2*ratio`
  (CSA overlap) or `ratio` (HCA). Token at absolute `pos` always lands at
  `kv_state[slot, pos % STATE_SIZE]` — no segment switching, no roll. The
  Compressor's softmax-pool consumer reads two halves whose A-side / B-side
  identity alternates by block-id parity; see `Compressor.forward` for that
  consumer-side logic.

Caller contract (`swa_write`):
- `kv`                  [T, head_dim] flat — full per-fwd KV (forward_vars).
- `positions`           [T] int — full positions buffer (forward_vars).
- `cu_seqlens_q`        [bs+1] int — per-fwd cumulative seqlens (so
                        seq `i` covers token rows `[cu_seqlens_q[i], cu_seqlens_q[i+1])`
                        in `kv` / `positions`). Per-seq token count is
                        derived inside the kernel as `cu_seqlens_q[i+1] -
                        cu_seqlens_q[i]`.
- `state_slot_per_seq`  [bs] int — `state_slot_mapping_gpu_i32`.
- `swa_kv`              [num_slots, cache_size, head_dim] in-place buffer.
- `cache_size`          int ring-slot count = `window_size + max_spec_steps`
                        (e.g. 128 + 0 = 128 non-MTP; 128 + 1 = 129 MTP-1).
- `write_per_batch`     int — max tokens to write per seq this fwd
                        (= `min(max_q_len, cache_size)`). Used as Triton
                        `constexpr` for grid sizing.

Grid = `(bs, write_per_batch)`; each program writes one (seq, row-in-seq)
token. Per-seq actual count is `min(token_num_per_seq[bs], write_per_batch)`;
threads whose `row_in_batch >= actual_count` bail. The kernel derives
`src_id = cu_seqlens_q[i+1] - actual_count + row_in_batch` — selects the
LAST `actual_count` tokens of seq `i` in `kv` / `positions`, no shared
GPU index buffer needed (no DMA race window).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_write_kernel(
    kv_ptr,  # [T, head_dim]
    positions_ptr,  # [T] int — full positions
    cu_seqlens_q_ptr,  # [bs+1] int — per-seq cumulative seqlens
    state_slot_per_seq_ptr,  # [bs] int — state_slot_mapping_gpu_i32
    swa_kv_ptr,  # [num_slots, cache_size, head_dim]
    swa_kv_slot_stride,  # = cache_size * head_dim
    swa_kv_pos_stride,  # = head_dim
    head_dim,
    cache_size,
    WRITE_PER_BATCH: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """2D grid `(bs, WRITE_PER_BATCH)`. Program `(b, r)` writes the `r`-th
    of the last-N tokens of seq `b`, where `N = min(tok_n_b, WRITE_PER_BATCH)`
    and `tok_n_b = cu_seqlens_q[b+1] - cu_seqlens_q[b]`. Threads with
    `r >= N` bail.

    `src_id = cu_seqlens_q[b+1] - N + r` — selects directly from `kv` /
    `positions` with NO shared GPU index buffer (no DMA race window).
    """
    batch_idx = tl.program_id(0)
    row_in_batch = tl.program_id(1)

    cu_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    cu_end = tl.load(cu_seqlens_q_ptr + batch_idx + 1)
    tok_n = cu_end - cu_start
    if tok_n <= 0:
        return
    write_n = tl.minimum(tok_n, WRITE_PER_BATCH)
    if row_in_batch >= write_n:
        return

    src_id = cu_end - write_n + row_in_batch

    slot = tl.load(state_slot_per_seq_ptr + batch_idx)
    pos = tl.load(positions_ptr + src_id)
    ring_idx = pos % cache_size

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    src = tl.load(
        kv_ptr + src_id * head_dim + d_offsets,
        mask=d_mask,
    )
    dst = (
        swa_kv_ptr
        + slot * swa_kv_slot_stride
        + ring_idx * swa_kv_pos_stride
        + d_offsets
    )
    tl.store(dst, src, mask=d_mask)


def swa_write(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    swa_kv: torch.Tensor,
    cache_size: int,
    write_per_batch: int,
) -> None:
    """In-place write `swa_kv[state_slot_per_seq[b], pos % cache_size, :] = kv[r, :]`
    for the last `min(tok_n_b, write_per_batch)` tokens of every seq
    `b ∈ [0, bs)` this fwd, where `tok_n_b = cu_seqlens_q[b+1] - cu_seqlens_q[b]`.
    `bs = state_slot_per_seq.shape[0]`.

    The kernel derives `r` from `cu_seqlens_q` diff inside the kernel,
    eliminating the prior `write_indices` GPU staging buffer (which had a DMA
    tearing race when its CPU mirror was rewritten by the next fwd before
    the H2D for the current fwd had completed; see `_swa_write_kernel` doc).

    Args:
        kv: [T, head_dim] per-fwd KV (BF16). `T = cu_seqlens_q[bs]`.
        positions: [T'] int — full forward_vars["positions"] (`T' >= T`).
        cu_seqlens_q: [bs+1] int — exact size (`bs == state_slot_per_seq.shape[0]`).
        state_slot_per_seq: [bs] int — per-seq state cache slot. Its
            `shape[0]` is the grid X dim and source-of-truth for `bs`.
        swa_kv: [num_slots, cache_size, head_dim] ring buffer.
        cache_size: ring-slot count = `window_size + max_spec_steps`.
        write_per_batch: `min(max_q_len, cache_size)` — max tokens written
            per seq this fwd (grid y dim, kernel `constexpr`).
    """
    assert kv.dim() == 2, f"kv must be [T, D], got {kv.shape}"
    assert positions.dim() == 1
    assert state_slot_per_seq.dim() == 1
    bs = state_slot_per_seq.shape[0]
    assert cu_seqlens_q.dim() == 1 and cu_seqlens_q.shape[0] >= bs + 1
    assert swa_kv.dim() == 3, f"swa_kv must be [S, C, D], got {swa_kv.shape}"
    T, head_dim = kv.shape
    assert positions.shape[0] >= T, f"positions {positions.shape[0]} < kv T={T}"
    assert (
        swa_kv.shape[1] == cache_size
    ), f"swa_kv ring dim {swa_kv.shape[1]} != cache_size {cache_size}"
    assert swa_kv.shape[2] == head_dim
    assert kv.is_contiguous() and swa_kv.is_contiguous()
    assert (
        bs > 0 and write_per_batch > 0
    ), f"bs={bs}, write_per_batch={write_per_batch} must be positive"

    # head_dim is small (e.g. 64-128 for V4 SWA layer), so a single Triton
    # block per token covers it. Round up to the next power of two for tl.
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (bs, write_per_batch)

    _swa_write_kernel[grid](
        kv,
        positions,
        cu_seqlens_q,
        state_slot_per_seq,
        swa_kv,
        swa_kv.stride(0),
        swa_kv.stride(1),
        head_dim,
        cache_size,
        WRITE_PER_BATCH=write_per_batch,
        BLOCK_D=BLOCK_D,
    )


def swa_write_reference(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    swa_kv: torch.Tensor,
    cache_size: int,
    write_per_batch: int,
) -> None:
    """Pure-PyTorch reference equivalent of `swa_write`. For tests / dump-bisect.

    Mirrors the kernel: for each seq `b ∈ [0, bs)`
    (`bs = state_slot_per_seq.shape[0]`), take the last
    `min(cu_seqlens_q[b+1] - cu_seqlens_q[b], write_per_batch)` rows of `kv`
    for that seq (via `cu_seqlens_q[b+1] - N + arange(N)`), look up state
    slot, ring write.
    """
    bs = state_slot_per_seq.shape[0]
    cu_cpu = cu_seqlens_q[: bs + 1].tolist()
    for b in range(bs):
        cu_start = int(cu_cpu[b])
        cu_end = int(cu_cpu[b + 1])
        tok_n = cu_end - cu_start
        write_n = min(tok_n, write_per_batch)
        if write_n <= 0:
            continue
        src_ids = torch.arange(
            cu_end - write_n, cu_end, dtype=torch.long, device=kv.device
        )
        src_kv = kv[src_ids]
        src_pos = positions[src_ids]
        slot = int(state_slot_per_seq[b].item())
        ring_idx = src_pos % cache_size
        swa_kv[slot, ring_idx] = src_kv


# === Unified Compressor state save (plan path) ==========================
# Paper §3.6.1: per-request fixed-size state cache for "uncompressed tail
# tokens + previous block as overlap context (B-side, eq 11)". ATOM keeps
# this as a single ring of size `STATE_SIZE = 2*ratio` (CSA overlap) or
# `ratio` (HCA). Each token at absolute `pos` writes to slot
# `pos % STATE_SIZE`; the consumer (`fused_compress.*` kernel) reads its K
# source rows per-source-position, dispatching INPUT vs state cache by the
# `k_static >= window_len` plan field (where `window_len` is the count of
# leading K-loop iterations that go to state cache, encoded per-boundary in
# `compress_plan`).
#
# Write window selection (HOST side, in compress_plan.make_compress_plans):
#   write_plan rows = tokens whose absolute `pos >= max(0, seq_len - STATE_SIZE)`.
#   This preserves the last STATE_SIZE absolute positions of this forward
#   regardless of how it was scheduled (fresh prefill, chunked prefill,
#   single decode, MTP-N). The kernel below writes those rows
#   unconditionally — no in-kernel mask.


@triton.jit
def _update_compressor_states_kernel(
    kv_ptr,  # [N, dim] (strided allowed)
    kv_row_stride,
    score_ptr,  # [N, dim] (strided allowed)
    score_row_stride,
    ape_ptr,  # [RATIO, dim]
    write_plan_ptr,  # [num_write, 4] int32 (ragged_id, batch_id, position, _)
    state_slot_mapping_ptr,  # [bs] int32 — per-seq state cache slot
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    dim,
    STATE_SIZE: tl.constexpr,  # ring buffer modulo = kv_state.shape[1] (≥ K_pool;
    #   V4-Pro spec decode: K_pool + max_spec_steps to keep R's rejected writes
    #   out of R+1's read window; non-spec or pre-spec models: exactly K_pool)
    OVERLAP: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SGLang plan-style write: one program per row in `write_plan_ptr`.

    Each plan row = (ragged_id, batch_id, position, _). The plan was
    pre-filtered on the host to include only tokens whose `position` falls in
    the per-seq "last STATE_SIZE absolute positions" window — so the kernel
    writes unconditionally (no in-kernel mask), keeping it minimal.

    Destination (uniform):
      dst = position % STATE_SIZE
      slot = state_slot_mapping[batch_id]

    Score write fuses ape lookup: `score + ape[position % RATIO]`.
    """
    pid = tl.program_id(0)
    plan_base = write_plan_ptr + pid * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)

    # Fixed-grid + sentinel for CUDAGraph compat: caller may pass a buffer
    # padded to max capacity; rows beyond `num_write` carry position = -1
    # and are skipped here.
    if position < 0:
        return

    slot = tl.load(state_slot_mapping_ptr + batch_id)
    dst = position % STATE_SIZE
    ring_idx_ape = position % RATIO

    d = tl.arange(0, BLOCK_D)
    m = d < dim

    kv_v = tl.load(kv_ptr + ragged_id * kv_row_stride + d, mask=m).to(tl.float32)
    sc_v = tl.load(score_ptr + ragged_id * score_row_stride + d, mask=m).to(tl.float32)
    ape_v = tl.load(ape_ptr + ring_idx_ape * dim + d, mask=m).to(tl.float32)

    tl.store(
        kv_state_ptr + slot * kv_state_slot_stride + dst * kv_state_pos_stride + d,
        kv_v,
        mask=m,
    )
    tl.store(
        score_state_ptr
        + slot * score_state_slot_stride
        + dst * score_state_pos_stride
        + d,
        sc_v + ape_v,
        mask=m,
    )


def update_compressor_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,  # [num_write, 4] int32
    num_write: int,
    state_slot_mapping: torch.Tensor,  # [bs] int32 — per-seq state slot
    ratio: int,
    overlap: bool,
) -> None:
    """In-place update of Compressor's per-request `kv_state`/`score_state`
    ring buffer (size ≥ `K_pool = (1+overlap)*ratio`; V4-Pro widens to
    `K_pool + max_spec_steps` for spec decode, keeps `K_pool` for non-spec),
    driven by a SGLang-style packed `write_plan`.

    The plan is pre-filtered on the host to include only tokens whose
    `position` falls in the per-seq "last K_pool absolute positions" window
    (`write_starts = max(0, context_lens - K_pool)` in `make_compress_plans`)
    — the kernel writes unconditionally, no in-kernel mask. Note that the
    write window is K_pool, NOT STATE_SIZE; the extra STATE_SIZE - K_pool
    slots exist purely as aliasing slack for spec rollback (see
    `csa_main_state_shape` comment in `deepseek_v4_attn.py`).

    Args:
      kv:           [N, dim] flat batched KV (typically fp32 or bf16, cast inside).
      score:        [N, dim] flat batched score (NOT pre-added with ape;
                    kernel fuses ape addition).
      ape:          [ratio, dim] absolute position embedding.
      kv_state:     [num_slots, S, dim] in-place ring buffer. S ≥ K_pool;
                    V4-Pro: S = K_pool + max_spec_steps.
      score_state:  same shape as kv_state.
      write_plan:   [num_write, 4] int32 — packed (ragged_id, batch_id,
                    position, _); each row = one token to write.
      num_write:    grid size (CPU scalar, == write_plan.shape[0] but kept
                    explicit to avoid GPU sync).
      state_slot_mapping: [bs] int32 — per-seq state cache slot.
      ratio, overlap: compress geometry.
    """
    assert kv.dim() == 2 and score.dim() == 2
    assert kv.shape == score.shape, f"{kv.shape} vs {score.shape}"
    assert ape.dim() == 2 and ape.shape[0] == ratio
    K_pool = (2 if overlap else 1) * ratio  # pool window (lower bound)
    state_size = kv_state.shape[1]  # ring buffer modulo (≥ K_pool)
    assert (
        state_size >= K_pool
    ), f"kv_state.shape[1]={state_size}, must be ≥ K_pool={K_pool}"
    dim = kv.shape[1]
    assert write_plan.dim() == 2 and write_plan.shape[1] == 4
    assert write_plan.dtype == torch.int32
    assert state_slot_mapping.dim() == 1 and state_slot_mapping.dtype == torch.int32
    # Grid = plan buffer capacity (fixed at builder __init__ time), NOT the
    # per-fwd `num_write`. Inactive rows past `num_write` carry sentinel
    # `position=-1` (filled host-side in `make_compress_plans`); the kernel
    # bails on those, so this is functionally identical to the variable-grid
    # version while keeping the launch CUDAGraph-capturable.
    grid_size = write_plan.shape[0]
    if grid_size == 0:
        return

    # Strided kv / score allowed (zero-copy split halves of fused upstream
    # GEMM); inner column stride must be 1 (kernel uses `+ d`).
    assert kv.stride(-1) == 1 and score.stride(-1) == 1
    BLOCK_D = triton.next_power_of_2(dim)
    _update_compressor_states_kernel[(grid_size,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        write_plan,
        state_slot_mapping,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        dim,
        STATE_SIZE=state_size,
        OVERLAP=int(overlap),
        RATIO=ratio,
        BLOCK_D=BLOCK_D,
    )


def update_compressor_states_reference(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    ratio: int,
    overlap: bool,
) -> None:
    """Pure-PyTorch reference equivalent of `update_compressor_states` (plan path).

    `write_plan[i] = (ragged_id, batch_id, position, _)` — each row is one
    token to write.  No mask (host filtered).
    """
    state_size = kv_state.shape[1]  # ring buffer modulo (≥ (1+overlap)*ratio)
    plan_cpu = write_plan.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    for i in range(plan_cpu.shape[0]):
        ragged_id, batch_id, position, _ = plan_cpu[i].tolist()
        slot = int(slot_map_cpu[batch_id].item())
        dst = position % state_size
        kv_state[slot, dst] = kv[ragged_id]
        score_state[slot, dst] = score[ragged_id] + ape[position % ratio]
