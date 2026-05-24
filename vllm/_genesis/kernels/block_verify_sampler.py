# SPDX-License-Identifier: Apache-2.0
"""Block-verify rejection sampler kernels for Genesis P71.

Backport of upstream PR vllm-project/vllm#40819 (Z. Golpayegani, OPEN draft)
implementing Sun et al. 2024 (arXiv 2403.10444) block verification rule.

================================================================
APPLIED FIXES (vs PR #40819 head, 2026-04-26)
================================================================

The upstream PR implementation has TWO critical bugs flagged by gemini-bot
in code review (both unresolved at time of writing). We apply both before
backporting:

**FIX 1 (CRITICAL)** — uniform draw must be SHARED across the block:
  PR loads `uniform_prob = tl.load(uniform_probs_ptr + token_idx)` for every
  position. Sun 2024 §3.2 algorithm requires a SINGLE Bernoulli decision
  per BLOCK (one shared u per request, not per-position). Fix: load
  `uniform_prob = tl.load(uniform_probs_ptr + start_idx)` once and reuse.

**FIX 2 (HIGH)** — perfect draft match must be ACCEPTED:
  PR computes `h_block = residual / denom if denom > 0 else 0.0`. When
  `denom == 0` this happens because `prefix_prob == 1.0` AND
  `residual_mass == 0.0` — i.e., the draft was a PERFECT match to target.
  Returning 0.0 here REJECTS perfect drafts (the worst possible behavior).
  Fix: return 1.0 (always accept on perfect match).

================================================================
ALGORITHM (Sun 2024)
================================================================

Per-token rule (Leviathan 2022, vLLM default):
  Accept token i iff u_i <= min(1, p_t(x_i) / p_d(x_i)) for u_i ~ U(0,1)

Block rule (Sun 2024, this kernel):
  At position k, accept iff u <= h_block(k) where:
    P_k = product_{j<=k} min(1, p_t(x_j) / p_d(x_j))   (clipped at 1)
    R_k = sum_v max(0, P_k * p_t(v) - p_d(v))          (residual mass)
    h_block(k) = R_k / (R_k + 1 - P_k)                  (intermediate)
    h_block(γ) = P_γ                                    (last position)

Recovered token sampled from argmax_v (P_i * p_t(v) - p_d(v))_+ / q_v
where q ~ Exp(1) shared per-request (Gumbel-style normalizer).

Theorem (Sun 2024 §4): block rule has expected accepted tokens >= per-token
rule for the same draft, with strict inequality whenever any per-position
ratio < 1. Same target marginal preserved (unbiased).

================================================================
GENESIS-SPECIFIC NOTES
================================================================

- Default OFF — opt-in via `GENESIS_ENABLE_P71_BLOCK_VERIFY=1`
- Realistic gain on 35B-A3B + Ampere: +0-3% wall-clock (PR's own Qwen3-32B
  bench shows parity at our model size). Treat as experimental.
- Safe fallback: any error in this kernel raises and is caught by an outer
  try/except in the wiring patch, which reverts to upstream per-token rule.
- Cudagraph compatibility: PIECEWISE only (data-dependent loop bounds).
  If P67b enables FULL_AND_PIECEWISE for spec-decode, P71 forces fallback
  to PIECEWISE for the rejection_sample call — unavoidable.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Bug-fixes: gemini-code-assist review on vllm#40819 (cited in code).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


# Vocab-dimension block size for the block-verify kernels. 8192 matches the
# upstream recovered-token kernel. Tunable for profiling.
_BLOCK_VERIFY_VOCAB_BLOCK = 8192

# PLACEHOLDER_TOKEN_ID matches upstream (vllm/v1/sample/rejection_sampler.py).
PLACEHOLDER_TOKEN_ID = -1


def generate_rejection_q(
    batch_size: int,
    vocab_size: int,
    num_draft_tokens: list[int],
    generators: dict,
    device: torch.device,
) -> torch.Tensor:
    """Per-request Exp(1) normalizer used for recovered-token sampling.

    Drawn once per request (shape [batch_size, vocab_size]); requests with a
    user-supplied seed share the same generator across their vocab row.
    """
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    for i, generator in generators.items():
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)
    return q


# ════════════════════════════════════════════════════════════════════════
# PyTorch reference (with both gemini bug-fixes applied).
# Used by parity tests + when GENESIS_P71_USE_PYTORCH=1.
# ════════════════════════════════════════════════════════════════════════

def rejection_random_sample_block_verify_pytorch(
    output_token_ids: torch.Tensor,      # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens: torch.Tensor,   # [batch_size]
    draft_token_ids: torch.Tensor,       # [num_tokens]
    draft_probs: torch.Tensor,           # [num_tokens, vocab_size]
    target_probs: torch.Tensor,          # [num_tokens, vocab_size]
    bonus_token_ids: torch.Tensor,       # [batch_size, 1]
    recovered_token_ids: torch.Tensor,   # [num_tokens]
    uniform_probs: torch.Tensor,         # [num_tokens]
    is_greedy: torch.Tensor,             # [batch_size]
    max_spec_len: int,
    vocab_size: int,
) -> None:
    """PyTorch reference for the block-verify acceptance kernel.

    Writes accept/recover/bonus tokens in-place on `output_token_ids`.
    """
    del vocab_size  # parity with triton signature
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    # v7.51.2: pre-allocate then slice-assign instead of torch.cat
    # (8-byte tensor is marginal but applies the same idiom we use
    # elsewhere — no double-alloc spike, no transient peak).
    cu_start = torch.empty_like(cu_num_draft_tokens)
    cu_start[0] = 0
    cu_start[1:] = cu_num_draft_tokens[:-1]
    num_draft_per_batch = cu_num_draft_tokens - cu_start
    gamma = num_draft_per_batch.to(torch.long)

    i_indices = torch.arange(1, max_spec_len + 1, device=device)[None, :]
    valid_mask = i_indices <= num_draft_per_batch[:, None]

    global_token_indices = (cu_start[:, None] + i_indices - 1).clamp(
        0, draft_token_ids.shape[0] - 1
    )
    draft_tokens = draft_token_ids[global_token_indices]
    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()

    draft_token_probs = draft_probs[flat_indices, flat_draft_tokens].view(
        batch_size, max_spec_len
    )
    target_token_probs = target_probs[flat_indices, flat_draft_tokens].view(
        batch_size, max_spec_len
    )

    # ─── [Genesis P71 FIX 1] SHARED u per request, not per position ────────
    # PR #40819 used `uniform_token_probs = uniform_probs[global_token_indices]`
    # which is per-position. Sun 2024 requires ONE Bernoulli per request.
    # Take the FIRST uniform draw per request (at cu_start) and broadcast.
    uniform_token_probs_shared = uniform_probs[cu_start.to(torch.long)][:, None]
    # ─────────────────────────────────────────────────────────────────────

    recovered_tokens = recovered_token_ids[global_token_indices]

    # Per-position acceptance ratio min(1, target_p / draft_p).
    ratio = torch.where(
        draft_token_probs > 0,
        target_token_probs / draft_token_probs.clamp(min=1e-10),
        torch.zeros_like(draft_token_probs),
    )

    # p_prefix[:, k+1] = min(p_prefix[:, k] * ratio[:, k], 1.0)
    p_prefix = torch.ones(
        (batch_size, max_spec_len + 1), dtype=torch.float32, device=device
    )
    for k in range(max_spec_len):
        p_prefix[:, k + 1] = (p_prefix[:, k] * ratio[:, k]).clamp(max=1.0)

    p_grid = p_prefix[:, 1:]
    h_block = torch.zeros(
        (batch_size, max_spec_len), dtype=torch.float32, device=device
    )
    intermediate_mask = i_indices < num_draft_per_batch[:, None]

    if torch.any(intermediate_mask):
        residual_mass = torch.zeros_like(p_grid)
        flat_intermediate_mask = intermediate_mask.flatten()
        flat_current_token_indices = flat_indices[flat_intermediate_mask]
        flat_p_grid = p_grid.flatten()[flat_intermediate_mask]
        flat_residual_mass = torch.clamp(
            flat_p_grid[:, None] * target_probs[flat_current_token_indices]
            - draft_probs[flat_current_token_indices],
            min=0.0,
        ).sum(dim=-1)
        residual_mass[intermediate_mask] = flat_residual_mass

        denom = residual_mass + (1.0 - p_grid)
        # ─── [Genesis P71 FIX 2] denom==0 means PERFECT match → ACCEPT (1.0) ─
        # PR #40819 returned 0.0 here, REJECTING perfect drafts. Wrong.
        h_block = torch.where(
            intermediate_mask,
            torch.where(denom > 0, residual_mass / denom, torch.ones_like(denom)),
            h_block,
        )
        # ─────────────────────────────────────────────────────────────────

    # Last draft position uses p_prefix[gamma] directly.
    batch_indices = torch.arange(batch_size, device=device)
    last_pos = (gamma - 1).clamp(min=0)
    h_block[batch_indices, last_pos] = p_prefix[
        batch_indices, gamma.clamp(max=max_spec_len)
    ]

    non_greedy_mask = (~is_greedy)[:, None]
    accepted_mask = (
        valid_mask
        & (uniform_token_probs_shared.to(torch.float32) <= h_block)
        & non_greedy_mask
    )

    last_accept_i = (
        torch.where(
            accepted_mask,
            i_indices.to(torch.long),
            torch.zeros_like(i_indices, dtype=torch.long),
        )
        .max(dim=1)
        .values
    )

    # Write accepted draft tokens.
    accept_span = (i_indices <= last_accept_i[:, None]) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(
        accept_span, draft_tokens, output_token_ids[:, :max_spec_len]
    )

    # Write recovered token at the first rejection position.
    reject_mask = (
        (i_indices == last_accept_i[:, None] + 1) & valid_mask & non_greedy_mask
    )
    output_token_ids[:, :max_spec_len] = torch.where(
        reject_mask, recovered_tokens, output_token_ids[:, :max_spec_len]
    )

    # Write bonus token when all drafts were accepted.
    all_positions = torch.arange(max_spec_len + 1, device=device)[None, :]
    bonus_mask = (
        (last_accept_i[:, None] >= num_draft_per_batch[:, None])
        & non_greedy_mask
        & (all_positions == num_draft_per_batch[:, None])
    )
    output_token_ids[:] = torch.where(
        bonus_mask,
        bonus_token_ids.expand(-1, max_spec_len + 1).to(output_token_ids.dtype),
        output_token_ids,
    )


def sample_recovered_tokens_blockwise_pytorch(
    output_token_ids: torch.Tensor,      # [num_tokens]
    cu_num_draft_tokens: torch.Tensor,   # [batch_size]
    draft_token_ids: torch.Tensor,       # [num_tokens]
    draft_probs: torch.Tensor,           # [num_tokens, vocab_size]
    target_probs: torch.Tensor,          # [num_tokens, vocab_size]
    q: torch.Tensor,                     # [batch_size, vocab_size]
    vocab_size: int,
) -> None:
    """Block-verify-aware recovered-token sampler (PyTorch reference)."""
    del vocab_size
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]
    batch_size = cu_num_draft_tokens.shape[0]
    if num_tokens == 0:
        return

    # v7.51.2: pre-allocate then slice-assign instead of torch.cat
    # (8-byte tensor is marginal but applies the same idiom we use
    # elsewhere — no double-alloc spike, no transient peak).
    cu_start = torch.empty_like(cu_num_draft_tokens)
    cu_start[0] = 0
    cu_start[1:] = cu_num_draft_tokens[:-1]
    token_indices = torch.arange(num_tokens, device=device)
    in_range_mask = (token_indices[:, None] >= cu_start[None, :]) & (
        token_indices[:, None] < cu_num_draft_tokens[None, :]
    )
    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)
    token_to_batch = torch.where(
        in_range_mask.any(dim=1),
        token_to_batch,
        torch.zeros_like(token_to_batch),
    )
    pos_in_seq = token_indices - cu_start[token_to_batch]

    max_spec_len = int((cu_num_draft_tokens - cu_start).max().item())

    draft_token_scalar_probs = draft_probs[token_indices, draft_token_ids]
    target_token_scalar_probs = target_probs[token_indices, draft_token_ids]
    per_token_ratio = torch.where(
        draft_token_scalar_probs > 0,
        target_token_scalar_probs / draft_token_scalar_probs.clamp(min=1e-10),
        torch.zeros_like(target_token_scalar_probs),
    )

    ratio_grid = torch.ones(
        (batch_size, max_spec_len), device=device, dtype=torch.float32
    )
    ratio_grid[token_to_batch, pos_in_seq] = per_token_ratio

    p_prefix = torch.ones(
        (batch_size, max_spec_len + 1), device=device, dtype=torch.float32
    )
    for k in range(max_spec_len):
        p_prefix[:, k + 1] = (p_prefix[:, k] * ratio_grid[:, k]).clamp(max=1.0)

    p_i = p_prefix[token_to_batch, pos_in_seq]
    p_i_expanded = p_i[:, None]

    residual = torch.clamp(p_i_expanded * target_probs - draft_probs, min=0.0)

    q_values = q[token_to_batch]
    eps = 1e-10
    q_values_safe = torch.where(q_values == 0, eps, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), eps, q_values_safe)
    prob_over_q = torch.where(
        (q_values == 0) | torch.isinf(q_values),
        torch.full_like(residual, -1e10),
        residual / q_values_safe,
    )

    output_token_ids[:] = torch.argmax(prob_over_q, dim=1).to(output_token_ids.dtype)


# ════════════════════════════════════════════════════════════════════════
# Triton kernels (with both gemini bug-fixes applied).
# Used in production path when GENESIS_ENABLE_P71_BLOCK_VERIFY=1.
# ════════════════════════════════════════════════════════════════════════

if _TRITON_OK:

    @triton.jit
    def sample_recovered_tokens_block_verify_kernel(
        output_token_ids_ptr,         # [num_tokens]
        cu_num_draft_tokens_ptr,      # [batch_size]
        draft_token_ids_ptr,          # [num_tokens]
        draft_probs_ptr,              # [num_tokens, vocab_size]
        target_probs_ptr,             # [num_tokens, vocab_size]
        q_ptr,                        # [batch_size, vocab_size]
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Block-verify variant of sample_recovered_tokens_kernel.

        Grid: (batch_size, max_spec_len). Each program walks prior positions
        to accumulate p_prefix, then picks argmax of
        (p_prefix·target - draft)/q over vocab in BLOCK_SIZE chunks.
        """
        req_idx = tl.program_id(0)
        start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
        num_draft_tokens = end_idx - start_idx

        pos = tl.program_id(1)
        if pos >= num_draft_tokens:
            return

        prefix_prob = 1.0
        for prev_pos in range(pos):
            prev_idx = start_idx + prev_pos
            prev_draft_id = tl.load(draft_token_ids_ptr + prev_idx)
            prev_target = tl.load(target_probs_ptr + prev_idx * vocab_size + prev_draft_id)
            prev_draft = tl.load(draft_probs_ptr + prev_idx * vocab_size + prev_draft_id)
            if prev_draft > 0:
                prefix_prob = min(prefix_prob * prev_target / prev_draft, 1.0)
            else:
                prefix_prob = 0.0

        token_idx = start_idx + pos

        global_max = -1.0e30
        global_id = 0
        for v in range(0, vocab_size, BLOCK_SIZE):
            vocab_offset = v + tl.arange(0, BLOCK_SIZE)
            vocab_mask = vocab_offset < vocab_size

            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask, other=0.0,
            )
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask, other=0.0,
            )
            prob = tl.maximum(prefix_prob * target_prob - draft_prob, 0.0)

            q = tl.load(
                q_ptr + req_idx * vocab_size + vocab_offset,
                mask=vocab_mask, other=float("inf"),
            )
            q_safe = tl.where(q <= 0, 1e-10, q)
            score = tl.where(vocab_mask, prob / q_safe, -1.0e30)
            local_max, local_id = tl.max(score, axis=0, return_indices=True)
            if local_max > global_max:
                global_max = local_max
                global_id = v + local_id

        tl.store(output_token_ids_ptr + token_idx, global_id)


    @triton.jit(do_not_specialize=["max_spec_len"])
    def rejection_random_sample_block_verify_kernel(
        output_token_ids_ptr,          # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,       # [batch_size]
        draft_token_ids_ptr,           # [num_tokens]
        draft_probs_ptr,               # [num_tokens, vocab_size]
        target_probs_ptr,              # [num_tokens, vocab_size]
        bonus_token_ids_ptr,           # [batch_size]
        recovered_token_ids_ptr,       # [num_tokens]
        uniform_probs_ptr,             # [num_tokens]
        is_greedy_ptr,                 # [batch_size]
        max_spec_len,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Block-verify acceptance rule — one program per request.

        Emits accepted draft tokens, recovered token at first rejection, or
        bonus token if every draft is accepted.
        """
        req_idx = tl.program_id(0)
        is_greedy = tl.load(is_greedy_ptr + req_idx)
        if is_greedy:
            return

        start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
        num_draft_tokens = end_idx - start_idx

        if num_draft_tokens == 0:
            bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1), bonus_token_id)
            return

        # ─── [Genesis P71 FIX 1] Load SHARED u once per request ─────────────
        # PR #40819 loaded uniform_prob inside the loop (per-position).
        # Sun 2024 requires ONE shared u for the whole block.
        uniform_prob_shared = tl.load(uniform_probs_ptr + start_idx)
        # ────────────────────────────────────────────────────────────────────

        accepted_len = 0
        prefix_prob = 1.0
        for pos in range(num_draft_tokens):
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + draft_token_id
            )
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + draft_token_id
            )

            if draft_prob > 0:
                prefix_prob = min(prefix_prob * target_prob / draft_prob, 1.0)
            else:
                prefix_prob = 0.0

            if pos == num_draft_tokens - 1:
                h_block = prefix_prob
            else:
                next_token_idx = token_idx + 1
                residual_mass = 0.0
                for v in range(0, vocab_size, BLOCK_SIZE):
                    vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                    vocab_mask = vocab_offset < vocab_size
                    next_draft = tl.load(
                        draft_probs_ptr + next_token_idx * vocab_size + vocab_offset,
                        mask=vocab_mask, other=0.0,
                    )
                    next_target = tl.load(
                        target_probs_ptr + next_token_idx * vocab_size + vocab_offset,
                        mask=vocab_mask, other=0.0,
                    )
                    local = tl.maximum(prefix_prob * next_target - next_draft, 0.0)
                    residual_mass += tl.sum(local, axis=0)
                denom = residual_mass + 1.0 - prefix_prob
                # ─── [Genesis P71 FIX 2] denom==0 → ACCEPT (1.0) ─────────
                # PR returned 0.0 here, rejecting perfect drafts. Wrong.
                h_block = residual_mass / denom if denom > 0 else 1.0
                # ────────────────────────────────────────────────────────

            # Use shared u (FIX 1)
            if uniform_prob_shared <= h_block:
                accepted_len = pos + 1

        for pos in range(accepted_len):
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                draft_token_id,
            )

        if accepted_len == num_draft_tokens:
            bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                bonus_token_id,
            )
        else:
            recovered_id = tl.load(recovered_token_ids_ptr + start_idx + accepted_len)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + accepted_len,
                recovered_id,
            )


# ════════════════════════════════════════════════════════════════════════
# Public entry point used by the wiring patch.
# ════════════════════════════════════════════════════════════════════════

def call_block_verify_sample(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor,
    num_draft_tokens: list[int],
    generators: dict,
    max_spec_len: int,
    vocab_size: int,
    use_pytorch: bool = False,
) -> torch.Tensor:
    """Block-verify rejection sampling entry point.

    Wraps the two-phase sampler:
      1. sample_recovered_tokens_blockwise (PyTorch or Triton)
      2. rejection_random_sample_block_verify (PyTorch or Triton)

    On any exception, the wiring patch's outer try/except falls back to the
    upstream per-token rejection sampler — no engine impact.
    """
    # ── A4 audit (PN13 follow-up) — defensive preconditions ────────────
    # Fail loudly with informative messages on shape / device / dtype
    # mismatches rather than letting Triton kernel raise cryptic errors.
    if output_token_ids.dim() < 1:
        raise ValueError(
            f"output_token_ids must be at least 1-D, got shape "
            f"{tuple(output_token_ids.shape)}"
        )
    if cu_num_draft_tokens.shape[0] != output_token_ids.shape[0] + 1:
        raise ValueError(
            f"cu_num_draft_tokens length {cu_num_draft_tokens.shape[0]} "
            f"must equal batch_size + 1 = {output_token_ids.shape[0] + 1}"
        )
    if draft_probs.shape != target_probs.shape:
        raise ValueError(
            f"draft_probs shape {tuple(draft_probs.shape)} must match "
            f"target_probs shape {tuple(target_probs.shape)}"
        )
    if draft_probs.shape[-1] != vocab_size:
        raise ValueError(
            f"draft_probs last dim {draft_probs.shape[-1]} must equal "
            f"vocab_size {vocab_size}"
        )
    if max_spec_len < 1:
        raise ValueError(f"max_spec_len must be >= 1, got {max_spec_len}")
    # All input tensors must be on the same device
    devs = {
        "output_token_ids": output_token_ids.device,
        "draft_probs": draft_probs.device,
        "target_probs": target_probs.device,
        "uniform_probs": uniform_probs.device,
    }
    if len(set(devs.values())) > 1:
        raise RuntimeError(
            f"All P71 input tensors must be on the same device. Got: "
            + ", ".join(f"{k}={v}" for k, v in devs.items())
        )
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    q = generate_rejection_q(
        batch_size, vocab_size, num_draft_tokens, generators, device,
    )
    recovered_token_ids = torch.empty_like(draft_token_ids)

    if use_pytorch or not _TRITON_OK:
        sample_recovered_tokens_blockwise_pytorch(
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
        )
        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
        )
    else:
        sample_recovered_tokens_block_verify_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
        )
        rejection_random_sample_block_verify_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
        )

    return output_token_ids


def is_active() -> bool:
    """Returns True if P71 is enabled via env."""
    import os
    return os.environ.get(
        "GENESIS_ENABLE_P71_BLOCK_VERIFY", ""
    ).strip().lower() in ("1", "true", "yes", "on")
