# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for GDN ``mamba_cache_mode='all'`` prefix caching.

The per-block SSM checkpoint scatter mirrors Mamba2 (mamba_mixer2.py:872-949) but
adapts to FLA's ``chunk_gated_delta_rule`` intermediate states:

  Mamba2 ``varlen_states[c]`` = state at the END of chunk c.
  FLA    ``h[c]``             = state at the START of chunk c (== end of chunk c-1).

So the strided gather index is shifted by +1 chunk vs Mamba2, and the final
(possibly partial) block is written from the sequence's final recurrent state
(``last_recurrent_state``) rather than from ``h``.
"""

import torch


def gdn_scatter_block_checkpoints(
    ssm_state: torch.Tensor,
    inter_states: torch.Tensor,
    final_states: torch.Tensor,
    block_table_p: torch.Tensor,
    block_idx_first_scheduled_token_p: torch.Tensor,
    block_idx_last_scheduled_token_p: torch.Tensor,
    num_computed_tokens_p: torch.Tensor,
    first_chunk_p: torch.Tensor,
    mamba_block_size: int,
    chunk_size: int,
) -> None:
    """Write per-block SSM checkpoints into ``ssm_state`` (in place) for all-mode.

    Args:
        ssm_state: [num_blocks, ...] recurrent-state cache (written in place).
        inter_states: FLA ``h`` squeezed to [NT, ...] — h[c] is the state BEFORE
            global chunk c (i.e. after the first c chunks).
        final_states: [num_prefills, ...] — each sequence's final recurrent state
            (``last_recurrent_state`` from chunk_gated_delta_rule).
        block_table_p: [num_prefills, max_blocks] — per-prefill-seq block ids.
        block_idx_first_scheduled_token_p: [num_prefills] — first scheduled block idx.
        block_idx_last_scheduled_token_p: [num_prefills] — last scheduled block idx.
        num_computed_tokens_p: [num_prefills] — already-computed tokens per seq.
        first_chunk_p: [num_prefills] — global index of each seq's first chunk
            (= chunk_offsets[seq]).
        mamba_block_size: cache block size in tokens.
        chunk_size: FLA chunk size in tokens (FLA_CHUNK_SIZE).
    """
    num_prefills = block_idx_last_scheduled_token_p.shape[0]

    # Final (possibly partial) block for every sequence == the seq's final state.
    last_blocks = block_table_p.gather(
        1, block_idx_last_scheduled_token_p.long().unsqueeze(1)
    ).squeeze(1)
    ssm_state[last_blocks] = final_states.to(ssm_state.dtype)

    # Interior full blocks [first, last) == per-block-boundary states from h.
    # Interior block j ends after (j+1)*mamba_block_size sequence tokens; that boundary
    # is ((j+1)*mamba_block_size - num_computed) *scheduled* tokens in, i.e. FLA chunk
    #   k = ((j+1)*mamba_block_size - num_computed) // chunk_size
    # and h[c] is the state at the START of chunk c, so the checkpoint is the global
    #   h index = first_chunk[s] + k.
    # This is exact when block boundaries coincide with chunk boundaries, i.e.
    # num_computed % chunk_size == 0 (guaranteed by the all-mode chunk-aligned
    # prefill split, which clips bites to chunk_size multiples). We index explicitly per
    # boundary (arange) and clamp into h's valid range: this replaces the previous
    # open-ended strided slice `inter_states[fla_first : fla_first+n*stride : stride]`,
    # which silently returned fewer than n rows once fla_first ran past the last
    # sequence's chunk range in h -> the [0]-vs-[n] shape-mismatch crash on long
    # cache-hit prompts. Sequences whose num_computed is not chunk-aligned are
    # skipped below (never write an approximate interior checkpoint into APC).
    nt = inter_states.shape[0]
    # An interior-block SSM checkpoint is only exact when the block boundary
    # coincides with an FLA chunk boundary, i.e. num_computed_tokens is a
    # multiple of chunk_size. In a valid all-mode config this always holds
    # (config validation requires mamba_block_size % FLA_CHUNK_SIZE == 0, and the
    # all-mode chunk-aligned prefill split keeps num_computed a multiple of
    # chunk_size). Writing an interior checkpoint at a non-boundary offset
    # would poison the content-hash-addressed prefix cache, so we SKIP the
    # interior scatter for any sequence whose num_computed is not chunk-aligned
    # rather than write a wrong ("nearest chunk") checkpoint. This also makes the
    # startup CUDA-graph memory dummy run safe: it feeds synthetic, unaligned
    # num_computed to throwaway state and must not crash. The final (possibly
    # partial) block is written from ``final_states`` above and is exact
    # regardless of alignment, so it is always kept.
    for s in range(num_prefills):
        first = int(block_idx_first_scheduled_token_p[s])
        last = int(block_idx_last_scheduled_token_p[s])
        n = last - first
        if n <= 0:
            continue
        ncomp = int(num_computed_tokens_p[s])
        if ncomp % chunk_size != 0:
            # Not chunk-aligned: cannot map interior block boundaries to exact FLA
            # chunk states. Skip (never write an approximate checkpoint into APC).
            continue
        cache_blocks = block_table_p[s, first:last]
        fc = int(first_chunk_p[s])
        # This sequence's chunks occupy h[fc : seq_hi_excl); clamp the gather to that
        # per-sequence range (not just the global [0, nt)) so a would-be overshoot can
        # never bleed into an adjacent sequence's chunks. Exact indices already fall in
        # range when block boundaries are chunk-aligned; this is defense-in-depth.
        seq_hi = (
            int(first_chunk_p[s + 1]) - 1
            if (s + 1) < first_chunk_p.shape[0]
            else nt - 1
        )
        j = torch.arange(first, last, device=cache_blocks.device)
        k = ((j + 1) * mamba_block_size - ncomp) // chunk_size
        chunk_idx = (fc + k).clamp_(fc, seq_hi)
        from_where = inter_states[chunk_idx]
        ssm_state[cache_blocks] = from_where.to(ssm_state.dtype)
