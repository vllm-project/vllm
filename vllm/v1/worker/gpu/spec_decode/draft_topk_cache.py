# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed


class SparseDraftCache:
    """Sparse draft cache for `speculative_config.draft_top_k`: per draft
    slot, the K (token id, log q') pairs the draft token was proposed from,
    instead of a dense [max_num_reqs, num_steps, vocab_size] cache of raw
    logits (`DraftLogitsCache`).

    q' is the drafter's distribution restricted to its top-K tokens
    (renormalized, temperature applied). Any proposal fixed before sampling
    is exactly lossless, so verifying against q' instead of the full
    distribution does not change the output.

    Indexed by persistent request-state index, like `DraftLogitsCache`'s
    replicated (non-sharded) mode: every TP rank holds every request's K
    pairs. `DraftLogitsCache.row_mapping`'s batch-sharded storage (each rank
    keeping only the rows it owns) is not implemented here yet; draft_top_k's
    own savings already come from exchanging only K pairs per row, rather
    than the full vocabulary, when the cache is written.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_steps: int,
        top_k: int,
        *,
        device: torch.device | str,
    ):
        # Row `max_num_reqs` absorbs padded / invalid draft rows (dummy runs,
        # CUDA-graph capture), mirroring DraftTokensHandler's sink row.
        shape = (max_num_reqs + 1, num_steps, top_k)
        self.ids = torch.zeros(shape, dtype=torch.int64, device=device)
        # Rows never written (padding) keep -inf everywhere, so they can
        # never match a sampled token id in the accept test.
        self.logq = torch.full(shape, float("-inf"), dtype=torch.float32, device=device)
        self.max_num_reqs = max_num_reqs
        self.top_k = top_k


@triton.jit(do_not_specialize=["num_tokens"])
def _topk_draft_sample_kernel(
    sampled_ptr,
    # [num_rows, num_steps, K]
    cache_ids_ptr,
    cache_ids_stride_0,
    cache_ids_stride_1,
    cache_logq_ptr,
    cache_logq_stride_0,
    cache_logq_stride_1,
    # [num_tokens, K]
    ids_ptr,
    ids_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    draft_step_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    sink_row,
    K,
    num_tokens,
    BLOCK_K: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """One program per draft row: samples the draft token from q' (the K-entry
    truncated, temperature-scaled distribution) via the same Gumbel-max
    mechanism as the dense path's `_gumbel_sample_kernel`, and writes the K
    (id, log q') pairs into the sparse cache — all in one kernel, instead of
    gumbel_sample + two index_put_ + log_softmax as separate launches.

    `ids` double as the noise key (so a token draws the same noise wherever it
    appears, matching `gumbel_block_argmax`'s use of the global vocab id), and
    the seed is step-mixed via `spec_step_seed` exactly as the dense path does
    when writing a cache column, which avoids the "revisited position" bug
    `test_dspark_revisited_position_uses_step_seed` guards against: without
    it, a request whose draft position repeats across speculative steps would
    draw the same noise twice.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, BLOCK_K)
    mask = block < K

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid, other=0)
    pos = tl.load(pos_ptr + token_idx)
    step = tl.load(draft_step_ptr + token_idx)
    seed = spec_step_seed(seed, step)

    ids = tl.load(ids_ptr + token_idx * ids_stride + block, mask=mask, other=0)
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block, mask=mask, other=float("-inf")
    ).to(tl.float32)

    _, local_idx = gumbel_noised_argmax(
        logits,
        ids,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=IS_DRAFTING,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=True,
    )
    sampled_id = tl.load(ids_ptr + token_idx * ids_stride + local_idx)
    tl.store(sampled_ptr + token_idx, sampled_id)

    # log q' = log_softmax(logits / T) over the K entries, for the verify-side
    # accept test. Greedy (temp == 0) rows: never read back (the accept test
    # checks the target argmax for those instead), so only needs to avoid a
    # div-by-zero here, not be meaningful.
    safe_temp = tl.maximum(temp, 1e-5)
    scaled = tl.where(mask, logits / safe_temp, float("-inf"))
    m = tl.max(scaled, axis=0)
    denom = tl.sum(tl.where(mask, tl.exp(scaled - m), 0.0), axis=0)
    logqp = scaled - (m + tl.log(denom))

    row = tl.where(is_valid, req_state_idx, sink_row)
    tl.store(
        cache_ids_ptr + row * cache_ids_stride_0 + step * cache_ids_stride_1 + block,
        ids,
        mask=mask,
    )
    tl.store(
        cache_logq_ptr + row * cache_logq_stride_0 + step * cache_logq_stride_1 + block,
        logqp,
        mask=mask,
    )


def sample_draft_topk_fused(
    cache: SparseDraftCache,
    ids: torch.Tensor,
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    draft_step: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Fused draft-side sample + sparse-cache write. `ids`/`logits` are the
    top-K (token id, logit) pairs from `get_top_k_tokens`, shape
    [num_tokens, K]; the logits need not be normalized (q' is renormalized
    over the K entries here, so any per-row shift cancels). `draft_step` is
    each row's draft slot, an int tensor of shape [num_tokens]."""
    num_tokens, K = ids.shape
    assert logits.shape == (num_tokens, K)
    ids, logits = ids.contiguous(), logits.contiguous()
    sampled = torch.empty(num_tokens, dtype=torch.int64, device=ids.device)
    BLOCK_K = triton.next_power_of_2(K)
    _topk_draft_sample_kernel[(num_tokens,)](
        sampled,
        cache.ids,
        cache.ids.stride(0),
        cache.ids.stride(1),
        cache.logq,
        cache.logq.stride(0),
        cache.logq.stride(1),
        ids,
        ids.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        draft_step,
        temperature,
        seeds,
        positions,
        cache.max_num_reqs,
        K,
        num_tokens,
        BLOCK_K=BLOCK_K,
        IS_DRAFTING=True,
        USE_FP64=use_fp64,
    )
    return sampled
