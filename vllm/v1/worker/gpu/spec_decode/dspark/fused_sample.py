# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused kernels for DSpark's sequential Markov sampling.

Per draft position, the eager loop runs an embedding lookup, the Markov bias
GEMM, a logits add, index/copy glue, the Gumbel kernel, an argmax reduction, a
gather and a scatter into ``draft_tokens``. The fused path keeps the bias GEMM
and replaces the rest with one kernel per position.

Each program of ``_markov_sample_kernel`` handles one ``(request, vocab block)``
pair. It forms the logits as ``base_logits[:, i] + bias`` does (fp32 add rounded
to the result dtype), writes them to the draft-logits cache and takes the
block's Gumbel-max winner with ``gumbel_noised_argmax``, as ``gumbel_sample``
does. The last program to finish a request (an arrival counter) reduces the
block winners with ``torch.argmax`` semantics, stores the draft token and copies
its Markov embedding row into the next position's GEMM input. The noise is a
hash of ``(seed, draft step, position, token)``, the step mixed into the seed
as ``gumbel_sample`` does for a cached column, so the fused loop draws exactly
the eager loop's tokens.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed

# gumbel_sample's block size, so block winners and tie-breaking agree.
_BLOCK_SIZE = 1024


@triton.jit
def _markov_embed_anchor_kernel(
    input_ids_ptr,
    query_per_req,
    weight_ptr,
    weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    BLOCK_R: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    token = tl.load(input_ids_ptr + req * query_per_req).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    row = tl.load(weight_ptr + token * weight_stride + r, mask=r < rank)
    tl.store(embed_ptr + req * embed_stride + r, row, mask=r < rank)


@triton.jit(do_not_specialize=["step"])
def _markov_sample_kernel(
    # [num_reqs, num_steps, vocab_size]
    base_logits_ptr,
    base_stride_0,
    base_stride_1,
    # [num_reqs, vocab_size]
    bias_ptr,
    bias_stride,
    step,
    num_steps,
    # [num_reqs * num_steps], ordered (req, step)
    idx_mapping_ptr,
    sample_pos_ptr,
    temp_ptr,
    seeds_ptr,
    # [max_num_reqs, num_steps, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    # [max_num_reqs, num_steps]
    draft_tokens_ptr,
    draft_tokens_stride,
    # [vocab_size, rank] and [num_reqs, rank]
    embed_weight_ptr,
    embed_weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    # [max_num_reqs, BLOCK_P] scratch and [max_num_reqs] arrival counters
    partial_max_ptr,
    partial_idx_ptr,
    counter_ptr,
    vocab_size,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_FP64: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    base = tl.load(
        base_logits_ptr + req * base_stride_0 + step * base_stride_1 + block,
        mask=mask,
        other=float("-inf"),
    )
    bias = tl.load(bias_ptr + req * bias_stride + block, mask=mask, other=0.0)
    # The eager add's rounding: fp32 sum, rounded once to the cache dtype.
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(
        logits_cache_ptr.dtype.element_ty
    )

    row = req * num_steps + step
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    tl.store(
        logits_cache_ptr
        + req_state_idx * logits_cache_stride_0
        + step * logits_cache_stride_1
        + block,
        logits,
        mask=mask & is_valid_req,
    )
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    seed = spec_step_seed(seed, step)
    # sample_pos is the predicted token's position; the draw is keyed by the
    # position before it.
    pos = tl.load(sample_pos_ptr + row) - 1
    value, idx = gumbel_noised_argmax(
        logits.to(tl.float32),
        block,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=True,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=True,
    )
    tl.store(partial_max_ptr + req * BLOCK_P + block_idx, value)
    tl.store(partial_idx_ptr + req * BLOCK_P + block_idx, block_idx * BLOCK_SIZE + idx)

    # The last block of this request to arrive finishes the reduction. The
    # barrier orders every thread's partial stores before the release.
    tl.debug_barrier()
    arrival = tl.atomic_add(counter_ptr + req, 1, sem="acq_rel", scope="gpu")
    if arrival != num_blocks - 1:
        return

    p = tl.arange(0, BLOCK_P)
    p_mask = p < num_blocks
    values = tl.load(
        partial_max_ptr + req * BLOCK_P + p,
        mask=p_mask,
        other=float("-inf"),
        volatile=True,
    )
    # torch.argmax semantics: the first NaN if any, else the first maximum.
    is_nan = values != values
    first_nan = tl.min(tl.where(is_nan, p, BLOCK_P))
    best = tl.max(tl.where(is_nan, float("-inf"), values))
    first_max = tl.min(tl.where(p_mask & (values == best), p, BLOCK_P))
    winner = tl.where(first_nan < BLOCK_P, first_nan, first_max)
    token = tl.load(partial_idx_ptr + req * BLOCK_P + winner, volatile=True).to(
        tl.int64
    )
    tl.store(draft_tokens_ptr + req * draft_tokens_stride + step, token)
    if HAS_NEXT:
        r = tl.arange(0, BLOCK_R)
        embed_row = tl.load(
            embed_weight_ptr + token * embed_weight_stride + r, mask=r < rank
        )
        tl.store(embed_ptr + req * embed_stride + r, embed_row, mask=r < rank)
    tl.atomic_xchg(counter_ptr + req, 0, sem="acq_rel", scope="gpu")


class DSparkFusedMarkovSampler:
    """Persistent buffers and launchers of the fused Markov sampling loop.

    ``embed_weight`` is the ``[vocab, rank]`` table ``markov_embed`` looks up.
    """

    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        embed_weight: torch.Tensor,
        use_fp64: bool,
    ):
        assert embed_weight.dim() == 2 and embed_weight.stride(1) == 1
        device = embed_weight.device
        self.embed_weight = embed_weight
        self.vocab_size = vocab_size
        self.use_fp64 = use_fp64
        self.num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
        self.block_p = triton.next_power_of_2(self.num_blocks)
        self.rank = embed_weight.shape[1]
        self.block_r = triton.next_power_of_2(self.rank)
        # Input of the Markov bias GEMM; rewritten by every sampling step.
        self.markov_embed = torch.zeros(
            max_num_reqs, self.rank, dtype=embed_weight.dtype, device=device
        )
        self.partial_max = torch.empty(
            max_num_reqs,
            self.block_p,
            dtype=torch.float64 if use_fp64 else torch.float32,
            device=device,
        )
        self.partial_idx = torch.empty(
            max_num_reqs, self.block_p, dtype=torch.int32, device=device
        )
        # Zero between launches: the last block of each request resets it.
        self.counter = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def embed_anchor(
        self, input_ids: torch.Tensor, query_per_req: int, num_reqs: int
    ) -> torch.Tensor:
        """Markov embedding of each request's anchor ``input_ids[r * q]``."""
        embed = self.markov_embed[:num_reqs]
        _markov_embed_anchor_kernel[(num_reqs,)](
            input_ids,
            query_per_req,
            self.embed_weight,
            self.embed_weight.stride(0),
            embed,
            embed.stride(0),
            self.rank,
            BLOCK_R=self.block_r,
        )
        return embed

    def sample_step(
        self,
        base_logits: torch.Tensor,  # [num_reqs, num_steps, vocab_size]
        bias: torch.Tensor,  # [num_reqs, vocab_size]
        step: int,
        idx_mapping: torch.Tensor,  # [num_reqs * num_steps]
        sample_pos: torch.Tensor,  # [num_reqs * num_steps]
        temperature: torch.Tensor,  # [max_num_reqs]
        seeds: torch.Tensor,  # [max_num_reqs]
        logits_cache: torch.Tensor,  # [max_num_reqs, num_steps, vocab_size]
        draft_tokens: torch.Tensor,  # [max_num_reqs, num_steps]
    ) -> None:
        """Sample position ``step`` into ``draft_tokens[:, step]`` and, unless
        it is the last position, leave the sampled tokens' Markov embeddings in
        ``self.markov_embed`` for the next bias GEMM."""
        num_reqs, num_steps, vocab_size = base_logits.shape
        assert vocab_size == self.vocab_size
        assert base_logits.stride(2) == 1 and bias.stride(1) == 1
        assert logits_cache.size(-1) >= vocab_size and logits_cache.stride(2) == 1
        # The eager path caches base_logits + bias in the cache's dtype.
        assert torch.promote_types(base_logits.dtype, bias.dtype) == (
            logits_cache.dtype
        )
        embed = self.markov_embed[:num_reqs]
        _markov_sample_kernel[(num_reqs, self.num_blocks)](
            base_logits,
            base_logits.stride(0),
            base_logits.stride(1),
            bias,
            bias.stride(0),
            step,
            num_steps,
            idx_mapping,
            sample_pos,
            temperature,
            seeds,
            logits_cache,
            logits_cache.stride(0),
            logits_cache.stride(1),
            draft_tokens,
            draft_tokens.stride(0),
            self.embed_weight,
            self.embed_weight.stride(0),
            embed,
            embed.stride(0),
            self.rank,
            self.partial_max,
            self.partial_idx,
            self.counter,
            vocab_size,
            self.num_blocks,
            BLOCK_SIZE=_BLOCK_SIZE,
            BLOCK_P=self.block_p,
            BLOCK_R=self.block_r,
            USE_FP64=self.use_fp64,
            HAS_NEXT=step + 1 < num_steps,
        )
