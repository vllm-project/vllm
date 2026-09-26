# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalents of the V2 sampler's Gumbel kernels."""

import torch

# splitmix64 constants as signed int64: torch has no uint64 arithmetic, and
# int64 multiply wraps, which is what the mixer wants.
_C1 = 0x9E3779B97F4A7C15 - (1 << 64)
_C2 = 0xBF58476D1CE4E5B9 - (1 << 64)
_C3 = 0x94D049BB133111EB - (1 << 64)
# Must stay in step with gpu/sample/gumbel.py, which offsets a draft's noise so
# it cannot share a vector with the residual it is resampled from.
_DRAFT_NOISE_SALT = 1 << 30


def _lsr(z: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift; torch's int64 shift propagates the sign bit."""
    return (z >> n) & ((1 << (64 - n)) - 1)


def _mix64(z: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer, so neighbouring inputs land far apart."""
    z = (z ^ _lsr(z, 30)) * _C2
    z = (z ^ _lsr(z, 27)) * _C3
    return z ^ _lsr(z, 31)


def _uniform(z: torch.Tensor, use_fp64: bool) -> torch.Tensor:
    """Mixed bits to a uniform in (0, 1).

    The half-step offset keeps the draw off both ends, which is what the
    Triton path's clamp against a zero draw is for.
    """
    if use_fp64:
        bits = _lsr(z, 11).to(torch.float64)
        return (bits + 0.5) * (1.0 / 9007199254740992.0)
    bits = _lsr(z, 32).to(torch.float32)
    return (bits + 0.5) * (1.0 / 4294967296.0)


def _add_gumbel(work, row_seed, vocab_size, use_fp64):
    """Add Gumbel noise keyed by (row_seed, token_id), as the Triton path does.

    Noise is a hash rather than a generator draw, so it stays addressable -- the
    same token in the same stream always gets the same noise, which is what lets
    a draft and its verification agree -- while remaining parallelisable.
    """
    keys = torch.arange(vocab_size, device=work.device).unsqueeze(0)
    u = _uniform(_mix64(row_seed.unsqueeze(1) * _C1 + keys), use_fp64)
    # log1p keeps the winning tail at u -> 0, where fp32 resolves it.
    return work - torch.log(-torch.log1p(-u))


def _blockwise_max(work, num_blocks, block_size, vocab_size):
    """Per-block max, reducing the tail separately to avoid a padded copy."""
    num_tokens = work.shape[0]
    full = (vocab_size // block_size) * block_size
    head = work[:, :full].reshape(num_tokens, full // block_size, block_size)
    values, idx = head.max(dim=-1)
    if full != vocab_size:
        tail_v, tail_i = work[:, full:].max(dim=-1, keepdim=True)
        values = torch.cat((values, tail_v), dim=-1)
        idx = torch.cat((idx, tail_i), dim=-1)
    offsets = torch.arange(num_blocks, device=work.device) * block_size
    return values, idx + offsets


def _sample_and_reduce(work, row_seed, vocab_size, num_blocks, block_size, use_fp64):
    """Noise and reduction together, so nothing [tokens, vocab] is materialised.

    Eager torch cannot fuse these and pays for a temporary per step; compiled,
    this becomes the single parallel pass over the logits that Triton runs.
    """
    work = _add_gumbel(work, row_seed, vocab_size, use_fp64)
    return _blockwise_max(work, num_blocks, block_size, vocab_size)


# dynamic=True: one graph serves every batch size, so a changing token count
# does not recompile.
_sample_and_reduce_compiled = torch.compile(_sample_and_reduce, dynamic=True)


@torch.inference_mode()
def warm_up(max_num_reqs: int, vocab_size: int, device: torch.device) -> None:
    """Compile the sampler graph at startup rather than mid-run.

    Driven through the real entry point so the graph's constants match the ones
    a request will bring. A single row compiles separately because dynamo always
    specialises a size-1 dimension, and a draining batch would otherwise pay for
    it mid-run.
    """
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    for num_reqs in dict.fromkeys((max_num_reqs, 1)):
        ones = torch.ones(num_reqs, device=device)
        zeros = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        gumbel_sample(
            torch.zeros(num_reqs, vocab_size, device=device),
            torch.arange(num_reqs, dtype=torch.int32, device=device),
            ones,  # temperature 1 takes the sampled path, not the greedy one
            zeros,
            zeros,
            apply_temperature=True,
            is_drafting=False,
        )


def apply_temperature(
    grid,
    logits,
    logits_stride,
    expanded_idx_mapping,
    temperature,
    vocab_size,
    **kwargs,
):
    num_tokens = grid[0]
    temp = temperature[expanded_idx_mapping[:num_tokens].long()].to(torch.float32)
    # The kernel returns early on both of these rather than dividing.
    scale = torch.where((temp == 0.0) | (temp == 1.0), torch.ones_like(temp), temp)
    rows = logits[:num_tokens, :vocab_size]
    rows.div_(scale.unsqueeze(1))


def gumbel_sample(
    grid,
    local_argmax,
    local_argmax_stride,
    local_max,
    local_max_stride,
    logits_cache,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col,
    logits,
    logits_stride,
    expanded_idx_mapping,
    seeds,
    pos,
    temperature,
    vocab_size,
    BLOCK_SIZE,
    IS_DRAFTING,
    APPLY_TEMPERATURE,
    USE_FP64,
    PER_TOKEN_COL,
    **kwargs,
):
    num_tokens, num_blocks = grid
    dtype = torch.float64 if USE_FP64 else torch.float32

    req = expanded_idx_mapping[:num_tokens].long()
    valid = req >= 0
    safe_req = req.clamp_min(0)
    temp = temperature[safe_req].to(torch.float32).masked_fill(~valid, 0.0)
    work = logits[:num_tokens, :vocab_size].to(torch.float32)

    if logits_cache is not None:
        col = logits_cache_col[:num_tokens] if PER_TOKEN_COL else logits_cache_col
        col = col.expand(num_tokens).long()
        rows = valid.nonzero(as_tuple=True)[0]
        # Cached before temperature: dividing first gives a value the cache
        # dtype generally cannot hold, and the rejection sampler divides on load.
        logits_cache[safe_req[rows], col[rows], :vocab_size] = work[rows].to(
            logits_cache.dtype
        )

    if APPLY_TEMPERATURE:
        scale = torch.where(temp != 0.0, temp, torch.ones_like(temp))
        work = work / scale.unsqueeze(1)
    work = work.to(dtype)

    # Temperature 0 is greedy, so those rows take the plain argmax below. An
    # invalid request was forced to 0 above and lands here too.
    salt = _DRAFT_NOISE_SALT if IS_DRAFTING else 0
    row_seed = _mix64(seeds[safe_req].long() * _C1 + pos[:num_tokens].long() + salt)
    noisy = temp != 0.0
    if bool(noisy.all()):
        values, idx = _sample_and_reduce_compiled(
            work, row_seed, vocab_size, num_blocks, BLOCK_SIZE, USE_FP64
        )
    else:
        if bool(noisy.any()):
            rows = noisy.nonzero(as_tuple=True)[0]
            work[rows] = _add_gumbel(work[rows], row_seed[rows], vocab_size, USE_FP64)
        values, idx = _blockwise_max(work, num_blocks, BLOCK_SIZE, vocab_size)
    local_argmax[:num_tokens, :num_blocks] = idx
    local_max[:num_tokens, :num_blocks] = values
