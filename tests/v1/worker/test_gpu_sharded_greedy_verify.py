# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded greedy verification must match native greedy rejection sampling.

TP ranks are emulated in one process: each vocabulary shard is reduced to
statistics, the per-rank statistics are concatenated in rank order (what
``all_gather(dim=0)`` produces), and the result is compared with
``rejection_sample`` on the full logits.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.sharded_greedy_verifier import (
    local_greedy_stats,
    verify_gathered_greedy_stats,
)

if not torch.cuda.is_available():
    pytest.skip("GPU required for Triton kernels", allow_module_level=True)

DEVICE = "cuda"
# Columns past the vocabulary on the last shard (the LM head pads its
# partitions). Filled with a huge value that would win any unmasked argmax.
PAD_VALUE = 1e30


def _make_batch(
    num_reqs: int,
    num_speculative_steps: int,
    full_logits_fn,
    vocab_size: int,
    accept_prob: float,
    seed: int,
):
    """Build a decode batch whose drafts often match the target argmax.

    Request i has 1 + d_i logits rows with d_i in [0, K]. Some requests are
    still prefilling or carry placeholder drafts, which both paths must
    reject identically.
    """
    gen = torch.Generator().manual_seed(seed)
    k = num_speculative_steps
    num_drafts = torch.randint(0, k + 1, (num_reqs,), generator=gen)
    # Keep at least one draft so the batch takes the verification path.
    num_drafts[:3] = k
    rows_per_req = num_drafts + 1
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu_num_logits[1:] = torch.cumsum(rows_per_req, 0)
    num_logits = int(cu_num_logits[-1])

    full_logits = full_logits_fn(num_logits, gen)
    target_argmax = full_logits.float().nan_to_num(nan=float("-inf")).argmax(-1)

    max_num_states = 2 * num_reqs + 3
    idx_mapping = torch.randperm(max_num_states, generator=gen)[:num_reqs]
    # Leave a gap of two tokens between requests so that the token and logits
    # indexing differ.
    num_tokens = num_logits + 2 * num_reqs
    input_ids = torch.randint(0, vocab_size, (num_tokens,), generator=gen)
    positions = torch.zeros(num_tokens, dtype=torch.int64)
    logits_indices = torch.empty(num_logits, dtype=torch.int64)
    expanded_idx_mapping = torch.empty(num_logits, dtype=torch.int64)
    expanded_local_pos = torch.empty(num_logits, dtype=torch.int32)
    seq_lens = torch.empty(num_reqs, dtype=torch.int32)
    prefill_len = torch.zeros(max_num_states, dtype=torch.int32)
    seeds = torch.randint(0, 2**31, (max_num_states,), generator=gen)

    for req in range(num_reqs):
        start, end = int(cu_num_logits[req]), int(cu_num_logits[req + 1])
        state = int(idx_mapping[req])
        context = int(torch.randint(16, 4096, (1,), generator=gen))
        # Request 1 is always prefilling and request 2 always has placeholder
        # drafts; request 0 is a regular decode.
        kind = [1.0, 0.0, 0.15][req] if req < 3 else torch.rand((), generator=gen)
        if kind < 0.1:
            # Still prefilling: nothing is sampled.
            prefill_len[state] = context + (end - start) + 5
        elif kind < 0.2:
            # First step after a remote prefill: drafts are placeholders.
            prefill_len[state] = context + 1
        else:
            prefill_len[state] = context // 2
        seq_lens[req] = context + (end - start)
        for row in range(start, end):
            token = row + 2 * req
            local = row - start
            logits_indices[row] = token
            expanded_idx_mapping[row] = state
            expanded_local_pos[row] = local
            positions[token] = context + local
            if local > 0 and torch.rand((), generator=gen).item() < accept_prob:
                input_ids[token] = target_argmax[row - 1]

    batch = SimpleNamespace(
        num_reqs=num_reqs,
        cu_num_logits=cu_num_logits.to(DEVICE),
        input_ids=input_ids.to(DEVICE),
        positions=positions.to(DEVICE),
        logits_indices=logits_indices.to(DEVICE),
        expanded_idx_mapping=expanded_idx_mapping.to(DEVICE),
        expanded_local_pos=expanded_local_pos.to(DEVICE),
        idx_mapping=idx_mapping.to(DEVICE),
        seq_lens=seq_lens.to(DEVICE),
    )
    return (
        batch,
        full_logits.to(DEVICE),
        prefill_len.to(DEVICE),
        seeds.to(DEVICE),
        max_num_states,
    )


def _native(batch, logits, prefill_len, seeds, max_num_states, k, rates):
    draft_sampled, pos = gather_draft_sampled(
        batch.input_ids,
        batch.positions,
        batch.logits_indices,
        batch.expanded_idx_mapping,
        batch.expanded_local_pos,
        prefill_len,
    )
    temperature = torch.zeros(max_num_states, dtype=torch.float32, device=DEVICE)
    sampled, num_sampled = rejection_sample(
        logits,
        None,
        draft_sampled,
        batch.cu_num_logits,
        pos,
        batch.idx_mapping,
        batch.expanded_idx_mapping,
        batch.expanded_local_pos,
        temperature,
        seeds,
        k,
        rates,
    )
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled,
        batch.seq_lens,
        batch.cu_num_logits,
        batch.idx_mapping,
        prefill_len,
    )
    return sampled, num_sampled, num_rejected


def _sharded(batch, logits, prefill_len, seeds, k, rates, tp_size, pad_multiple):
    num_logits, vocab_size = logits.shape
    padded = -(-vocab_size // pad_multiple) * pad_multiple
    width = -(-padded // tp_size)
    full = torch.full(
        (num_logits, width * tp_size), PAD_VALUE, dtype=logits.dtype, device=DEVICE
    )
    full[:, :vocab_size] = logits
    stats = [
        local_greedy_stats(
            full[:, rank * width : (rank + 1) * width].contiguous(),
            batch,
            rank * width,
            vocab_size,
            k,
        )
        for rank in range(tp_size)
    ]
    output = verify_gathered_greedy_stats(
        torch.cat(stats, dim=0),
        batch,
        seeds,
        prefill_len,
        vocab_size,
        k,
        rates,
        tp_size,
    )
    return output.sampled_token_ids, output.num_sampled, output.num_rejected


def _assert_same(native, sharded):
    sampled, num_sampled, num_rejected = native
    s_sampled, s_num_sampled, s_num_rejected = sharded
    torch.testing.assert_close(s_num_sampled, num_sampled, rtol=0, atol=0)
    torch.testing.assert_close(s_num_rejected, num_rejected, rtol=0, atol=0)
    # Only the first num_sampled tokens of each request are consumed.
    for req, n in enumerate(num_sampled.tolist()):
        assert torch.equal(s_sampled[req, :n], sampled[req, :n]), req


def _random_logits(dtype):
    def make(num_logits, vocab_size, gen):
        return (torch.randn(num_logits, vocab_size, generator=gen) * 4).to(dtype)

    return make


def _run(
    tp_size,
    vocab_size,
    k,
    logits_fn,
    *,
    num_reqs=24,
    accept_prob=0.7,
    synthetic=False,
    pad_multiple=64,
    seed=0,
):
    batch, logits, prefill_len, seeds, max_num_states = _make_batch(
        num_reqs,
        k,
        lambda n, gen: logits_fn(n, vocab_size, gen),
        vocab_size,
        accept_prob,
        seed,
    )
    rates = None
    if synthetic:
        rates = torch.linspace(0.9, 0.5, k, dtype=torch.float32, device=DEVICE)
    native = _native(batch, logits, prefill_len, seeds, max_num_states, k, rates)
    sharded = _sharded(
        batch, logits, prefill_len, seeds, k, rates, tp_size, pad_multiple
    )
    _assert_same(native, sharded)
    return native


@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
@pytest.mark.parametrize("vocab_size", [1000, 50257, 200064])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_matches_native_greedy(tp_size, vocab_size, k):
    _, num_sampled, num_rejected = _run(
        tp_size, vocab_size, k, _random_logits(torch.float32)
    )
    # The batch exercises acceptance, rejection and prefilling requests.
    assert num_sampled.max() > 1
    assert num_rejected.max() > 0
    assert (num_sampled == 0).any()


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("k", [1, 3])
def test_matches_native_synthetic_acceptance(tp_size, k):
    _run(tp_size, 200064, k, _random_logits(torch.float32), synthetic=True)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_matches_native_bf16_ties(tp_size):
    """bf16 logits on a coarse grid tie often, also across shard and block
    boundaries; both paths must pick the lowest token id."""

    def make(num_logits, vocab_size, gen):
        logits = torch.randint(0, 4, (num_logits, vocab_size), generator=gen)
        return logits.to(torch.bfloat16)

    for seed in range(4):
        _run(tp_size, 200064, 3, make, seed=seed)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_matches_native_with_planted_cross_shard_ties(tp_size):
    vocab_size = 200064
    width = -(-vocab_size // 64) * 64 // tp_size

    def make(num_logits, vocab, gen):
        logits = torch.randn(num_logits, vocab, generator=gen)
        for row in range(num_logits):
            # The same maximum just before and just after a shard boundary,
            # and inside a later block.
            boundary = width * (1 + row % (tp_size - 1))
            logits[row, [boundary - 1, boundary, vocab - 1]] = 100.0
        return logits

    _run(tp_size, vocab_size, 3, make)


@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
@pytest.mark.parametrize("vocab_size", [1000, 50257, 200064])
def test_nan_rows_yield_valid_token_ids(tp_size, vocab_size):
    """NaN logits have no defined argmax (the native path returns whatever
    tl.max reports), but the token must stay inside the vocabulary."""
    k = 3

    def make(num_logits, vocab, gen):
        logits = torch.randn(num_logits, vocab, generator=gen)
        logits[::3, vocab // 2] = float("nan")
        logits[1::5, :] = float("nan")
        return logits

    batch, logits, prefill_len, seeds, _ = _make_batch(
        24, k, lambda n, gen: make(n, vocab_size, gen), vocab_size, 0.7, 0
    )
    sampled, num_sampled, _ = _sharded(
        batch, logits, prefill_len, seeds, k, None, tp_size, 64
    )
    for req, n in enumerate(num_sampled.tolist()):
        tokens = sampled[req, :n]
        assert ((tokens >= 0) & (tokens < vocab_size)).all(), (req, tokens)


def test_all_accepted_and_all_rejected():
    _, num_sampled, _ = _run(4, 50257, 3, _random_logits(torch.float32), accept_prob=1)
    assert (num_sampled == 4).any()
    _, num_sampled, _ = _run(4, 50257, 3, _random_logits(torch.float32), accept_prob=0)
    assert num_sampled.max() <= 1


def test_padding_columns_are_ignored():
    """An unaligned pad multiple leaves every rank but the last with a full
    shard and the last with padding beyond the vocabulary."""
    _run(4, 50257, 3, _random_logits(torch.float32), pad_multiple=1000)
