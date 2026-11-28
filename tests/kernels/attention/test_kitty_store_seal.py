# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `vllm/v1/attention/ops/kitty/store_seal.py`."""

import numpy as np
import pytest
import torch

from vllm.v1.attention.ops.kitty.store_seal import (
    _block_ranges,
    kitty_boost_ranks,
    kitty_gather_blocks,
    kitty_write_residual,
    plan_kitty_step,
)
from vllm.v1.attention.ops.kitty.tiles import (
    K_TAIL_ROW,
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_ROWS,
    V_TAIL_ROW,
)

PAGE = KITTY_PAGE_SIZE


# _block_ranges


@pytest.mark.parametrize(
    "starts, ends, expected",
    [
        ([], [], []),  # no requests
        ([1], [1], []),  # empty range
        ([3], [1], []),  # reversed range counts as empty
        ([1], [4], [[0, 1], [0, 2], [0, 3]]),
        ([2, 1], [3, 3], [[0, 2], [1, 1], [1, 2]]),  # by request, then block
        ([1, 4, 5], [2, 4, 7], [[0, 1], [2, 5], [2, 6]]),  # request 1 has none
    ],
)
def test_block_ranges_known_values(starts, ends, expected):
    got = _block_ranges(np.array(starts, np.int64), np.array(ends, np.int64))
    assert got.tolist() == expected


def test_block_ranges_no_jobs_is_still_two_columns():
    """`kitty_seal` reads `jobs.shape[0]` and the kernels read jobs as pairs."""
    got = _block_ranges(np.array([1, 2]), np.array([1, 2]))
    assert got.shape == (0, 2)
    assert got.dtype == np.int64


# plan_kitty_step


def plan_one(old: int, new: int):
    """Plan a step that grows one request's cache from `old` to `new` tokens."""
    return plan_kitty_step(
        np.array([0, new - old], np.int32), np.array([new], np.int32)
    )


def written_rows(writes: np.ndarray, old: int) -> dict[int, int]:
    """{position: residual row} for a one-request plan's K or V writes."""
    tok, _, row = writes
    return dict(zip((old + tok).tolist(), row.tolist()))


def test_plan_old_lens_are_cache_lengths_before_the_step():
    plan = plan_kitty_step(np.array([0, 1, 601, 611]), np.array([384, 600, 20]))
    assert plan.old_lens.tolist() == [383, 0, 10]


# (old length, new length, K blocks sealed, V blocks sealed)
SEAL_JOBS = [
    (0, 20, [], []),  # short prompt stays in the sink
    (0, 255, [], []),  # block 1 one token short of full
    (255, 256, [1], []),  # decode fills block 1
    (383, 384, [2], [1]),  # decode fills block 2, so V seals block 1
    (384, 385, [], []),  # decode starts block 3
    (200, 300, [1], []),  # prefill chunk crossing one page boundary
    (0, 600, [1, 2, 3], [1, 2]),  # one-step prefill across several pages
    (256, 512, [2, 3], [1, 2]),
]


@pytest.mark.parametrize("old, new, k_blocks, v_blocks", SEAL_JOBS)
def test_plan_seal_jobs(old, new, k_blocks, v_blocks):
    plan = plan_one(old, new)
    assert plan.k_jobs.tolist() == [[0, b] for b in k_blocks]
    assert plan.v_jobs.tolist() == [[0, b] for b in v_blocks]


@pytest.mark.parametrize(
    "chunks",
    [
        [1000],  # one-step prefill
        [100] * 10,  # chunked prefill, not page aligned
        [128] * 7 + [104],  # chunked prefill, page aligned
        [300] + [1] * 700,  # prefill then decodes
        [130, 1, 250, 1, 127, 1, 490],
    ],
)
def test_plan_over_steps_seals_each_block_once(chunks):
    """However 1000 tokens are split into steps, the steps together seal K
    blocks 1-6 and V blocks 1-5, in order and each once."""
    k_sealed, v_sealed, old = [], [], 0
    for chunk in chunks:
        plan = plan_one(old, old + chunk)
        k_sealed += plan.k_jobs[:, 1].tolist()
        v_sealed += plan.v_jobs[:, 1].tolist()
        old += chunk
    assert k_sealed == [1, 2, 3, 4, 5, 6]
    assert v_sealed == [1, 2, 3, 4, 5]


# (old length, new length, K positions kept fp16, V positions kept fp16)
KEPT_POSITIONS = [
    (0, 20, range(20), range(20)),  # all sink
    (0, 200, range(200), range(200)),  # sink plus tail, nothing sealed
    # Blocks sealed in this step are quantized straight from the step's K/V.
    (0, 600, [*range(128), *range(512, 600)], [*range(128), *range(384, 600)]),
    # Fullest tails: 127 K and 255 V tokens, one short of their rings.
    (0, 511, [*range(128), *range(384, 511)], [*range(128), *range(256, 511)]),
    (255, 256, [], [255]),  # token completes K block 1, which is sealed
    (300, 400, range(384, 400), range(300, 400)),
]


@pytest.mark.parametrize("old, new, k_kept, v_kept", KEPT_POSITIONS)
def test_plan_writes_only_unsealed_tokens(old, new, k_kept, v_kept):
    """Kept tokens also get distinct rows: `kitty_write_residual` does a
    step's writes in one assignment, where a repeated row has no defined
    winner."""
    plan = plan_one(old, new)
    assert sorted(written_rows(plan.k_writes, old)) == list(k_kept)
    assert sorted(written_rows(plan.v_writes, old)) == list(v_kept)
    for rows in (plan.k_writes[2], plan.v_writes[2]):
        assert len(set(rows.tolist())) == len(rows)


@pytest.mark.parametrize(
    "pos, k_row, v_row",
    [
        (0, 0, 256),  # sink
        (127, 127, 383),  # last sink token
        (128, 128, 512),  # first tail token
        (255, None, 639),  # completes K block 1, which is sealed instead
        (256, 128, 384),  # K and V tail rings both back at their first row
        (383, None, 511),  # completes K block 2
        (384, 128, 512),
        (600, 216, 472),
    ],
)
def test_plan_residual_row_of_decoded_token(pos, k_row, v_row):
    plan = plan_one(pos, pos + 1)
    assert written_rows(plan.k_writes, pos) == ({} if k_row is None else {pos: k_row})
    assert written_rows(plan.v_writes, pos) == {pos: v_row}


def test_plan_batch_indexes_the_flattened_step():
    """Request 0 decodes, request 1 prefills 600 tokens, request 2 continues
    a prefill from position 10."""
    plan = plan_kitty_step(
        np.array([0, 1, 601, 611], np.int32), np.array([384, 600, 20], np.int32)
    )
    assert plan.k_jobs.tolist() == [[0, 2], [1, 1], [1, 2], [1, 3]]
    assert plan.v_jobs.tolist() == [[0, 1], [1, 1], [1, 2]]

    k_tok, k_req, k_row = plan.k_writes
    assert 0 not in k_req  # its token completes K block 2
    assert k_tok[k_req == 1].tolist() == [*range(1, 129), *range(513, 601)]
    assert k_tok[k_req == 2].tolist() == list(range(601, 611))
    assert k_row[k_req == 2].tolist() == list(range(10, 20))

    v_tok, v_req, v_row = plan.v_writes
    assert v_tok[v_req == 0].tolist() == [0]
    assert v_row[v_req == 0].tolist() == [511]  # V tail row of position 383


# kitty_gather_blocks

NUM_HEADS, HEAD_SIZE = 2, 8


def label(positions, req: int = 0) -> torch.Tensor:
    """[len(positions), H, D] values naming the request, position, head and
    channel, so each gathered value shows exactly where it was read from."""
    pos = torch.as_tensor(positions)
    h = torch.arange(NUM_HEADS)
    d = torch.arange(HEAD_SIZE)
    return (req * 1_000_000 + pos[:, None, None] * 1000 + h[:, None] * 100 + d).float()


def block_labels(kind: str, block: int, req: int = 0) -> torch.Tensor:
    """[H, 128, D] expected gather of `block`: its positions' labels in order,
    negated for V."""
    labels = label(range(block * PAGE, (block + 1) * PAGE), req).transpose(0, 1)
    return labels if kind == "k" else -labels


def gather_last_step(steps, kind, residual_block_ids=None):
    """Run the batch through `steps`, the cache lengths of each request after
    each step, and gather the `kind` blocks the last step seals.

    Earlier steps fill the residual with `kitty_write_residual`, as the backend
    does. Keys are the labels of their positions and values the negated
    labels. Residual rows that no step wrote are NaN.
    """
    num_reqs = len(steps[0])
    if residual_block_ids is None:
        residual_block_ids = list(range(num_reqs))
    residual_block_ids = torch.tensor(residual_block_ids)
    residual = torch.full(
        (num_reqs, NUM_HEADS, KITTY_RESIDUAL_ROWS, HEAD_SIZE), float("nan")
    )

    def step(old, new):
        query_start_loc = np.cumsum([0] + [n - o for o, n in zip(old, new)])
        plan = plan_kitty_step(query_start_loc, np.array(new))
        key = torch.cat(
            [label(range(o, n), r) for r, (o, n) in enumerate(zip(old, new))]
        )
        return query_start_loc, plan, key

    old = [0] * num_reqs
    for new in steps[:-1]:
        _, plan, key = step(old, new)
        kitty_write_residual(
            key,
            -key,
            residual,
            residual_block_ids,
            torch.from_numpy(plan.k_writes),
            torch.from_numpy(plan.v_writes),
        )
        old = new

    query_start_loc, plan, key = step(old, steps[-1])
    if kind == "k":
        src, jobs, tail_row, ring = key, plan.k_jobs, K_TAIL_ROW, PAGE
    else:
        src, jobs, tail_row, ring = -key, plan.v_jobs, V_TAIL_ROW, 2 * PAGE
    gathered = kitty_gather_blocks(
        src,
        residual,
        residual_block_ids=residual_block_ids,
        jobs=torch.from_numpy(jobs),
        old_lens=torch.from_numpy(plan.old_lens),
        query_start_loc=torch.from_numpy(query_start_loc),
        tail_row=tail_row,
        ring=ring,
    )
    return gathered, jobs.tolist()


# (K or V, cache length after each step, block the last step seals)
GATHER_ONE_BLOCK = [
    ("k", [384, 512], 3),  # all from this step
    ("k", [255, 256], 1),  # 127 tokens from the residual, the last from the step
    ("k", [200, 300], 1),  # split mid-block
    ("k", [200, 300, 384], 2),  # rows of 300-383 still hold block 1, step wins
    ("v", [383, 384], 1),  # all from the residual
    ("v", [450, 512], 2),  # all from the residual, which also holds block 3
]


@pytest.mark.parametrize("kind, lengths, block", GATHER_ONE_BLOCK)
def test_gather_returns_the_blocks_positions_in_order(kind, lengths, block):
    got, _ = gather_last_step([[n] for n in lengths], kind)
    want = block_labels(kind, block)[None]
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_gather_output_is_contiguous():
    """The seal kernels address token t, channel d of a page as `t * D + d`."""
    got, _ = gather_last_step([[600]], "k")
    assert got.shape == (3, NUM_HEADS, PAGE, HEAD_SIZE)
    assert got.is_contiguous()


@pytest.mark.parametrize(
    "kind, jobs",
    [
        ("k", [[0, 2], [1, 1], [1, 2], [1, 3], [2, 3]]),
        ("v", [[0, 1], [1, 1], [1, 2], [2, 2]]),
    ],
)
def test_gather_batch_reads_each_requests_own_rows(kind, jobs):
    """Requests decode, prefill and continue a prefill in one step, with
    residual blocks in a different order than the requests."""
    steps = [[383, 0, 450], [384, 600, 512]]
    got, got_jobs = gather_last_step(steps, kind, residual_block_ids=[2, 0, 1])
    assert got_jobs == jobs
    want = torch.stack([block_labels(kind, b, req) for req, b in jobs])
    torch.testing.assert_close(got, want, rtol=0, atol=0)


# kitty_boost_ranks


def constant_k_block(channel_values: list[float]) -> torch.Tensor:
    """[1, 1, 128, D] K block where channel c is `channel_values[c]` at every token."""
    values = torch.tensor(channel_values)
    return values.expand(PAGE, -1)[None, None]


@pytest.mark.parametrize(
    "d_boost, expected",
    [
        (0, [0, 0, 0, 0, 0, 0, 0, 0]),  # nothing boosted, all ranks are DB
        (1, [1, 1, 0, 1, 1, 1, 1, 1]),
        (3, [2, 3, 0, 3, 3, 3, 1, 3]),
        (8, [2, 6, 0, 4, 7, 5, 1, 3]),  # every channel boosted
    ],
)
def test_boost_ranks_known_values(d_boost, expected):
    """Rank 0 is the largest |K|, here channel 2 even though it is negative.
    Channels that are not boosted get rank `d_boost`."""
    k = constant_k_block([5.0, -1.0, -7.0, 3.0, 0.0, 2.0, 6.0, -4.0])
    ranks = kitty_boost_ranks(k, d_boost)
    assert ranks.dtype == torch.uint8
    assert ranks.flatten().tolist() == expected


def test_boost_ranks_score_is_mean_over_tokens():
    """One outlier token does not outweigh a channel that is large everywhere."""
    k = torch.zeros(1, 1, PAGE, 2)
    k[0, 0, 0, 0] = 100.0  # mean |K| = 100 / 128
    k[0, 0, :, 1] = 1.0  # mean |K| = 1
    assert kitty_boost_ranks(k, 1).flatten().tolist() == [1, 0]


def test_boost_ranks_each_page_picks_its_own_channels():
    """Page (job j, head h) is large only in channel 2 * j + h."""
    k = torch.zeros(2, 2, PAGE, 4)
    for j in range(2):
        for h in range(2):
            k[j, h, :, 2 * j + h] = 1.0
    ranks = kitty_boost_ranks(k, 1)
    assert ranks.tolist() == [
        [[0, 1, 1, 1], [1, 0, 1, 1]],
        [[1, 1, 0, 1], [1, 1, 1, 0]],
    ]


# kitty_write_residual

SENTINEL = 1000.0  # residual value that no write produces


def writes(*triples) -> torch.Tensor:
    """[3, n] writes from (step token, request, residual row) triples."""
    return torch.tensor(triples, dtype=torch.int64).view(-1, 3).T


def written_residual_rows(residual: torch.Tensor) -> list[tuple[int, int]]:
    """(residual block, row) pairs that no longer hold SENTINEL."""
    changed = (residual != SENTINEL).any(dim=3).any(dim=1)
    return sorted(map(tuple, changed.nonzero().tolist()))


def sentinel_residual(dtype=torch.float16) -> torch.Tensor:
    shape = (3, NUM_HEADS, KITTY_RESIDUAL_ROWS, HEAD_SIZE)
    return torch.full(shape, SENTINEL, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_write_residual_copies_tokens_to_their_rows(dtype):
    """Request 0 lives in residual block 2 and request 1 in block 0."""
    torch.manual_seed(0)
    key = torch.randn(4, NUM_HEADS, HEAD_SIZE).to(dtype)
    value = torch.randn(4, NUM_HEADS, HEAD_SIZE).to(dtype)
    residual = sentinel_residual(dtype)

    kitty_write_residual(
        key,
        value,
        residual,
        residual_block_ids=torch.tensor([2, 0]),
        k_writes=writes((0, 0, 5), (1, 0, 130), (3, 1, 0)),
        v_writes=writes((1, 0, 300), (2, 1, 639)),
    )

    expected = {
        (2, 5): key[0],
        (2, 130): key[1],
        (0, 0): key[3],
        (2, 300): value[1],
        (0, 639): value[2],
    }
    for (block, row), want in expected.items():
        assert torch.equal(residual[block, :, row], want)
    assert written_residual_rows(residual) == sorted(expected)


def test_write_residual_overwrites_a_reused_row():
    """Tail rows are a ring, so a later step writes over an older token."""
    key = torch.ones(1, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
    residual = sentinel_residual()
    k_writes = writes((0, 0, 128))
    kitty_write_residual(key, key, residual, torch.tensor([0]), k_writes, writes())
    kitty_write_residual(2 * key, key, residual, torch.tensor([0]), k_writes, writes())
    assert (residual[0, :, 128] == 2).all()
    assert written_residual_rows(residual) == [(0, 128)]


@pytest.mark.parametrize(
    "v_writes, written",
    [((), []), (((0, 0, 256),), [(0, 256)])],
    ids=["nothing", "v-only"],
)
def test_write_residual_with_no_k_writes(v_writes, written):
    """A step whose tokens all go to sealed K blocks keeps no K token."""
    key = torch.ones(1, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
    residual = sentinel_residual()
    kitty_write_residual(
        key, -key, residual, torch.tensor([0]), writes(), writes(*v_writes)
    )
    assert written_residual_rows(residual) == written
    for block, row in written:
        assert (residual[block, :, row] == -1).all()
