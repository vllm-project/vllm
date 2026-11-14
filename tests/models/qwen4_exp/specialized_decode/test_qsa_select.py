# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CuTe QSA decode select (`qsa_select_expand`, a CuTe DSL cluster
kernel) against the stock top-k and index expansion of a decode step's
indexer, at Qwen3.8-Flash-Next's selection (512 blocks of 4 tokens): rows of
0 to 512 visible blocks (every block, in order: the stock packed row bitwise)
and of more (the stock set of blocks, with distinct scores), the causal tail
and count columns, scores past a row's visible blocks, every cluster shape of
the steps the select serves, rows built to take each path of a long row
(CTA 0's candidates or the cluster rounds, with each number of radix rounds,
and blocks tied with the 512th score, zeros among them), eager and replayed
in a CUDA graph, and the swap of the layers' indexers within
`QsaRows.active` (`runtime/qsa.py`)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="the QSA select requires SM90+ (thread block clusters)",
)

if current_platform.is_cuda():
    # The model module first: it imports the QSA module (which imports it back).
    from vllm.models.qwen4_exp.nvidia import model  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        _TOPK_WORKSPACE_BYTES,
        _topk,
        expand_qsa_block_indices,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.qsa import (
        PACKED_WIDTH,
        RATIO,
        TOKENS,
        default_variant,
        qsa_select_expand,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.qsa import (
        SELECT_ROWS,
        DecodeIndexer,
        QsaRows,
    )

COLUMNS = 41 * 400  # 41 compressed pages of 400 blocks: 65,600 tokens
# Visible blocks of the rows, in turn: none, a few, the selection's edge, a
# 65,536-token context and others.
VISIBLE = (0, 3, 511, 512, 513, 16384, 9000, 2000, 16400, 1)


def _case(rows: int, seed: int):
    """Logits with distinct scores below each row's visible blocks (and large
    scores past them, which neither select may take), visible blocks and
    positions (each row's tail: 0 to 3 tokens)."""
    gen = torch.Generator().manual_seed(seed)
    visible = torch.tensor(
        [VISIBLE[(row + seed) % len(VISIBLE)] for row in range(rows)],
        dtype=torch.int32,
    )
    tails = torch.randint(0, RATIO, (rows,), generator=gen)
    positions = (visible.long() * RATIO + tails - 1).clamp(min=0)
    logits = torch.full((rows, COLUMNS), 1.0e30)
    for row in range(rows):
        order = torch.randperm(COLUMNS, generator=gen).float() - COLUMNS / 2
        n = int(visible[row])
        logits[row, :n] = order[:n] * 1.0e-3
    return logits.cuda(), visible.cuda(), positions.cuda()


def _stock(logits, visible, positions):
    """The stock top-k and expansion's packed rows."""
    rows = logits.shape[0]
    blocks = torch.empty(rows, TOKENS // RATIO, dtype=torch.int32, device="cuda")
    workspace = torch.empty(_TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    _topk(logits, visible, TOKENS, RATIO, blocks, workspace)
    packed = torch.empty(rows, PACKED_WIDTH, dtype=torch.int32, device="cuda")
    expand_qsa_block_indices(blocks, positions, visible, RATIO, TOKENS, packed)
    return packed


def _assert_selects(out, expected, visible):
    """`out` has `expected`'s count and causal tail columns, its packed
    row bitwise for rows of at most 512 visible blocks, and its set of tokens
    for longer rows (whose order is the kernel's own, from its shared-memory
    atomics, as the stock top-k's is from its own)."""
    torch.accelerator.synchronize()
    assert torch.equal(out[:, TOKENS:], expected[:, TOKENS:])
    for row in range(out.shape[0]):
        if int(visible[row]) <= TOKENS // RATIO:
            assert torch.equal(out[row], expected[row]), row
        else:
            got = out[row, :TOKENS].sort().values
            want = expected[row, :TOKENS].sort().values
            assert torch.equal(got, want), row


@pytest.mark.parametrize("rows", [1, 4, 9, 16, 19, SELECT_ROWS])
def test_select_matches_the_stock_top_k(rows):
    """Every row's count and causal tail are stock's; a row of at most 512
    visible blocks is stock's packed row bitwise, a longer one holds the stock
    set of tokens (in its own order). The output's prior contents do not
    matter."""
    logits, visible, positions = _case(rows, seed=rows)
    expected = _stock(logits, visible, positions)
    for fill in (-7, 123456):
        out = torch.full((rows, PACKED_WIDTH), fill, dtype=torch.int32, device="cuda")
        qsa_select_expand(logits, visible, positions, out)
        _assert_selects(out, expected, visible)


def test_select_replays_in_a_cuda_graph():
    """A captured select's replays select as the eager launch does."""
    logits, visible, positions = _case(4, seed=3)
    expected = _stock(logits, visible, positions)
    out = torch.full((4, PACKED_WIDTH), -5, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_select_expand(logits, visible, positions, out)
    for _ in range(3):
        out.fill_(-5)
        graph.replay()
        _assert_selects(out, expected, visible)


def test_every_cluster_shape_of_small_steps_selects():
    """Each cluster shape the steps of up to `SELECT_ROWS` rows take, forced
    on a 4-row case, selects the stock set."""
    logits, visible, positions = _case(4, seed=8)
    expected = _stock(logits, visible, positions)
    variants = {default_variant(rows) for rows in range(1, SELECT_ROWS + 1)}
    for variant in variants:
        out = torch.empty(4, PACKED_WIDTH, dtype=torch.int32, device="cuda")
        qsa_select_expand(logits, visible, positions, out, variant)
        _assert_selects(out, expected, visible)


TOP = 512  # blocks a row selects
ONE = 0x3F800000  # the FP32 bit pattern of 1.0


def _bits(patterns: list[int]) -> torch.Tensor:
    """FP32 scores of positive bit patterns."""
    return torch.tensor(patterns, dtype=torch.int32).view(torch.float32)


def _spread(start: float, step: float, count: int) -> torch.Tensor:
    """`count` distinct FP32 scores from `start`, `step` apart."""
    return start + step * torch.arange(count, dtype=torch.float32)


def _ladder(bins: int, per_bin: int) -> torch.Tensor:
    """`per_bin` scores in each of the top `bins` bins of key bits 20-10
    of [1, 1.25) (one top-11-bit bin), distinct in bits 9-0."""
    return _bits([ONE + (d << 10) + low for d in range(bins) for low in range(per_bin)])


# Long rows built to take each path of the select, by (path, radix rounds,
# blocks tied with the 512th score beyond the slots left for them). The top
# 11 key bits are a quarter octave of positive scores; "gather" is CTA 0's
# candidates (at most 4,096 keys in the 512th key's top-bit bin, more than
# needed), "general" the cluster rounds.
PATH_ROWS = (
    # 500 above the 512th key's bin, which holds the 12 needed.
    (
        ("general", 1, False),
        torch.cat(
            [_spread(16.0, 0.004, 500), _spread(8.0, 0.1, 12), _spread(0, 0.004, 1488)]
        ),
    ),
    # 8,192 keys in one top-bit bin, 4 a bits 20-10 bin: 128 bins hold 512.
    (("general", 2, False), _ladder(2048, 4)),
    # 3 a bits 20-10 bin: 2 of the 512th key's bin needed.
    (("general", 3, False), _ladder(2048, 3)),
    # 300 above 5,700 equal scores, 212 of which are taken.
    (
        ("general", 3, True),
        torch.cat(
            [_bits([ONE + ((d + 1) << 10) for d in range(300)]), torch.ones(5700)]
        ),
    ),
    # Sum-of-ReLU scores: 300 positive, 4,700 zeros.
    (("general", 3, True), torch.cat([_spread(0.5, 0.001, 300), torch.zeros(4700)])),
    # 1,200 keys in one top-bit bin, 3 a bits 20-10 bin.
    (("gather", 3, False), _ladder(400, 3)),
    # 400 above the bin; in it 700 scores of 1.125, 112 of them taken, and
    # 300 below them; 1,600 below the bin.
    (
        ("gather", 3, True),
        torch.cat(
            [
                _spread(4.0, 0.002, 400),
                torch.full((700,), 1.125),
                _bits([ONE + (d << 10) for d in range(300)]),
                _spread(0.5, 1e-4, 1600),
            ]
        ),
    ),
    # Sum-of-ReLU scores: 400 positive, 2,600 zeros.
    (("gather", 3, True), torch.cat([_spread(0.5, 0.001, 400), torch.zeros(2600)])),
)


def _keys(scores: torch.Tensor) -> torch.Tensor:
    """The select's order-preserving 32-bit keys of FP32 scores (int64)."""
    bits = scores.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)


def _path(scores: torch.Tensor) -> tuple[str, int, bool]:
    """The path, radix rounds and ties of a long row (`PATH_ROWS`)."""
    keys = _keys(scores)
    kth = keys.sort(descending=True).values[TOP - 1]

    def split(shift: int) -> tuple[int, int]:
        """Keys sharing the 512th key's bits above `shift`, and how many of
        them are needed."""
        held = int(((keys >> shift) == (kth >> shift)).sum())
        return held, TOP - int(((keys >> shift) > (kth >> shift)).sum())

    (held0, remain0), (held1, remain1), (held2, remain2) = map(split, (21, 10, 0))
    path = "gather" if held0 != remain0 and held0 <= 4096 else "general"
    rounds = 1 if held0 == remain0 else 2 if held1 == remain1 else 3
    return path, rounds, held2 != remain2


def _cluster_order(index: int, variant: tuple[int, ...]) -> tuple[int, int, int]:
    """(CTA of the cluster, thread, key) of block `index` in `variant`."""
    cluster, threads, _ = variant
    vector, key = divmod(index, 4)
    group, thread = divmod(vector, threads)
    step, cta = divmod(group, cluster)
    return cta, thread, 4 * step + key


def _blocks(row: torch.Tensor) -> list[int]:
    """The blocks of a packed row, each of whose four tokens it must hold in
    four consecutive columns."""
    tokens = row[:TOKENS].view(-1, RATIO)
    blocks = tokens[:, 0].div(RATIO, rounding_mode="floor")
    assert torch.equal(tokens, blocks[:, None] * RATIO + torch.arange(RATIO))
    return blocks.tolist()


def _assert_paths(out, expected, logits, variant):
    """`out`'s count and tail columns are stock's (`expected`) and each
    row selects every block above its 512th score and, of the blocks tied
    with it, the first in the path's order: lowest block index on CTA 0's
    candidates, (CTA, thread, key) on the cluster rounds; stock's blocks are
    the same up to the choice among the ties."""
    torch.accelerator.synchronize()
    out, expected = out.cpu(), expected.cpu()
    assert torch.equal(out[:, TOKENS:], expected[:, TOKENS:])
    for row, ((path, _, _), scores) in enumerate(PATH_ROWS):
        n = scores.numel()
        row_scores = logits[row, :n]
        kth = row_scores.topk(TOP).values[-1]
        above = (row_scores > kth).nonzero().flatten().tolist()
        tied = (row_scores == kth).nonzero().flatten().tolist()
        if path == "general":
            tied.sort(key=lambda index: _cluster_order(index, variant))
        want = sorted(above + tied[: TOP - len(above)])
        assert sorted(_blocks(out[row])) == want, (row, variant)
        stock = _blocks(expected[row])
        assert len(set(stock)) == TOP and set(above) <= set(stock)
        assert bool((row_scores[stock] >= kth).all())


def test_every_path_of_long_rows():
    """Rows built to take each path of a long row (`PATH_ROWS`), shuffled
    over the cluster, select as `_assert_paths` says, in every cluster shape
    of the steps of up to `SELECT_ROWS` rows, eager and replayed in a CUDA
    graph; the cluster rounds' rows (no atomics) replay bitwise."""
    gen = torch.Generator().manual_seed(5)
    rows = len(PATH_ROWS)
    logits = torch.full((rows, COLUMNS), 1.0e30)
    visible = torch.empty(rows, dtype=torch.int32)
    for row, (path, scores) in enumerate(PATH_ROWS):
        n = scores.numel()
        logits[row, :n] = scores[torch.randperm(n, generator=gen)]
        visible[row] = n
        assert _path(logits[row, :n]) == path, row
    positions = (
        visible.long() * RATIO + torch.randint(0, RATIO, (rows,), generator=gen) - 1
    )
    args = (logits.cuda(), visible.cuda(), positions.cuda())
    expected = _stock(*args)
    for variant in {default_variant(r) for r in range(1, SELECT_ROWS + 1)}:
        eager = torch.full((rows, PACKED_WIDTH), -3, dtype=torch.int32, device="cuda")
        qsa_select_expand(*args, eager, variant)
        _assert_paths(eager, expected, logits, variant)
        out = torch.full_like(eager, -5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qsa_select_expand(*args, out, variant)
        for _ in range(2):
            out.fill_(-5)
            graph.replay()
            _assert_paths(out, expected, logits, variant)
            for row, ((path, _, _), _) in enumerate(PATH_ROWS):
                if path == "general":
                    assert torch.equal(out[row], eager[row]), (row, variant)


class _StockImpl:
    """A stock implementation stand-in."""


def test_indexers_take_the_select_in_small_steps():
    """`QsaRows.active` gives the indexers of the layers with a fused
    prepare a `DecodeIndexer` as their `forward` in steps of at most
    `SELECT_ROWS` rows (only when the model length fits the kernel), and
    restores them on exit."""
    layers = []
    for fused in (True, False):
        layer = torch.nn.Module()
        layer.impl = _StockImpl()
        layer.num_kv_heads = 2
        layer.use_fused_qsa_prepare = fused
        layer.indexer = torch.nn.Module()
        layer.indexer.compress_ratio = RATIO
        layer.indexer.token_topk = TOKENS
        layers.append(layer)
    for max_model_len, selecting in ((262144, True), (262148, False), (None, False)):
        qsa = QsaRows(layers, max_model_len)  # type: ignore[arg-type]
        for rows in (1, SELECT_ROWS, SELECT_ROWS + 1):
            with qsa.active(rows):
                taken = selecting and rows <= SELECT_ROWS
                forward = layers[0].indexer.__dict__.get("forward")
                assert isinstance(forward, DecodeIndexer) == taken
                assert "forward" not in layers[1].indexer.__dict__
            assert all("forward" not in layer.indexer.__dict__ for layer in layers)
