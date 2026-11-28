# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cooperative_topk's grid launch (FP32 rows, a persistent `state` passed in):
for 1-13, 16-22 and 34-49 rows, rows at least as long as the row count's
threshold take the grid path over the whole GPU and shorter rows the cluster
path, chosen on the device from the row lengths; 23-33 rows take the
dynamic-split launch (a few long rows split among the CTAs not busy with a
short row); 14-15 and 50-64 rows take the cluster launch. Every result must be
the exact top-k (same value multiset as torch.topk, indices in range and
distinct), including under CUDA graph replay with changing lengths and row
counts sharing one state."""

import pytest
import torch

from vllm.model_executor.layers.indexer_topk import (
    COOPERATIVE_TOPK_STATE_SIZE as STATE_SIZE,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and not current_platform.is_device_capability_family(120)
    ),
    reason="cooperative_topk's grid launch requires SM100 (16-CTA clusters)",
)

WIDTH = 1_048_576  # logits buffer width (max_model_len), as in decode
WORKSPACE_SIZE = 1024 * 1024
# kGridFrom in csrc/attention/dsa/topk/cooperative_topk.cu: the shortest row
# that takes the grid path, per row count (None: the cluster path only).
GRID_FROM = (
    [458752, 524289, 524289, 524289]
    + [655360] * 4
    + [393216] * 3
    + [524288] * 2
    + [None] * 2
    + [458752, 655360, 589824, 786432, 786432, 786432, 786432]
    + [None] * 11
    + [262144] * 4
    + [655360] * 12
    + [None] * 15
)
assert len(GRID_FROM) == 64


def _state() -> torch.Tensor:
    return torch.zeros(STATE_SIZE, dtype=torch.uint8, device="cuda")


def _run(logits, lengths, k, state):
    out = torch.full((logits.shape[0], k), -7, dtype=torch.int32, device="cuda")
    ws = torch.empty(WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
    torch.ops._C.cooperative_topk(logits, lengths, out, ws, k, logits.shape[1], state)
    return out


def _check(logits, lengths, out, k):
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        o = out[r]
        if n <= k:
            exp = torch.arange(k, device="cuda", dtype=torch.int32)
            exp[n:] = -1
            torch.testing.assert_close(o, exp)
            continue
        idx = o.long()
        assert bool(((idx >= 0) & (idx < n)).all()), f"row {r}: index out of range"
        assert idx.unique().numel() == k, f"row {r}: duplicate indices"
        ref = torch.topk(logits[r, :n].float(), k).values
        got = logits[r].float()[idx].sort(descending=True).values
        torch.testing.assert_close(
            got,
            ref,
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=f"row {r} (length {n}): not the exact top-{k}",
        )


def _logits(rows, lengths, kind, gen, dtype=torch.float32):
    x = torch.full((rows, WIDTH), float("-inf"), device="cuda")
    for r, n in enumerate(lengths):
        v = torch.randn(n, device="cuda", generator=gen)
        if kind == "ties":
            v = torch.randint(0, 4, (n,), device="cuda", generator=gen).float()
        elif kind == "constant":
            v = torch.zeros(n, device="cuda")
        elif kind == "spike":
            # k - 1 large scores, the rest one tied value
            v = torch.zeros(n, device="cuda")
            v[torch.randperm(n, device="cuda", generator=gen)[:2047]] = 10.0
        elif kind == "masked":
            v[torch.rand(n, device="cuda", generator=gen) < 0.99] = float("-inf")
        elif kind == "sample_trap":
            # high values exactly at half the grid path's sample positions,
            # so the sampled cutoff keeps fewer than k scores
            s = 2048
            pos = ((2 * torch.arange(s, device="cuda") + 1) * n) // (2 * s)
            v[pos[::2]] = 10 + torch.rand(s // 2, device="cuda", generator=gen)
        elif kind == "block":
            # 4096 top scores packed contiguously in the middle of the row
            v[n // 2 : n // 2 + 4096] += 10.0
        elif kind == "recent":
            # top scores concentrated in the latest tokens, as in decode
            v[-16384:] += 3.0
        elif kind == "sorted":
            v = v.sort().values
        elif kind == "nan":
            # NaN of both signs ranks above +inf, as in the cluster path
            p = torch.randperm(n, device="cuda", generator=gen)
            v[p[:8]] = float("nan")
            v[p[8:16]] = -float("nan")
            v[p[16:24]] = float("inf")
        x[r, :n] = v
    return x.to(dtype)


# Every cluster size and launch: 16-CTA (1-4) and 8-CTA (5-13) grid, the
# cluster launch (14-15), 4-CTA grid (16-22), the cluster launch (23-33), 2-CTA
# grid (34-49), the cluster launch (50-64), at lengths on both sides of the
# thresholds.
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "rows", [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 22, 23, 33, 34, 40, 49, 50, 64]
)
@pytest.mark.parametrize("length", [16_385, 262_145, 524_289, 1_040_001])
def test_grid_rows_lengths(rows: int, length: int, k: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(rows * 1000 + length % 997)
    logits = _logits(rows, [length] * rows, "gaussian", gen)
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    _check(logits, lengths, _run(logits, lengths, k, _state()), k)


@pytest.mark.parametrize(
    "rows",
    [r for r in range(1, 50) if GRID_FROM[r - 1] is not None and r % 3 == 1]
    + [2, 9, 12, 18, 38],
)
def test_grid_threshold_lengths(rows: int) -> None:
    """Rows one below, at and one above the row count's threshold in one call:
    the cluster and grid paths split the launch at that exact length."""
    t = GRID_FROM[rows - 1]
    gen = torch.Generator(device="cuda").manual_seed(rows)
    lens = [(t - 1, t, t + 1)[i % 3] for i in range(rows)]
    logits = _logits(rows, lens, "gaussian", gen)
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    _check(logits, lengths, _run(logits, lengths, 2048, _state()), 2048)


@pytest.mark.parametrize("rows", [2, 5, 9, 15, 18, 28, 34, 45, 64])
def test_grid_mixed_lengths(rows: int) -> None:
    """Short and long rows in one call: clusters and the grid path together."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    pick = [1_000, 2_048, 70_000, 300_000, 600_000, 1_040_001]
    lens = [pick[i % len(pick)] for i in range(rows)]
    logits = _logits(rows, lens, "gaussian", gen)
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    _check(logits, lengths, _run(logits, lengths, 2048, _state()), 2048)


@pytest.mark.parametrize(
    "kind",
    [
        "ties",
        "constant",
        "spike",
        "masked",
        "sample_trap",
        "block",
        "recent",
        "sorted",
        "nan",
    ],
)
@pytest.mark.parametrize("rows", [1, 4, 8, 16, 34, 48])
@pytest.mark.parametrize("length", [524_289, 1_040_001])
def test_grid_adversarial(kind: str, rows: int, length: int) -> None:
    """Inputs that misjudge the sampled cutoff, overflow a CTA's survivors or
    fill the threshold bin with ties (the grid-wide redo rounds)."""
    gen = torch.Generator(device="cuda").manual_seed(len(kind) * rows)
    logits = _logits(rows, [length] * rows, kind, gen)
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    _check(logits, lengths, _run(logits, lengths, 2048, _state()), 2048)


def test_grid_matches_without_state() -> None:
    """With and without the state: both exact (the cluster launch)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    lens = [900_000, 20_000, 1_040_001]
    logits = _logits(3, lens, "gaussian", gen)
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.empty(3, 2048, dtype=torch.int32, device="cuda")
    ws = torch.empty(WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
    torch.ops._C.cooperative_topk(logits, lengths, out, ws, 2048, WIDTH)
    _check(logits, lengths, out, 2048)
    _check(logits, lengths, _run(logits, lengths, 2048, _state()), 2048)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [9, 12, 15, 16])
@pytest.mark.parametrize("length", [131_073, 1_040_001])
def test_cluster_launch_cs8_rows(dtype: torch.dtype, rows: int, length: int) -> None:
    """The cluster launch (no state, or BF16) takes 8-CTA clusters for 9-15
    rows while they all fit at once, 4-CTA clusters from 16 rows."""
    gen = torch.Generator(device="cuda").manual_seed(rows + length)
    logits = _logits(rows, [length] * rows, "gaussian", gen, dtype)
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    for k in (512, 2048):
        out = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
        ws = torch.empty(WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
        torch.ops._C.cooperative_topk(logits, lengths, out, ws, k, WIDTH)
        _check(logits, lengths, out, k)


def test_grid_state_shared_across_calls() -> None:
    """One state through calls of different row counts, k and lengths, eager,
    as the workspace manager shares it across layers and steps."""
    state = _state()
    gen = torch.Generator(device="cuda").manual_seed(3)
    for step, (rows, k, n) in enumerate(
        [(1, 2048, 1_040_001), (8, 512, 700_000), (34, 2048, 300_000)] * 3
    ):
        lens = [max(4_097, n - 1_000 * r * (step + 1)) for r in range(rows)]
        logits = _logits(rows, lens, "recent", gen)
        lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
        _check(logits, lengths, _run(logits, lengths, k, state), k)


def test_grid_graph_replay_changing_lengths() -> None:
    """One captured graph per row count, replayed with new lengths each step,
    all row counts sharing one state (the generation counters carry over)."""
    state = _state()
    ws = torch.empty(WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
    graphs = {}
    for rows in (1, 4, 6, 16, 28, 34, 64):
        logits = torch.full((rows, WIDTH), float("-inf"), device="cuda")
        lengths = torch.full((rows,), 1_040_001, dtype=torch.int32, device="cuda")
        out = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")

        def fn(x=logits, n=lengths, o=out):
            torch.ops._C.cooperative_topk(x, n, o, ws, 2048, WIDTH, state)

        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs[rows] = (g, logits, lengths, out)
    gen = torch.Generator(device="cuda").manual_seed(0)
    steps = [30_000, 500_000, 1_040_001, 200_000, 800_000, 1_040_000]
    for step, n in enumerate(steps):
        for rows, (g, logits, lengths, out) in graphs.items():
            lens = [max(2_049, n - 97 * r * (step + 1)) for r in range(rows)]
            logits.fill_(float("-inf"))
            for r, m in enumerate(lens):
                logits[r, :m] = torch.randn(m, device="cuda", generator=gen)
            lengths.copy_(torch.tensor(lens, dtype=torch.int32))
            g.replay()
            torch.cuda.synchronize()
            _check(logits, lengths, out, 2048)


# ---------------------------------------------------------------------------
# 23-33 rows: dynamic split (cooperative_grid_dyn.cuh)
# ---------------------------------------------------------------------------

DYN_FROM = 524288
LONG = 1_040_000


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("short", [3_000, 131_072])
@pytest.mark.parametrize("n_long", [0, 1, 2, 8, 9])
@pytest.mark.parametrize("rows", [23, 24, 28, 33])
def test_dyn_ragged_rows(rows: int, n_long: int, short: int, k: int) -> None:
    """A few long rows among short ones: up to 8 split dynamically, 9 or more
    each in its own cluster."""
    gen = torch.Generator(device="cuda").manual_seed(rows * 10 + n_long)
    lens = [LONG - 7 * i for i in range(n_long)] + [short] * (rows - n_long)
    logits = _logits(rows, lens, "recent", gen)
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    state = _state()
    for _ in range(2):  # the second call checks the state reset
        _check(logits, lengths, _run(logits, lengths, k, state), k)


@pytest.mark.parametrize(
    "kind", ["sample_trap", "ties", "constant", "spike", "masked", "nan"]
)
@pytest.mark.parametrize("n_long", [1, 8])
@pytest.mark.parametrize("rows", [24, 33])
def test_dyn_misjudged_cutoff(kind: str, n_long: int, rows: int) -> None:
    """Long rows that defeat the sampled cutoff: the row's own cluster redoes
    it with the cluster algorithm."""
    gen = torch.Generator(device="cuda").manual_seed(len(kind) + rows + n_long)
    lens = [LONG - 11 * i for i in range(n_long)] + [16_384] * (rows - n_long)
    logits = torch.cat(
        [
            _logits(n_long, lens[:n_long], kind, gen),
            _logits(rows - n_long, lens[n_long:], "gaussian", gen),
        ]
    )
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    state = _state()
    for _ in range(2):
        _check(logits, lengths, _run(logits, lengths, 2048, state), 2048)


@pytest.mark.parametrize("rows", [24, 33])
def test_dyn_graph_replay(rows: int) -> None:
    """One captured graph, replayed with no, a few (split), misjudged and too
    many long rows in turn: the per-row state is re-zeroed by each launch."""
    state = _state()
    ws = torch.empty(WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
    logits = torch.full((rows, WIDTH), float("-inf"), device="cuda")
    lengths = torch.full((rows,), 16_384, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")

    def fn():
        torch.ops._C.cooperative_topk(logits, lengths, out, ws, 2048, WIDTH, state)

    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    gen = torch.Generator(device="cuda").manual_seed(rows)
    steps = [(0, "gaussian"), (2, "recent"), (8, "sample_trap"), (9, "gaussian")]
    steps += [(1, "ties"), (4, "recent"), (0, "gaussian")]
    for step, (n_long, kind) in enumerate(steps):
        lens = [LONG - 11 * i * (step + 1) for i in range(n_long)]
        lens += [10_000 + 37 * i * (step + 1) for i in range(rows - n_long)]
        new = torch.cat(
            [
                _logits(n_long, lens[:n_long], kind, gen),
                _logits(rows - n_long, lens[n_long:], "gaussian", gen),
            ]
        )
        logits.copy_(new)
        lengths.copy_(torch.tensor(lens, dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        _check(logits, lengths, out, 2048)
