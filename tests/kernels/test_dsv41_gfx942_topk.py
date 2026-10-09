# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 top-512 helpers of the DeepSeek-V4.1 sparse indexer
(vllm/model_executor/layers/dsv41_gfx942/topk.py) against vLLM's ops.

Each helper must choose the columns or blocks that vLLM's op chooses. Equal
values at the cut may be chosen differently, and then the chosen values must
be the same multiset. The steps are one request with its 5 DSpark drafts
(the step that the gfx942 mono decode path takes), 3 such requests, and 8
requests without drafts, because the helpers also take the steps that are
too large for the mono layers. The logits rows are as wide as the AgentX
server's max_model_len (1M).
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


pytestmark = pytest.mark.skipif(
    not _on_gfx942(), reason="the extension is built for gfx942"
)

WIDTH = 1 << 20
TOPK_BLOCKS, BLOCK = 2048, 8
# (requests, rows a request) of a decode step.
STEPS = [(1, 6), (3, 6), (8, 1)]
STEP_IDS = ["1x6", "3x6", "8x1"]


@pytest.fixture
def gfx942_decode(monkeypatch):
    """Turns on the gfx942 decode helpers for one test."""
    from vllm.model_executor.layers.dsv41_gfx942 import enabled

    monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1")
    enabled.cache_clear()
    yield
    enabled.cache_clear()


def _lengths(length: int, reqs: int) -> list[int]:
    """The requests' lengths: the first is ``length`` and the others are
    shorter by a few hundred tokens each, so the rows end at different
    places."""
    return [max(0, length - 397 * i) for i in range(reqs)]


def _seq_lens(lengths: list[int], next_n: int, layout: str) -> torch.Tensor:
    """The lengths per row (2d) or per request (1d), the two forms that
    vLLM's decode indexer uses."""
    if layout == "2d":
        ends = [max(0, n - next_n + 1 + j) for n in lengths for j in range(next_n)]
        return torch.tensor(ends, dtype=torch.int32, device="cuda").view(-1, next_n)
    return torch.tensor(lengths, dtype=torch.int32, device="cuda")


def _row_ends(seq_lens: torch.Tensor, next_n: int, rows: int) -> list[int]:
    """Each row's end, the way vLLM's top_k_per_row_decode computes it."""
    lens = seq_lens.reshape(-1).tolist()
    if seq_lens.dim() == 2:
        return [max(0, v) for v in lens[:rows]]
    return [max(0, lens[r // next_n] - next_n + r % next_n + 1) for r in range(rows)]


def _visible(seq_lens: torch.Tensor, next_n: int, rows: int) -> torch.Tensor:
    """The int64 row ends that vLLM's decode indexer passes to its candidate
    kernels."""
    visible = seq_lens.reshape(-1)
    if visible.numel() != rows:
        visible = visible.repeat_interleave(next_n)
    return visible[:rows].to(torch.int64)


def _row_sets(indices: torch.Tensor, ends: list[int]) -> list[set[int]]:
    out = []
    for r, e in enumerate(ends):
        got = indices[r, :512]
        live = got[got >= 0]
        assert (live < e).all(), f"row {r}: an index past the row end {e}"
        assert live.unique().numel() == live.numel(), f"row {r}: repeated indices"
        out.append(set(live.tolist()))
    return out


def _assert_same_top512(x, ref, got, ends, name):
    """Got must hold vLLM's columns, or other columns with the same values
    where values at the cut are equal, and -1 past a short row's end."""
    want, have = _row_sets(ref, ends), _row_sets(got, ends)
    for r, e in enumerate(ends):
        pad = got[r, min(512, e) :]
        assert (pad == -1).all(), f"{name} row {r}: missing -1 padding"
        if have[r] != want[r]:
            vg = sorted(x[r, list(have[r])].tolist())
            vw = sorted(x[r, list(want[r])].tolist())
            assert vg == vw, f"{name} row {r}: different top-512 values"


def _logits(ends: list[int], masked: bool, gen) -> torch.Tensor:
    """Random logits. ``masked`` keeps 2048 random blocks of 8 and the newest
    block of each row and sets the rest of the row to -inf, as layers 24 to
    36 see their logits after the DSpark candidate mask."""
    x = torch.randn(len(ends), WIDTH, device="cuda", generator=gen) * 4.0
    for r, e in enumerate(ends):
        nblocks = (e + BLOCK - 1) // BLOCK
        if not masked or nblocks <= TOPK_BLOCKS:
            continue
        keep = torch.zeros(nblocks, dtype=torch.bool, device="cuda")
        pick = torch.randperm(nblocks - 1, device="cuda", generator=gen)
        keep[pick[: TOPK_BLOCKS - 1]] = True
        keep[nblocks - 1] = True
        live = keep.repeat_interleave(BLOCK)[:e]
        x[r, :e] = torch.where(live, x[r, :e], float("-inf"))
    return x


@pytest.mark.parametrize("register_select", [True, False])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [100, 517, 16384, 131072, 300000])
@pytest.mark.parametrize("reqs,next_n", STEPS, ids=STEP_IDS)
def test_decode_top_k_matches_vllm(reqs, next_n, length, masked, register_select):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.dsv41_gfx942 import topk

    gen = torch.Generator(device="cuda").manual_seed(length + reqs)
    rows = reqs * next_n
    seq_lens = _seq_lens(_lengths(length, reqs), next_n, "2d")
    ends = _row_ends(seq_lens, next_n, rows)
    x = _logits(ends, masked, gen)
    ref = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")
    got = torch.full_like(ref, -7)
    ops.top_k_per_row_decode(
        x, next_n, seq_lens, ref, rows, x.stride(0), x.stride(1), 512
    )
    topk.top_k_per_row_decode_512(
        x, next_n, seq_lens, got, register_select=register_select
    )
    torch.accelerator.synchronize()
    _assert_same_top512(x, ref, got, ends, f"length {length}")


def _compact_rows(kind: str, rows: int, length: int, gen) -> torch.Tensor:
    if kind == "normal":
        return torch.randn(rows, length, generator=gen)
    if kind == "wide":
        return torch.randn(rows, length, generator=gen) * torch.exp(
            torch.randn(rows, length, generator=gen) * 4
        )
    if kind == "few values":
        return torch.randint(0, 40, (rows, length), generator=gen).float()
    if kind == "all equal":
        return torch.full((rows, length), 0.75)
    if kind == "near max":
        # Most values sit just below the largest, in the first bins of the
        # select by distance below the max.
        return 1000.0 - torch.rand(rows, length, generator=gen) * 1e-3
    x = torch.randn(rows, length, generator=gen)
    x[:, 1000:1500] = float("-inf")
    x[2, : length - 1000] = float("-inf")
    x[3, : length - 300] = float("-inf")
    return x


@pytest.mark.parametrize(
    "kind", ["normal", "wide", "few values", "all equal", "near max", "with -inf"]
)
def test_compact_register_select_matches_radix_job(kind):
    """compact_top_k_512_regs (selectFromRegisters) against
    compact_top_k_512 (vLLM's radix job) on compact candidate rows of 16384
    logits, whose ids are their positions."""
    from vllm.model_executor.layers.dsv41_gfx942 import topk

    ext = topk._load()
    rows, length = 6, TOPK_BLOCKS * BLOCK
    x = _compact_rows(kind, rows, length, torch.Generator().manual_seed(3))
    logits = x.cuda().contiguous()
    ids = torch.arange(length, dtype=torch.int32).repeat(rows, 1).cuda()
    lens = torch.full((rows,), 1 << 20, dtype=torch.int32, device="cuda")
    old = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    new = torch.empty_like(old)
    ext.compact_top_k_512(logits, ids, lens, 1, old)
    ext.compact_top_k_512_regs(logits, ids, lens, 1, new)
    torch.accelerator.synchronize()
    old, new = old.cpu(), new.cpu()
    top = torch.topk(x, 512, dim=1).values
    for r in range(rows):
        pos = new[r].long()
        assert pos.min() >= 0 and pos.unique().numel() == 512, f"row {r}"
        chosen = torch.sort(x[r, pos], descending=True).values
        assert torch.equal(chosen, top[r]), f"row {r}: not the 512 largest values"
        cut = top[r, -1]
        if int((x[r] == cut).sum()) == int((top[r] == cut).sum()):
            # Without equal values at the cut the set is unique.
            assert torch.equal(new[r].sort().values, old[r].sort().values), f"row {r}"


@pytest.mark.parametrize("layout", ["2d", "1d"])
@pytest.mark.parametrize("length", [100, 512, 517, 16390, 131072, 300000])
@pytest.mark.parametrize("reqs,next_n", STEPS, ids=STEP_IDS)
def test_candidate_top_k_matches_mask_and_top_k(
    gfx942_decode, reqs, next_n, length, layout
):
    """topk.candidate_top_k against vLLM's decode path on layers 24 to 36:
    the candidate mask, then top_k_per_row_decode on the masked rows. The
    candidate lists come from vLLM's own block selection over another row,
    as layer 20 writes them."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.dsv41_gfx942 import topk
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _select_candidate_blocks,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _apply_candidate_mask_strided,
    )

    gen = torch.Generator(device="cuda").manual_seed(length + reqs + len(layout))
    rows = reqs * next_n
    seq_lens = _seq_lens(_lengths(length, reqs), next_n, layout)
    ends = _row_ends(seq_lens, next_n, rows)
    visible = _visible(seq_lens, next_n, rows)
    x = torch.randn(rows, WIDTH, device="cuda", generator=gen) * 4.0
    src = torch.randn(rows, WIDTH, device="cuda", generator=gen) * 4.0
    cand = torch.empty(rows, TOPK_BLOCKS, dtype=torch.int32, device="cuda")
    _select_candidate_blocks(
        src, torch.zeros_like(visible), visible, TOPK_BLOCKS, BLOCK, cand
    )
    masked = x.clone()
    ref = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")
    got = torch.full_like(ref, -7)
    _apply_candidate_mask_strided(
        masked, torch.zeros_like(visible), visible, cand, BLOCK
    )
    ops.top_k_per_row_decode(
        masked, next_n, seq_lens, ref, rows, masked.stride(0), masked.stride(1), 512
    )
    assert topk.candidate_top_k(x, next_n, seq_lens, cand, BLOCK, got, 512)
    torch.accelerator.synchronize()
    for r, e in enumerate(ends):
        if e <= 512:
            # The readers take the first min(end, 512) entries, so a short row
            # must hold columns 0 to end - 1 and then -1.
            row = got[r].tolist()
            assert sorted(row[:e]) == list(range(e)) and set(row[e:]) <= {-1}
            continue
        cols = got[r].long()
        assert (cols >= 0).all(), f"row {r}: fewer than 512 columns"
        assert torch.isfinite(masked[r, cols]).all(), f"row {r}: a masked column"
    _assert_same_top512(masked, ref, got, ends, f"length {length} {layout}")


def _block_scores(row: torch.Tensor, end: int) -> torch.Tensor:
    """VLLM's block scores of one row before its end: the largest logit of a
    block (NaN when the block holds a NaN), and +inf for the newest block."""
    nblocks = math.ceil(end / BLOCK)
    padded = torch.full((nblocks * BLOCK,), -math.inf, device=row.device)
    padded[:end] = row[:end]
    scores = padded.view(nblocks, BLOCK).amax(dim=1)
    if end > 0:
        scores[-1] = math.inf
    return scores.cpu()


def _sort_key(v: float):
    # NaN sorts above +inf, as both torch.topk and vLLM's radix job pick it.
    return (1, 0.0) if v != v else (0, v)


@pytest.mark.parametrize("case", ["random", "ties", "special"])
@pytest.mark.parametrize("layout", ["2d", "1d"])
@pytest.mark.parametrize("length", [0, 5, 100, 16390, 131072, 300000])
@pytest.mark.parametrize("reqs,next_n", STEPS, ids=STEP_IDS)
def test_select_candidates_matches_vllm(
    gfx942_decode, reqs, next_n, length, layout, case
):
    """topk.select_candidates (layer 20) against vLLM's
    select_candidate_blocks. The block ids are the same, in no particular
    order. "ties" rounds the logits so that many block scores are equal, and
    "special" puts NaN and whole blocks of -inf inside the rows."""
    from vllm.model_executor.layers.dsv41_gfx942 import topk
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _select_candidate_blocks,
    )

    gen = torch.Generator(device="cuda").manual_seed(length + reqs + 7 * len(case))
    rows = reqs * next_n
    seq_lens = _seq_lens(_lengths(length, reqs), next_n, layout)
    visible = _visible(seq_lens, next_n, rows)
    x = torch.randn(rows, WIDTH, device="cuda", generator=gen) * 4.0
    if case == "ties":
        x = torch.round(x * 2.0) / 2.0
    elif case == "special":
        cols = torch.randint(0, WIDTH, (rows, 16), device="cuda", generator=gen)
        x.scatter_(1, cols, math.nan)
        blocks = torch.randint(
            0, WIDTH // BLOCK, (rows, 64), device="cuda", generator=gen
        )
        for j in range(BLOCK):
            x.scatter_(1, blocks * BLOCK + j, -math.inf)
    ref = torch.full((rows, TOPK_BLOCKS), -7, dtype=torch.int32, device="cuda")
    got = torch.full_like(ref, -7)
    _select_candidate_blocks(
        x, torch.zeros_like(visible), visible, TOPK_BLOCKS, BLOCK, ref
    )
    assert topk.select_candidates(x, next_n, seq_lens, BLOCK, got)
    torch.accelerator.synchronize()
    for r in range(rows):
        w, g = ref[r].tolist(), got[r].tolist()
        ws, gs = sorted(v for v in w if v >= 0), sorted(v for v in g if v >= 0)
        assert len(gs) == len(set(gs)), f"row {r}: a block id twice"
        assert w.count(-1) == g.count(-1), f"row {r}: different -1 counts"
        assert min(g) >= -1, f"row {r}: an entry below -1"
        if ws != gs:
            scores = _block_scores(x[r], int(visible[r]))
            vw = sorted(scores[ws].tolist(), key=_sort_key)
            vg = sorted(scores[gs].tolist(), key=_sort_key)
            assert all((a != a and b != b) or a == b for a, b in zip(vw, vg)), (
                f"row {r}: different chosen block scores"
            )


def test_failed_build_leaves_the_top_k_to_vllm(monkeypatch):
    """A failed build of topk512_gfx942.cu, for example without hipcc, must not
    fail a decode step. build returns False, every hook returns False, and the
    indexer keeps its -1 fill, so vLLM runs its own top-k ops."""
    from vllm.model_executor.layers.dsv41_gfx942 import topk

    def fail():
        raise RuntimeError("hipcc not found")

    monkeypatch.setattr(topk, "enabled", lambda: True)
    monkeypatch.setattr(topk, "_ext", None)
    monkeypatch.setattr(topk, "_build_failed", False)
    monkeypatch.setattr(topk, "_compile", fail)
    assert not topk.build()
    logits = torch.zeros(6, 4096)
    indices = torch.zeros(6, 512, dtype=torch.int32)
    seq_lens = torch.full((1,), 4096, dtype=torch.int32)
    assert not topk.decode_top_k(logits, 6, seq_lens, indices, 512)
    candidates = torch.zeros(6, 2048, dtype=torch.int32)
    assert not topk.candidate_top_k(logits, 6, seq_lens, candidates, 8, indices, 512)
    assert not topk.select_candidates(logits, 6, seq_lens, 8, candidates)
    decode = SimpleNamespace(requires_padding=False)
    assert not topk.skip_decode_fill(False, 6, 6, decode)
    with pytest.raises(RuntimeError, match="did not build"):
        topk.top_k_per_row_decode_512(logits, 6, seq_lens, indices)
