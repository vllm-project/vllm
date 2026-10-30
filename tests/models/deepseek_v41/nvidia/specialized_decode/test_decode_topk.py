# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``decode_topk`` (``decode_topk_cute.py``) against ``torch.topk``.

Token mode: each row's output must be an exact top-k of its BF16 logits in
the kernel's key order (every index whose key exceeds the k-th largest key,
then indices whose key equals it), or ``0 .. n-1`` and -1 for rows of at most
k values. Rows cover the sampled path, short rows (the direct path), crowded
and constant values, values rising toward the row's end (survivors in the last
CTA), NaN and infinities, the forced direct path, and a sample that misses the
row's top values. Block mode must
select an exact top-k of the 8-token block maxima, as the stock
``select_candidate_blocks`` on the upcast logits does, with the newest block
pinned and -inf blocks written as -1. ``cluster_topk`` (``cluster_topk.py``)
must keep token mode's contract, and select ``torch.topk``'s index set where no
value ties the k-th past k. ``sparse_topk_small``
(``sparse_topk_small.py``) writes the remapped top-k in ascending column order
with ties broken toward the lowest column.
"""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.cluster_topk import (  # noqa: E501
    cluster_for as cluster_topk_cluster_for,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.cluster_topk import (  # noqa: E501
    cluster_topk,
    supports_cluster_topk,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.decode_topk_cute import (  # noqa: E501
    SAMPLES,
    _resident_clusters,
    cluster_for,
    decode_topk,
    supports_decode_topk,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.sparse_topk_small import (  # noqa: E501
    sparse_topk_small,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="needs an SM100-family GPU",
)

DEVICE = "cuda"
WIDTH = 262144


def order_keys(values: torch.Tensor) -> torch.Tensor:
    """The kernel's order-preserving key of each BF16 value, as int32."""
    bits = values.view(torch.int16).int() & 0xFFFF
    return torch.where(bits >= 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)


def make_logits(rows: int, kind: str, seed: int, width: int = WIDTH) -> torch.Tensor:
    """[rows, width] BF16 logits of one distribution."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, width, device=DEVICE, generator=g)
    if kind == "crowded":  # few distinct values: long exact ties
        x = torch.randint(0, 6, (rows, width), device=DEVICE, generator=g).float()
    elif kind == "constant":
        x = torch.full((rows, width), 3.0, device=DEVICE)
    elif kind == "rising":  # the largest values at the row's end
        x = x * 0.1 + torch.linspace(0, 8, width, device=DEVICE)
    elif kind == "special":
        x[:, 7] = float("nan")
        x[:, 1000::977] = float("inf")
        x[:, 2000::991] = float("-inf")
    elif kind == "distinct":  # each positive finite value once, the rest -2
        bits = torch.full((width,), -16384, dtype=torch.int16, device=DEVICE)
        bits[:0x7F7F] = torch.arange(1, 0x7F80, dtype=torch.int16, device=DEVICE)
        order = torch.stack(
            [torch.randperm(width, device=DEVICE, generator=g) for _ in range(rows)]
        )
        return bits[order].view(torch.bfloat16)
    return x.bfloat16()


def check_tokens(logits: torch.Tensor, lengths: list[int], out: torch.Tensor, k: int):
    """Assert each row's output is an exact top-k (module docstring)."""
    for r, n in enumerate(lengths):
        got = out[r]
        if n <= k:
            want = torch.arange(k, device=DEVICE, dtype=torch.int32)
            want[n:] = -1
            assert torch.equal(got, want), f"row {r} (n {n})"
            continue
        assert int((got < 0).sum()) == 0 and int(got.max()) < n, (
            f"row {r}: ids out of range"
        )
        assert got.unique().numel() == k, f"row {r}: repeated ids"
        keys = order_keys(logits[r, :n])
        kth = keys.topk(k).values[-1]
        chosen = keys[got.long()]
        assert int((chosen < kth).sum()) == 0, f"row {r}: a key below the k-th"
        assert int((chosen > kth).sum()) == int((keys > kth).sum()), (
            f"row {r}: missed a key above"
        )


@pytest.mark.parametrize("kind", ["normal", "crowded", "constant", "rising", "special"])
# One row, 8 CTAs a row (12), 4 CTAs a row (20), the most rows (64).
@pytest.mark.parametrize("rows", [1, 12, 20, 64])
@pytest.mark.parametrize(("k", "fallback"), [(512, False), (2048, False), (2048, True)])
def test_token_topk(kind, rows, k, fallback):
    lengths_all = [
        WIDTH,
        0,
        k - 1,
        k,
        k + 1,
        5000,
        8192,
        8193,
        30000,
        162000,
        99991,
        250000,
    ]
    lengths = [lengths_all[(r * 5) % len(lengths_all)] for r in range(rows)]
    logits = make_logits(rows, kind, rows * 31 + k)
    assert supports_decode_topk(logits, rows, k, block_mode=False)
    ends = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    out = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
    decode_topk(logits, ends, out, block_mode=False, fallback=fallback)
    torch.accelerator.synchronize()
    check_tokens(logits, lengths, out, k)


def check_torch_topk_sets(
    logits: torch.Tensor, lengths: list[int], out: torch.Tensor, k: int
):
    """Assert that each row whose top k values tie no other value selects
    ``torch.topk``'s index set."""
    for r, n in enumerate(lengths):
        if n <= k:
            continue
        values = logits[r, :n].float()
        top = values.topk(k)
        if int((values >= top.values[-1]).sum()) == k:
            want = top.indices.int().sort().values
            assert torch.equal(out[r].sort().values, want), f"row {r} (n {n})"


@pytest.mark.parametrize(
    "kind", ["normal", "crowded", "constant", "rising", "distinct"]
)
@pytest.mark.parametrize("shape", ["ratio2", "source", "cluster8"])
def test_cluster_topk(kind, shape):
    """The token top-k of one request's 6 verify rows at the ratio-2 layers
    (512Ki wide at a 1M context, ~34K valid at a 64K context) and at the
    candidate source (1Mi wide, ~69K valid; rows past 512Ki stream through the
    kernel's ring), and of more rows than 16-CTA clusters fit at once (8 CTAs
    a row)."""
    device = torch.accelerator.current_device_index()
    k = 512
    if shape == "ratio2":
        rows, width, cluster = 6, 2 * WIDTH, 16
        lengths_all = [34000, 33000, 36001, k + 1, k, 0]
    elif shape == "source":
        rows, width, cluster = 6, 4 * WIDTH, 16
        lengths_all = [69000, 65600, 72001, 4 * WIDTH, 600001, k - 1]
    else:
        rows, width, cluster = _resident_clusters(device, 16) + 1, 2 * WIDTH, 8
        lengths_all = [2 * WIDTH, 34000, 400003, k - 1, 9000]
    lengths = [lengths_all[r % len(lengths_all)] for r in range(rows)]
    logits = make_logits(rows, kind, rows * 13 + width // WIDTH, width)
    ends = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    out = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
    assert supports_cluster_topk(logits, out)
    assert cluster_topk_cluster_for(rows, device) == cluster
    cluster_topk(logits, ends, out)
    torch.accelerator.synchronize()
    check_tokens(logits, lengths, out, k)
    check_torch_topk_sets(logits, lengths, out, k)


def test_cluster_topk_graph():
    """A CUDA graph replay of ``cluster_topk`` on new logits and lengths."""
    rows, k, width = 6, 512, 2 * WIDTH
    logits = make_logits(rows, "normal", 3, width)
    ends = torch.tensor([34000] * rows, dtype=torch.int32, device=DEVICE)
    out = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
    cluster_topk(logits, ends, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cluster_topk(logits, ends, out)
    logits.copy_(make_logits(rows, "distinct", 4, width))
    lengths = [35000, 2 * WIDTH, 300, 69001, 34000, 0]
    ends.copy_(torch.tensor(lengths, dtype=torch.int32, device=DEVICE))
    graph.replay()
    torch.accelerator.synchronize()
    check_tokens(logits, lengths, out, k)
    check_torch_topk_sets(logits, lengths, out, k)


def test_token_topk_strided_output_and_graph():
    """A wider output buffer (row stride > k) and a CUDA graph replay on new
    logits."""
    rows, k = 12, 512
    lengths = [162000, 140000, 600, 9000] * 3
    logits = make_logits(rows, "normal", 1)
    ends = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    buffer = torch.full((rows, k + 64), 7, dtype=torch.int32, device=DEVICE)
    out = buffer[:, :k]
    decode_topk(logits, ends, out, block_mode=False)
    torch.accelerator.synchronize()
    check_tokens(logits, lengths, out, k)
    assert torch.all(buffer[:, k:] == 7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_topk(logits, ends, out, block_mode=False)
    logits.copy_(make_logits(rows, "rising", 2))
    graph.replay()
    torch.accelerator.synchronize()
    check_tokens(logits, lengths, out, k)


@pytest.mark.parametrize(
    "rows,wide,cluster",
    [(4, 2 * WIDTH, 8), (20, 2 * WIDTH, 8), (4, 4 * WIDTH, 16), (12, 4 * WIDTH, 16)],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_wide_rows(rows, wide, cluster, fallback):
    """Rows of up to 524,288 logits (the ratio-2 layers' width at a 1M
    context), which only 8 CTAs a row hold, and of up to 1,048,576 (the
    candidate source's), which take 16; at 20 and 12 rows more clusters than
    a GB300 runs at once, so they run in waves. The blocks again as the
    deferred candidate-block selection launches them: without PDL, a few rows
    (here 3) a launch."""
    lengths = [
        [wide, wide * 3 // 4 + 1, wide // 2 + 1, 9000][r % 4] for r in range(rows)
    ]
    g = torch.Generator(device=DEVICE).manual_seed(11)
    logits = (torch.randn(rows, wide, device=DEVICE, generator=g) - 1.5).bfloat16()
    assert supports_decode_topk(logits, rows, 512, block_mode=False)
    assert supports_decode_topk(logits, rows, 2048, block_mode=True)
    assert cluster_for(rows, wide, torch.accelerator.current_device_index()) == cluster
    ends = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    out = torch.full((rows, 512), 7, dtype=torch.int32, device=DEVICE)
    decode_topk(logits, ends, out, block_mode=False, fallback=fallback)
    blocks = torch.full((2, rows, 2048), 7, dtype=torch.int32, device=DEVICE)
    decode_topk(logits, ends, blocks[0], block_mode=True)
    decode_topk(logits, ends, blocks[1], block_mode=True, pdl=False, max_rows=3)
    torch.accelerator.synchronize()
    check_tokens(logits, lengths, out, 512)
    for got in blocks:
        for r, end in enumerate(lengths):
            nblocks = (end + 7) // 8
            if nblocks <= 2048:  # every block, in order (no -inf blocks here)
                want = torch.arange(2048, device=DEVICE, dtype=torch.int32)
                want[nblocks:] = -1
                assert torch.equal(got[r], want), f"row {r}"
                continue
            keys = block_keys(logits[r], end)
            best = keys.topk(2048).values.sort().values
            assert torch.equal(selected_keys(got[r], keys), best), f"row {r}"


def test_candidate_source_waves():
    """Rows that need 16 CTAs are taken for at most two waves of clusters."""
    waves = 2 * _resident_clusters(torch.accelerator.current_device_index(), 16)
    logits = torch.empty(waves + 1, 4 * WIDTH, dtype=torch.bfloat16, device=DEVICE)
    for block_mode, k in ((False, 512), (True, 2048)):
        assert supports_decode_topk(logits, waves, k, block_mode)
        assert not supports_decode_topk(logits, waves + 1, k, block_mode)
    narrow = logits[:, : 2 * WIDTH]  # 8 CTAs a row: up to 64 rows
    assert supports_decode_topk(narrow, waves + 1, 512, block_mode=False)


def test_token_topk_sampler_miss():
    """Large values only at the sampled positions: the sampled cutoff keeps
    fewer than k keys, so the kernel must recover through the direct path."""
    rows, k, n = 2, 512, WIDTH
    cluster = cluster_for(rows, WIDTH, torch.accelerator.current_device_index())
    per_cta = SAMPLES // cluster
    chunk = ((n + cluster - 1) // cluster + 7) // 8 * 8
    positions = []
    for c in range(cluster):
        length = min(n - c * chunk, chunk)
        valid = per_cta * length // chunk
        positions += [
            c * chunk + (2 * t + 1) * length // (2 * valid) for t in range(valid)
        ]
    x = torch.full((rows, WIDTH), -1.0, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(7)
    x[:, positions] = torch.randn(rows, len(positions), device=DEVICE, generator=g) + 10
    logits = x.bfloat16()
    ends = torch.tensor([n] * rows, dtype=torch.int32, device=DEVICE)
    out = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
    decode_topk(logits, ends, out, block_mode=False)
    torch.accelerator.synchronize()
    check_tokens(logits, [n] * rows, out, k)


NEG_INF_KEY = 0x007F


def block_keys(logits: torch.Tensor, end: int) -> torch.Tensor:
    """Keys of a row's valid 8-token block maxima, the newest pinned at +inf."""
    nblocks = (end + 7) // 8
    padded = torch.full((nblocks * 8,), float("-inf"), device=DEVICE)
    padded[:end] = logits[:end].float()
    scores = padded.view(nblocks, 8).amax(dim=1)
    scores[nblocks - 1] = float("inf")
    return order_keys(scores.bfloat16())


def selected_keys(ids: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Sorted keys of a candidate row's selected blocks, -1 slots as -inf."""
    assert int(ids.max()) < keys.numel()
    valid = ids[ids >= 0]
    assert valid.unique().numel() == valid.numel(), "repeated ids"
    chosen = torch.full_like(ids, NEG_INF_KEY)
    chosen[ids >= 0] = keys[valid.long()]
    return chosen.sort().values


@pytest.mark.parametrize("rows", [3, 12, 20])
@pytest.mark.parametrize("fallback", [False, True])
def test_candidate_blocks_match_stock(workspace_init, rows, fallback):
    """Block mode selects an exact top-k of the block maxima, as the stock
    ``select_candidate_blocks`` on the upcast logits does: the same multiset
    of block keys (BF16 maxima tie often, so the block sets may differ only
    among equal maxima), the pinned newest block, -inf blocks as -1."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )

    k, block = 2048, 8
    ends_all = [WIDTH, 162000, 3000 * 8 + 5, 2048 * 8, 100 * 8 + 3, 0, 70001, 131072]
    ends_list = [ends_all[r % len(ends_all)] for r in range(rows)]
    g = torch.Generator(device=DEVICE).manual_seed(rows)
    logits = torch.randn(rows, WIDTH, device=DEVICE, generator=g).bfloat16()
    logits[0, : 50 * block] = float("-inf")
    last = rows - 1
    end = ends_list[last]
    if end:
        logits[last, (end - 1) // block * block : end] = -100.0  # lowest, but pinned
    ends = torch.tensor(ends_list, dtype=torch.int32, device=DEVICE)
    assert supports_decode_topk(logits, rows, k, block_mode=True)
    got = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
    decode_topk(logits, ends, got, block_mode=True, fallback=fallback)
    want = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
    select_candidate_blocks(logits.float(), None, ends, k, block, want)
    torch.accelerator.synchronize()
    for r, end in enumerate(ends_list):
        nblocks = (end + 7) // 8
        if nblocks <= k:
            # Every block in order, -inf ones as -1; the stock path pads alike.
            assert torch.equal(got[r].sort().values, want[r].sort().values), f"row {r}"
            continue
        keys = block_keys(logits[r], end)
        best = keys.topk(k).values.sort().values
        assert torch.equal(selected_keys(got[r], keys), best), f"row {r}"
        assert torch.equal(selected_keys(want[r], keys), best), f"row {r} (stock)"
        assert int(got[r].max()) == nblocks - 1, f"row {r}: newest block not pinned"


@pytest.mark.parametrize("kind", ["normal", "crowded", "constant", "rising", "special"])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("block_size", [8, 16])
def test_sparse_topk_small(kind, k, block_size):
    """The candidate consumers' top-k on their rows (2048 candidate blocks of
    8 tokens), eagerly and replayed from a CUDA graph on new logits: in
    ascending column order, the columns above the k-th value and the
    lowest-numbered ones equal to it (``torch.topk``'s set where the k-th value
    is unique), through the block table. A positive NaN is never picked and
    leaves its slot -1; a row of at most k columns gets them all in order, then
    -1, into a wider output buffer past which nothing is written."""
    width = 16384
    lengths = [width, width - 3, 12345, 5000, k + 1, k, k - 1, 0]
    rows = len(lengths)
    g = torch.Generator(device=DEVICE).manual_seed(block_size)
    blocks = torch.randint(
        0, 1 << 20, (rows, width // block_size), device=DEVICE, generator=g
    ).int()
    ends = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    buffer = torch.full((rows, k + 64), 7, dtype=torch.int32, device=DEVICE)

    def check(logits: torch.Tensor) -> None:
        torch.accelerator.synchronize()
        for r, n in enumerate(lengths):
            values = logits[r, :n].float()
            cols = torch.arange(n, device=DEVICE)
            if n > k:
                real = ~values.isnan()
                picks = k - int((~real).sum())
                values, cols = values[real], cols[real]
                kth = values.topk(picks).values[-1]
                above = cols[values > kth]
                ties = cols[values == kth][: picks - above.numel()]
                cols = torch.cat([above, ties]).sort().values
            want = torch.full((k,), -1, dtype=torch.int32, device=DEVICE)
            want[: cols.numel()] = blocks[r, cols // block_size] * block_size + (
                cols % block_size
            )
            assert torch.equal(buffer[r, :k], want), f"row {r} (n {n})"
        assert torch.all(buffer[:, k:] == 7)

    logits = make_logits(rows, kind, k + block_size, width)
    sparse_topk_small(logits, ends, blocks, block_size, buffer[:, :k])
    check(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_topk_small(logits, ends, blocks, block_size, buffer[:, :k])
    logits.copy_(make_logits(rows, kind, k + block_size + 1, width))
    graph.replay()
    check(logits)
