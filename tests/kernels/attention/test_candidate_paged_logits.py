# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 candidate-only decode scoring for the DeepSeek V4.1 indexer.

Covers the three properties the path has to have to be usable:

* **Numerics** — the selected top-k matches what the dense score-then-mask
  path selects, including the ``-1`` padding convention and the causal and
  candidate bounds.
* **KV cache layout** — the paged FP8 indexer cache is read correctly through
  the strided, gapped views the allocator hands out, not just a contiguous
  tensor.
* **CUDA graphs** — a graph captured on one context length replays correctly
  at others, which is the property the dense path's host-side bounds break.

Run with::

    pytest tests/kernels/attention/test_candidate_paged_logits.py
"""

import os
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.candidate_paged_logits import (
    INDEXER_SCALE_BYTES,
    candidate_paged_decode_available,
    candidate_paged_prepare,
    candidate_paged_score_decode,
    candidate_paged_workspace_specs,
    paged_candidate_logits,
)
from vllm.platforms import current_platform
from vllm.v1.worker import workspace

HEAD_DIM = 128
NUM_HEADS = 32
TOPK = 512

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(90),
    reason="SM90 CUDA device required",
)


@pytest.fixture
def ws():
    """A workspace manager isolated from the process-wide one."""
    with patch.object(workspace, "_manager", None):
        device = torch.device("cuda", torch.cuda.current_device())
        workspace.init_workspace_manager(device)
        yield workspace.current_workspace_manager()
        workspace.reset_workspace_manager()


def _enable_env():
    return patch.dict(os.environ, {"VLLM_DSA_CANDIDATE_PAGED_DECODE": "1"})


class Cache:
    """A paged FP8 indexer cache plus the bf16 values it decodes to.

    ``pad_bytes`` emulates the real allocation, where the indexer's pages are
    a strided window into a larger block shared with other layers, so there
    are gaps between logical pages and ``view(-1)`` is invalid.
    """

    def __init__(self, num_blocks, block_size, device, pad_bytes=0, seed=0):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.block_size = block_size
        self.fp8 = current_platform.fp8_dtype()
        entry = HEAD_DIM + INDEXER_SCALE_BYTES
        page_bytes = block_size * entry
        backing = torch.zeros(
            (num_blocks, page_bytes + pad_bytes), dtype=torch.uint8, device=device
        )
        values = torch.randn(
            (num_blocks, block_size, HEAD_DIM), device=device, generator=gen
        ).to(self.fp8)
        # Non-trivial per-position scales: a wrong scale offset must not pass.
        scales = (
            torch.rand((num_blocks, block_size), device=device, generator=gen) + 0.5
        )
        flat = backing[:, :page_bytes].view(num_blocks, block_size, entry)
        # Per block: all value bytes, then all fp32 scales.
        packed = torch.cat(
            [
                values.view(num_blocks, block_size * HEAD_DIM).view(torch.uint8),
                scales.view(num_blocks, block_size).view(torch.uint8),
            ],
            dim=1,
        )
        backing[:, :page_bytes] = packed
        self.tensor = flat
        self.values = values
        self.scales = scales
        assert pad_bytes == 0 or not self.tensor.is_contiguous()

    def dequantized(self, phys, offset):
        """The bf16 K vector and fp32 scale of one physical slot."""
        return (
            self.values[phys, offset].to(torch.float32),
            self.scales[phys, offset],
        )


def _reference_scores(q, weights, cache, block_table, causal_len, device):
    """Dense reference: score every causal position, in fp32.

    Mirrors the decode reference's reduction order (dot, relu, weight, sum
    over heads, then the per-position dequantization multiply).
    """
    rows = q.shape[0]
    out = torch.full((rows, causal_len), float("-inf"), device=device)
    q_f = q.to(torch.float32)
    w_f = weights.to(torch.float32)
    for r in range(rows):
        for pos in range(causal_len):
            phys = int(block_table[r, pos // cache.block_size])
            k, scale = cache.dequantized(phys, pos % cache.block_size)
            dots = (q_f[r] * k[None, :]).sum(-1)
            out[r, pos] = (dots.relu() * w_f[r]).sum() * scale
    return out


def _reference_topk(scores, candidate_blocks, block_size, causal_len, topk):
    """Top-k inside the candidate blocks, as the dense path's mask produces."""
    rows = scores.shape[0]
    masked = torch.full_like(scores, float("-inf"))
    for r in range(rows):
        for blk in candidate_blocks[r].tolist():
            if blk < 0:
                continue
            lo = blk * block_size
            hi = min(lo + block_size, causal_len)
            if lo < hi:
                masked[r, lo:hi] = scores[r, lo:hi]
        finite = int(torch.isfinite(masked[r]).sum())
        if finite == 0:
            continue
    keep = min(topk, causal_len)
    idx = masked.topk(keep, dim=-1).indices
    # Positions the mask excluded are -inf; the real path reports -1 there.
    picked = masked.gather(1, idx)
    return torch.where(torch.isfinite(picked), idx, torch.full_like(idx, -1))


def _setup(
    device,
    *,
    rows,
    num_candidates=128,
    block_size=8,
    causal_len=4096,
    cache_block_size=64,
    pad_bytes=0,
    seed=0,
    topk=64,
):
    gen = torch.Generator(device=device).manual_seed(seed + 100)
    num_blocks = causal_len // cache_block_size + 2
    cache = Cache(num_blocks, cache_block_size, device, pad_bytes, seed)
    # A shuffled page table: a kernel that assumed identity mapping fails.
    block_table = torch.randperm(num_blocks, device=device, generator=gen).to(
        torch.int32
    )[None].expand(rows, -1).contiguous()
    total_blocks = causal_len // block_size
    cands = torch.stack(
        [
            torch.randperm(total_blocks, device=device, generator=gen)[:num_candidates]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    q = torch.randn(
        (rows, NUM_HEADS, HEAD_DIM), device=device, generator=gen
    ).to(cache.fp8)
    weights = torch.rand((rows, NUM_HEADS), device=device, generator=gen)
    seq_lens = torch.full((rows, 1), causal_len, dtype=torch.int32, device=device)
    out = torch.full((rows, topk), -7, dtype=torch.int32, device=device)
    return dict(
        cache=cache,
        block_table=block_table,
        candidate_blocks=cands,
        q=q,
        weights=weights,
        seq_lens=seq_lens,
        topk_indices=out,
        causal_len=causal_len,
        block_size=block_size,
        num_candidates=num_candidates,
        topk=topk,
        rows=rows,
    )


def _reserve(ws, s):
    ws.get_simultaneous(
        *candidate_paged_workspace_specs(
            s["rows"], s["num_candidates"], s["block_size"], s["topk"]
        )
    )


def _run(s, backend="persistent", next_n=1, plan=None):
    candidate_paged_score_decode(
        s["q"],
        s["cache"].tensor,
        s["weights"],
        s["seq_lens"],
        s["block_table"],
        s["candidate_blocks"],
        s["block_size"],
        next_n,
        s["topk"],
        backend,
        s["topk_indices"],
        HEAD_DIM,
        plan=plan,
    )


def _as_sets(indices):
    return [{int(v) for v in row.tolist() if v >= 0} for row in indices]


# ---------------------------------------------------------------------------
# Availability is a configuration-level predicate
# ---------------------------------------------------------------------------


def test_availability_is_off_without_the_env_flag():
    with patch.dict(os.environ, {"VLLM_DSA_CANDIDATE_PAGED_DECODE": "0"}):
        assert not candidate_paged_decode_available(
            use_fp4_cache=False,
            dcp_world_size=1,
            use_pcp=False,
            candidate_block_size=8,
            num_candidates=2048,
            topk_tokens=TOPK,
            head_dim=HEAD_DIM,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"use_fp4_cache": True},
        {"dcp_world_size": 2},
        {"use_pcp": True},
        {"num_candidates": 2047},  # not a power of two
        {"candidate_block_size": 6},  # not a power of two
        {"candidate_block_size": 0},
        {"topk_tokens": 2048 * 8 + 1},  # cannot select more than the width
        {"head_dim": 130},  # not a power of two
    ],
)
def test_unsupported_configurations_fall_back(override):
    """Each rejection keeps the dense path rather than failing at runtime."""
    kwargs = dict(
        use_fp4_cache=False,
        dcp_world_size=1,
        use_pcp=False,
        candidate_block_size=8,
        num_candidates=2048,
        topk_tokens=TOPK,
        head_dim=HEAD_DIM,
    )
    kwargs.update(override)
    with _enable_env():
        assert not candidate_paged_decode_available(**kwargs)


@requires_sm90
def test_supported_configuration_is_admitted():
    with _enable_env():
        assert candidate_paged_decode_available(
            use_fp4_cache=False,
            dcp_world_size=1,
            use_pcp=False,
            candidate_block_size=8,
            num_candidates=2048,
            topk_tokens=TOPK,
            head_dim=HEAD_DIM,
        )


def test_availability_does_not_depend_on_runtime_shapes():
    """The signature must stay free of per-step values.

    This is what lets a captured CUDA graph keep the path it was captured
    with: nothing the predicate reads can change between replays.
    """
    import inspect

    params = set(inspect.signature(candidate_paged_decode_available).parameters)
    runtime_shaped = {"num_rows", "rows", "max_seq_len", "seq_lens", "context"}
    assert not params & runtime_shaped


# ---------------------------------------------------------------------------
# Numerics against the dense reference
# ---------------------------------------------------------------------------


@requires_sm90
@torch.inference_mode()
def test_logits_match_dense_reference_on_candidate_columns(ws):
    """The compact logits equal the dense scores at the positions they map to,
    and are -inf everywhere else."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=4, num_candidates=128, causal_len=2048, topk=TOPK)
    _reserve(ws, s)

    rows, width = s["rows"], s["num_candidates"] * s["block_size"]
    sorted_blocks = s["candidate_blocks"].sort(dim=-1).values.contiguous()
    end = torch.full((rows,), width, dtype=torch.int32, device=device)
    causal = torch.full((rows,), s["causal_len"], dtype=torch.int32, device=device)
    logits = torch.empty((rows, width), dtype=torch.float32, device=device)

    paged_candidate_logits(
        s["q"],
        s["cache"].tensor,
        s["weights"],
        sorted_blocks,
        end,
        causal,
        s["block_table"],
        s["block_size"],
        HEAD_DIM,
        logits,
    )
    torch.cuda.synchronize()

    ref = _reference_scores(
        s["q"],
        s["weights"],
        s["cache"],
        s["block_table"],
        s["causal_len"],
        device,
    )
    for r in range(rows):
        for c in range(width):
            blk = int(sorted_blocks[r, c // s["block_size"]])
            pos = blk * s["block_size"] + c % s["block_size"]
            expected = ref[r, pos]
            # fp8 inputs on tensor cores vs an fp32 reference on the same
            # dequantized values: compare with a relative tolerance.
            torch.testing.assert_close(
                logits[r, c], expected, rtol=2e-2, atol=1e-2
            )


@requires_sm90
@torch.inference_mode()
@pytest.mark.parametrize("backend", ["persistent", "auto"])
def test_selected_topk_matches_dense_mask_path(ws, backend):
    """The end-to-end selection equals the dense score-then-mask selection."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=6, num_candidates=128, causal_len=4096, topk=TOPK, seed=3)
    _reserve(ws, s)
    _run(s, backend=backend)
    torch.cuda.synchronize()

    ref_scores = _reference_scores(
        s["q"], s["weights"], s["cache"], s["block_table"], s["causal_len"], device
    )
    ref = _reference_topk(
        ref_scores,
        s["candidate_blocks"],
        s["block_size"],
        s["causal_len"],
        s["topk"],
    )
    assert _as_sets(s["topk_indices"]) == _as_sets(ref)


@requires_sm90
@torch.inference_mode()
def test_selection_stays_inside_the_candidate_blocks(ws):
    """Every selected position must belong to one of the row's candidates."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=4, num_candidates=64, causal_len=4096, topk=TOPK, seed=5)
    _reserve(ws, s)
    _run(s)
    torch.cuda.synchronize()

    for r in range(s["rows"]):
        allowed = set()
        for blk in s["candidate_blocks"][r].tolist():
            allowed.update(range(blk * s["block_size"], (blk + 1) * s["block_size"]))
        picked = {int(v) for v in s["topk_indices"][r].tolist() if v >= 0}
        assert picked <= allowed
        # The candidate set is smaller than the context, so the row fills
        # its top-k from the candidates alone.
        assert len(picked) == min(s["topk"], len(allowed))


@requires_sm90
@torch.inference_mode()
def test_shared_plan_matches_fresh_plan(ws):
    """Consumer layers share one expansion plan per step: scoring with the
    reused plan must select the same set as scoring with a freshly computed
    one. Selection is a set: fp8 scores tie often and the top-k backends do
    not promise a stable order across buffer layouts."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=6, num_candidates=128, causal_len=4096, topk=TOPK, seed=5)
    _reserve(ws, s)

    rows, num_candidates = s["candidate_blocks"].shape
    plan = (
        torch.empty(rows, dtype=torch.int32, device=device),
        torch.empty(rows, dtype=torch.int32, device=device),
        torch.empty(rows, num_candidates, dtype=torch.int32, device=device),
        torch.empty(rows, dtype=torch.int32, device=device),
    )
    candidate_paged_prepare(
        s["seq_lens"], s["candidate_blocks"], s["block_size"], 1, rows, out=plan
    )

    _run(s)
    torch.cuda.synchronize()
    fresh = s["topk_indices"].clone()

    s["topk_indices"].fill_(-7)
    candidate_paged_score_decode(
        s["q"],
        s["cache"].tensor,
        s["weights"],
        s["seq_lens"],
        s["block_table"],
        s["candidate_blocks"],
        s["block_size"],
        1,
        s["topk"],
        "persistent",
        s["topk_indices"],
        HEAD_DIM,
        plan=plan,
    )
    torch.cuda.synchronize()
    assert _as_sets(fresh) == _as_sets(s["topk_indices"])


# ---------------------------------------------------------------------------
# KV cache layout
# ---------------------------------------------------------------------------


@requires_sm90
@torch.inference_mode()
@pytest.mark.parametrize("pad_bytes", [0, 64, 260])
@pytest.mark.parametrize("cache_block_size", [16, 64])
def test_strided_and_gapped_cache_layouts(ws, pad_bytes, cache_block_size):
    """Gapped pages (the real allocation) and several page sizes all decode
    to the same selection as the dense reference."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(
        device,
        rows=4,
        num_candidates=128,
        causal_len=2048,
        cache_block_size=cache_block_size,
        pad_bytes=pad_bytes,
        topk=TOPK,
        seed=7,
    )
    _reserve(ws, s)
    _run(s)
    torch.cuda.synchronize()

    ref_scores = _reference_scores(
        s["q"], s["weights"], s["cache"], s["block_table"], s["causal_len"], device
    )
    ref = _reference_topk(
        ref_scores,
        s["candidate_blocks"],
        s["block_size"],
        s["causal_len"],
        s["topk"],
    )
    assert _as_sets(s["topk_indices"]) == _as_sets(ref)


@requires_sm90
@torch.inference_mode()
def test_wrong_entry_width_is_rejected(ws):
    """A cache whose entry is not head_dim+4 must assert, not read garbage."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=2, causal_len=2048, topk=TOPK)
    _reserve(ws, s)
    bad = s["cache"].tensor[:, :, :-1]
    with pytest.raises(AssertionError, match="head_dim"):
        paged_candidate_logits(
            s["q"],
            bad,
            s["weights"],
            s["candidate_blocks"].sort(dim=-1).values.contiguous(),
            torch.zeros(s["rows"], dtype=torch.int32, device=device),
            torch.zeros(s["rows"], dtype=torch.int32, device=device),
            s["block_table"],
            s["block_size"],
            HEAD_DIM,
            torch.empty(
                (s["rows"], s["num_candidates"] * s["block_size"]),
                dtype=torch.float32,
                device=device,
            ),
        )


# ---------------------------------------------------------------------------
# Bounds and degenerate rows
# ---------------------------------------------------------------------------


@requires_sm90
@torch.inference_mode()
def test_causal_bound_and_empty_and_padded_rows(ws):
    """Rows with zero context, a short context, and -1 padded candidates."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=4, num_candidates=128, causal_len=4096, topk=TOPK, seed=11)
    _reserve(ws, s)

    # Row 0: nothing visible. Row 1: only the first candidate block's worth.
    s["seq_lens"][0] = 0
    s["seq_lens"][1] = s["block_size"]
    # Row 2: all candidates padded away.
    s["candidate_blocks"][2].fill_(-1)
    # Row 3: half the candidates padded, and one beyond the causal length.
    s["candidate_blocks"][3, 4:] = -1
    s["candidate_blocks"][3, 0] = s["causal_len"] // s["block_size"] + 5
    _run(s)
    torch.cuda.synchronize()

    picked = _as_sets(s["topk_indices"])
    assert picked[0] == set(), "no context must select nothing"
    assert picked[2] == set(), "fully padded candidates must select nothing"
    assert all(p < s["block_size"] for p in picked[1]), (
        f"row 1 selected past its causal length: {picked[1]}"
    )
    allowed_3 = set()
    for blk in s["candidate_blocks"][3].tolist():
        if blk >= 0:
            allowed_3.update(
                range(blk * s["block_size"], (blk + 1) * s["block_size"])
            )
    assert picked[3] <= allowed_3
    assert all(p < s["causal_len"] for p in picked[3])
    # Unselected slots are -1, never the sentinel the buffer was filled with.
    assert int(s["topk_indices"].min()) >= -1


@requires_sm90
@torch.inference_mode()
def test_speculative_rows_get_descending_causal_bounds(ws):
    """With one length per request, speculative row s of a request sees
    ``next_n - 1 - s`` fewer positions than the request's newest row."""
    device = torch.device("cuda", torch.cuda.current_device())
    next_n = 3
    requests = 2
    rows = requests * next_n
    s = _setup(
        device, rows=rows, num_candidates=128, causal_len=4096, topk=TOPK, seed=13
    )
    _reserve(ws, s)
    # One length per request, so the kernel must derive the per-row bounds.
    request_len = s["block_size"] * 2
    s["seq_lens"] = torch.full(
        (requests, 1), request_len, dtype=torch.int32, device=device
    )
    # The page table is per request, like the real decode metadata. Candidates
    # stay per decode row, which is how the dense path indexes them.
    s["block_table"] = s["block_table"][:requests].contiguous()
    s["candidate_blocks"].fill_(-1)
    s["candidate_blocks"][:, 0] = 0
    s["candidate_blocks"][:, 1] = 1
    _run(s, next_n=next_n)
    torch.cuda.synchronize()

    picked = _as_sets(s["topk_indices"])
    bounds = [max(request_len - next_n + 1 + step, 0) for step in range(next_n)]
    assert bounds == sorted(bounds) and bounds[0] < bounds[-1], bounds
    for req in range(requests):
        for step in range(next_n):
            row = req * next_n + step
            assert all(p < bounds[step] for p in picked[row]), (
                f"row {row} exceeded its causal bound {bounds[step]}: {picked[row]}"
            )
            # The two candidate blocks cover [0, 2 * block_size), so each row
            # sees exactly its own causal prefix of them.
            assert len(picked[row]) == bounds[step], (
                f"row {row} saw {len(picked[row])} positions, expected "
                f"{bounds[step]}"
            )


# ---------------------------------------------------------------------------
# CUDA graphs
# ---------------------------------------------------------------------------


@requires_sm90
@torch.inference_mode()
def test_graph_captured_short_replays_long(ws):
    """A graph captured on a short context must stay correct when replayed at
    longer ones.

    This is the property that fails if any context-dependent value reaches the
    host: capture would bake the warmup batch's bound.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=6, num_candidates=128, causal_len=4096, topk=TOPK, seed=17)
    _reserve(ws, s)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Warm up at the long context, then capture at a deliberately short
        # one, mirroring the runner's warmup/capture sequence.
        _run(s)
        s["seq_lens"].fill_(s["block_size"])
        for _ in range(3):
            _run(s)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    ws.lock()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _run(s)

    try:
        for causal in (s["causal_len"], 0, s["block_size"] * 3, s["causal_len"] - 3):
            s["seq_lens"].fill_(causal)
            s["candidate_blocks"].copy_(
                torch.stack(
                    [
                        torch.randperm(
                            max(1, s["causal_len"] // s["block_size"]), device=device
                        )[: s["num_candidates"]]
                        for _ in range(s["rows"])
                    ]
                ).to(torch.int32)
            )
            graph.replay()
            torch.cuda.synchronize()
            replayed = s["topk_indices"].clone()

            # The same inputs run eagerly must select the same positions.
            s["topk_indices"].fill_(-7)
            _run(s)
            torch.cuda.synchronize()
            assert _as_sets(replayed) == _as_sets(s["topk_indices"]), (
                f"replay diverged from eager at causal={causal}"
            )
            if causal == 0:
                assert _as_sets(replayed) == [set()] * s["rows"]
    finally:
        del graph
        torch.cuda.synchronize()


@requires_sm90
@torch.inference_mode()
def test_graph_replay_is_stable_across_page_table_changes(ws):
    """Replays must follow the current page table, not the captured one."""
    device = torch.device("cuda", torch.cuda.current_device())
    s = _setup(device, rows=4, num_candidates=128, causal_len=2048, topk=TOPK, seed=19)
    _reserve(ws, s)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(s)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    ws.lock()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _run(s)

    try:
        num_blocks = s["cache"].tensor.shape[0]
        s["block_table"].copy_(
            torch.randperm(num_blocks, device=device)
            .to(torch.int32)[None]
            .expand(s["rows"], -1)
        )
        graph.replay()
        torch.cuda.synchronize()
        replayed = s["topk_indices"].clone()

        ref_scores = _reference_scores(
            s["q"], s["weights"], s["cache"], s["block_table"], s["causal_len"], device
        )
        ref = _reference_topk(
            ref_scores,
            s["candidate_blocks"],
            s["block_size"],
            s["causal_len"],
            s["topk"],
        )
        assert _as_sets(replayed) == _as_sets(ref)
    finally:
        del graph
        torch.cuda.synchronize()


@requires_sm90
@torch.inference_mode()
def test_candidate_gather_is_bandwidth_efficient(ws):
    """The candidate K stream must run at a useful fraction of peak bandwidth.

    This is a regression guard, not a benchmark. The gather only vectorizes if
    Triton can prove a position's value bytes are 16-byte aligned, which it
    needs the page stride's divisibility for. Adding the stride to the
    kernel's ``do_not_specialize`` list silently costs ~20x here while leaving
    every correctness test green, so pin a floor well below what a correctly
    vectorized kernel achieves.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    # Production candidate width, so the K stream dominates the measurement
    # rather than the per-step costs of the expansion and top-k stages.
    rows, num_candidates, block_size = 24, 2048, 8
    s = _setup(
        device,
        rows=rows,
        num_candidates=num_candidates,
        block_size=block_size,
        causal_len=32768,
        topk=TOPK,
        seed=23,
    )
    _reserve(ws, s)

    # Both plan modes must stay vectorized: the shared plan's buffers come
    # from the builder rather than the workspace arena, and a different
    # stride/alignment proof would silently slow the gather the same way.
    for use_plan in (False, True):
        plan = None
        if use_plan:
            plan = (
                torch.empty(rows, dtype=torch.int32, device=device),
                torch.empty(rows, dtype=torch.int32, device=device),
                torch.empty(rows, num_candidates, dtype=torch.int32, device=device),
                torch.empty(rows, dtype=torch.int32, device=device),
            )
            candidate_paged_prepare(
                s["seq_lens"], s["candidate_blocks"], block_size, 1, rows, out=plan
            )
        for _ in range(5):
            _run(s, plan=plan)
        torch.cuda.synchronize()

        iters = 20
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            _run(s, plan=plan)
        stop.record()
        torch.cuda.synchronize()
        micros = start.elapsed_time(stop) * 1e3 / iters

        bytes_moved = (
            rows * num_candidates * block_size * (HEAD_DIM + INDEXER_SCALE_BYTES)
        )
        gb_per_s = bytes_moved / micros / 1e3
        assert gb_per_s > 300, (
            f"candidate gather (plan={use_plan}) only reached {gb_per_s:.0f} "
            f"GB/s ({micros:.0f} us for the K stream); the loads are "
            "probably no longer vectorized"
        )


@requires_sm90
@torch.inference_mode()
def test_workspace_reservation_covers_the_hot_path(ws):
    """Reserving for the largest row count must let smaller steps run after
    the workspace is locked, which is what the real profiling run does."""
    device = torch.device("cuda", torch.cuda.current_device())
    big = _setup(device, rows=16, num_candidates=128, causal_len=2048, topk=TOPK)
    _reserve(ws, big)
    ws.lock()
    for rows in (1, 4, 16):
        s = _setup(
            device, rows=rows, num_candidates=128, causal_len=2048, topk=TOPK, seed=rows
        )
        _run(s)
    torch.cuda.synchronize()
