# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V4 in-image bidirectional SWA visibility (vision variant).

The reference semantics are a torch port of the official
``get_image_visible`` / ``get_window_topk_idxs_visible``: inside an image span
[span_start, span_end] (inclusive), a token at ``pos`` sees
``min(pos - span_start, max_image_tokens - 1)`` extra tokens to the left and
``min(span_end - pos, max_image_tokens)`` to the right; the window then starts
at ``max(pos - (window - 1) - max(left - (window - 1), 0), 0)`` and ends at
``pos + right`` (inclusive).
"""

import inspect

import pytest
import torch
from transformers import DeepseekV4Config
from typing_extensions import TypedDict

from tests.v1.attention.utils import create_vllm_config
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    build_flashinfer_mixed_sparse_indices,
    combine_topk_swa_indices,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
    _compute_image_visibility_kernel,
    _compute_swa_indices_and_lens_kernel,
)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

WINDOW = 8
MAX_IMG = 6
WIDTH = WINDOW + MAX_IMG
BLOCK_SIZE = 64


def ref_left_right(
    seq_lens: list[int],
    query_lens: list[int],
    spans_per_req: list[list[tuple[int, int]]],
    max_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Per-token (left, right) for the flattened decode-first token stream."""
    lefts: list[int] = []
    rights: list[int] = []
    for seq_len, query_len, spans in zip(seq_lens, query_lens, spans_per_req):
        prefix_len = seq_len - query_len
        for i in range(query_len):
            pos = prefix_len + i
            left = right = 0
            for span_start, span_end in spans:
                if span_start <= pos <= span_end:
                    left = min(pos - span_start, max_image_tokens - 1)
                    right = min(span_end - pos, max_image_tokens)
            lefts.append(left)
            rights.append(right)
    return lefts, rights


def ref_swa_bounds(
    pos: int, window: int, left: int, right: int, replay_start: int = 0
) -> tuple[int, int]:
    """Reference [start, end) window bounds for one query token. Under SWA
    bounded replay no window KV exists below ``replay_start``."""
    left_add = max(left - (window - 1), 0)
    start = max(pos - (window - 1) - left_add, 0, replay_start)
    return start, pos + right + 1


def make_batch(
    seq_lens: list[int],
    query_lens: list[int],
    device: torch.device,
):
    """Build the kernel inputs for a decode-first batch."""
    num_reqs = len(seq_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.tensor(query_lens, dtype=torch.int32, device=device),
    )
    max_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=device
    ).view(num_reqs, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    return query_start_loc, seq_lens_t, token_to_req, slot_mapping, block_table


def ref_swa_slot_rows(
    seq_lens: list[int],
    query_lens: list[int],
    spans_per_req: list[list[tuple[int, int]]],
    block_table: torch.Tensor,
    window: int,
    max_image_tokens: int,
    width: int,
    replay_starts: list[int] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Reference paged slot-id rows and lens for every token in the batch."""
    block_table_cpu = block_table.cpu()
    lefts, rights = ref_left_right(
        seq_lens, query_lens, spans_per_req, max_image_tokens
    )
    replay_starts = replay_starts or [0] * len(seq_lens)
    rows: list[list[int]] = []
    lens: list[int] = []
    token = 0
    for req, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        prefix_len = seq_len - query_len
        for i in range(query_len):
            pos = prefix_len + i
            start, end = ref_swa_bounds(
                pos, window, lefts[token], rights[token], replay_starts[req]
            )
            row = []
            for p in range(start, end):
                blk = int(block_table_cpu[req, p // BLOCK_SIZE])
                row.append(blk * BLOCK_SIZE + p % BLOCK_SIZE)
            lens.append(len(row))
            row.extend([-1] * (width - len(row)))
            rows.append(row)
            token += 1
    return rows, lens


def run_swa_kernel(
    seq_lens: list[int],
    query_lens: list[int],
    spans_per_req: list[list[tuple[int, int]]],
    window: int = WINDOW,
    max_image_tokens: int = MAX_IMG,
    with_image: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    query_start_loc, seq_lens_t, token_to_req, slot_mapping, block_table = make_batch(
        seq_lens, query_lens, device
    )
    num_tokens = int(query_start_loc[-1])
    width = window + (max_image_tokens if with_image else 0)
    swa_indices = torch.zeros(num_tokens, 1, width, dtype=torch.int32, device=device)
    swa_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    is_valid = slot_mapping >= 0
    replay_start_t = torch.zeros(len(seq_lens), dtype=torch.int32, device=device)

    if with_image:
        lefts, rights = ref_left_right(
            seq_lens, query_lens, spans_per_req, max_image_tokens
        )
        left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
        right_t = torch.tensor(rights, dtype=torch.int32, device=device)
    else:
        left_t = right_t = swa_lens  # unused dummies

    _compute_swa_indices_and_lens_kernel[(num_tokens,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        window,
        width,
        left_t,
        right_t,
        query_start_loc,
        seq_lens_t,
        token_to_req,
        is_valid,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        replay_start_t,
        token_offset=0,
        HAS_IMAGE=with_image,
        TRITON_BLOCK_SIZE=1024,
    )
    return swa_indices[:, 0], swa_lens


# seq_lens, query_lens, spans per request (positions are prompt-absolute,
# inclusive on both ends, matching the reference's sentinel-bracketed spans).
class _Case(TypedDict):
    seq_lens: list[int]
    query_lens: list[int]
    spans: list[list[tuple[int, int]]]


CASES: list[_Case] = [
    # two image spans in one request, one pure-text request
    {
        "seq_lens": [30, 12],
        "query_lens": [30, 12],
        "spans": [[(4, 12), (20, 24)], []],
    },
    # span larger than max_image_tokens, straddling the window boundary
    {
        "seq_lens": [26],
        "query_lens": [26],
        "spans": [[(2, 17)]],
    },
    # chunked prefill: span fully inside the second chunk's query
    {
        "seq_lens": [40, 9],
        "query_lens": [16, 9],
        "spans": [[(30, 38)], []],
    },
    # span ending exactly at the prompt end; tiny request
    {
        "seq_lens": [7, 15],
        "query_lens": [7, 15],
        "spans": [[(0, 6)], [(10, 14)]],
    },
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", CASES)
def test_swa_indices_kernel_with_image_spans(case):
    rows, lens = ref_swa_slot_rows(
        case["seq_lens"],
        case["query_lens"],
        case["spans"],
        make_batch(case["seq_lens"], case["query_lens"], torch.device("cuda"))[4],
        WINDOW,
        MAX_IMG,
        WIDTH,
    )
    indices, actual_lens = run_swa_kernel(
        case["seq_lens"], case["query_lens"], case["spans"], with_image=True
    )
    assert actual_lens.cpu().tolist() == lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", CASES)
def test_swa_indices_kernel_without_image_unchanged(case):
    """HAS_IMAGE=False must reproduce the plain causal sliding window."""
    rows, lens = ref_swa_slot_rows(
        case["seq_lens"],
        case["query_lens"],
        [[] for _ in case["seq_lens"]],
        make_batch(case["seq_lens"], case["query_lens"], torch.device("cuda"))[4],
        WINDOW,
        MAX_IMG,
        WINDOW,
    )
    indices, actual_lens = run_swa_kernel(
        case["seq_lens"], case["query_lens"], case["spans"], with_image=False
    )
    assert indices.shape[-1] == WINDOW
    assert actual_lens.cpu().tolist() == lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_image_visibility_kernel():
    """The builder's visibility kernel must match get_image_visible."""
    device = torch.device("cuda")
    seq_lens = [30, 12]
    query_lens = [30, 12]
    spans = [[(4, 12), (20, 24)], []]
    query_start_loc, seq_lens_t, token_to_req, _, _ = make_batch(
        seq_lens, query_lens, device
    )
    num_tokens = int(query_start_loc[-1])

    # CSR span layout: request -> contiguous [start, end) span list.
    indptr = [0, 2, 2]
    starts = [4, 20]
    ends = [12, 24]
    left = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    right = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    _compute_image_visibility_kernel[(num_tokens,)](
        left,
        right,
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(ends, dtype=torch.int32, device=device),
        query_start_loc,
        seq_lens_t,
        token_to_req,
        MAX_IMG,
        token_offset=0,
    )
    ref_left, ref_right = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
    assert left.cpu().tolist() == ref_left
    assert right.cpu().tolist() == ref_right


def combine_case(
    compress_ratio: int,
    topk: int,
    seq_lens: list[int],
    query_lens: list[int],
    spans: list[list[tuple[int, int]]],
    with_image: bool,
    replay_starts: list[int] | None = None,
    combine_fn=combine_topk_swa_indices,
    in_range_topk: bool = False,
):
    """Run combine_topk_swa_indices and return (indices, lens, expected).

    ``in_range_topk`` draws top-k indices below N, for combiners that drop
    out-of-range ones.
    """
    device = torch.device("cuda")
    num_reqs = len(seq_lens)
    replay_starts = replay_starts or [0] * num_reqs
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    # The builder's gather covers only the context above the replay start.
    gather_lens = torch.tensor(
        [
            q + min(max(s - q - r, 0), WINDOW - 1)
            for s, q, r in zip(seq_lens, query_lens, replay_starts)
        ],
        dtype=torch.int32,
        device=device,
    )
    N = (max(seq_lens) + compress_ratio - 1) // compress_ratio
    M = N + int(gather_lens.max()) + 8
    gen = torch.Generator(device="cpu").manual_seed(0)
    topk_indices = torch.randint(
        0,
        N if in_range_topk else 4096,
        (num_tokens, max(topk, 1)),
        generator=gen,
        dtype=torch.int32,
    ).to(device)
    topk_indices = topk_indices[:, : max(topk, 1)]

    if with_image:
        lefts, rights = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
        left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
        right_t = torch.tensor(rights, dtype=torch.int32, device=device)
    else:
        left_t = right_t = None

    # V4.1's combine has no image windows (causal SWA only).
    has_image = "left_visible" in inspect.signature(combine_fn).parameters
    image_kwargs = (
        dict(left_visible=left_t, right_visible=right_t, max_image_tokens=MAX_IMG)
        if has_image
        else {}
    )
    combined_indices, combined_lens = combine_fn(
        topk_indices,
        query_start_loc,
        seq_lens_t,
        gather_lens,
        WINDOW,
        compress_ratio,
        topk,
        M,
        N,
        **image_kwargs,
    )

    # Reference rows.
    lefts, rights = ref_left_right(
        seq_lens,
        query_lens,
        spans if with_image else [[] for _ in seq_lens],
        MAX_IMG,
    )
    topk_cpu = topk_indices.cpu()
    width = WINDOW + (MAX_IMG if has_image else 0)
    combined_topk = (topk + width + 127) // 128 * 128
    rows = []
    lens = []
    token = 0
    for b, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        prefix_len = seq_len - query_len
        gather_start = seq_len - int(gather_lens[b])
        for i in range(query_len):
            pos = prefix_len + i
            topk_len = min((pos + 1) // compress_ratio, topk)
            start, end = ref_swa_bounds(pos, WINDOW, lefts[token], rights[token])
            # The window never reaches below the gathered buffer.
            start = max(start, gather_start)
            swa_len = end - start
            row = [-1] * combined_topk
            for j in range(topk_len):
                row[j] = int(topk_cpu[token, j]) + M * b
            for j in range(swa_len):
                row[topk_len + j] = M * b + N + start + j - gather_start
            rows.append(row)
            lens.append(topk_len + swa_len)
            token += 1
    return combined_indices, combined_lens, rows, lens


COMBINE_CASES = [
    dict(compress_ratio=1, topk=0),  # SWA-only layer
    dict(compress_ratio=4, topk=16),  # C4A layer
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cfg", COMBINE_CASES)
def test_combine_topk_swa_with_image_spans(cfg):
    case = CASES[0]
    indices, lens, rows, exp_lens = combine_case(
        cfg["compress_ratio"],
        cfg["topk"],
        case["seq_lens"],
        case["query_lens"],
        case["spans"],
        with_image=True,
    )
    assert lens.cpu().tolist() == exp_lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cfg", COMBINE_CASES)
@pytest.mark.parametrize("query_len", [24, 20])
def test_v41_combine_topk_swa_stops_at_replay_start(cfg, query_len):
    """SWA bounded replay: the gathered buffer starts at replay_start, so the
    window never indexes below it."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        combine_topk_swa_indices as combine_v41,
    )

    # Request 0 replays from 16: a 24-token chunk starts there, a 20-token
    # chunk leaves only 4 context tokens above it (fewer than the window).
    indices, lens, rows, exp_lens = combine_case(
        cfg["compress_ratio"],
        cfg["topk"],
        seq_lens=[40, 12],
        query_lens=[query_len, 12],
        spans=[[], []],
        with_image=False,
        replay_starts=[16, 0],
        combine_fn=combine_v41,
        in_range_topk=True,
    )
    assert lens.cpu().tolist() == exp_lens
    assert indices.cpu().tolist() == rows


def _require_bi_kernels():
    from vllm.model_executor.kernels.batch_invariant import (
        batch_invariant_quant_kernel_available,
    )

    if not torch.cuda.is_available() or not batch_invariant_quant_kernel_available():
        pytest.skip("needs CUDA and the batch-invariant extension")


def _swa_first_chunk(prefix_query_replay, compress_ratio, ragged, base, gen):
    """One combine chunk: query_start_loc offset by ``base``, per-request
    gather lengths stopping at the replay start, and top-k rows that are
    ascending (compact) or random with -1 holes, negatives and ids >= N
    past and inside each row's length (ragged)."""
    device = torch.device("cuda")
    window, width = 128, 512
    seq_lens = [p + q for p, q, _ in prefix_query_replay]
    query_lens = [q for _, q, _ in prefix_query_replay]
    gather_lens = [
        q + min(max(p - r, 0), window - 1) for p, q, r in prefix_query_replay
    ]
    n = max(s // compress_ratio for s in seq_lens) if compress_ratio else 0
    m = n + max(gather_lens)
    starts = [base]
    for q in query_lens:
        starts.append(starts[-1] + q)
    positions = torch.cat(
        [torch.arange(p, p + q) for p, q, _ in prefix_query_replay]
    ).view(-1, 1)
    visible = (positions + 1) // max(compress_ratio, 1)
    tokens = positions.shape[0]
    if ragged:
        topk = torch.randint(-1, n + 64, (tokens, width), generator=gen)
        topk[torch.rand(tokens, width, generator=gen) < 0.05] = -7
    else:
        draw = torch.randint(0, 1 << 30, (tokens, width), generator=gen)
        topk = (draw % visible.clamp(min=1)).sort(-1).values
        topk[torch.arange(width).view(1, -1) >= visible] = -1

    def int32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return dict(
        topk_indices=topk.to(device=device, dtype=torch.int32),
        query_start_loc=int32(starts),
        seq_lens=int32(seq_lens),
        gather_lens=int32(gather_lens),
        window_size=window,
        compress_ratio=compress_ratio,
        topk=width if compress_ratio else 0,
        M=m,
        N=n,
    )


@pytest.mark.parametrize("compress_ratio", [0, 1, 2, 4])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("length", [1, 127, 128, 129, 700, 2600, 16384])
def test_v41_bi_combine_swa_first_matches_combine(length, ragged, compress_ratio):
    """The batch-invariant SWA-first op returns combine_topk_swa_indices' ids
    and lengths in the mega decode kernel's slot space, [SWA padded to the
    window with -1; top-k], length window + topk_len, for fresh, continued
    and replay-bounded requests split over two chunks of one workspace."""
    from vllm.model_executor.kernels.batch_invariant import (
        combine_swa_first_indices,
    )
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        combine_topk_swa_indices as combine_v41,
    )

    _require_bi_kernels()
    gen = torch.Generator().manual_seed(length * 8 + compress_ratio * 2 + ragged)
    chunks = [
        _swa_first_chunk([(0, length, 0)], compress_ratio, ragged, 7, gen),
        _swa_first_chunk(
            [(50, length, 0), (5000, length, 4990), (300, length, 0)],
            compress_ratio,
            ragged,
            7 + length,
            gen,
        ),
    ]
    workspace = torch.full((3 * length, 640), 12345, dtype=torch.int32).cuda()
    lens_workspace = torch.full((3 * length,), 12345, dtype=torch.int32).cuda()
    for chunk in chunks:
        tokens = chunk["topk_indices"].shape[0]
        indices, lens = combine_swa_first_indices(
            **chunk, out=(workspace[:tokens], lens_workspace[:tokens])
        )
        combined, combined_lens = combine_v41(**chunk)
        width = combined.shape[1]

        starts = chunk["query_start_loc"]
        query_lens = (starts[1:] - starts[:-1]).long()
        prefix = torch.repeat_interleave(chunk["seq_lens"] - query_lens, query_lens)
        first = torch.repeat_interleave(starts[:-1] - starts[0], query_lens)
        pos = prefix + torch.arange(tokens, device=prefix.device) - first
        topk_len = torch.zeros_like(pos)
        if compress_ratio:
            topk_len = ((pos + 1) // compress_ratio).clamp(0, chunk["topk"])
        swa_len = combined_lens - topk_len
        slot = torch.arange(width, device=pos.device).view(1, -1)
        in_swa = slot < 128
        source = torch.where(in_swa, topk_len[:, None] + slot, slot - 128)
        valid = torch.where(
            in_swa, slot < swa_len[:, None], slot - 128 < topk_len[:, None]
        )
        expected = torch.where(
            valid, combined.gather(1, source.clamp(0, width - 1)), -1
        )
        assert torch.equal(indices[:, :width], expected)
        assert torch.equal(lens, (128 + topk_len).int())


def test_v41_bi_combine_swa_first_replays_in_cuda_graph():
    """The SWA-first op captures into a CUDA graph and its replay follows
    new top-k and metadata contents."""
    from vllm.model_executor.kernels.batch_invariant import (
        combine_swa_first_indices,
    )

    _require_bi_kernels()
    gen = torch.Generator().manual_seed(0)
    requests = [(40, 300, 0), (900, 64, 850), (0, 5, 0)]
    static = _swa_first_chunk(requests, 2, True, 3, gen)
    out = (
        torch.empty((369, 640), dtype=torch.int32, device="cuda"),
        torch.empty(369, dtype=torch.int32, device="cuda"),
    )
    combine_swa_first_indices(**static, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combine_swa_first_indices(**static, out=out)
    # Same M and N, so the captured scalars still hold.
    fresh = _swa_first_chunk([(40, 300, 0), (900, 64, 870), (0, 5, 0)], 2, True, 9, gen)
    for name in ("topk_indices", "query_start_loc", "seq_lens", "gather_lens"):
        static[name].copy_(fresh[name])
    graph.replay()
    expected = combine_swa_first_indices(**fresh)
    assert torch.equal(out[0], expected[0]) and torch.equal(out[1], expected[1])


@pytest.mark.parametrize("compress_ratio", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 127, 128, 129, 700, 2600, 16384])
def test_v41_bi_topk_lens_by_position(seq_len, compress_ratio):
    """Decode rows (1 to 7 per request, as spec verify gives) get the top-k
    length prefill rows get at the same position, max(min((pos + 1) //
    compress_ratio, top_k), 0); padding rows get 0."""
    from vllm.model_executor.kernels.batch_invariant import topk_lens_by_position

    _require_bi_kernels()
    query_lens = [1, 7, 3, 1]
    seq_lens = [seq_len, seq_len + 6, max(seq_len, 3), 2 * seq_len]
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + q)
    tokens = starts[-1] + 5  # padding rows past the batch
    token_to_req = torch.cat(
        [torch.full((q,), r) for r, q in enumerate(query_lens)] + [torch.full((5,), 77)]
    )
    is_valid = torch.arange(tokens) < starts[-1]
    is_valid[2] = False
    lengths = torch.full((tokens,), 12345, dtype=torch.int32, device="cuda")
    topk_lens_by_position(
        token_to_req.int().cuda(),
        is_valid.cuda(),
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        compress_ratio,
        512,
        lengths,
    )
    expected = []
    for token in range(tokens):
        if not is_valid[token]:
            expected.append(0)
            continue
        r = int(token_to_req[token])
        pos = seq_lens[r] - query_lens[r] + token - starts[r]
        expected.append(max(min((pos + 1) // compress_ratio, 512), 0))
    assert lengths.tolist() == expected


def _paged_indices_batch(requests, compress_ratio, topk_rows, gen):
    """Metadata for (request, prefix, query_len, replay_start) rows, with
    per-request shuffled pages; top-k rows come from ``topk_rows[request]``
    at absolute positions."""
    window, swa_block, top_k = 128, 32, 512
    compressed_block = 64
    starts, seq_lens, replay, token_to_req, topk = [0], [], [], [], []
    swa_tables, compressed_tables = [], []
    for i, (request, prefix, query_len, replay_start) in enumerate(requests):
        starts.append(starts[-1] + query_len)
        seq_lens.append(prefix + query_len)
        replay.append(replay_start)
        token_to_req += [i] * query_len
        topk.append(topk_rows[request][prefix : prefix + query_len])
        g = torch.Generator().manual_seed(1000 + request)
        swa_tables.append(torch.randperm(1024, generator=g)[:640])
        compressed_tables.append(torch.randperm(1024, generator=g)[:320])
    padding = 3
    token_to_req += [0] * padding
    is_valid = torch.arange(len(token_to_req)) < starts[-1]
    topk.append(torch.full((padding, top_k), -1))

    def cuda_int(values):
        return torch.as_tensor(values).to(device="cuda", dtype=torch.int32)

    return dict(
        topk=cuda_int(torch.cat(topk)) if compress_ratio else None,
        token_to_req=cuda_int(token_to_req),
        is_valid=is_valid.cuda(),
        query_start_loc=cuda_int(starts),
        seq_lens=cuda_int(seq_lens),
        replay_start=cuda_int(replay),
        compressed_table=(
            cuda_int(torch.stack(compressed_tables)) if compress_ratio else None
        ),
        compressed_block_size=compressed_block,
        swa_table=cuda_int(torch.stack(swa_tables)),
        swa_block_size=swa_block,
        window=window,
        compress_ratio=compress_ratio,
        top_k=top_k if compress_ratio else 0,
    )


def _run_paged_indices(batch, token_offset=0, num_tokens=None):
    from vllm.model_executor.kernels.batch_invariant import sparse_paged_indices

    if num_tokens is None:
        num_tokens = batch["token_to_req"].numel() - token_offset
    compressed = batch["compress_ratio"] > 0
    out = torch.full((num_tokens, 640), 12345, dtype=torch.int32, device="cuda")
    lens = torch.full((2, num_tokens), 12345, dtype=torch.int32, device="cuda")
    sparse_paged_indices(
        **{k: v for k, v in batch.items() if k not in ("window", "top_k")},
        token_offset=token_offset,
        num_tokens=num_tokens,
        window=batch["window"],
        top_k=batch["top_k"],
        swa_indices=out[:, :128],
        swa_lens=lens[0],
        extra_indices=out[:, 128:] if compressed else None,
        extra_lens=lens[1] if compressed else None,
    )
    return out, lens


@pytest.mark.parametrize("compress_ratio", [0, 1, 2, 4])
def test_v41_bi_sparse_paged_indices(compress_ratio):
    """Each causal row gets the paged SWA and compressed slots the eager
    prefill path exposes -- the window clipped by the gathered range under
    replay, top-k cut at its position with out-of-range ids as holes -- and
    a (request, position) gets the same lists and lengths whether it is a
    prefill row, a chunk row or a decode / verify row of another batch."""
    _require_bi_kernels()
    gen = torch.Generator().manual_seed(compress_ratio)
    lengths = [16384, 700, 129, 5]
    topk_rows = []
    for length in lengths:
        # Causal ids as the indexer gives them, plus holes and ids past the
        # request's compressed length.
        visible = ((torch.arange(length) + 1) // max(compress_ratio, 1)).clamp(min=1)
        rows = torch.randint(0, 1 << 30, (length, 512), generator=gen)
        rows %= visible[:, None]
        noise = torch.rand(length, 512, generator=gen)
        rows[noise < 0.03] = -1
        rows[(noise >= 0.03) & (noise < 0.05)] = -7
        rows[(noise >= 0.05) & (noise < 0.07)] = length + 7
        topk_rows.append(rows)
    prefill = [(0, 0, 16384, 0), (1, 0, 700, 0), (2, 0, 129, 0), (3, 0, 5, 0)]
    mixed = [
        (1, 699, 1, 0),
        (0, 9000, 6, 0),
        (2, 64, 65, 0),
        (0, 5000, 300, 4990),
        (3, 2, 3, 0),
    ]
    results: dict[tuple[int, int], list[list]] = {}
    for name, requests in (("prefill", prefill), ("mixed", mixed)):
        batch = _paged_indices_batch(requests, compress_ratio, topk_rows, gen)
        out, lens = _run_paged_indices(batch)
        tokens = batch["token_to_req"].numel()
        split = tokens // 3
        head, head_lens = _run_paged_indices(batch, 0, split)
        tail, tail_lens = _run_paged_indices(batch, split)
        assert torch.equal(torch.cat([head, tail]), out)
        assert torch.equal(torch.cat([head_lens, tail_lens], 1), lens)
        out, lens = out.cpu(), lens.cpu()
        if not compress_ratio:
            assert (out[:, 128:] == 12345).all() and (lens[1] == 12345).all()
        swa_table = batch["swa_table"].cpu()
        compressed_table = batch["compressed_table"]
        row = 0
        for i, (request, prefix, query_len, replay_start) in enumerate(requests):
            seq_len = prefix + query_len
            gather = query_len + min(max(prefix - replay_start, 0), 127)
            for pos in range(prefix, seq_len):
                swa_start = max(pos - 127, 0, seq_len - gather)
                swa = [
                    int(swa_table[i, p // 32]) * 32 + p % 32
                    for p in range(swa_start, pos + 1)
                ]
                assert lens[0, row] == len(swa)
                assert out[row, : len(swa)].tolist() == swa
                assert (out[row, len(swa) : 128] == -1).all()
                # Slots differ by batch only through the page tables.
                seen = [swa_start, lens[:, row].tolist()]
                if compress_ratio:
                    topk_len = min((pos + 1) // compress_ratio, 512)
                    local = topk_rows[request][pos, :topk_len]
                    ok = (local >= 0) & (local < seq_len // compress_ratio)
                    pages = compressed_table[i].cpu()[local.clamp(0) // 64]
                    slots = torch.where(ok, pages * 64 + local.clamp(0) % 64, -1)
                    assert lens[1, row] == topk_len
                    assert torch.equal(out[row, 128 : 128 + topk_len], slots.int())
                    assert (out[row, 128 + topk_len :] == -1).all()
                    seen.append(local.where(ok, -1).tolist())
                if not replay_start:
                    results.setdefault((request, pos), []).append(seen)
                row += 1
        assert (lens[0, row:] == 0).all() and (out[row:, :128] == -1).all()
        if compress_ratio:
            assert (lens[1, row:] == 0).all() and (out[row:, 128:] == -1).all()
    shared = [seen for seen in results.values() if len(seen) > 1]
    assert len(shared) == 1 + 6 + 65 + 3
    assert all(seen[0] == seen[1] for seen in shared)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cfg", COMBINE_CASES)
def test_combine_topk_swa_without_image_unchanged(cfg):
    """left_visible=None must reproduce the plain causal combined indices."""
    case = CASES[0]
    indices, lens, rows, exp_lens = combine_case(
        cfg["compress_ratio"],
        cfg["topk"],
        case["seq_lens"],
        case["query_lens"],
        case["spans"],
        with_image=False,
    )
    assert lens.cpu().tolist() == exp_lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_mixed_sparse_indices_with_image_spans():
    device = torch.device("cuda")
    # 1 decode token (req 0) + two prefill requests (reqs 1, 2).
    seq_lens = [20, 30, 9]
    query_lens = [1, 30, 9]
    spans = [[], [(4, 12)], [(0, 6)]]
    query_start_loc = torch.tensor([0, 1, 31, 40], dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0] + [1] * 30 + [2] * 9, dtype=torch.int32, device=device
    )
    block_table = torch.arange(3, dtype=torch.int32, device=device).view(3, 1)
    decode_swa = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107]], dtype=torch.int32, device=device
    )
    prefill_topk = torch.zeros(39, 0, dtype=torch.int32, device=device)

    lefts, rights = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
    left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
    right_t = torch.tensor(rights, dtype=torch.int32, device=device)

    sparse_indices, sparse_lens = build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=decode_swa,
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=prefill_topk,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        token_to_req_indices=token_to_req,
        swa_block_table=block_table,
        swa_block_size=BLOCK_SIZE,
        compressed_block_table=None,
        compressed_block_size=BLOCK_SIZE,
        window_size=WINDOW,
        compress_ratio=1,
        topk=0,
        prefill_left_visible=left_t,
        prefill_right_visible=right_t,
        max_image_tokens=MAX_IMG,
    )

    swa_total = WINDOW + MAX_IMG
    assert sparse_indices.shape == (40, swa_total)
    assert sparse_lens.cpu().tolist() == [swa_total] * 40
    # Decode row: slots copied, image-extension columns padded with -1.
    decode_row = sparse_indices[0].cpu().tolist()
    assert decode_row[:WINDOW] == list(range(100, 108))
    assert decode_row[WINDOW:] == [-1] * MAX_IMG
    # Prefill rows: paged slot ids over the widened window.
    exp_rows, _ = ref_swa_slot_rows(
        seq_lens, query_lens, spans, block_table, WINDOW, MAX_IMG, swa_total
    )
    actual = sparse_indices[1:].cpu().tolist()
    assert actual == exp_rows[1:]


def make_builder(
    vision: bool,
    window: int = WINDOW,
    max_image_tokens: int = MAX_IMG,
    max_num_batched_tokens: int = 64,
    model_name: str = "Qwen/Qwen3-0.6B",
    builder_cls: type[
        DeepseekSparseSWAMetadataBuilder
    ] = DeepseekSparseSWAMetadataBuilder,
    deepseek_v4_config: bool = True,
    bounded_replay: bool = False,
) -> DeepseekSparseSWAMetadataBuilder:
    vision_fields = dict(vision_n_layers=2, vision_max_n_token=max_image_tokens)
    vllm_config = create_vllm_config(
        model_name=model_name,
        max_model_len=4096,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=8,
        hf_config_override={"sliding_window": window}
        | (vision_fields if vision and not deepseek_v4_config else {}),
    )
    if vision and deepseek_v4_config:
        # Only V4 configs widen the SWA window in-kernel over image spans
        vllm_config.model_config.hf_config = DeepseekV4Config(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=window,
            **vision_fields,
        )
    spec = SlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=window,
        cache_dtype_str="auto",
        model_version="deepseek_v4",
        bounded_replay=bounded_replay,
    )
    return builder_cls(
        kv_cache_spec=spec,
        layer_names=["layer0"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )


def build_metadata(
    builder: DeepseekSparseSWAMetadataBuilder,
    seq_lens: list[int],
    query_lens: list[int],
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None,
    replay_starts: list[int] | None = None,
):
    device = torch.device("cuda")
    query_start_loc, seq_lens_t, _, slot_mapping, block_table = make_batch(
        seq_lens, query_lens, device
    )
    replay_start = (
        None
        if replay_starts is None
        else torch.tensor(replay_starts, dtype=torch.int32, device=device)
    )
    return builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens_t,
            seq_lens_cpu_upper_bound=seq_lens_t.cpu(),
            num_reqs=len(seq_lens),
            num_actual_tokens=int(query_start_loc[-1]),
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            mm_req_doc_ranges=mm_req_doc_ranges,
        ),
        replay_start=replay_start,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_in_image_visibility():
    seq_lens = [30, 12]
    query_lens = [30, 12]
    spans = [[(4, 12), (20, 24)], []]
    builder = make_builder(vision=True)
    md = build_metadata(builder, seq_lens, query_lens, {0: spans[0], 1: spans[1]})
    assert md.num_prefills == 2
    assert md.num_decode_tokens == 0
    assert md.prefill_swa_indices.shape[-1] == WIDTH
    assert md.prefill_left_visible is not None
    assert md.prefill_right_visible is not None

    ref_left, ref_right = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
    assert md.prefill_left_visible.cpu().tolist() == ref_left
    assert md.prefill_right_visible.cpu().tolist() == ref_right

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens, query_lens, spans, block_table, WINDOW, MAX_IMG, WIDTH
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_no_image_spans_fast_path():
    """Vision model, image-free batch: no visibility tensors, plain window."""
    seq_lens = [30, 12]
    query_lens = [30, 12]
    builder = make_builder(vision=True)
    md = build_metadata(builder, seq_lens, query_lens, {0: [], 1: []})
    assert md.prefill_left_visible is None
    assert md.prefill_right_visible is None

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens, query_lens, [[], []], block_table, WINDOW, MAX_IMG, WIDTH
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_text_model_unchanged():
    """Text-only model: buffers stay window-sized and spans are ignored."""
    seq_lens = [30, 12]
    query_lens = [30, 12]
    builder = make_builder(vision=False)
    assert builder.max_image_tokens == 0
    md = build_metadata(builder, seq_lens, query_lens, None)
    assert md.prefill_swa_indices.shape[-1] == WINDOW
    assert md.prefill_left_visible is None

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens, query_lens, [[], []], block_table, WINDOW, MAX_IMG, WINDOW
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compress_ratio", [0, 1, 2])
@pytest.mark.parametrize("query_len", [100, 4000])
def test_v41_image_prefill_uses_causal_swa(compress_ratio, query_len):
    """V4.1 image tokens use the same causal SWA as text, including in chunks."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        combine_topk_swa_indices as combine_v41,
    )
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV41SparseSWAMetadataBuilder

    seq_len, window, max_image_tokens = 4000, 128, 2048
    spans = [(1900, 3947)]
    builder = make_builder(
        True,
        window,
        max_image_tokens,
        query_len,
        builder_cls=DeepseekV41SparseSWAMetadataBuilder,
        deepseek_v4_config=False,
    )
    assert builder.max_image_tokens == 0
    md = build_metadata(builder, [seq_len], [query_len], {0: spans})
    assert md.prefill_left_visible is None
    assert md.prefill_right_visible is None
    assert md.prefill_swa_indices.shape[-1] == window
    plan = md.get_prefill_chunk_plan(
        compress_ratio, prefill_chunk_size=4, has_compressed=compress_ratio > 0
    )
    [(start, end, n, m)] = plan
    gather_len = int(md.prefill_gather_lens[0])
    assert (start, end, m - n) == (0, 1, gather_len)
    assert gather_len == query_len + min(seq_len - query_len, window - 1)
    indices, lens = combine_v41(
        torch.empty(query_len, 0, dtype=torch.int32, device="cuda"),
        md.query_start_loc,
        md.prefill_seq_lens,
        md.prefill_gather_lens,
        window,
        compress_ratio,
        0,
        m,
        n,
    )
    indices, lens = indices.cpu(), lens.cpu().tolist()
    paged_indices = md.prefill_swa_indices.cpu()
    paged_lens = md.prefill_swa_lens.cpu().tolist()
    for token, pos in enumerate(range(seq_len - query_len, seq_len)):
        lo, hi = max(0, pos - window + 1), pos + 1
        row = indices[token, : lens[token]]
        assert lens[token] == hi - lo
        assert torch.all((row >= n) & (row < m))
        absolute_keys = row - n + seq_len - gather_len
        assert absolute_keys.tolist() == list(range(lo, hi))
        assert paged_lens[token] == hi - lo
        assert paged_indices[token, 0, : hi - lo].tolist() == list(range(lo, hi))


# SWA bounded replay: every prefill index path clamps the window at replay_start.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_replay_start_bounds_prefill_window_and_gather():
    """Paged-direct prefill indices stop at replay_start and the FlashMLA
    gather only covers the context the request may see."""
    seq_lens = [40, 12, 30]
    query_lens = [24, 12, 10]
    replay_starts = [12, 0, 0]
    builder = make_builder(vision=False)
    md = build_metadata(builder, seq_lens, query_lens, None, replay_starts)
    assert md.replay_start is not None
    assert md.replay_start.cpu().tolist() == replay_starts

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens,
        query_lens,
        [[], [], []],
        block_table,
        WINDOW,
        MAX_IMG,
        WINDOW,
        replay_starts=replay_starts,
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows
    # Request 0: the first replayed row (pos 16) sees [12, 16]; its last row
    # (pos 39) is a full window above the replay start and sees all of it.
    assert md.prefill_swa_lens[0].item() == 16 - 12 + 1
    assert md.prefill_swa_lens[23].item() == WINDOW
    # gather_len = query_len + min(prefix_len - replay_start, WINDOW - 1).
    assert md.prefill_gather_lens.cpu().tolist() == [
        24 + min(16 - 12, WINDOW - 1),
        12,
        10 + min(20, WINDOW - 1),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_replaying_group_builds_without_replay_start():
    """Graph captures build without model-specific kwargs: a replaying group's
    metadata then bounds nothing."""
    builder = make_builder(vision=False, bounded_replay=True)
    md = build_metadata(builder, [40, 12], [24, 12], None)
    assert not md.replay_start.any()
    assert md.prefill_swa_lens[0].item() == WINDOW


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v41_flashinfer_mixed_sparse_indices_respect_replay_start():
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices as build_v41,
    )

    device = torch.device("cuda")
    # 1 decode token (req 0) + two prefill requests; req 1 resumes at 16.
    seq_lens = [20, 40, 9]
    query_lens = [1, 24, 9]
    replay_starts = [0, 16, 0]
    query_start_loc = torch.tensor([0, 1, 25, 34], dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0] + [1] * 24 + [2] * 9, dtype=torch.int32, device=device
    )
    block_table = torch.arange(3, dtype=torch.int32, device=device).view(3, 1)
    kwargs = dict(
        decode_swa_indices=torch.zeros((1, WINDOW), dtype=torch.int32, device=device),
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=torch.empty((33, 0), dtype=torch.int32, device=device),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        token_to_req_indices=token_to_req,
        swa_block_table=block_table,
        swa_block_size=BLOCK_SIZE,
        compressed_block_table=None,
        compressed_block_size=BLOCK_SIZE,
        window_size=WINDOW,
        compress_ratio=1,
        topk=0,
    )
    plain, _ = build_v41(
        replay_start=torch.zeros(3, dtype=torch.int32, device=device), **kwargs
    )
    bounded, _ = build_v41(
        replay_start=torch.tensor(replay_starts, dtype=torch.int32, device=device),
        **kwargs,
    )
    rows, _ = ref_swa_slot_rows(
        seq_lens,
        query_lens,
        [[], [], []],
        block_table,
        WINDOW,
        MAX_IMG,
        WINDOW,
        replay_starts=replay_starts,
    )
    # Decode row is copied through; prefill rows follow the bounded window.
    assert bounded[0].cpu().tolist() == plain[0].cpu().tolist()
    assert bounded[1:].cpu().tolist() == rows[1:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_noncausal_draft_decode_update_matches_build():
    """A DSpark draft graph refreshes its non-causal SWA metadata in place
    instead of rebuilding it, so the refresh must match a fresh build."""
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )

    num_spec = 4
    query_lens = [num_spec] * 3

    def make_dspark_builder():
        builder = make_builder(vision=False)
        builder.is_dspark = True
        builder.noncausal_index_width = get_dspark_swa_index_width(WINDOW, num_spec)
        builder.decode_threshold = num_spec
        return builder

    def common_metadata(seq_lens):
        query_start_loc, seq_lens_t, _, slot_mapping, block_table = make_batch(
            seq_lens, query_lens, torch.device("cuda")
        )
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens_t,
            seq_lens_cpu_upper_bound=seq_lens_t.cpu(),
            num_reqs=len(seq_lens),
            num_actual_tokens=int(query_start_loc[-1]),
            max_query_len=num_spec,
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=False,
        )

    captured_seq_lens = [6, 40, 130]
    replay_seq_lens = [130, 9, 90]
    builder = make_dspark_builder()
    captured = common_metadata(captured_seq_lens)
    md = builder.build(0, captured)
    replay = common_metadata(replay_seq_lens)
    expected = make_dspark_builder().build(0, replay)

    # A replay sees new inputs through the same device buffers.
    captured.seq_lens.copy_(replay.seq_lens)
    captured.block_table_tensor.copy_(replay.block_table_tensor)
    builder.update_draft_decode_metadata(md)

    assert md.decode_swa_width == builder.noncausal_index_width
    torch.testing.assert_close(md.decode_swa_lens, expected.decode_swa_lens)
    torch.testing.assert_close(md.decode_swa_indices, expected.decode_swa_indices)


class _StubLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight_block_size = None


def _prefix_replay_attention_cls():
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention

    class _Backend:
        @staticmethod
        def get_name() -> str:
            return "STUB"

    class Layer(DeepseekV4Attention):
        backend_cls = _Backend

        @classmethod
        def get_padded_num_q_heads(cls, num_heads: int) -> int:
            return num_heads

        def forward_mqa(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            positions: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
            return None

        def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor):
            return attn_out

    return Layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "is_rocm,cdna,flag,replays_window",
    [
        (False, 0, True, True),
        (True, 4, True, True),
        (True, 3, True, False),
        (True, 4, False, False),
    ],
)
def test_prefix_cache_replays_the_swa_window(
    monkeypatch, is_rocm, cdna, flag, replays_window
):
    """Prefix caching stays on, and the window is replayed only on CDNA4.

    tests/v1/core/test_prefix_replay.py already checks that a
    bounded-replay SWA spec makes a prefix hit recompute the window.
    This is the spec the attention layer actually publishes. Forcing
    the window back into the prefix cache, on CDNA4 or off it, fails here.
    The replay_start clamp used by the gather kernels is covered by
    test_builder_replay_start_bounds_prefill_window_and_gather.
    """
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v41 import attention as attn
    from vllm.platforms import current_platform

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(attn, "MergedColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(attn, "ColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(attn, "RowParallelLinear", _StubLinear)
    monkeypatch.setattr(attn, "RMSNorm", _StubLinear)
    monkeypatch.setattr(attn, "build_deepseek_v4_rope", lambda *a, **k: None)
    monkeypatch.setattr(attn, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: object())
    monkeypatch.setattr(current_platform, "is_rocm", lambda: is_rocm)
    if is_rocm:
        import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "get_cdna_version", lambda: cdna)

    window = 128
    vllm_config = create_vllm_config(
        max_model_len=256, max_num_seqs=4, max_num_batched_tokens=64
    )
    vllm_config.cache_config.swa_bounded_replay = flag
    assert vllm_config.cache_config.enable_prefix_caching
    vllm_config.model_config.hf_config = DeepseekV4Config(
        num_hidden_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        q_lora_rank=32,
        o_lora_rank=32,
        head_dim=64,
        qk_rope_head_dim=16,
        o_groups=8,
        sliding_window=window,
        compress_ratios=[0],
    )
    layer_cls = _prefix_replay_attention_cls()
    with set_current_vllm_config(vllm_config):
        layer = layer_cls(vllm_config, prefix="model.layers.0.self_attn")
    spec = layer.swa_cache_layer.get_kv_cache_spec(vllm_config)
    assert spec.bounded_replay is replays_window
    assert spec.prefix_cacheable is not replays_window
    assert spec.prefix_replay_tokens == (window if replays_window else 0)


@pytest.mark.parametrize(
    "batch_invariant,expected", [("1", "nvfp4_ds_mla"), ("0", "fp8_ds_mla")]
)
def test_ds41_flashmla_bi_auto_kv_cache_dtype(monkeypatch, batch_invariant, expected):
    """Under BI, FlashMLA sparse resolves auto to the default path's NVFP4 cache."""
    from vllm.config import CacheConfig
    from vllm.models.deepseek_v41.attention import _resolve_dsv4_kv_cache_dtype
    from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", batch_invariant)
    attention = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
    cache_config = CacheConfig(cache_dtype="auto")
    resolved, dtype = _resolve_dsv4_kv_cache_dtype(
        True, "auto", cache_config, attention.packed_kv_cache_dtype
    )
    assert (resolved, dtype, cache_config.cache_dtype) == (
        expected,
        torch.uint8,
        expected,
    )


def _ds41_swa_bi_config(lengths):
    from types import SimpleNamespace

    from vllm.config import CacheConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    window, block_size, dim = 128, 32, 512
    max_tokens = sum(lengths)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sliding_window=window, compress_ratios=[0]),
            max_model_len=max(lengths),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_tokens, max_num_seqs=3
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache_config = CacheConfig(cache_dtype="nvfp4_ds_mla")
    with set_current_vllm_config(config):
        swa = DeepseekV4SWACache(
            head_dim=dim,
            window_size=window,
            dtype=torch.uint8,
            prefix="swa",
            cache_config=cache_config,
            backend_cls=DeepseekV4FlashMLAAttention.swa_backend_cls,
            block_size=block_size,
            packed_bytes_per_token=528,
            packed_page_alignment=512,
        )
    return config, swa


def _ds41_cr2_bi_builder(config, device):
    from vllm.models.deepseek_v41.sparse_mla import (
        DeepseekV4FlashMLAMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=2,
        state_content_bytes=288,
        cache_dtype_str="nvfp4_ds_mla",
        model_version="deepseek_v4",
    )
    return DeepseekV4FlashMLAMetadataBuilder(spec, ["compressed"], config, device)


def test_ds41_flashmla_cr2_bi_builder_constructs_on_cpu(monkeypatch):
    """Construct the same CR2 builder before requesting a GPU arithmetic proof."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config, _ = _ds41_swa_bi_config([12290, 65, 129])
    config.model_config.hf_config.compress_ratios = [0, 0, 2, 2]
    config.model_config.hf_config.index_topk = 512
    builder = _ds41_cr2_bi_builder(config, torch.device("cpu"))
    assert builder.compress_ratio == 2 and builder.topk_tokens == 512
    assert builder.req_id_per_token_buffer.numel() == 12484
    assert builder.compressed_slot_mapping_buffer.device.type == "cpu"


@pytest.mark.parametrize("num_spec", [0, 5])
def test_ds41_flashmla_swa_bi_builder_policy(monkeypatch, num_spec):
    """BI keeps decode and verify rows as decodes and graphs uniform batches.

    UNIFORM_BATCH lets spec decode (1 + num_spec rows per request) keep FULL
    decode graphs instead of falling back to PIECEWISE.
    """
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
    )
    from vllm.v1.attention.backend import AttentionCGSupport

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config, swa = _ds41_swa_bi_config([12290, 65, 129])
    if num_spec:
        config.speculative_config = SimpleNamespace(
            num_speculative_tokens=num_spec,
            parallel_drafting=False,
            use_dspark=lambda: False,
        )
    spec = swa.get_kv_cache_spec(config)
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
        spec, [swa.prefix], config, torch.device("cpu")
    )
    assert builder.decode_threshold == 1 + num_spec
    assert (
        builder.get_cudagraph_support(config, spec) == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize("major,supported", [(9, False), (10, True), (12, False)])
def test_ds41_swa_bi_declaration_stays_scoped(monkeypatch, major, supported):
    """The tested SM100 route must not grant BI to DS4 or FlashInfer."""
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        DeepseekSparseSWAFlashInferBackend,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLABackend,
    )
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, 0)),
    )
    assert DeepseekSparseSWAFlashMLABackend.supports_batch_invariance() == supported
    assert not DeepseekSparseSWABackend.supports_batch_invariance()
    assert not DeepseekSparseSWAFlashInferBackend.supports_batch_invariance()


@pytest.mark.parametrize("major,supported", [(9, False), (10, True), (12, False)])
def test_ds41_cr2_indexer_bi_declaration_stays_scoped(monkeypatch, major, supported):
    """The CR2 proof must not grant BI to DS4, mega attention or sparse indexer."""
    from vllm.models.deepseek_v41.sparse_mla import (
        DeepseekV4FlashMLABackend,
        FlashMLAMegaAttnBackend,
    )
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV4IndexerBackend,
        DeepseekV41IndexerBackend,
    )
    from vllm.v1.attention.backends.mla.sparse_indexer import (
        DeepseekV41SparseIndexerBackend,
    )

    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, 0)),
    )
    assert DeepseekV4FlashMLABackend.supports_batch_invariance() == supported
    assert DeepseekV41IndexerBackend.supports_batch_invariance() == supported
    assert not DeepseekV4IndexerBackend.supports_batch_invariance()
    assert not FlashMLAMegaAttnBackend.supports_batch_invariance()
    assert not DeepseekV41SparseIndexerBackend.supports_batch_invariance()


@pytest.mark.parametrize("length", [127, 129, 12290])
def test_ds41_flashmla_swa_bi_batch_chunk_pd_graph(monkeypatch, length):
    _assert_ds41_flashmla_bi_schedule(monkeypatch, length, compress_ratio=0)


@pytest.mark.parametrize("length", [129, 12290])
def test_ds41_flashmla_cr2_indexer_bi_batch_chunk_pd_graph(monkeypatch, length):
    """Actual BF16 DeepGEMM -> stable Top-K -> CR2 attention is schedule invariant."""
    _assert_ds41_flashmla_bi_schedule(monkeypatch, length, compress_ratio=2)


def _assert_ds41_flashmla_bi_schedule(monkeypatch, length, compress_ratio):
    """Scheduling cannot change actual DS41 SWA attention output bits.

    Cover window edges, all query rows across 12288, request permutation,
    chunked prefill, static-metadata graph replay, and single-token decode
    rows: eager, and replayed from one padded FULL-style decode graph while
    the batch, positions and padding change in place.
    """
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        DeepseekV4FlashMLAAttention,
    )
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    if not torch.cuda.is_available():
        pytest.skip("DS41 FlashMLA requires CUDA")
    assert torch.cuda.get_device_capability()[0] == 10
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(1949)
    device = torch.device("cuda")
    window, block_size, heads, dim = 128, 32, 64, 512
    lengths = [length, 65, 129]
    max_tokens = sum(lengths)
    config, swa = _ds41_swa_bi_config(lengths)
    config.model_config.hf_config.compress_ratios = (
        [0, 0, 2, 2] if compress_ratio else [0]
    )
    config.model_config.hf_config.index_topk = 512
    spec = swa.get_kv_cache_spec(config)
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
        spec, [swa.prefix], config, device
    )
    assert builder.decode_threshold == 1
    assert (
        builder.get_cudagraph_support(config, spec) == AttentionCGSupport.UNIFORM_BATCH
    )

    pages = (max(lengths) + block_size - 1) // block_size
    tables = torch.arange(3 * pages, dtype=torch.int32, device=device).view(3, pages)
    packed = torch.zeros(3 * pages, block_size * 528, device=device, dtype=torch.uint8)
    swa.kv_cache = packed.view(3 * pages, block_size, 528)
    queries = []
    for request, size in enumerate(lengths):
        queries.append(
            torch.randn(size, heads, dim, device=device, dtype=torch.bfloat16)
        )
        values = torch.randn(size, dim, device=device, dtype=torch.bfloat16)
        slots = torch.arange(size, device=device, dtype=torch.int64)
        slots += request * pages * block_size
        quantize_and_insert_k_cache(
            values, packed, slots, block_size=block_size, bytes_per_token=528
        )

    # Exercise the production attention subpath without allocating projections,
    # experts or embedding tables unrelated to its scheduling contract.
    attention = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.compress_ratio = compress_ratio
    attention.use_pcp = False
    attention.compressed_cache_prefix = "compressed" if compress_ratio else None
    attention.swa_cache_layer = swa
    attention.window_size = window
    attention.max_num_batched_tokens = max_tokens
    attention.topk_indices_buffer = torch.full(
        (max_tokens, 512), -1, device=device, dtype=torch.int32
    )
    attention.scale = dim**-0.5
    attention.attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    compressed_builder = None
    if compress_ratio:
        from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill
        from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

        compressed_pages = (max(lengths) + 127) // 128
        compressed_tables = torch.arange(
            3 * compressed_pages, device=device, dtype=torch.int32
        ).view(3, compressed_pages)
        compressed = torch.empty(
            3 * compressed_pages, 64 * 288, device=device, dtype=torch.uint8
        )
        # Interleaved V4.1 NVFP4 records: 256 packed bytes, then 32 scales.
        records = compressed.view(3 * compressed_pages, 64, 288)
        records[..., :256] = torch.randint(
            256, (3 * compressed_pages, 64, 256), device=device, dtype=torch.uint8
        )
        scales = (torch.rand(3 * compressed_pages, 64, 32, device=device) + 0.25).to(
            torch.float8_e4m3fn
        )
        records[..., 256:] = scales.view(torch.uint8)
        attention.is_kv_source = True
        attention.kv_cache = compressed.view(-1, 64, 288)
        attention._static_forward_context = {"compressed": attention}
        compressed_builder = _ds41_cr2_bi_builder(config, device)
        key_lengths = [(size + 1) // 2 for size in lengths]
        key_offsets = [0, key_lengths[0], sum(key_lengths[:2])]
        index_keys = torch.randint(
            -128, 128, (sum(key_lengths), 64), device=device, dtype=torch.int8
        )
        index_key_scales = torch.full(
            (sum(key_lengths),), 0x7F7F7F7F, device=device, dtype=torch.int32
        )
        index_queries = [
            torch.randint(-128, 128, (size, 32, 64), device=device, dtype=torch.int8)
            for size in lengths
        ]
        index_weights = [torch.randn(size, 32, device=device) for size in lengths]
        for weights in index_weights:
            weights[::97].zero_()
        if length > 12288:
            index_weights[0][12288].zero_()
    init_workspace_manager(device)

    def prepare(schedule):
        sizes = [end - start for _, start, end in schedule]
        ends = [end for _, _, end in schedule]
        starts_cpu = torch.tensor([0, *sizes], dtype=torch.int32).cumsum(0).int()
        starts = starts_cpu.to(device)
        q = torch.cat([queries[r][start:end] for r, start, end in schedule])
        positions = torch.cat(
            [torch.arange(start, end, device=device) for _, start, end in schedule]
        )
        common = CommonAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=starts_cpu,
            seq_lens=torch.tensor(ends, device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor(ends, dtype=torch.int32),
            num_reqs=len(schedule),
            num_actual_tokens=sum(sizes),
            max_query_len=max(sizes),
            max_seq_len=max(ends),
            block_table_tensor=torch.stack([tables[r] for r, _, _ in schedule]),
            slot_mapping=torch.arange(sum(sizes), device=device, dtype=torch.int64),
            causal=True,
        )
        metadata = builder.build(0, common)
        decodes = 1 if sizes == [1] else 0
        assert metadata.num_decodes == decodes
        assert metadata.num_prefills == len(schedule) - decodes
        assert metadata.num_decode_tokens + metadata.num_prefill_tokens == sum(sizes)
        all_metadata = {swa.prefix: metadata}
        if compressed_builder is not None:
            from dataclasses import replace

            compressed_common = replace(
                common,
                block_table_tensor=torch.stack(
                    [compressed_tables[r] for r, _, _ in schedule]
                ),
            )
            all_metadata["compressed"] = compressed_builder.build(0, compressed_common)
            index_q = torch.cat(
                [index_queries[r][start:end] for r, start, end in schedule]
            )
            weights = torch.cat(
                [index_weights[r][start:end] for r, start, end in schedule]
            )
            row_starts = torch.cat(
                [
                    torch.full(
                        (end - start,), key_offsets[r], device=device, dtype=torch.int32
                    )
                    for r, start, end in schedule
                ]
            )
            row_ends = row_starts + (positions.int() + 1) // 2
            valid_ends = row_ends - row_starts
            query_scales, native_weights = mqa_logits_mx_inputs(index_q, None, weights)
        context = ForwardContext({}, all_metadata, {})
        output = torch.empty_like(q)

        def forward():
            if compressed_builder is not None:
                scores = fp8_fp4_mqa_logits(
                    (index_q, query_scales),
                    (index_keys, index_key_scales),
                    native_weights,
                    row_starts,
                    row_ends,
                    max(key_lengths),
                )
                assert scores.dtype == torch.bfloat16
                logits = as_fp32_logits(scores, valid_ends)
                assert attention.topk_indices_buffer is not None
                top_k_per_row_prefill(
                    logits,
                    torch.zeros_like(valid_ends),
                    valid_ends,
                    attention.topk_indices_buffer[: q.shape[0]],
                    q.shape[0],
                    logits.stride(0),
                    logits.stride(1),
                    512,
                )
            with override_forward_context(context):
                attention.forward_mqa(q, q[:, 0], positions, output)

        return forward, output, metadata, q

    def run(schedule):
        forward, output, _, _ = prepare(schedule)
        forward()
        return output.clone()

    try:
        expected = [run([(r, 0, size)]) for r, size in enumerate(lengths)]
        schedule = [(1, 0, lengths[1]), (0, 0, length), (2, 0, lengths[2])]
        for order in (schedule, schedule[::-1]):
            actual = run(order)
            assert torch.equal(actual, torch.cat([expected[r] for r, _, _ in order]))

        attention.PREFILL_CHUNK_SIZE = 1  # type: ignore[misc]
        forward, output, metadata, _ = prepare(schedule)
        plan = metadata.get_prefill_chunk_plan(
            compress_ratio, 1, has_compressed=bool(compress_ratio)
        )
        # Small cases can fit one workspace chunk; the long case must split.
        if length > 12288:
            assert len(plan) > 1
        forward()
        wanted = torch.cat([expected[r] for r, _, _ in schedule])
        assert torch.equal(output, wanted)
        attention.PREFILL_CHUNK_SIZE = 4  # type: ignore[misc]
        if compress_ratio:
            attention.is_kv_source = False
            assert torch.equal(run([(0, 0, length)]), expected[0])
            attention.is_kv_source = True

        cuts = sorted({0, 64, 127, 128, length - 2, length - 1, length})
        cuts = [cut for cut in cuts if 0 <= cut <= length]
        chunks = [run([(0, start, end)]) for start, end in zip(cuts, cuts[1:])]
        assert torch.equal(torch.cat(chunks), expected[0])
        for position in sorted({0, 126, 127, 128, length - 1}):
            if position < length:
                assert torch.equal(
                    run([(0, position, position + 1)]),
                    expected[0][position : position + 1],
                )

        forward, output, _, q = prepare(schedule)
        forward()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        for multiplier in (1, -1):
            q.mul_(multiplier)
            forward()
            eager = output.clone()
            output.fill_(float("nan"))
            graph.replay()
            assert torch.equal(output, eager)
            assert torch.isfinite(output).all()

        for query_len in (1, 6):
            _assert_ds41_bi_decode_graph(
                locals(), [length, *lengths[1:]], compress_ratio, expected, query_len
            )
    finally:
        torch.accelerator.synchronize()
        reset_workspace_manager()


def _assert_ds41_bi_decode_graph(fixture, lengths, compress_ratio, expected, query_len):
    """Decode rows replayed from one padded graph equal their prefill rows.

    A FULL decode graph is captured once; later steps rewrite the persistent
    metadata, queries and indexer inputs in place and replay. Each request
    contributes ``query_len`` rows (6 is a spec-decode verify step with 5
    draft tokens). Every real row must match the eager prefill result for the
    same (request, position) bit for bit; padding rows stay inert.
    """
    from dataclasses import replace

    from vllm.forward_context import ForwardContext, override_forward_context

    attention, builder, swa = fixture["attention"], fixture["builder"], fixture["swa"]
    queries, tables, device = fixture["queries"], fixture["tables"], fixture["device"]
    heads, dim = fixture["heads"], fixture["dim"]
    capture_tokens = 24
    capture_reqs = capture_tokens // query_len
    q = torch.zeros(capture_tokens, heads, dim, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(capture_tokens, dtype=torch.int64, device=device)
    starts = torch.arange(capture_reqs + 1, dtype=torch.int32, device=device)
    starts *= query_len
    seq_lens = torch.zeros(capture_reqs, dtype=torch.int32, device=device)
    table = torch.zeros(capture_reqs, tables.shape[1], dtype=torch.int32, device=device)
    slots = torch.full((capture_tokens,), -1, dtype=torch.int64, device=device)
    output = torch.empty_like(q)
    if compress_ratio:
        from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill
        from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

        compressed_tables = fixture["compressed_tables"]
        key_offsets, key_lengths = fixture["key_offsets"], fixture["key_lengths"]
        index_queries, index_weights = (
            fixture["index_queries"],
            fixture["index_weights"],
        )
        index_keys, index_key_scales = (
            fixture["index_keys"],
            fixture["index_key_scales"],
        )
        compressed_table = torch.zeros(
            capture_reqs, compressed_tables.shape[1], dtype=torch.int32, device=device
        )
        index_q = torch.zeros(
            capture_tokens,
            *index_queries[0].shape[1:],
            device=device,
            dtype=torch.int8,
        )
        weights = torch.zeros(capture_tokens, index_weights[0].shape[1], device=device)
        row_starts = torch.zeros(capture_tokens, dtype=torch.int32, device=device)
        row_ends = torch.zeros_like(row_starts)

    def tokens(reqs):
        return [
            (i, request, position)
            for i, (request, last) in enumerate(reqs)
            for position in range(last - query_len + 1, last + 1)
        ]

    def load(reqs):
        assert 0 < len(reqs) <= capture_reqs
        q.zero_()
        seq_lens.zero_()
        slots.fill_(-1)
        for i, (request, last) in enumerate(reqs):
            seq_lens[i] = last + 1
            table[i] = tables[request]
        for t, (_, request, position) in enumerate(tokens(reqs)):
            q[t] = queries[request][position]
            positions[t] = position
            slots[t] = t
        common = CommonAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=starts.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens.cpu(),
            num_reqs=capture_reqs,
            num_actual_tokens=capture_tokens,
            max_query_len=query_len,
            max_seq_len=max(last + 1 for _, last in reqs),
            block_table_tensor=table,
            slot_mapping=slots,
            causal=True,
        )
        metadata = builder.build(0, common)
        assert metadata.num_decodes == capture_reqs and metadata.num_prefills == 0
        all_metadata = {swa.prefix: metadata}
        if compress_ratio:
            index_q.zero_()
            weights.zero_()
            row_starts.zero_()
            row_ends.zero_()
            for i, (request, _) in enumerate(reqs):
                compressed_table[i] = compressed_tables[request]
            for t, (_, request, position) in enumerate(tokens(reqs)):
                index_q[t] = index_queries[request][position]
                weights[t] = index_weights[request][position]
                row_starts[t] = key_offsets[request]
                row_ends[t] = key_offsets[request] + (position + 1) // 2
            all_metadata["compressed"] = fixture["compressed_builder"].build(
                0, replace(common, block_table_tensor=compressed_table)
            )
        return ForwardContext({}, all_metadata, {})

    def forward(context):
        if compress_ratio:
            query_scales, native_weights = mqa_logits_mx_inputs(index_q, None, weights)
            valid_ends = row_ends - row_starts
            logits = as_fp32_logits(
                fp8_fp4_mqa_logits(
                    (index_q, query_scales),
                    (index_keys, index_key_scales),
                    native_weights,
                    row_starts,
                    row_ends,
                    max(key_lengths),
                ),
                valid_ends,
            )
            top_k_per_row_prefill(
                logits,
                torch.zeros_like(valid_ends),
                valid_ends,
                attention.topk_indices_buffer[:capture_tokens],
                capture_tokens,
                logits.stride(0),
                logits.stride(1),
                512,
            )
        with override_forward_context(context):
            attention.forward_mqa(q, q[:, 0], positions, output)

    def check(reqs):
        rows = tokens(reqs)
        wanted = torch.stack([expected[r][p] for _, r, p in rows])
        assert torch.equal(output[: len(rows)], wanted)

    # Spec-decode builders treat 1 + num_speculative_tokens rows as a decode.
    builder.decode_threshold = query_len
    try:
        low, last = query_len - 1, lengths[0] - 1
        first = [(0, last), (1, 64), (2, 128)]
        context = load(first)
        forward(context)
        check(first)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(context)
        batches = [
            first,
            [(2, low), (0, 127), (0, 128), (1, 30), (0, last // 2)]
            + [(2, 77), (1, low), (0, last)],
            [(1, 64)],
            [(0, low), (0, low + 1), (0, 126)],
        ]
        for reqs in batches:
            reqs = [(r, min(p, lengths[r] - 1)) for r, p in reqs][:capture_reqs]
            context = load(reqs)
            forward(context)
            check(reqs)
            output.fill_(float("nan"))
            graph.replay()
            check(reqs)
            assert torch.isfinite(output[: len(tokens(reqs))]).all()
    finally:
        builder.decode_threshold = 1


@pytest.mark.parametrize("bounded", [False, True])
def test_ds41_bi_nvfp4_gather_matches_paged_record_and_graph(monkeypatch, bounded):
    """BI gathers the same packed records, preserving offsets and page strides."""
    if not torch.cuda.is_available():
        pytest.skip("DS41 NVFP4 CUDA gather")
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        dequantize_and_gather_k_cache,
    )

    torch.manual_seed(1959)
    device = torch.device("cuda")
    block, requests, pages = 64, 3, 3
    page_stride = block * 288 + 512
    packed = torch.full(
        (requests * pages, page_stride), 165, dtype=torch.uint8, device=device
    )
    # Interleaved V4.1 NVFP4 records: 256 packed bytes, then 32 scales.
    records = packed[:, : block * 288].view(requests * pages, block, 288)
    records[..., :256] = torch.randint(
        256, (requests * pages, block, 256), dtype=torch.uint8, device=device
    )
    scales = (torch.rand(requests * pages, block, 32, device=device) + 0.25).to(
        torch.float8_e4m3fn
    )
    records[..., 256:] = scales.view(torch.uint8)
    cache = packed.as_strided((requests * pages, block, 288), (page_stride, 288, 1))
    table = torch.randperm(requests * pages, device=device).int().view(requests, pages)
    seq_lens = torch.tensor([130, 127, 65], dtype=torch.int32, device=device)
    gather_lens = seq_lens.clamp(max=63) if bounded else None
    offset, max_gather = 17, 63 if bounded else 130
    shape = (requests, offset + max_gather + 3, 512)
    strides = (shape[1] * 512 + 64, 512, 1)

    def allocate(count=requests):
        return torch.empty_strided(
            (count, *shape[1:]), strides, dtype=torch.bfloat16, device=device
        ).fill_(-7.25)

    def gather(output, lengths=seq_lens, tables=table, limits=gather_lens):
        dequantize_and_gather_k_cache(
            output, cache, lengths, limits, tables, block, offset
        )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    expected = allocate()
    gather(expected)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    actual = allocate()
    gather(actual)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    for request in range(requests):
        alone = allocate(1)
        gather(
            alone,
            seq_lens[request : request + 1],
            table[request : request + 1],
            None if gather_lens is None else gather_lens[request : request + 1],
        )
        assert torch.equal(alone, expected[request : request + 1])

    permutation = torch.tensor([2, 0, 1], device=device)
    shuffled = allocate()
    gather(
        shuffled,
        seq_lens[permutation],
        table[permutation],
        None if gather_lens is None else gather_lens[permutation],
    )
    assert torch.equal(shuffled, expected[permutation])
    gather(actual)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather(actual)
    seq_lens.sub_(1)
    if gather_lens is not None:
        gather_lens.clamp_(max=62)
    expected.fill_(-7.25)
    gather(expected)
    actual.fill_(-7.25)
    graph.replay()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("candidates", [False, True])
def test_ds41_bi_indexer_decode_rows_match_prefill_rows(
    monkeypatch, compress_ratio, candidates
):
    """BI indexer decode rows select exactly what prefill rows select.

    Decode rows score with paged MQA logits, prefill rows gather K and score
    densely; both then take the same FP32 cast, candidate mask and
    deterministic top-k. Decode rows are also replayed from one padded graph
    while the batch changes in place.
    """
    from types import SimpleNamespace

    from vllm.config.attention import AttentionConfig
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
    from vllm.utils.deep_gemm import _ceil_to_ue8m0
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    if not torch.cuda.is_available():
        pytest.skip("DS41 indexer requires CUDA")
    assert torch.cuda.get_device_capability()[0] == 10
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(2003)
    device = torch.device("cuda")
    heads, dim, top_k, block_size = 32, 128, 512, 128
    states = block_size // compress_ratio
    lengths = [3000, 700, 1100]
    max_model_len, max_tokens, capture_size = 4096, sum(lengths), 8
    attention_config = AttentionConfig()
    attention_config.sparse_mla_prefill_workspace_factor = 5
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len, hf_text_config=None),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_tokens, max_num_seqs=capture_size
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        attention_config=attention_config,
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        speculative_config=None,
        num_speculative_tokens=0,
    )
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=dim,
        dtype=torch.uint8,
        tokens_per_state=compress_ratio,
    )
    pages = (max_model_len + block_size - 1) // block_size
    builder = DeepseekV32IndexerMetadataBuilder(
        spec, ["indexer"], config, device, block_table_width=pages
    )
    assert builder.decode_threshold == 1

    # FP8 keys with power-of-two scales, paged as DeepGEMM reads them: a
    # page's 128-byte rows, then its four-byte packed UE8M0 scales.
    tables = torch.randperm(3 * pages, device=device).int().view(3, pages)
    cache = torch.zeros(3 * pages, states * (dim + 4), device=device, dtype=torch.uint8)
    for request, size in enumerate(lengths):
        keys = torch.randn(size // compress_ratio, dim, device=device)
        scale = _ceil_to_ue8m0(keys.abs().amax(-1, keepdim=True).clamp(1e-4) / 448)
        values = (keys / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        packed = (scale.view(torch.int32) >> 23).to(torch.uint8).repeat(1, 4)
        for row in range(values.shape[0]):
            page, slot = tables[request, row // states], row % states
            cache[page, slot * dim : (slot + 1) * dim] = values[row]
            cache[page, states * dim + slot * 4 : states * dim + slot * 4 + 4] = packed[
                row
            ]
    cache = cache.view(3 * pages, states, dim + 4)
    queries = [
        (torch.randn(size, heads, dim, device=device) / 4).to(torch.float8_e4m3fn)
        for size in lengths
    ]
    # Indexer weights are BF16-valued, so equal scores (ties) are common.
    weights = [
        torch.randn(size, heads, device=device).bfloat16().float() for size in lengths
    ]
    topk = torch.full((max_tokens, top_k), -1, device=device, dtype=torch.int32)
    blocks = torch.full((max_tokens, 2048), -1, device=device, dtype=torch.int32)
    init_workspace_manager(device)

    def common(schedule, padded=None):
        sizes = [end - start for _, start, end in schedule]
        count = padded or len(schedule)
        sizes += [1] * (count - len(schedule))
        ends = [end for _, _, end in schedule] + [0] * (count - len(schedule))
        rows = [tables[r] for r, _, _ in schedule]
        rows += [tables[0]] * (count - len(schedule))
        starts = torch.tensor([0, *sizes], dtype=torch.int32).cumsum(0).int()
        slots = torch.arange(sum(sizes), device=device, dtype=torch.int64)
        slots[sum(sizes[: len(schedule)]) :] = -1
        return CommonAttentionMetadata(
            query_start_loc=starts.to(device),
            query_start_loc_cpu=starts,
            seq_lens=torch.tensor(ends, device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor(ends, dtype=torch.int32),
            num_reqs=count,
            num_actual_tokens=sum(sizes),
            max_query_len=max(sizes),
            max_seq_len=max(ends),
            block_table_tensor=torch.stack(rows),
            slot_mapping=slots,
            causal=True,
        )

    def score(metadata, q, w, write):
        context = ForwardContext({}, {"indexer": metadata}, {})
        with override_forward_context(context):
            sparse_attn_indexer(
                q.new_empty(q.shape[0], 1),
                "indexer",
                cache,
                q,
                None,
                None,
                w,
                128,
                "ue8m0",
                top_k,
                dim,
                max_model_len // compress_ratio,
                max_tokens // compress_ratio + 3 * block_size,
                topk,
                True,
                False,
                False,
                None,
                candidate_blocks=blocks if candidates else None,
                candidate_block_size=8 if candidates else 0,
                candidate_write=write,
                topk_backend="deep_select",
            )

    def run(schedule, padded=None):
        """Top-k (and candidates) per row: write pass, then masked pass."""
        metadata = builder.build(0, common(schedule, padded))
        q = torch.cat([queries[r][s:e] for r, s, e in schedule])
        w = torch.cat([weights[r][s:e] for r, s, e in schedule])
        if padded:
            pad = padded - q.shape[0]
            q = torch.cat([q, q.new_zeros(pad, heads, dim)])
            w = torch.cat([w, w.new_zeros(pad, heads)])
        results = []
        for write in (True, False) if candidates else (False,):
            score(metadata, q, w, write)
            results.append(topk[: q.shape[0]].clone())
        return metadata, results

    try:
        prefill = [run([(r, 0, size)])[1] for r, size in enumerate(lengths)]
        metadata, _ = run([(r, 0, size) for r, size in enumerate(lengths)])
        assert metadata.num_decodes == 0 and metadata.num_prefills == 3

        def check(rows, results):
            for out, ref in zip(results, zip(*prefill)):
                wanted = torch.stack([ref[r][p] for r, p in rows])
                assert torch.equal(out[: len(rows)], wanted)

        rows = [(0, 2999), (1, 0), (2, 513), (0, 1024), (1, 699), (2, 1)]
        metadata, results = run([(r, p, p + 1) for r, p in rows])
        assert metadata.num_decodes == len(rows) and metadata.num_prefills == 0
        check(rows, results)

        # One padded decode graph, replayed with new rows written in place.
        first = [(0, 1500), (1, 64), (2, 1099)]
        schedule = [(r, p, p + 1) for r, p in first]
        metadata = builder.build(0, common(schedule, capture_size))
        q = torch.zeros(capture_size, heads, dim, device=device).to(torch.float8_e4m3fn)
        w = torch.zeros(capture_size, heads, device=device)
        static = common(schedule, capture_size)

        def load(rows):
            batch = common([(r, p, p + 1) for r, p in rows], capture_size)
            for name in ("query_start_loc", "seq_lens", "block_table_tensor"):
                getattr(static, name).copy_(getattr(batch, name))
            static.slot_mapping.copy_(batch.slot_mapping)
            q.zero_()
            w.zero_()
            for i, (r, p) in enumerate(rows):
                q[i] = queries[r][p]
                w[i] = weights[r][p]
            return builder.build(0, static)

        def forward(metadata):
            results = []
            for write in (True, False) if candidates else (False,):
                score(metadata, q, w, write)
                results.append(topk[:capture_size].clone())
            return results

        metadata = load(first)
        check(first, forward(metadata))
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for write in (True, False) if candidates else (False,):
                score(metadata, q, w, write)
        for rows in (
            first,
            [(2, 1000), (0, 7)],
            [(i % 3, 99 * i + 5) for i in range(8)],
        ):
            load(rows)
            topk.fill_(-7)
            graph.replay()
            # Only the last pass's indices survive in the shared buffer.
            assert torch.equal(
                topk[: len(rows)],
                torch.stack([prefill[r][-1][p] for r, p in rows]),
            )
    finally:
        torch.accelerator.synchronize()
        reset_workspace_manager()


@pytest.mark.parametrize(
    "length", [1, 129, 254, 255, 256, 382, 383, 509, 510, 700, 1500, 4000, 12288]
)
def test_ds41_bi_exact_replay_matches_full_forward(monkeypatch, length):
    """Bounded replay keeps enough rows to equal the full forward, bit for bit.

    A stack of real BI SWA attention layers (per-row maps between them) runs a
    prompt in full, then (a) with the layers past a cut on its trailing kept
    rows, as one stage or in layer-wise stages that each keep fewer rows,
    (b) after a prefix hit that replays the hit's tail at every layer, and
    (c) both. The last window's KV at every layer and the new rows' top outputs
    must match the full forward; the one-window replay, and stages keeping one
    row too few, must not.
    """
    if not torch.cuda.is_available():
        pytest.skip("DS41 FlashMLA requires CUDA")
    from vllm.config import CacheConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.models.deepseek_v41.decoder_replay_layers import (
        exact_replay_rows,
        exact_replay_stages,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        DeepseekV4FlashMLAAttention,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    assert torch.cuda.get_device_capability()[0] == 10
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(length)
    device = torch.device("cuda")
    window, block_size, heads, dim, record = 128, 32, 64, 512, 528
    num_layers, cut = 6, 2
    config, _ = _ds41_swa_bi_config([max(length, 1024)])
    pages = (length + block_size - 1) // block_size
    caches = []
    with set_current_vllm_config(config):
        for layer in range(num_layers):
            cache = DeepseekV4SWACache(
                head_dim=dim,
                window_size=window,
                dtype=torch.uint8,
                prefix=f"swa{layer}",
                cache_config=CacheConfig(cache_dtype="nvfp4_ds_mla"),
                backend_cls=DeepseekV4FlashMLAAttention.swa_backend_cls,
                block_size=block_size,
                packed_bytes_per_token=record,
                packed_page_alignment=512,
            )
            cache.kv_cache = torch.empty(
                pages, block_size, record, device=device, dtype=torch.uint8
            )
            caches.append(cache)
    spec = caches[0].get_kv_cache_spec(config)
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
        spec, [c.prefix for c in caches], config, device
    )
    attention = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.compress_ratio = 0
    attention.use_pcp = False
    attention.compressed_cache_prefix = None
    attention.window_size = window
    attention.max_num_batched_tokens = max(length, 1024)
    attention.topk_indices_buffer = torch.full(
        (max(length, 1024), 512), -1, device=device, dtype=torch.int32
    )
    attention.scale = dim**-0.5
    sinks = torch.randn(num_layers, heads, device=device)
    q_weights = torch.randn(num_layers, heads, dim, device=device)
    k_weights = torch.randn(num_layers, dim, device=device)
    embeddings = torch.randn(length, dim, device=device).bfloat16()
    table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    init_workspace_manager(device)

    def scramble():
        for cache in caches:
            cache.kv_cache.random_(0, 256)

    def forward(cache, layer, hidden, rows, run_from, kv_start):
        """The rows from ``run_from`` attend, reading no window KV below
        ``kv_start``; returns their per-row output map."""
        num = rows.numel() - run_from
        q = (hidden[run_from:, None].float() * q_weights[layer]).bfloat16()
        starts = torch.tensor([0, num], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=starts.to(device),
            query_start_loc_cpu=starts,
            seq_lens=torch.tensor([length], device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([length], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=num,
            max_query_len=num,
            max_seq_len=length,
            block_table_tensor=table,
            slot_mapping=rows[run_from:],
            causal=True,
        )
        replay_start = torch.tensor([kv_start], device=device, dtype=torch.int32)
        metadata = builder.build(0, common, replay_start=replay_start)
        attention.swa_cache_layer = cache
        attention.attn_sink = sinks[layer]
        output = torch.empty_like(q)
        attn_metadata = {cache.prefix: metadata}
        with override_forward_context(ForwardContext({}, attn_metadata, {})):  # type: ignore[arg-type]
            attention.forward_mqa(q, q[:, 0], rows[run_from:], output)
        mixed = (output[:, 0].float() + output[:, 1].float()) * 0.5
        return torch.tanh(mixed + hidden[run_from:].float()).bfloat16()

    def run(start, decoder_starts):
        """Rows [start, length) of one request, replaying from ``start``. Layers
        below the cut run every row; the cut writes every row's KV and, like the
        layers past it, runs the rows from its ``decoder_starts`` entry; those
        layers write only those rows' KV and read no window KV below them."""
        hidden = embeddings[start:]
        rows = torch.arange(start, length, device=device)
        for layer, cache in enumerate(caches):
            decoder_start = decoder_starts.get(layer, start)
            kept_from = decoder_start - start
            write_from = kept_from if layer > cut else 0
            values = (hidden.float() * k_weights[layer]).bfloat16()
            quantize_and_insert_k_cache(
                values[write_from:],
                cache.kv_cache.view(pages, -1),
                rows[write_from:],
                block_size=block_size,
                bytes_per_token=record,
            )
            run_from = kept_from if layer >= cut else 0
            kv_start = decoder_start if layer > cut else start
            out = forward(cache, layer, hidden, rows, run_from, kv_start)
            hidden = torch.zeros_like(hidden)
            hidden[run_from:] = out
        return hidden

    def all_kv():
        return [cache.kv_cache.view(-1, record)[:length].clone() for cache in caches]

    def layer_rows(stages):
        return {
            layer: rows for first, stop, rows in stages for layer in range(first, stop)
        }

    try:
        scramble()
        full = run(0, {})
        full_kv = all_kv()
        hit = length * 3 // 5 // block_size * block_size
        kept = exact_replay_rows(window, num_layers - cut - 1)
        replayed = exact_replay_rows(window, num_layers)

        def matches(start, decoder_rows):
            """``decoder_rows``: rows each layer from the cut on keeps (an int
            for all of them)."""
            scramble()
            if isinstance(decoder_rows, int):
                decoder_rows = dict.fromkeys(range(cut, num_layers), decoder_rows)
            starts = {
                layer: max(start, length - rows) for layer, rows in decoder_rows.items()
            }
            decoder_start = starts[num_layers - 1]
            out = run(start, starts)
            # What later steps read: the last window's KV at every layer (from
            # the hit's window when nothing trims), and the outputs of the new
            # rows the replay layers ran.
            if decoder_start > start:
                lo = out_lo = max(length - window, 0)
            else:
                lo, out_lo = max(hit - window, 0) if start else 0, start and hit
            return torch.equal(out[out_lo - start :], full[out_lo:]) and all(
                torch.equal(a[lo:], b[lo:]) for a, b in zip(all_kv(), full_kv)
            )

        assert matches(0, kept)
        start = max(hit - replayed, 0)
        assert matches(start, length)
        assert matches(start, kept)
        for layers_per_stage in (1, 2):
            stages = exact_replay_stages(window, cut, num_layers, layers_per_stage)
            assert stages[0][2] == kept and stages[-1][2] == 2 * window - 1
            assert matches(0, layer_rows(stages))
            assert matches(start, layer_rows(stages))
        # Negative controls: the one-window replay of either side, and a stage
        # keeping one row fewer than its layers need.
        if length > window:
            assert not matches(0, window)
        if hit > window:
            assert not matches(hit - window, length)
        short = layer_rows(exact_replay_stages(window, cut, num_layers, 1))
        short[cut + 2] -= 1
        if length >= short[cut + 2] + 1:
            assert not matches(0, short)
    finally:
        torch.accelerator.synchronize()
        reset_workspace_manager()
