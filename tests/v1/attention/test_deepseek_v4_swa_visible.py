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

import pytest
import torch
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


def ref_swa_bounds(pos: int, window: int, left: int, right: int) -> tuple[int, int]:
    """Reference [start, end) window bounds for one query token."""
    left_add = max(left - (window - 1), 0)
    start = max(pos - (window - 1) - left_add, 0)
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
) -> tuple[list[list[int]], list[int]]:
    """Reference paged slot-id rows and lens for every token in the batch."""
    block_table_cpu = block_table.cpu()
    lefts, rights = ref_left_right(
        seq_lens, query_lens, spans_per_req, max_image_tokens
    )
    rows: list[list[int]] = []
    lens: list[int] = []
    token = 0
    for req, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        prefix_len = seq_len - query_len
        for i in range(query_len):
            pos = prefix_len + i
            start, end = ref_swa_bounds(pos, window, lefts[token], rights[token])
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
):
    """Run combine_topk_swa_indices and return (indices, lens, expected)."""
    device = torch.device("cuda")
    num_reqs = len(seq_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    gather_lens = torch.tensor(
        [q + min(s - q, WINDOW - 1) for s, q in zip(seq_lens, query_lens)],
        dtype=torch.int32,
        device=device,
    )
    N = (max(seq_lens) + compress_ratio - 1) // compress_ratio
    M = N + int(gather_lens.max()) + 8
    gen = torch.Generator(device="cpu").manual_seed(0)
    topk_indices = torch.randint(
        0, 4096, (num_tokens, max(topk, 1)), generator=gen, dtype=torch.int32
    ).to(device)
    topk_indices = topk_indices[:, : max(topk, 1)]

    if with_image:
        lefts, rights = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
        left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
        right_t = torch.tensor(rights, dtype=torch.int32, device=device)
    else:
        left_t = right_t = None

    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens_t,
        gather_lens,
        WINDOW,
        compress_ratio,
        topk,
        M,
        N,
        left_visible=left_t,
        right_visible=right_t,
        max_image_tokens=MAX_IMG,
    )

    # Reference rows.
    lefts, rights = ref_left_right(
        seq_lens,
        query_lens,
        spans if with_image else [[] for _ in seq_lens],
        MAX_IMG,
    )
    topk_cpu = topk_indices.cpu()
    width = WINDOW + MAX_IMG
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


def make_builder(vision: bool) -> DeepseekSparseSWAMetadataBuilder:
    overrides: dict = {"sliding_window": WINDOW}
    if vision:
        overrides.update(vision_n_layers=2, vision_max_n_token=MAX_IMG)
    vllm_config = create_vllm_config(
        max_model_len=4096,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        hf_config_override=overrides,
    )
    spec = SlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        cache_dtype_str="auto",
        model_version="deepseek_v4",
    )
    return DeepseekSparseSWAMetadataBuilder(
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
):
    device = torch.device("cuda")
    query_start_loc, seq_lens_t, _, slot_mapping, block_table = make_batch(
        seq_lens, query_lens, device
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
