# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill chunk planning with query-length-0 requests (coverage matrix B8).

Two kinds of request reach the prefill region with no query tokens:
- FULL CUDA graph request padding (seq_len 0), whenever the builder declares
  ``ALWAYS`` (DSv4.1 on ROCm with adaptive verification);
- a request whose KV an async connector is loading (seq_len > 0,
  vllm-project/vllm#51486).

``split_decodes_and_prefills`` counts every row after the first prefill as a
prefill, and ``get_prefill_chunk_plan`` charges each chunk at its widest row,
so once four long-context prefills fill a chunk the empty rows get a chunk of
their own. ``_forward_prefill`` then views zero rows and crashes
(vllm-project/vllm#55695). The plan is built here from the same CPU fields the
SWA builder fills (``sparse_swa.py``), on host tensors.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

# DeepSeek-V4.1-Flash serving limits.
MAX_MODEL_LEN = 65536
MAX_NUM_BATCHED_TOKENS = 8192
WINDOW = 128
DECODE_THRESHOLD = 1
CHUNK_SIZE = DeepseekV41ROCMAiterMLAAttention.PREFILL_CHUNK_SIZE

# Four chunked prefills at full context: together they fill one chunk.
LONG_PREFILLS = [(MAX_MODEL_LEN, 512)] * CHUNK_SIZE
DECODES = [(3000, 1)] * 3


def _plan(requests, compress_ratio):
    """(seq_len, query_len) per request, decodes first, as the runner orders
    them; returns the prefill query lengths and the chunk plan."""
    seq_lens = torch.tensor([s for s, _ in requests], dtype=torch.int32)
    query_lens = torch.tensor([q for _, q in requests], dtype=torch.int32)
    query_start_loc = torch.zeros(len(requests) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(query_lens, 0)
    common = SimpleNamespace(
        max_query_len=int(query_lens.max()),
        num_reqs=len(requests),
        num_actual_tokens=int(query_start_loc[-1]),
        query_start_loc_cpu=query_start_loc,
    )
    num_decodes, num_prefills, _, _ = split_decodes_and_prefills(
        common, decode_threshold=DECODE_THRESHOLD
    )
    prefill_query_lens = (
        query_start_loc[num_decodes + 1 : num_decodes + num_prefills + 1]
        - query_start_loc[num_decodes : num_decodes + num_prefills]
    )
    md = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0),
        slot_mapping=torch.empty(0),
        block_size=128,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        prefill_seq_lens_cpu=seq_lens[num_decodes:],
        prefill_query_lens_cpu=prefill_query_lens,
        prefill_window_size=WINDOW,
        prefill_max_model_len=MAX_MODEL_LEN,
        prefill_max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )
    plan = md.get_prefill_chunk_plan(
        compress_ratio=compress_ratio,
        prefill_chunk_size=CHUNK_SIZE,
        has_compressed=compress_ratio > 0,
    )
    return prefill_query_lens.tolist(), plan


def _assert_plan_ok(query_lens, plan):
    real = [i for i, q in enumerate(query_lens) if q > 0]
    covered = [i for start, end, _, _ in plan for i in range(start, end)]
    # Every real row is planned once, in order.
    assert [i for i in covered if query_lens[i] > 0] == real, plan
    # No chunk without query tokens: the loop would view zero rows.
    for start, end, _, _ in plan:
        assert sum(query_lens[start:end]) > 0, (start, end, plan)


# Layer kinds of the stand-in: SWA only, ratio 2 and ratio 1.
RATIOS = [0, 2, 1]


@pytest.mark.parametrize("compress_ratio", RATIOS)
def test_long_prefills_fill_one_chunk(compress_ratio):
    """Control: without empty rows the plan is valid, and the four long
    prefills of the cases below share one chunk."""
    query_lens, plan = _plan(DECODES + LONG_PREFILLS, compress_ratio)
    _assert_plan_ok(query_lens, plan)
    assert plan[-1][:2] == (0, CHUNK_SIZE)


# Only layers with a compressed region reach the chunk's area limit.
EMPTY_CHUNK = pytest.mark.xfail(
    strict=True,
    reason="get_prefill_chunk_plan gives query-length-0 rows their own chunk; "
    "fixed by vllm-project/vllm#51489 (planner), guarded in _forward_prefill "
    "by vllm-project/vllm#55695",
)


@pytest.mark.parametrize(
    "compress_ratio",
    [r if r == 0 else pytest.param(r, marks=EMPTY_CHUNK) for r in RATIOS],
)
@pytest.mark.parametrize("num_empty", [1, 3])
@pytest.mark.parametrize(
    "empty_seq_len",
    [
        pytest.param(0, id="graph_padding"),
        pytest.param(5000, id="async_kv_load"),
    ],
)
def test_query_len_0_rows_never_form_a_chunk(compress_ratio, num_empty, empty_seq_len):
    requests = DECODES + LONG_PREFILLS + [(empty_seq_len, 0)] * num_empty
    query_lens, plan = _plan(requests, compress_ratio)
    # The split counts the empty trailing rows as prefills.
    assert len(query_lens) == CHUNK_SIZE + num_empty
    _assert_plan_ok(query_lens, plan)
