# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels of the P/D hidden-state handoff: records into and out of the KV
cache slots past the prompt, and the sample-only rows appended to a batch."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.hidden_state_record import RecordLayout
from vllm.v1.worker.gpu.hidden_state_handoff import (
    HiddenStateHandoff,
    _DraftPlan,
    _SampleOnlyRows,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
BLOCK_SIZE = 4
MAX_REQS = 8


def _handoff(
    layout: RecordLayout, num_blocks: int = 48, num_heads: int = 2, split: int = 1
):
    """A handoff over two fp16 layers' [blocks, heads, tokens, K + V] caches
    of different head sizes, with each request's blocks scattered. Each block
    of the block table is ``split`` of the caches' (kernel) blocks."""
    caches = [
        torch.zeros(
            num_blocks,
            num_heads,
            BLOCK_SIZE,
            2 * size,
            dtype=torch.float16,
            device=DEVICE,
        )
        for size in (8, 4)
    ]
    handoff = object.__new__(HiddenStateHandoff)
    handoff.device = DEVICE
    handoff.dtype = torch.float16
    handoff.layout = layout
    handoff.record_bytes = layout.num_slots * layout.hidden_size * 2
    views = [cache.view(torch.uint8) for cache in caches]
    handoff.head_bytes = sum(v.shape[-1] for v in views)
    handoff.tail_tokens = cdiv(2 * handoff.record_bytes, handoff.head_bytes)
    perm = torch.randperm(
        num_blocks // split, generator=torch.Generator().manual_seed(0)
    )
    block_table = perm.view(MAX_REQS, -1).to(torch.int32).to(DEVICE)
    handoff._init_record_slots(views, block_table, split * BLOCK_SIZE, BLOCK_SIZE)
    return handoff, caches, block_table


@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("num_heads_read", [2, 1])
def test_record_round_trip_through_kv_slots(num_heads_read, split):
    """Records written past the prompt into every KV head read back whole,
    also from a head slice (a decoder at a different TP size), laid out one
    nibble per byte, split over the layers of each token slot, also with the
    block table's blocks split into kernel blocks."""
    torch.manual_seed(0)
    layout = RecordLayout(hidden_size=24, window=1, num_aux=2)
    handoff, caches, block_table = _handoff(layout, split=split)
    # 192 bytes, stored as 384, at 48 per token slot.
    assert handoff.tail_tokens == 8

    req_idx = torch.tensor([1, 4, 6], dtype=torch.int64, device=DEVICE)
    prompt_len = torch.tensor([3, 9, 4], dtype=torch.int64, device=DEVICE)
    records = torch.randn(3, layout.num_slots, layout.hidden_size).half().to(DEVICE)
    handoff._write_records(req_idx, prompt_len, records.view(3, -1).view(torch.uint8))

    payload = records.view(3, -1).view(torch.uint8)
    encoded = torch.stack([payload & 0xF, payload >> 4], dim=-1).view(3, -1)
    expected = encoded.view(3, handoff.tail_tokens, handoff.head_bytes)
    written = torch.zeros(len(caches), *caches[0].shape[:3], dtype=torch.bool)
    for r, (req, n) in enumerate(zip(req_idx.tolist(), prompt_len.tolist())):
        start = 0
        for layer, cache in enumerate(caches):
            view = cache.view(torch.uint8)
            size = view.shape[-1]
            for t in range(handoff.tail_tokens):
                pos = n + t
                block = int(block_table[req, pos // (split * BLOCK_SIZE)])
                block = block * split + pos % (split * BLOCK_SIZE) // BLOCK_SIZE
                offset = pos % BLOCK_SIZE
                written[layer, block, :, offset] = True
                want = expected[r, t, start : start + size]
                assert torch.equal(view[block, :, offset], want.expand(2, -1))
            start += size
    for layer, cache in enumerate(caches):
        # No other token slot is touched (the prompts' included).
        assert not cache.view(torch.uint8)[~written[layer].to(DEVICE)].any()

    if num_heads_read == 1:
        handoff._init_record_slots(
            [cache.view(torch.uint8)[:, 1:] for cache in caches],
            block_table,
            split * BLOCK_SIZE,
            BLOCK_SIZE,
        )
    assert torch.equal(handoff._read_records(req_idx, prompt_len), records)


def test_save_gathers_records():
    """A completed prompt's record holds the model output at its last position
    and each drafter input at the window's positions, right-aligned (zeros
    before the prompt's start); unfinished and decode rows save none."""
    torch.manual_seed(0)
    layout = RecordLayout(hidden_size=24, window=2, num_aux=1)
    handoff, _, _ = _handoff(layout)
    handoff.saves_records = True
    handoff.group_id = 0
    handoff.block_tables = SimpleNamespace(
        num_blocks=SimpleNamespace(np=np.full((1, MAX_REQS), 6, dtype=np.int32))
    )
    # Rows: a whole 5-token prompt, a decode, a prompt's final chunk (tokens
    # 4..8 of 9), a 1-token prompt and an unfinished chunk (tokens 0..2 of 9).
    num_scheduled = np.array([5, 1, 5, 1, 3], dtype=np.int32)
    query_start_loc = np.concatenate([[0], np.cumsum(num_scheduled)])
    prefill_len = np.array([5, 4, 9, 1, 9], dtype=np.int32)
    num_computed = np.array([0, 4, 4, 0, 0], dtype=np.int32)
    batch = SimpleNamespace(
        num_reqs=5,
        idx_mapping_np=np.array([3, 0, 5, 2, 7]),
        num_scheduled_tokens=num_scheduled,
        query_start_loc_np=query_start_loc,
        prefill_len_np=prefill_len,
        num_computed_prefill_tokens_np=num_computed,
        is_prefilling_np=num_computed < prefill_len,
    )
    num_tokens = int(query_start_loc[-1])
    hidden, drafter, aux = (
        torch.randn(num_tokens, 24).half().to(DEVICE) for _ in range(3)
    )
    handoff._save(batch, hidden, drafter, [aux])
    assert handoff._pending is not None
    req_idx, prompt_len, records = handoff._pending
    assert req_idx.tolist() == [3, 5, 2]
    assert prompt_len.tolist() == [5, 9, 1]
    records = records.view(torch.float16).view(3, layout.num_slots, 24)
    zero = torch.zeros(24, dtype=torch.float16, device=DEVICE)
    # Last prompt token: row 0's token 4, row 2's token 6 + 4, row 3's 11.
    for r, last in enumerate([4, 10, 11]):
        assert torch.equal(records[r, 0], hidden[last])
        before = r != 2
        assert torch.equal(records[r, 1], drafter[last - 1] if before else zero)
        assert torch.equal(records[r, 2], drafter[last])
        assert torch.equal(records[r, 3], aux[last - 1] if before else zero)
        assert torch.equal(records[r, 4], aux[last])


def _decoder_handoff(layout: RecordLayout, decode_query_len: int):
    handoff = object.__new__(HiddenStateHandoff)
    handoff.device = DEVICE
    handoff.dtype = torch.float16
    handoff.layout = layout
    handoff.decode_query_len = decode_query_len
    handoff.input_buffers = InputBuffers(MAX_REQS, 64, DEVICE)
    handoff._no_padding = torch.zeros(64, dtype=torch.bool, device=DEVICE)
    handoff._zero_rows = torch.zeros(
        0, layout.hidden_size, dtype=torch.float16, device=DEVICE
    )
    handoff.model = SimpleNamespace()
    handoff.draft_attn_groups = None
    # Token p of request r is 1000 * r + p.
    token_ids = 1000 * torch.arange(MAX_REQS)[:, None] + torch.arange(32)[None, :]
    handoff.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=token_ids.to(torch.int32).to(DEVICE))
    )
    return handoff


def _rows(handoff, idx_mapping, prefill_len, draft):
    num_rows = len(idx_mapping)
    layout = handoff.layout
    to_gpu = lambda x: torch.tensor(x, dtype=torch.int64, device=DEVICE)  # noqa: E731
    return _SampleOnlyRows(
        req_ids=[f"so{i}" for i in range(num_rows)],
        idx_mapping_np=np.array(idx_mapping),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        req_idx=to_gpu(idx_mapping),
        prompt_len=to_gpu(prefill_len),
        records=torch.randn(num_rows, layout.num_slots, layout.hidden_size)
        .half()
        .to(DEVICE),
        draft=draft,
    )


@pytest.mark.parametrize("with_forward", [True, False])
def test_sample_only_rows_appended(with_forward):
    """Sample-only rows follow the forward batch's, one token and logit each
    for sampling, and for drafting padded to full decode rows of placeholders
    (rejected) up to a full CUDA graph's padded shape."""
    torch.manual_seed(0)
    layout = RecordLayout(hidden_size=16, window=1)
    handoff = _decoder_handoff(layout, decode_query_len=3)
    fwd = None
    if with_forward:
        # Two decode rows of 1 + 2 drafts.
        fwd = InputBatch.make_dummy(2, 6, handoff.input_buffers)
        fwd.logits_indices = torch.arange(6, device=DEVICE)
        fwd.cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32, device=DEVICE)
        fwd.cu_num_logits_np = np.array([0, 3, 6], dtype=np.int32)
        fwd.expanded_idx_mapping = torch.tensor(
            [0, 0, 0, 1, 1, 1], dtype=torch.int32, device=DEVICE
        )
        fwd.expanded_local_pos = torch.tensor(
            [0, 1, 2, 0, 1, 2], dtype=torch.int32, device=DEVICE
        )
    na, nt, nl = (2, 6, 6) if with_forward else (0, 0, 0)
    draft = _DraftPlan(
        num_tokens_np=np.array([3, 3], dtype=np.int32),
        num_real_np=np.array([1, 1]),
        cu_num_tokens_np=np.array([0, 3, 6]),
        token_row_np=np.array([0, 0, 0, 1, 1, 1]),
        token_slot_np=np.ones((1, 6), dtype=np.int64),
        full_cudagraph=True,
        num_tokens_padded=nt + 10,
        num_reqs_padded=na + 4,
    )
    to_gpu = lambda x: torch.tensor(x, dtype=torch.int64, device=DEVICE)  # noqa: E731
    draft.cu_num_tokens = to_gpu(draft.cu_num_tokens_np)
    draft.num_real = to_gpu(draft.num_real_np)
    draft.token_row = to_gpu(draft.token_row_np)
    draft.token_slot = to_gpu(draft.token_slot_np)
    rows = _rows(handoff, [3, 5], [10, 7], draft)
    handoff._rows = rows

    batch = handoff._sampling_batch(fwd, rows, False)
    fwd_req_idx = [0, 1] if with_forward else []
    assert batch.num_reqs == na + 2 and batch.num_tokens == nt + 2
    assert batch.idx_mapping.tolist() == fwd_req_idx + [3, 5]
    assert batch.query_start_loc.tolist() == ([0, 3, 6] if fwd else [0]) + [
        nt + 1,
        nt + 2,
    ]
    assert batch.query_start_loc_np.tolist() == batch.query_start_loc.tolist()
    assert batch.seq_lens.tolist() == [3, 3] * with_forward + [10, 7]
    assert batch.cu_num_logits.tolist() == ([0, 3, 6] if fwd else [0]) + [
        nl + 1,
        nl + 2,
    ]
    assert batch.cu_num_logits_np.tolist() == batch.cu_num_logits.tolist()
    assert batch.logits_indices.tolist() == list(range(nl)) + [nt, nt + 1]
    assert batch.expanded_idx_mapping.tolist() == [0, 0, 0, 1, 1, 1] * (
        with_forward
    ) + [3, 5]
    assert batch.expanded_local_pos.tolist() == [0, 1, 2] * 2 * with_forward + [0, 0]
    assert batch.positions[nt:].tolist() == [9, 6]
    assert batch.input_ids[nt:].tolist() == [3009, 5006]

    num_rejected = torch.tensor(
        [0, 1] * with_forward + [0, 0], dtype=torch.int32, device=DEVICE
    )
    fwd_hidden = torch.randn(nt, 16).half().to(DEVICE) if fwd else None
    draft_batch, hidden, aux, _, _, draft_rejected = handoff.prepare_drafting(
        fwd, fwd_hidden, None, batch, num_rejected
    )
    assert aux is None
    assert draft_rejected.tolist() == [0, 1] * with_forward + [2, 2]
    assert draft_batch.num_tokens == nt + 6
    assert draft_batch.num_tokens_after_padding == nt + 10
    assert draft_batch.num_reqs_after_padding == na + 4
    assert draft_batch.decode_graph_eligible
    assert draft_batch.max_query_len == 3
    qsl = ([0, 3, 6] if fwd else [0]) + [nt + 3, nt + 6] + [nt + 6] * 2
    assert draft_batch.query_start_loc.tolist() == qsl
    assert draft_batch.query_start_loc_np.tolist() == qsl
    seq_lens = [3, 3] * with_forward + [12, 9, 0, 0]
    assert draft_batch.seq_lens.tolist() == seq_lens
    assert draft_batch.seq_lens_cpu_upper_bound.tolist() == seq_lens
    assert draft_batch.num_draft_tokens_per_req.tolist() == [0, 0] * with_forward + [
        2,
        2,
    ]
    assert draft_batch.positions[nt : nt + 6].tolist() == [9, 10, 11, 6, 7, 8]
    assert draft_batch.input_ids[nt : nt + 6].tolist() == [3009, 0, 0, 5006, 0, 0]
    # The forward's hidden states, each row's drafter input (slot 1) for its
    # last prompt position and placeholders, then zeros.
    assert hidden.shape[0] == nt + 10
    if fwd_hidden is not None:
        assert torch.equal(hidden[:nt], fwd_hidden)
    for r in range(2):
        start = nt + 3 * r
        want = rows.records[r, 1].expand(3, -1)
        assert torch.equal(hidden[start : start + 3], want)
    assert not hidden[nt + 6 :].any()
