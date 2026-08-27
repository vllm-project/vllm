# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the kpool tail slot mapping (no GPU required).

The kpool tail cache is a 1-block-per-request circular ring addressed by
``pos % kpool`` (``KpoolTailSpec`` / ``KpoolTailManager``: exactly one block
allocated per request, never grown, so only column 0 of its block table is
ever written; the rest stays zero-initialized).

The generic per-group slot kernel cannot express that layout: it maps
``pos -> bt[req][pos // bs] * bs + pos % bs`` (``_compute_slot_mappings_kernel``
in vllm/v1/worker/gpu/block_table.py), so every token at ``pos >= kpool``
reads a zero column and collapses onto physical tail block 0. All concurrent
requests then share one ``kpool``-slot ring and corrupt each other's pool
compression.

These tests pin that defect's arithmetic, verify the circular replacement
(``compute_kpool_tail_slot_mapping``), and mirror the tail kernels' index
math to show cross-request pollution before the fix and isolation after.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    KpoolTailBackend,
    KpoolTailMetadataBuilder,
    compute_kpool_tail_slot_mapping,
)
from vllm.v1.kv_cache_interface import KpoolTailSpec, compute_layout_strides
from vllm.v1.kv_cache_layout import KVCacheLayout

KPOOL = 4


def test_tail_backend_layout_matches_kernel_pointer_arithmetic():
    (layout,) = KpoolTailBackend.supported_kv_cache_layouts()
    spec = KpoolTailSpec(
        block_size=KPOOL,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=KPOOL,
    )
    strides = compute_layout_strides(spec, num_blocks=8, num_layers=3, layout=layout)
    _, _, head_stride, state_stride, content_stride = strides

    assert layout is KVCacheLayout.LBHNC
    assert head_stride == KPOOL * 128 * torch.bfloat16.itemsize
    assert state_stride == 128 * torch.bfloat16.itemsize
    assert content_stride == 1


def make_tail_block_table(own_blocks, width=64):
    """Tail-group block table as BlockTables produces it: column 0 holds the
    request's single KpoolTailManager block, the remaining columns are never
    written and stay zero."""
    bt = torch.zeros(len(own_blocks), width, dtype=torch.int32)
    bt[:, 0] = torch.tensor(own_blocks, dtype=torch.int32)
    return bt


def legacy_generic_tail_slots(block_table, query_start_loc, positions):
    """Reference of the generic ``_compute_slot_mappings_kernel`` arithmetic
    (block_table.py:305-313) applied to the tail group's table."""
    slots = []
    for req in range(block_table.shape[0]):
        for i in range(query_start_loc[req], query_start_loc[req + 1]):
            pos = int(positions[i])
            block_number = int(block_table[req, pos // KPOOL])
            slots.append(block_number * KPOOL + pos % KPOOL)
    return torch.tensor(slots, dtype=torch.int64)


def circular_tail_slots(
    slot_mapping, block_table, query_start_loc, positions, num_actual, num_reqs
):
    return compute_kpool_tail_slot_mapping(
        slot_mapping,
        block_table,
        query_start_loc,
        positions,
        num_actual,
        num_reqs,
        KPOOL,
    )


def make_batch(per_req_positions, padded_len=None):
    positions = torch.cat(
        [torch.tensor(p, dtype=torch.int64) for p in per_req_positions]
    )
    num_actual = positions.numel()
    num_reqs = len(per_req_positions)
    lens = [len(p) for p in per_req_positions]
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(lens, dtype=torch.int64), 0, out=qsl[1:])
    if padded_len is None:
        padded_len = num_actual
    slot_mapping = torch.full((padded_len,), -1, dtype=torch.int64)
    return positions, qsl, slot_mapping, num_actual, num_reqs


def test_legacy_generic_mapping_collapses_onto_block_zero():
    """The bug: with the manager's 1-column block table, the generic kernel
    maps every pos >= kpool onto tail block 0, and distinct requests collide."""
    own_blocks = [5, 9]
    per_req = [list(range(10)), list(range(12))]  # prompts of len 10 and 12
    positions, qsl, _, num_actual, num_reqs = make_batch(per_req)
    bt = make_tail_block_table(own_blocks)

    legacy = legacy_generic_tail_slots(bt, qsl, positions)

    # Every token at pos >= kpool resolves to block 0, not the request's own.
    off = 0
    for req, prompt in enumerate(per_req):
        req_slots = legacy[off : off + len(prompt)]
        for pos in range(len(prompt)):
            slot = int(req_slots[pos])
            if pos >= KPOOL:
                assert slot // KPOOL == 0, (
                    f"expected collapse onto block 0 at pos {pos}"
                )
                assert slot // KPOOL != own_blocks[req]
        off += len(prompt)

    # The two requests share ring slots -> cross-request pollution.
    a_slots = set(legacy[: len(per_req[0])].tolist())
    b_slots = set(legacy[len(per_req[0]) :].tolist())
    assert a_slots & b_slots, "legacy mapping must collide across requests"


def test_circular_mapping_isolates_requests():
    """The fix: every token lands in its own request's block at pos % kpool,
    and no slot is ever shared by two different requests (slots do recur
    within a request every kpool positions -- that is the circular design)."""
    own_blocks = [5, 9]
    per_req = [list(range(10)), list(range(12))]
    positions, qsl, slot_mapping, num_actual, num_reqs = make_batch(per_req)
    bt = make_tail_block_table(own_blocks)

    out = circular_tail_slots(slot_mapping, bt, qsl, positions, num_actual, num_reqs)

    off = 0
    per_req_slots = []
    for req, prompt in enumerate(per_req):
        req_slots = set()
        for pos in range(len(prompt)):
            slot = int(out[off + pos])
            assert slot // KPOOL == own_blocks[req], (
                f"req {req} pos {pos} left its tail block"
            )
            assert slot % KPOOL == pos % KPOOL
            req_slots.add(slot)
        per_req_slots.append(req_slots)
        off += len(prompt)
    assert not per_req_slots[0] & per_req_slots[1]


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 4])
def test_circular_mapping_matches_generic_for_short_requests(prompt_len):
    """For pos < kpool the generic kernel already picks the own block, so the
    two mappings agree while every position fits the request's first block
    (single-request behavior is unchanged)."""
    own_blocks = [7]
    per_req = [list(range(prompt_len))]
    positions, qsl, slot_mapping, num_actual, num_reqs = make_batch(per_req)
    bt = make_tail_block_table(own_blocks)

    legacy = legacy_generic_tail_slots(bt, qsl, positions)
    out = circular_tail_slots(slot_mapping, bt, qsl, positions, num_actual, num_reqs)
    assert torch.equal(out, legacy)


def test_circular_mapping_preserves_padding_and_empty_batch():
    own_blocks = [5, 9]
    per_req = [list(range(10)), list(range(12))]
    padded_len = sum(len(p) for p in per_req) + 8
    positions, qsl, slot_mapping, num_actual, num_reqs = make_batch(
        per_req, padded_len=padded_len
    )
    bt = make_tail_block_table(own_blocks)

    out = circular_tail_slots(slot_mapping, bt, qsl, positions, num_actual, num_reqs)
    assert out.shape == slot_mapping.shape
    assert torch.equal(out[num_actual:], torch.full_like(out[num_actual:], -1))

    empty = circular_tail_slots(slot_mapping, bt, qsl, positions[:0], 0, num_reqs)
    assert torch.equal(empty, slot_mapping)


def make_common_metadata(per_req_positions, own_blocks, with_positions=True):
    positions, qsl, slot_mapping, num_actual, num_reqs = make_batch(
        per_req_positions, padded_len=sum(len(p) for p in per_req_positions) + 4
    )
    bt = make_tail_block_table(own_blocks)
    seq_lens = torch.tensor(
        [max(p) + 1 if p else 1 for p in per_req_positions], dtype=torch.int64
    )
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual,
        max_query_len=max((len(p) for p in per_req_positions), default=1),
        max_seq_len=int(seq_lens.max()) if num_reqs else 1,
        block_table_tensor=bt,
        slot_mapping=slot_mapping,
        positions=positions if with_positions else None,
    )


def make_tail_builder(block_size=KPOOL):
    builder = object.__new__(KpoolTailMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(block_size=block_size)
    return builder


def test_builder_build_uses_circular_mapping():
    per_req = [list(range(10)), list(range(12))]
    own_blocks = [5, 9]
    cam = make_common_metadata(per_req, own_blocks)
    meta = KpoolTailMetadataBuilder.build(make_tail_builder(), 0, cam)

    out = meta.slot_mapping
    off = 0
    for req, prompt in enumerate(per_req):
        for pos in range(len(prompt)):
            slot = int(out[off + pos])
            assert slot // KPOOL == own_blocks[req]
            assert slot % KPOOL == pos % KPOOL
        off += len(prompt)
    # Padding tail of the buffer keeps the -1 sentinel.
    assert torch.equal(
        out[cam.num_actual_tokens :], torch.full_like(out[cam.num_actual_tokens :], -1)
    )


def test_builder_build_falls_back_without_positions():
    """Capture / dummy builds without positions keep the generic mapping."""
    per_req = [list(range(10))]
    cam = make_common_metadata(per_req, [5], with_positions=False)
    meta = KpoolTailMetadataBuilder.build(make_tail_builder(), 0, cam)
    assert meta.slot_mapping is cam.slot_mapping


# ---------------------------------------------------------------------------
# Index-level mirror of the tail kernels: seed / stash / pool completion
# (addressing replicated from kpool_compress.py's Triton kernels).
# ---------------------------------------------------------------------------


class TailRingMirror:
    """Mirror of _kpool_tail_seed_kernel / _kpool_decode_update_batched_kernel
    addressing: block = tail_slot // kpool, ring offset = pos % kpool; a pool
    completing at pos reads ring slots (pool_start + s) % kpool and uses the
    current token's own K/score for the last member."""

    def __init__(self, num_blocks, kpool=KPOOL):
        self.kpool = kpool
        self.k = torch.full((num_blocks, kpool, 3), float("nan"))
        self.s = torch.full((num_blocks, kpool, 3), float("nan"))

    def stash(self, tail_slot, pos, k, s):
        blk, off = tail_slot // self.kpool, pos % self.kpool
        self.k[blk, off] = k
        self.s[blk, off] = s

    seed = stash  # the seed kernel writes with the same addressing

    def complete(self, tail_slot, pos, k, s):
        blk = tail_slot // self.kpool
        start = pos - (self.kpool - 1)
        kk = torch.stack(
            [self.k[blk, (start + i) % self.kpool] for i in range(self.kpool)]
        )
        ss = torch.stack(
            [self.s[blk, (start + i) % self.kpool] for i in range(self.kpool)]
        )
        kk[-1], ss[-1] = k, s  # is_current for the completing token
        w = torch.softmax(ss, dim=0)
        return (kk * w).sum(0)


def token_kv(req, pos):
    k = torch.tensor([pos + 100.0 * req, pos + 0.5, 2.0 * pos + 0.25])
    s = torch.tensor([0.1 * (pos + 1) + req, 0.2 * pos, 0.05 * pos])
    return k, s


def tail_slot_for(mapping, req, pos, own_block):
    if mapping == "legacy":
        bt_val = own_block if pos < KPOOL else 0
        return bt_val * KPOOL + pos % KPOOL
    return own_block * KPOOL + pos % KPOOL


def run_scenario(mapping, interleave):
    """Requests A (block 5, prompt len 9) and B (block 9, prompt len 11)
    decode concurrently; returns A's boundary pool [8, 9, 10, 11]."""
    ring = TailRingMirror(num_blocks=16)
    blocks = {"A": 5, "B": 9}
    prompts = {"A": 9, "B": 11}

    def slot(req, pos):
        return tail_slot_for(mapping, 0 if req == "A" else 1, pos, blocks[req])

    # Prefill: seed each request's trailing incomplete pool.
    for req, L in prompts.items():
        for pos in range(L - (L % KPOOL or KPOOL), L):
            if pos < 0:
                continue
            ring.stash(slot(req, pos), pos, *token_kv(0 if req == "A" else 1, pos))

    # Decode: A emits pos 9, 10, 11; B emits 11, 12, 13. `interleave`
    # processes B before A within a step, which is exactly what concurrent
    # Triton programs do when they share tail block 0.
    order = ["B", "A"] if interleave else ["A", "B"]
    decode = {"A": [9, 10, 11], "B": [11, 12, 13]}
    result = None
    for step in range(3):
        for req in order:
            pos = decode[req][step]
            k, s = token_kv(0 if req == "A" else 1, pos)
            if pos % KPOOL == KPOOL - 1:
                pool = ring.complete(slot(req, pos), pos, k, s)
                if req == "A":
                    result = pool
            ring.stash(slot(req, pos), pos, k, s)
    return result


def test_interleaved_decode_pollution_legacy_vs_circular():
    """Ground truth: request A decoding alone (its ring used exclusively)."""
    torch.manual_seed(0)
    ground_truth = run_scenario("circular", interleave=False)

    legacy = run_scenario("legacy", interleave=True)
    circular = run_scenario("circular", interleave=True)

    # The old mapping lets request B's tokens into A's ring: A's boundary
    # pool is compressed from 2 of B's tokens -> wrong.
    assert not torch.allclose(legacy, ground_truth), (
        f"legacy mapping unexpectedly clean: {legacy} vs {ground_truth}"
    )

    # The circular mapping keeps the rings isolated under interleaving.
    torch.testing.assert_close(circular, ground_truth)
