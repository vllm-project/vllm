# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""1D-RoPE (text) streaming sessions: position-offset bookkeeping.

Text sessions share the retention/eviction mechanics with mRoPE sessions;
instead of a per-token position array, a text token's RoPE position is its
index plus a per-request scalar offset (the cumulative width of evicted
ranges). These tests pin that bookkeeping: eviction grows the offset by the
block-aligned width, re-prefill resets it, the wire carries it, and the
re-prefill trigger sees the derived watermark.
"""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.streaming.eviction import evict_segment
from vllm.v1.streaming.reprefill import should_trigger_reprefill
from vllm.v1.streaming.retention import HistorySegment, StreamingRetentionParams

BLOCK = 4


def _make_managers(num_blocks: int = 64):
    kv = KVCacheManager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=1024,
        enable_caching=False,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
    )
    return kv, EncoderCacheManager(cache_size=4096)


def _make_text_session(request_id: str, num_tokens: int) -> Request:
    retention = StreamingRetentionParams(
        max_video_segments=2, max_session_tokens=2048, reprefill_threshold=0.7
    )
    req = Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(
            max_tokens=16, extra_args={"streaming_retention": retention}
        ),
        pooling_params=None,
        resumable=True,
        first_chunk=True,
    )
    req.num_computed_tokens = num_tokens
    return req


def test_eviction_accumulates_position_offset():
    """Evicting a text segment grows position_offset by exactly the
    block-aligned width that was freed."""
    kv, enc = _make_managers()
    req = _make_text_session("txt-1", 3 * BLOCK)
    blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, 3 * BLOCK, 0, blocks)
    assert req.position_offset == 0

    seg = HistorySegment(
        segment_type="user_text", token_range=(0, 2 * BLOCK), age_chunks=3
    )
    req.session_history.append(seg)
    evict_segment(req, seg, kv, enc)

    # [0, 8) with block 4 is already aligned: full width evicted.
    assert req.position_offset == 2 * BLOCK
    assert req.pending_evicted_token_ranges == [(0, 2 * BLOCK)]

    # A second eviction accumulates on top.
    req.num_computed_tokens = len(req._all_token_ids)
    seg2 = HistorySegment(
        segment_type="user_text", token_range=(0, BLOCK), age_chunks=3
    )
    req.session_history.append(seg2)
    evict_segment(req, seg2, kv, enc)
    assert req.position_offset == 3 * BLOCK


def test_eviction_offset_uses_block_aligned_width():
    """A non-aligned segment only frees whole blocks; the offset must grow
    by the freed (aligned) width, not the raw segment width — otherwise the
    RoPE positions and the token indices desynchronize."""
    kv, enc = _make_managers()
    req = _make_text_session("txt-2", 4 * BLOCK)
    blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, 4 * BLOCK, 0, blocks)

    # Raw range [1, 2*BLOCK+1) -> inward alignment frees [BLOCK, 2*BLOCK).
    seg = HistorySegment(
        segment_type="user_text", token_range=(1, 2 * BLOCK + 1), age_chunks=3
    )
    req.session_history.append(seg)
    evict_segment(req, seg, kv, enc)
    assert req.position_offset == BLOCK


def test_new_request_data_carries_position_offset():
    req = _make_text_session("txt-3", 8)
    req.position_offset = 12
    data = NewRequestData.from_request(req, block_ids=([],))
    assert data.position_offset == 12
    # And a plain request defaults to 0.
    other = _make_text_session("txt-4", 8)
    data2 = NewRequestData.from_request(other, block_ids=([],))
    assert data2.position_offset == 0


def test_reprefill_trigger_uses_derived_text_watermark():
    """For text sessions the highest RoPE position is exactly
    num_tokens - 1 + position_offset; the trigger must fire on that
    derived watermark even though max_cached_position stays -1 (the
    worker never reports it for non-mRoPE models)."""
    retention = StreamingRetentionParams(
        max_video_segments=2, max_session_tokens=2048, reprefill_threshold=0.5
    )
    req = _make_text_session("txt-5", 100)
    assert req.max_cached_position == -1

    model_max_position = 1000
    # Below threshold: 100 - 1 + 300 = 399 <= 500.
    assert not should_trigger_reprefill(
        req,
        retention,
        model_max_position,
        highest_position=req.num_tokens - 1 + 300,
    )
    # Above threshold: 100 - 1 + 450 = 549 > 500.
    assert should_trigger_reprefill(
        req,
        retention,
        model_max_position,
        highest_position=req.num_tokens - 1 + 450,
    )
    # Disabled threshold never fires.
    retention_off = StreamingRetentionParams(
        max_video_segments=2, max_session_tokens=2048, reprefill_threshold=1.0
    )
    assert not should_trigger_reprefill(
        req, retention_off, model_max_position, highest_position=10**9
    )
