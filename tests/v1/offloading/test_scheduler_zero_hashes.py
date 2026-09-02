# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for update_offload_keys with non-hashable groups.

A group whose tokens_per_chunk < tokens_per_hash resolves to
hashes_per_chunk == 0 (e.g. GLM5Next's KpoolTail scratch group: 4 tokens
per block vs a 16-token hash granularity). islice(step=0) raises
ValueError, crashing the engine on the first scheduled request. Such
groups carry no hash-addressable offload blocks by construction.
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    RequestOffloadState,
)
from vllm.v1.kv_offload.base import ReqContext


def _make_state(hashes_per_chunk, tokens_per_chunk, block_hashes):
    """Build a RequestOffloadState with two groups: a normal MLA group
    (hash-addressable) and a KpoolTail-shaped group (zero hashes per
    chunk)."""
    common = SimpleNamespace(
        num_prompt_tokens=1024,
        block_hashes=block_hashes,
        kv_transfer_params={},
        request_id="test-req",
    )
    config = SimpleNamespace(
        kv_group_configs=(
            GroupOffloadConfig(
                group_idx=0,
                tokens_per_block=16,
                tokens_per_chunk=16,
                hashes_per_chunk=1,
                kv_event_group_spec=None,
                sliding_window_size_in_chunks=None,
            ),
            GroupOffloadConfig(
                group_idx=1,
                tokens_per_block=4,
                tokens_per_chunk=tokens_per_chunk,
                hashes_per_chunk=hashes_per_chunk,
                kv_event_group_spec=None,
                sliding_window_size_in_chunks=None,
            ),
        ),
        blocks_per_chunk=1,
        tokens_per_hash=16,
        num_workers=1,
        offload_prompt_only=True,
        supports_partial_tail=False,
    )
    return RequestOffloadState(
        config=config,
        req=common,
        req_context=ReqContext(req_id="test-req"),
        offloading_context=SimpleNamespace(),
    )


def test_update_offload_keys_zero_hashes_per_chunk_group():
    """A zero hashes_per_chunk group must be skipped, not crash.

    On stock code this raises:
      ValueError: Indices for islice() must be None or an integer:
      0 <= x <= sys.maxsize.
    because islice is called with step=0.
    """
    # 16 block hashes of 16 tokens each (the MLA group's granularity).
    block_hashes = [bytes([i]) * 16 for i in range(16)]
    state = _make_state(
        hashes_per_chunk=0,  # 4-token chunks // 16-token hash = 0
        tokens_per_chunk=4,
        block_hashes=block_hashes,
    )
    # Must not raise; the zero-chunk group accumulates no offload keys.
    state.update_offload_keys()
    assert len(state.group_states[1].offload_keys) == 0
    # The normal MLA group still accumulates keys at its own granularity.
    assert len(state.group_states[0].offload_keys) > 0


def test_update_offload_keys_normal_group_unaffected():
    """Sanity: a positive hashes_per_chunk group still accumulates keys."""
    block_hashes = [bytes([i]) * 16 for i in range(16)]
    state = _make_state(
        hashes_per_chunk=1,
        tokens_per_chunk=16,
        block_hashes=block_hashes,
    )
    state.update_offload_keys()
    # 16 hashes / 1 hash per chunk, starting from chunk 1 (offset -1 + 1).
    assert len(state.group_states[0].offload_keys) >= 1
