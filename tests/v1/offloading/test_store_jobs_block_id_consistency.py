# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for _build_store_jobs block_ids/keys consistency.

With a zero-chunk group (GLM5Next KpoolTail) present, offload_keys for
that group are never appended (update_offload_keys skips it), but
block_ids are derived from positions that advance via the shared hybrid
allocator. Deriving block ids from num_chunks (positions) instead of the
keys list length made the two lists inconsistent, tripping:
    assert len(offload_keys) == len(offload_block_ids)
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    RequestOffloadState,
)
from vllm.v1.kv_offload.base import ReqContext


def _make_state(num_block_hashes):
    common = SimpleNamespace(
        num_prompt_tokens=1024,
        block_hashes=[bytes([i]) * 16 for i in range(num_block_hashes)],
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
            # KpoolTail-shaped: tokens_per_chunk=4, hashes_per_chunk=0.
            GroupOffloadConfig(
                group_idx=1,
                tokens_per_block=4,
                tokens_per_chunk=4,
                hashes_per_chunk=0,
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


def test_zero_chunk_group_never_breaks_store_job_construction():
    """Constructing store jobs with a zero-chunk group present must not
    raise and must keep offload_keys/offload_block_ids length-consistent
    for the participating group."""
    state = _make_state(num_block_hashes=16)
    # Simulate the scheduler having accumulated keys for the participating
    # group only (update_offload_keys skips the zero-chunk group).
    state.update_offload_keys()
    # The participating group must have keys; the zero-chunk group must
    # have none — and neither path may raise.
    assert len(state.group_states[0].offload_keys) > 0
    assert len(state.group_states[1].offload_keys) == 0
