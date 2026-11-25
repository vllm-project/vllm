# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builtin LMCache MP ranges under lookahead block hashes."""

import pytest

pytest.importorskip("lmcache")

from vllm import SamplingParams  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request  # noqa: E402

BLOCK_SIZE = 16
BLOCKS_IN_CHUNK = 2
NUM_TOKENS = 4 * BLOCK_SIZE


def _tracker(lookahead: int) -> LMCacheMPRequestTracker:
    """A chunk-aligned prompt whose last block waits for a lookahead token."""
    init_none_hash(sha256)
    request = Request(
        "r",
        list(range(NUM_TOKENS)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256, lookahead),
    )
    assert len(request.block_hashes) == 4 - lookahead
    tracker = LMCacheMPRequestTracker(request)
    tracker.append_block_ids(list(range(4)))
    return tracker


@pytest.mark.parametrize("lookahead", [0, 1])
def test_retrieve_range_follows_tokens(lookahead):
    tracker = _tracker(lookahead)
    tracker.num_lmcache_hit_blocks = 4
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, BLOCKS_IN_CHUNK, BLOCK_SIZE
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, NUM_TOKENS)


@pytest.mark.parametrize("lookahead", [0, 1])
def test_store_range_follows_tokens(lookahead):
    tracker = _tracker(lookahead)
    tracker.increase_num_scheduled_tokens(NUM_TOKENS)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, BLOCKS_IN_CHUNK, BLOCK_SIZE
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, NUM_TOKENS)
