# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builtin LMCache MP ranges under lookahead block hashes."""

from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from vllm import SamplingParams  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    get_dcp_decorated_model_name,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request  # noqa: E402

BLOCK_SIZE = 16
TOKENS_PER_CHUNK = 2 * BLOCK_SIZE
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
    tracker.append_block_ids((list(range(4)),))
    return tracker


@pytest.mark.parametrize("lookahead", [0, 1])
def test_retrieve_range_follows_tokens(lookahead):
    tracker = _tracker(lookahead)
    tracker.num_lmcache_hit_tokens = NUM_TOKENS
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, TOKENS_PER_CHUNK, [BLOCK_SIZE]
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, NUM_TOKENS)


@pytest.mark.parametrize("lookahead", [0, 1])
def test_store_range_follows_tokens(lookahead):
    tracker = _tracker(lookahead)
    tracker.increase_num_scheduled_tokens(NUM_TOKENS)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, TOKENS_PER_CHUNK, [BLOCK_SIZE]
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, NUM_TOKENS)


def test_dcp_model_name_uses_the_interleave_vllm_resolves():
    """The name carries the interleave vLLM resolves; its config is restored."""
    groups = [object()]
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=2, cp_kv_cache_interleave_size=1
    )

    def adjust(kv_cache_groups):
        assert kv_cache_groups is groups
        parallel_config.cp_kv_cache_interleave_size = BLOCK_SIZE

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="m"),
        parallel_config=parallel_config,
        adjust_dcp_kv_cache_interleave_size=adjust,
    )

    name = get_dcp_decorated_model_name(
        vllm_config, SimpleNamespace(kv_cache_groups=groups)
    )

    assert name.endswith(f"d2-interleave{BLOCK_SIZE}")
    assert parallel_config.cp_kv_cache_interleave_size == 1
