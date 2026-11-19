# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builtin LMCache MP connector: token ranges and KV cache group metadata."""

import pytest

pytest.importorskip("lmcache", minversion="0.5.7.post1")

from vllm import SamplingParams  # noqa: E402
from vllm.config.kv_transfer import KVCacheAccess  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnectorUpstream,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    get_group_tokens_per_block,
    get_prefix_replay_tokens,
    get_resolved_attention_block_sizes,
    get_vllm_scheduler_block_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_kv_cache_group_edits import (  # noqa: E402, E501
    validate_kv_cache_groups,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_kv_cache_groups import (  # noqa: E402
    group_layer_specs,
    group_mamba_cache_mode,
    group_sliding_window,
    is_scratch_group,
    tokens_per_group_block,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request  # noqa: E402

from .utils import create_vllm_config, make_kv_cache_config  # noqa: E402

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


def test_builtin_declares_legacy_ipc():
    """The builtin shares the KV cache with the server through CUDA IPC."""
    config = create_vllm_config(kv_connector="LMCacheMPConnector")
    assert (
        LMCacheMPConnectorUpstream.kv_cache_access(config.kv_transfer_config)
        == KVCacheAccess.LEGACY_IPC
    )


def test_group_metadata_reads_kv_cache_groups():
    """Group metadata comes from `KVCacheGroup` and its manager config."""
    sw_size = 128
    kv_cache_config = make_kv_cache_config(
        BLOCK_SIZE,
        swa_enabled=True,
        mamba_enabled=True,
        sw_size=sw_size,
        mamba_cache_mode="align",
    )
    full, sliding, mamba = kv_cache_config.kv_cache_groups

    assert list(group_layer_specs(full)) == ["layer0", "layer2"]
    assert [tokens_per_group_block(g, 1) for g in (full, sliding, mamba)] == [
        BLOCK_SIZE
    ] * 3
    assert not any(is_scratch_group(g) for g in (full, sliding, mamba))

    full_spec = next(iter(group_layer_specs(full).values()))
    sliding_spec = next(iter(group_layer_specs(sliding).values()))
    mamba_spec = next(iter(group_layer_specs(mamba).values()))
    assert group_sliding_window(full, full_spec) is None
    assert group_sliding_window(sliding, sliding_spec) == sw_size
    assert group_mamba_cache_mode(mamba, mamba_spec) == "align"

    vllm_config = create_vllm_config(block_size=BLOCK_SIZE)
    assert get_group_tokens_per_block(vllm_config, kv_cache_config) == [BLOCK_SIZE] * 3
    assert get_vllm_scheduler_block_size(vllm_config, kv_cache_config) == BLOCK_SIZE
    assert get_resolved_attention_block_sizes(vllm_config, kv_cache_config) == {
        BLOCK_SIZE
    }
    assert get_prefix_replay_tokens(kv_cache_config) == 0
    validate_kv_cache_groups(kv_cache_config)


def test_group_validation_rejects_stateless_mamba():
    """Mamba groups that keep no per-block snapshots cannot be served."""
    kv_cache_config = make_kv_cache_config(
        BLOCK_SIZE, mamba_enabled=True, mamba_cache_mode="none"
    )
    with pytest.raises(ValueError, match="mamba_cache_mode='none'"):
        validate_kv_cache_groups(kv_cache_config)
