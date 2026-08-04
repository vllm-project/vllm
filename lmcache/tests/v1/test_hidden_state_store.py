# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`HiddenStateStore` and engine integration.

These tests construct a :class:`HiddenStateStore` directly so they don't need
a full :class:`LMCacheEngine`, GPU connector, or storage manager. The KV
presence side of "coupled eviction" is exercised via a tiny fake storage
manager that exposes the single ``contains`` method the store consults.
"""

# Standard
from typing import Optional, Set

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.hidden_state_store import HiddenStateStore
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import ChunkedTokenDatabase

CHUNK_SIZE = 8
HIDDEN_DIM = 16


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(1, 2, CHUNK_SIZE, 1, HIDDEN_DIM),
    )


def _config(
    *,
    enable_hidden_state_cache: bool = True,
    pool_gb: float = 0.05,
    layers=None,
) -> LMCacheEngineConfig:
    """Minimal CPU-only config for HS-store tests."""
    return LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_SIZE,
        local_cpu=True,
        max_local_cpu_size=0.05,
        local_disk=None,
        max_local_disk_size=0,
        remote_url=None,
        save_unfull_chunk=False,
        enable_hidden_state_cache=enable_hidden_state_cache,
        max_hidden_state_cpu_size=pool_gb,
        hidden_state_layers=layers,
    )


def _token_db(config: LMCacheEngineConfig) -> ChunkedTokenDatabase:
    db = ChunkedTokenDatabase(config, _metadata())
    return db


class _Fixture:
    """Bundle a config, one token database, and a fake SM.

    A shared :class:`ChunkedTokenDatabase` instance is essential here: the
    database lazily resolves its hash function on first use, so two separate
    instances can pick different ``NONE_HASH`` seeds and produce different
    keys for the same tokens.
    """

    def __init__(self, *, layers=None, pool_gb: float = 0.05) -> None:
        self.config = _config(layers=layers, pool_gb=pool_gb)
        self.db = _token_db(self.config)
        self.sm = _FakeStorageManager()
        self.store = HiddenStateStore(self.config, self.db)
        self.store.bind_storage_manager(self.sm)  # type: ignore[arg-type]

    def keys_for(self, token_ids):
        return [key for _, _, key in self.db.process_tokens(tokens=token_ids)]

    def mark_kv_present(self, keys) -> None:
        self.sm.present.update(keys)

    def close(self) -> None:
        self.store.close()


class _FakeStorageManager:
    """Implements just the ``contains(key)`` slice the store consults."""

    def __init__(self, present_keys: Optional[Set[CacheEngineKey]] = None) -> None:
        self.present: Set[CacheEngineKey] = (
            set(present_keys) if present_keys is not None else set()
        )

    def contains(self, key, search_range=None, pin: bool = False):  # noqa: D401
        return "LocalCPUBackend" if key in self.present else None


# ---------------------------------------------------------------------------
# Alignment with KV chunking
# ---------------------------------------------------------------------------


def test_chunks_align_with_kv_keys():
    """HS uses the engine's TokenDatabase, so chunk keys equal KV chunk keys."""
    fix = _Fixture()
    token_ids = list(range(3 * CHUNK_SIZE))  # exactly three full chunks
    expected_keys = fix.keys_for(token_ids)
    assert len(expected_keys) == 3
    fix.mark_kv_present(expected_keys)

    n = fix.store.store_hidden_states(
        token_ids, torch.randn(len(token_ids), HIDDEN_DIM)
    )
    assert n == 3
    for k in expected_keys:
        assert fix.store.has_chunk(k, layer_idx=0)
    fix.close()


# ---------------------------------------------------------------------------
# Round-trip retrieve
# ---------------------------------------------------------------------------


def test_store_then_retrieve_roundtrip():
    fix = _Fixture()
    token_ids = list(range(2 * CHUNK_SIZE))
    fix.mark_kv_present(fix.keys_for(token_ids))

    rows = torch.randn(len(token_ids), HIDDEN_DIM)
    assert fix.store.store_hidden_states(token_ids, rows) == 2

    out = fix.store.retrieve_hidden_states(token_ids)
    assert out is not None
    assert out.shape == (len(token_ids), HIDDEN_DIM)
    torch.testing.assert_close(out, rows.to(torch.float32))
    fix.close()


# ---------------------------------------------------------------------------
# Coupled eviction (KV evict implies HS evict on next read)
# ---------------------------------------------------------------------------


def test_kv_eviction_truncates_retrieved_prefix():
    fix = _Fixture()
    token_ids = list(range(3 * CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    rows = torch.randn(len(token_ids), HIDDEN_DIM)
    fix.store.store_hidden_states(token_ids, rows)
    assert fix.store.num_cached_chunks() == 3

    fix.sm.present.discard(keys[1])  # simulate KV eviction of chunk #1

    out = fix.store.retrieve_hidden_states(token_ids)
    assert out is not None
    assert out.shape == (CHUNK_SIZE, HIDDEN_DIM)
    torch.testing.assert_close(out, rows[:CHUNK_SIZE].to(torch.float32))
    # KV evict -> HS evict (orphan dropped lazily on retrieve).
    assert not fix.store.has_chunk(keys[1])
    fix.close()


# ---------------------------------------------------------------------------
# Retrieve refreshes LRU in reverse prefix order
# ---------------------------------------------------------------------------


def test_retrieve_refreshes_lru_in_reverse_prefix_order():
    """After a prefix retrieve, earliest chunks are MRU and suffix chunks LRU.

    ``retrieve_hidden_states`` collects hit keys during the prefix walk, then
    touches ``_lru`` in reverse order so that under HS-only pool pressure the
    suffix is evicted before the prefix (matching KV cache LRU semantics).
    """
    fix = _Fixture()
    token_ids = list(range(3 * CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    fix.store.store_hidden_states(token_ids, torch.randn(len(token_ids), HIDDEN_DIM))
    # Store order: oldest -> newest.
    assert list(fix.store._lru.keys()) == keys

    fix.store.retrieve_hidden_states(token_ids)
    # Reverse touch on hit: suffix oldest, prefix newest.
    assert list(fix.store._lru.keys()) == [keys[2], keys[1], keys[0]]
    fix.close()


def test_retrieve_reverse_lru_evicts_suffix_before_prefix():
    """HS-only eviction after retrieve drops the suffix chunk, not the prefix."""
    fix = _Fixture()
    token_ids = list(range(3 * CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    fix.store.store_hidden_states(token_ids, torch.randn(len(token_ids), HIDDEN_DIM))
    fix.store.retrieve_hidden_states(token_ids)

    assert fix.store._evict_one_lru()
    assert fix.store.has_chunk(keys[0])
    assert fix.store.has_chunk(keys[1])
    assert not fix.store.has_chunk(keys[2])
    fix.close()


# ---------------------------------------------------------------------------
# HS-only eviction does not touch KV
# ---------------------------------------------------------------------------


def test_hs_eviction_does_not_imply_kv_eviction():
    fix = _Fixture()
    token_ids = list(range(2 * CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    fix.store.store_hidden_states(token_ids, torch.randn(len(token_ids), HIDDEN_DIM))
    assert fix.store.drop_key(keys[0])  # HS-only LRU-style eviction
    # KV-side state untouched.
    assert keys[0] in fix.sm.present
    assert keys[1] in fix.sm.present
    fix.close()


# ---------------------------------------------------------------------------
# Partial restore cutoff (HS missing in the middle)
# ---------------------------------------------------------------------------


def test_partial_restore_cutoff_when_hs_missing():
    fix = _Fixture()
    token_ids = list(range(3 * CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    rows = torch.randn(len(token_ids), HIDDEN_DIM)
    fix.store.store_hidden_states(token_ids, rows)
    fix.store.drop_key(keys[1])  # HS hole in the middle, KV intact

    out = fix.store.retrieve_hidden_states(token_ids)
    assert out is not None
    # prefix_strict: stop at the missing-HS chunk.
    assert out.shape == (CHUNK_SIZE, HIDDEN_DIM)
    torch.testing.assert_close(out, rows[:CHUNK_SIZE].to(torch.float32))
    fix.close()


# ---------------------------------------------------------------------------
# Multi-layer storage and per-layer retrieve
# ---------------------------------------------------------------------------


def test_multi_layer_store_and_retrieve():
    fix = _Fixture()
    token_ids = list(range(2 * CHUNK_SIZE))
    fix.mark_kv_present(fix.keys_for(token_ids))

    layer0 = torch.randn(len(token_ids), HIDDEN_DIM)
    layer3 = torch.randn(len(token_ids), HIDDEN_DIM)
    assert fix.store.store_hidden_states(token_ids, layer0, layer_idx=0) > 0
    assert fix.store.store_hidden_states(token_ids, layer3, layer_idx=3) > 0

    out0 = fix.store.retrieve_hidden_states(token_ids, layer_idx=0)
    out3 = fix.store.retrieve_hidden_states(token_ids, layer_idx=3)
    assert out0 is not None and out3 is not None
    torch.testing.assert_close(out0, layer0.to(torch.float32))
    torch.testing.assert_close(out3, layer3.to(torch.float32))
    assert fix.store.retrieve_hidden_states(token_ids, layer_idx=99) is None
    fix.close()


# ---------------------------------------------------------------------------
# Layer allowlist filters writes
# ---------------------------------------------------------------------------


def test_layer_allowlist_filters_writes():
    fix = _Fixture(layers=[0])
    token_ids = list(range(CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    rows = torch.randn(len(token_ids), HIDDEN_DIM)
    assert fix.store.store_hidden_states(token_ids, rows, layer_idx=0) == 1
    assert fix.store.store_hidden_states(token_ids, rows, layer_idx=1) == 0
    assert fix.store.has_chunk(keys[0], layer_idx=0)
    assert not fix.store.has_chunk(keys[0], layer_idx=1)
    fix.close()


# ---------------------------------------------------------------------------
# Separate memory pool isolation
# ---------------------------------------------------------------------------


def test_hs_pool_is_independent_object():
    """Each HS store owns its own ``MixedMemoryAllocator`` (and pinned buffer).

    Filling one store's pool does not affect a peer store's contents.
    """
    fix_a = _Fixture()
    fix_b = _Fixture()
    assert fix_a.store._allocator is not fix_b.store._allocator

    n_toks = 4 * CHUNK_SIZE
    rows = torch.randn(n_toks, HIDDEN_DIM)
    token_ids = list(range(n_toks))
    fix_a.mark_kv_present(fix_a.keys_for(token_ids))
    fix_a.store.store_hidden_states(token_ids, rows)
    assert fix_a.store.num_cached_chunks() > 0
    assert fix_b.store.num_cached_chunks() == 0
    fix_a.close()
    fix_b.close()


# ---------------------------------------------------------------------------
# Without a bound storage manager, retrieve assumes KV is present.
# ---------------------------------------------------------------------------


def test_retrieve_without_bound_sm_returns_full_prefix():
    cfg = _config()
    db = _token_db(cfg)
    store = HiddenStateStore(cfg, db)  # no SM bound

    token_ids = list(range(2 * CHUNK_SIZE))
    rows = torch.randn(len(token_ids), HIDDEN_DIM)
    store.store_hidden_states(token_ids, rows)

    out = store.retrieve_hidden_states(token_ids)
    assert out is not None
    assert out.shape == (len(token_ids), HIDDEN_DIM)
    store.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_store_rejects_wrong_shape():
    fix = _Fixture()
    with pytest.raises(ValueError):
        fix.store.store_hidden_states([1, 2, 3], torch.randn(2, HIDDEN_DIM))
    with pytest.raises(ValueError):
        fix.store.store_hidden_states([1, 2, 3], torch.randn(3))
    fix.close()


# ---------------------------------------------------------------------------
# token_offset: incremental store
# ---------------------------------------------------------------------------


def test_store_with_token_offset_aligns_with_kv_keys():
    """Calling store incrementally via token_offset stores all chunks with
    correct KV-aligned keys, and retrieve returns the full concatenated rows.

    Simulates the vllm-omni pattern where:
    - call 1 covers the first CHUNK_SIZE tokens (offset=0, short token_ids)
    - call 2 covers the next CHUNK_SIZE tokens (offset=CHUNK_SIZE, full
      token_ids so chunk keys are KV-aligned across the whole prefix)
    """
    fix = _Fixture()
    token_ids = list(range(2 * CHUNK_SIZE))
    keys = fix.keys_for(token_ids)
    fix.mark_kv_present(keys)

    rows = torch.randn(len(token_ids), HIDDEN_DIM)

    # First call: only the first chunk's tokens are available so far.
    # Pass just that segment; chunk key is computed identically because
    # the first chunk's key depends only on token_ids[0:CHUNK_SIZE].
    n1 = fix.store.store_hidden_states(
        token_ids[:CHUNK_SIZE], rows[:CHUNK_SIZE], token_offset=0
    )
    assert n1 == 1

    # Second call: full token_ids for correct KV-aligned chunk keys,
    # but hidden_states covers only token_ids[CHUNK_SIZE:] (the new tokens).
    n2 = fix.store.store_hidden_states(
        token_ids, rows[CHUNK_SIZE:], token_offset=CHUNK_SIZE
    )
    assert n2 == 1

    # Both chunk keys should now be present.
    for k in keys:
        assert fix.store.has_chunk(k, layer_idx=0)

    # Retrieve should return the full prefix, matching original rows.
    out = fix.store.retrieve_hidden_states(token_ids)
    assert out is not None
    assert out.shape == (len(token_ids), HIDDEN_DIM)
    torch.testing.assert_close(out, rows.to(torch.float32))
    fix.close()


def test_store_token_offset_rejects_bad_args():
    """token_offset out of range or mismatched hidden_states length is rejected."""
    fix = _Fixture()
    token_ids = list(range(CHUNK_SIZE))

    # Negative offset.
    with pytest.raises(ValueError):
        fix.store.store_hidden_states(
            token_ids, torch.randn(CHUNK_SIZE, HIDDEN_DIM), token_offset=-1
        )

    # Offset beyond sequence length.
    with pytest.raises(ValueError):
        fix.store.store_hidden_states(
            token_ids,
            torch.randn(0, HIDDEN_DIM),
            token_offset=CHUNK_SIZE + 1,
        )

    # hidden_states rows don't match n_toks - token_offset.
    with pytest.raises(ValueError):
        fix.store.store_hidden_states(
            token_ids,
            torch.randn(CHUNK_SIZE, HIDDEN_DIM),  # should be CHUNK_SIZE - 2 rows
            token_offset=2,
        )
    fix.close()
