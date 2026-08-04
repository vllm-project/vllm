# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``lmcache.v1.distributed.serde.utils``."""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.utils import make_temp_key


def _orig(salt: str = "", object_group_id: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=b"\x00" * 16,
        model_name="model",
        kv_rank=0,
        object_group_id=object_group_id,
        cache_salt=salt,
    )


def test_make_temp_key_propagates_cache_salt() -> None:
    """Temp keys inherit ``cache_salt`` so per-tenant L1 byte accounting
    keeps temp buffers attributed to the same bucket as their originals."""
    salt = "tenant-A"
    temp = make_temp_key(_orig(salt=salt))
    assert temp.cache_salt == salt


def test_make_temp_key_propagates_other_fields() -> None:
    """Non-hash identity fields are preserved verbatim."""
    orig = _orig(salt="tenant-X", object_group_id=3)
    temp = make_temp_key(orig)
    assert temp.model_name == orig.model_name
    assert temp.kv_rank == orig.kv_rank
    assert temp.object_group_id == orig.object_group_id


def test_make_temp_key_differs_from_original() -> None:
    """Suffix ensures temp_hash ≠ original_hash so temps cannot collide
    with real keys in the L1 namespace."""
    orig = _orig()
    temp = make_temp_key(orig)
    assert temp.chunk_hash != orig.chunk_hash
    assert temp.chunk_hash.startswith(orig.chunk_hash)


def test_make_temp_key_unique_across_calls() -> None:
    """Two temp keys for the same original are distinct."""
    orig = _orig()
    a = make_temp_key(orig)
    b = make_temp_key(orig)
    assert a.chunk_hash != b.chunk_hash
