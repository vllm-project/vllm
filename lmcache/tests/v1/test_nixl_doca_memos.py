# SPDX-License-Identifier: Apache-2.0
"""Tests for the DOCA_MEMOS NIXL storage backend.

Covers backend/device validation, the 128-bit-hex object-key format
(``_format_object_key_b128``) and the URL-safe regression guard, static
object-pool slot names and the ``createPool`` routing, and the dynamic
backend's formatter dispatch.

All tests run without NIXL hardware: the formatters do not touch instance
state, so the real unbound methods are invoked directly with a
``Mock(spec=...)`` standing in for ``self``.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock, Mock, patch
import hashlib
import re
import uuid

# Third Party
import pytest
import torch

pytest.importorskip("nixl")

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.nixl_storage_backend import (
    B128_MAX_POOL_SIZE,
    NixlDynamicStorageBackend,
    NixlObjectPool,
    NixlStaticStorageBackend,
    NixlStorageConfig,
)
import lmcache.v1.storage_backend.nixl_storage_backend as nsb

_HEX_128 = re.compile(r"^[0-9a-f]{32}$")


def _make_key(
    model_name: str = "org/test_model", chunk_hash: int = 0
) -> CacheEngineKey:
    """A CacheEngineKey whose to_string() contains both '/' and '@'."""
    return CacheEngineKey(
        model_name=model_name,
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


class TestFormatObjectKeyB128:
    """``_format_object_key_b128`` -> 32-char hex (sha256 truncated to 128 bits)."""

    @staticmethod
    def _fmt(key: CacheEngineKey) -> str:
        backend = Mock(spec=NixlDynamicStorageBackend)
        return NixlDynamicStorageBackend._format_object_key_b128(backend, key)

    def test_returns_32_char_lowercase_hex(self) -> None:
        out = self._fmt(_make_key())
        assert _HEX_128.match(out), f"not 128-bit hex: {out!r}"

    def test_matches_documented_algorithm(self) -> None:
        key = _make_key()
        expected = hashlib.sha256(key.to_string().encode("utf-8")).hexdigest()[:32]
        assert self._fmt(key) == expected

    def test_deterministic_for_same_key(self) -> None:
        key = _make_key(chunk_hash=42)
        assert self._fmt(key) == self._fmt(_make_key(chunk_hash=42))

    def test_distinct_keys_differ(self) -> None:
        assert self._fmt(_make_key(chunk_hash=1)) != self._fmt(_make_key(chunk_hash=2))


class TestFormatObjectKeyUrlSafe:
    """Regression guard for the legacy URL-safe formatter (non-DOCA backends)."""

    @staticmethod
    def _fmt(key: CacheEngineKey) -> str:
        backend = Mock(spec=NixlDynamicStorageBackend)
        return NixlDynamicStorageBackend._format_object_key_url_safe(backend, key)

    def test_no_raw_slash_or_at(self) -> None:
        out = self._fmt(_make_key())
        assert "/" not in out and "@" not in out

    def test_matches_legacy_transform(self) -> None:
        # Standard
        from urllib.parse import quote

        key = _make_key()
        flat = key.to_string().replace("/", "_").replace("@", "_")
        assert self._fmt(key) == quote(flat, safe="")


class TestNixlObjectPoolKeys:
    """Static object-pool slot names per backend."""

    def test_b128_keys_are_128bit_hex(self) -> None:
        pool = NixlObjectPool(64, b128=True)
        assert len(pool.keys) == 64
        assert all(_HEX_128.match(k) for k in pool.keys)

    def test_b128_keys_are_unique(self) -> None:
        pool = NixlObjectPool(256, b128=True)
        assert len(set(pool.keys)) == len(pool.keys)

    def test_b128_first_8_hex_encode_slot_index(self) -> None:
        size = 16
        pool = NixlObjectPool(size, b128=True)
        # Keys are generated for i in reversed(range(size)); the leading
        # 8 hex chars must recover every slot index exactly once.
        indices = sorted(int(k[:8], 16) for k in pool.keys)
        assert indices == list(range(size))

    def test_default_keys_use_obj_prefix(self) -> None:
        pool = NixlObjectPool(8)
        assert all(k.startswith("obj_") for k in pool.keys)

    def test_b128_rejects_pool_size_above_max(self) -> None:
        with pytest.raises(ValueError, match="at most"):
            NixlObjectPool(B128_MAX_POOL_SIZE + 1, b128=True)

    def test_b128_max_pool_size_is_2_32(self) -> None:
        # The guard accepts up to 2**32 slots because the largest valid slot
        # index (B128_MAX_POOL_SIZE - 1 == 0xffffffff) still formats to exactly
        # 8 hex chars, keeping the full slot name at 32 hex chars (128 bits).
        # Constructing a 2**32-slot pool is infeasible in a test, so this
        # asserts the boundary invariant directly rather than the constructor.
        assert B128_MAX_POOL_SIZE == 2**32
        max_index = B128_MAX_POOL_SIZE - 1
        key = f"{max_index:08x}{uuid.uuid4().hex[:24]}"
        assert len(key) == 32 and _HEX_128.match(key)


class TestValidateNixlBackend:
    """NixlStorageConfig.validate_nixl_backend — the config-acceptance path."""

    def test_doca_memos_cpu_is_valid(self) -> None:
        assert NixlStorageConfig.validate_nixl_backend("DOCA_MEMOS", "cpu") is True

    def test_doca_memos_cuda_is_rejected(self) -> None:
        assert NixlStorageConfig.validate_nixl_backend("DOCA_MEMOS", "cuda") is False

    def test_doca_memos_cuda_with_index_is_rejected(self) -> None:
        # device strings may carry an index suffix (e.g. "cuda:0")
        assert NixlStorageConfig.validate_nixl_backend("DOCA_MEMOS", "cuda:0") is False


class TestCreatePool:
    """NixlStaticStorageBackend.createPool — the static configured path."""

    def test_doca_memos_creates_b128_object_pool(self) -> None:
        pool = NixlStaticStorageBackend.createPool(
            "DOCA_MEMOS",
            size=8,
            path="/tmp/unused",
            use_direct_io=False,
            path_sharding="by_gpu",
            dst_device="cpu",
        )
        assert isinstance(pool, NixlObjectPool)
        # b128 slot names are 32-char lowercase hex (no "obj_" prefix).
        assert all(_HEX_128.match(k) for k in pool.keys)
        assert not any(k.startswith("obj_") for k in pool.keys)

    def test_obj_creates_non_b128_object_pool(self) -> None:
        pool = NixlStaticStorageBackend.createPool(
            "OBJ",
            size=8,
            path="/tmp/unused",
            use_direct_io=False,
            path_sharding="by_gpu",
            dst_device="cpu",
        )
        assert isinstance(pool, NixlObjectPool)
        assert all(k.startswith("obj_") for k in pool.keys)


class TestDynamicFormatterDispatch:
    """A dynamic backend must route _format_object_key to b128 for DOCA_MEMOS."""

    @staticmethod
    def _make_dynamic_backend(backend_name: str) -> nsb.NixlDynamicStorageBackend:
        cfg = MagicMock()
        cfg.backend = backend_name
        cfg.use_direct_io = False
        cfg.enable_async_put = False
        cfg.enable_presence_cache = False
        cfg.path = "/tmp/nixl-unused"

        def fake_super_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # super().__init__ normally allocates the NIXL buffer/allocator;
            # the agent constructor reads self.memory_allocator.
            self.memory_allocator = MagicMock()

        with (
            patch.object(nsb.NixlStorageBackend, "__init__", fake_super_init),
            # Skip init_chunk_meta so the mock metadata is never dereferenced.
            patch.object(nsb.NixlDynamicStorageBackend, "init_chunk_meta"),
            patch.object(nsb, "NixlDynamicStorageAgent", return_value=MagicMock()),
        ):
            return nsb.NixlDynamicStorageBackend(
                cfg, MagicMock(), MagicMock(), MagicMock()
            )

    def test_doca_memos_uses_b128(self) -> None:
        backend = self._make_dynamic_backend("DOCA_MEMOS")
        key = _make_key()
        assert backend._format_object_key(key) == backend._format_object_key_b128(key)

    def test_obj_uses_url_safe(self) -> None:
        backend = self._make_dynamic_backend("OBJ")
        key = _make_key()
        assert backend._format_object_key(key) == backend._format_object_key_url_safe(
            key
        )
