# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import types
from unittest.mock import patch

import pytest

_real_import = builtins.__import__


@pytest.mark.parametrize("algo", ["xxhash", "xxhash_cbor"])
def test_xxhash_missing_raises_import_error(algo: str):
    """CacheConfig must fail fast when xxhash is required but not installed."""
    from vllm.config.cache import CacheConfig

    def _mock_import(name, *args, **kwargs):
        if name == "xxhash":
            raise ModuleNotFoundError("No module named 'xxhash'",
                                      name="xxhash")
        return _real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        with pytest.raises(ImportError, match="xxhash"):
            CacheConfig(prefix_caching_hash_algo=algo)


@pytest.mark.parametrize("algo", ["xxhash", "xxhash_cbor"])
def test_xxhash_outdated_raises_import_error(algo: str):
    """CacheConfig must reject xxhash < 3.0.0 (missing xxh3_128_digest)."""
    from vllm.config.cache import CacheConfig

    fake_xxhash = types.ModuleType("xxhash")

    def _mock_import(name, *args, **kwargs):
        if name == "xxhash":
            return fake_xxhash
        return _real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        with pytest.raises(ImportError, match="xxhash >= 3.0.0"):
            CacheConfig(prefix_caching_hash_algo=algo)


@pytest.mark.parametrize("algo", ["xxhash", "xxhash_cbor"])
def test_xxhash_not_checked_when_prefix_caching_disabled(algo: str):
    """No error when prefix caching is off even if xxhash is missing."""
    from vllm.config.cache import CacheConfig

    def _mock_import(name, *args, **kwargs):
        if name == "xxhash":
            raise ModuleNotFoundError("No module named 'xxhash'",
                                      name="xxhash")
        return _real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        config = CacheConfig(prefix_caching_hash_algo=algo,
                             enable_prefix_caching=False)
        assert config.prefix_caching_hash_algo == algo
