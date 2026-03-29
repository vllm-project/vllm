# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering-aware prefix cache key generation."""

from unittest.mock import Mock

import pytest

from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import (
    _gen_steering_extra_hash_keys,
    hash_block_tokens,
    init_none_hash,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    """Ensure NONE_HASH is initialized for hash_block_tokens tests."""
    init_none_hash(sha256_cbor)


def make_mock_request(
    prefill_hash: int = 0,
    has_attr: bool = True,
) -> Mock:
    """Create a mock request with optional prefill_steering_config_hash."""
    req = Mock()
    if has_attr:
        req.prefill_steering_config_hash = prefill_hash
    else:
        del req.prefill_steering_config_hash
    return req


class TestGenSteeringExtraHashKeys:
    """Tests for _gen_steering_extra_hash_keys helper."""

    def test_returns_empty_list_when_hash_is_zero(self):
        req = make_mock_request(prefill_hash=0)
        result = _gen_steering_extra_hash_keys(req)
        assert result == []

    def test_returns_hash_when_nonzero(self):
        req = make_mock_request(prefill_hash=12345)
        result = _gen_steering_extra_hash_keys(req)
        assert result == [12345]

    def test_getattr_fallback_when_attribute_missing(self):
        req = make_mock_request(has_attr=False)
        result = _gen_steering_extra_hash_keys(req)
        assert result == []


class TestSteeringBlockHashes:
    """Tests for block hash behavior with steering extra keys."""

    def test_block_hashes_differ_with_different_prefill_steering(self):
        """Blocks with different prefill steering produce different hashes."""
        tokens = [1, 2, 3, 4]
        hash_no_steering = hash_block_tokens(
            sha256_cbor,
            None,
            tokens,
            extra_keys=None,
        )
        hash_with_steering = hash_block_tokens(
            sha256_cbor,
            None,
            tokens,
            extra_keys=(12345,),
        )
        assert hash_no_steering != hash_with_steering

    def test_block_hashes_identical_when_only_decode_steering_differs(self):
        """Decode steering is not included in extra keys, so block hashes
        should be identical when only decode steering differs."""
        req_a = make_mock_request(prefill_hash=0)
        req_a.decode_steering_config_hash = 111

        req_b = make_mock_request(prefill_hash=0)
        req_b.decode_steering_config_hash = 222

        keys_a = _gen_steering_extra_hash_keys(req_a)
        keys_b = _gen_steering_extra_hash_keys(req_b)

        assert keys_a == keys_b == []

        tokens = [10, 20, 30, 40]
        extra_a = tuple(keys_a) if keys_a else None
        extra_b = tuple(keys_b) if keys_b else None

        hash_a = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=extra_a)
        hash_b = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=extra_b)
        assert hash_a == hash_b

    def test_block_hashes_identical_when_no_steering(self):
        """A request with no steering attributes should produce the same
        block hash as a request with explicit prefill_steering_config_hash=0."""
        req_no_attr = make_mock_request(has_attr=False)
        req_zero = make_mock_request(prefill_hash=0)

        keys_no_attr = _gen_steering_extra_hash_keys(req_no_attr)
        keys_zero = _gen_steering_extra_hash_keys(req_zero)

        assert keys_no_attr == keys_zero == []

        tokens = [100, 200, 300]
        extra_no_attr = tuple(keys_no_attr) if keys_no_attr else None
        extra_zero = tuple(keys_zero) if keys_zero else None

        hash_no_attr = hash_block_tokens(
            sha256_cbor, None, tokens, extra_keys=extra_no_attr
        )
        hash_zero = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=extra_zero)
        assert hash_no_attr == hash_zero
