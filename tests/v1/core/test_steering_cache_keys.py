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
    decode_hash: int = 0,
    num_prompt_tokens: int = 10,
    has_prefill_attr: bool = True,
    has_decode_attr: bool = True,
) -> Mock:
    """Create a mock request with optional steering config hashes."""
    req = Mock()
    req.num_prompt_tokens = num_prompt_tokens
    if has_prefill_attr:
        req.prefill_steering_config_hash = prefill_hash
    else:
        del req.prefill_steering_config_hash
    if has_decode_attr:
        req.decode_steering_config_hash = decode_hash
    else:
        del req.decode_steering_config_hash
    return req


class TestGenSteeringExtraHashKeys:
    """Tests for _gen_steering_extra_hash_keys helper."""

    # --- Pure prompt block tests ---
    # (start < num_prompt_tokens, end <= num_prompt_tokens)

    def test_prompt_block_returns_empty_list_when_prefill_hash_is_zero(self):
        req = make_mock_request(prefill_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=0, end_token_idx=8)
        assert result == []

    def test_prompt_block_returns_prefill_hash_when_nonzero(self):
        req = make_mock_request(prefill_hash=12345, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=0, end_token_idx=8)
        assert result == [12345]

    def test_prompt_block_getattr_fallback_when_attribute_missing(self):
        req = make_mock_request(has_prefill_attr=False, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=0, end_token_idx=8)
        assert result == []

    def test_prompt_block_ignores_decode_hash(self):
        """Prompt blocks should use prefill hash, not decode hash."""
        req = make_mock_request(prefill_hash=0, decode_hash=99999, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=5, end_token_idx=10)
        assert result == []

    # --- Pure decode block tests (start >= num_prompt_tokens) ---

    def test_generated_block_returns_decode_hash_when_nonzero(self):
        req = make_mock_request(decode_hash=67890, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(
            req, start_token_idx=10, end_token_idx=18
        )
        assert result == [67890]

    def test_generated_block_returns_empty_when_decode_hash_is_zero(self):
        req = make_mock_request(decode_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(
            req, start_token_idx=15, end_token_idx=23
        )
        assert result == []

    def test_generated_block_ignores_prefill_hash(self):
        """Generated blocks should use decode hash, not prefill hash.
        The prefill hash is already embedded in the parent hash chain."""
        req = make_mock_request(prefill_hash=12345, decode_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(
            req, start_token_idx=10, end_token_idx=18
        )
        assert result == []

    def test_generated_block_getattr_fallback_when_decode_attr_missing(self):
        req = make_mock_request(has_decode_attr=False, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(
            req, start_token_idx=10, end_token_idx=18
        )
        assert result == []

    # --- Boundary exact and non-steered tests ---

    def test_boundary_exact_prompt_length_is_generated(self):
        """start_token_idx == num_prompt_tokens is a decode block."""
        req = make_mock_request(prefill_hash=111, decode_hash=222, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(
            req, start_token_idx=10, end_token_idx=18
        )
        assert result == [222]

    def test_non_steered_request_returns_empty_for_prompt_block(self):
        req = make_mock_request(prefill_hash=0, decode_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=0, end_token_idx=8)
        assert result == []

    def test_non_steered_request_returns_empty_for_generated_block(self):
        req = make_mock_request(prefill_hash=0, decode_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(
            req, start_token_idx=15, end_token_idx=23
        )
        assert result == []

    # --- Boundary-straddling block tests ---

    def test_boundary_block_returns_both_hashes(self):
        """A block straddling prompt/decode boundary should include both
        prefill and decode hashes."""
        req = make_mock_request(prefill_hash=111, decode_hash=222, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=5, end_token_idx=15)
        assert result == [111, 222]

    def test_boundary_block_prefill_only(self):
        """Boundary block with only prefill steering returns [prefill, 0]."""
        req = make_mock_request(prefill_hash=111, decode_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=5, end_token_idx=15)
        assert result == [111, 0]

    def test_boundary_block_decode_only(self):
        """Boundary block with only decode steering returns [0, decode]."""
        req = make_mock_request(prefill_hash=0, decode_hash=222, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=5, end_token_idx=15)
        assert result == [0, 222]

    def test_boundary_block_prefill_only_vs_decode_only_are_distinct(self):
        """prefill_hash=X, decode_hash=0 must produce different keys than
        prefill_hash=0, decode_hash=X to prevent false cache sharing."""
        X = 42
        req_prefill = make_mock_request(
            prefill_hash=X, decode_hash=0, num_prompt_tokens=10
        )
        req_decode = make_mock_request(
            prefill_hash=0, decode_hash=X, num_prompt_tokens=10
        )
        keys_prefill = _gen_steering_extra_hash_keys(
            req_prefill, start_token_idx=5, end_token_idx=15
        )
        keys_decode = _gen_steering_extra_hash_keys(
            req_decode, start_token_idx=5, end_token_idx=15
        )
        assert keys_prefill == [X, 0]
        assert keys_decode == [0, X]
        assert keys_prefill != keys_decode

    def test_boundary_block_no_steering(self):
        """Boundary block with no steering returns empty list."""
        req = make_mock_request(prefill_hash=0, decode_hash=0, num_prompt_tokens=10)
        result = _gen_steering_extra_hash_keys(req, start_token_idx=5, end_token_idx=15)
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

    def test_prompt_block_hashes_differ_when_decode_steering_differs(self):
        """For prompt blocks, decode steering is irrelevant, so hashes
        should be identical when only decode steering differs."""
        req_a = make_mock_request(prefill_hash=0, decode_hash=111, num_prompt_tokens=10)
        req_b = make_mock_request(prefill_hash=0, decode_hash=222, num_prompt_tokens=10)

        keys_a = _gen_steering_extra_hash_keys(
            req_a, start_token_idx=0, end_token_idx=8
        )
        keys_b = _gen_steering_extra_hash_keys(
            req_b, start_token_idx=0, end_token_idx=8
        )

        assert keys_a == keys_b == []

        tokens = [10, 20, 30, 40]
        extra_a = tuple(keys_a) if keys_a else None
        extra_b = tuple(keys_b) if keys_b else None

        hash_a = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=extra_a)
        hash_b = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=extra_b)
        assert hash_a == hash_b

    def test_generated_block_hashes_differ_with_different_decode_steering(self):
        """For generated blocks, different decode steering should produce
        different block hashes."""
        req_a = make_mock_request(decode_hash=111, num_prompt_tokens=5)
        req_b = make_mock_request(decode_hash=222, num_prompt_tokens=5)

        keys_a = _gen_steering_extra_hash_keys(
            req_a, start_token_idx=5, end_token_idx=13
        )
        keys_b = _gen_steering_extra_hash_keys(
            req_b, start_token_idx=5, end_token_idx=13
        )

        assert keys_a == [111]
        assert keys_b == [222]

        tokens = [10, 20, 30, 40]
        hash_a = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=tuple(keys_a))
        hash_b = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=tuple(keys_b))
        assert hash_a != hash_b

    def test_block_hashes_identical_when_no_steering(self):
        """A request with no steering attributes should produce the same
        block hash as a request with explicit hashes=0."""
        req_no_attr = make_mock_request(
            has_prefill_attr=False, has_decode_attr=False, num_prompt_tokens=10
        )
        req_zero = make_mock_request(
            prefill_hash=0, decode_hash=0, num_prompt_tokens=10
        )

        # Prompt blocks
        keys_no_attr = _gen_steering_extra_hash_keys(
            req_no_attr, start_token_idx=0, end_token_idx=8
        )
        keys_zero = _gen_steering_extra_hash_keys(
            req_zero, start_token_idx=0, end_token_idx=8
        )
        assert keys_no_attr == keys_zero == []

        tokens = [100, 200, 300]
        extra_no_attr = tuple(keys_no_attr) if keys_no_attr else None
        extra_zero = tuple(keys_zero) if keys_zero else None

        hash_no_attr = hash_block_tokens(
            sha256_cbor, None, tokens, extra_keys=extra_no_attr
        )
        hash_zero = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=extra_zero)
        assert hash_no_attr == hash_zero

        # Generated blocks
        keys_no_attr_gen = _gen_steering_extra_hash_keys(
            req_no_attr, start_token_idx=15, end_token_idx=23
        )
        keys_zero_gen = _gen_steering_extra_hash_keys(
            req_zero, start_token_idx=15, end_token_idx=23
        )
        assert keys_no_attr_gen == keys_zero_gen == []

    def test_boundary_block_hashes_differ_when_decode_steering_differs(self):
        """For boundary blocks straddling prompt/decode, different decode
        steering should produce different block hashes even when prefill
        steering is identical."""
        req_a = make_mock_request(
            prefill_hash=111, decode_hash=333, num_prompt_tokens=10
        )
        req_b = make_mock_request(
            prefill_hash=111, decode_hash=444, num_prompt_tokens=10
        )

        # Both blocks straddle: start=5 < 10, end=15 > 10
        keys_a = _gen_steering_extra_hash_keys(
            req_a, start_token_idx=5, end_token_idx=15
        )
        keys_b = _gen_steering_extra_hash_keys(
            req_b, start_token_idx=5, end_token_idx=15
        )

        assert keys_a == [111, 333]
        assert keys_b == [111, 444]

        tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        hash_a = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=tuple(keys_a))
        hash_b = hash_block_tokens(sha256_cbor, None, tokens, extra_keys=tuple(keys_b))
        assert hash_a != hash_b
