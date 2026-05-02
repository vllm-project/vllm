# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Request steering hash invalidation.

Covers:
- invalidate_steering_hashes: clears cached_property values so they recompute
- Scheduler ordering: _update_request_as_session swaps sampling_params and
  invalidates before recomputing block hashes
"""

import unittest
from unittest.mock import MagicMock

import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, StreamingUpdate
from vllm.v1.structured_output import StructuredOutputManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEERING_A = {"post_mlp": {0: [1.0, 2.0]}}
STEERING_B = {"post_mlp": {0: [99.0, 100.0]}}

init_none_hash(sha256_cbor)


def _make_request(
    request_id: str = "req-1",
    steering_vectors=None,
    prefill_steering_vectors=None,
    decode_steering_vectors=None,
    max_tokens: int = 16,
    resumable: bool = True,
    prompt_token_ids: list[int] | None = None,
) -> Request:
    sp = SamplingParams(
        max_tokens=max_tokens,
        steering_vectors=steering_vectors,
        prefill_steering_vectors=prefill_steering_vectors,
        decode_steering_vectors=decode_steering_vectors,
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=(
            prompt_token_ids if prompt_token_ids is not None else [1, 2, 3]
        ),
        sampling_params=sp,
        pooling_params=None,
        resumable=resumable,
    )


def _create_scheduler() -> Scheduler:
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.model_config.is_multimodal_model = False
    vllm_config.model_config.is_encoder_decoder = False
    vllm_config.model_config.max_model_len = 1024
    vllm_config.model_config.enable_return_routed_experts = False
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.num_gpu_blocks = 1000
    vllm_config.cache_config.enable_prefix_caching = False
    kv_cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
    )


# ---------------------------------------------------------------------------
# Tests: invalidate_steering_hashes on Request
# ---------------------------------------------------------------------------


class TestInvalidateSteeringHashes(unittest.TestCase):
    """Verify that invalidate_steering_hashes clears @cached_property
    entries so they recompute from current sampling_params."""

    def test_hashes_recompute_after_invalidation(self):
        """After swapping sampling_params and invalidating, the cached
        properties must return values based on the *new* params."""
        req = _make_request(steering_vectors=STEERING_A)

        # Populate caches
        hash_prefill_old = req.prefill_steering_config_hash
        hash_decode_old = req.decode_steering_config_hash
        assert hash_prefill_old != 0
        assert hash_decode_old != 0

        # Replace sampling_params with different steering vectors
        req.sampling_params = SamplingParams(
            max_tokens=16,
            steering_vectors=STEERING_B,
        )

        # Without invalidation the cached hashes are stale
        assert req.prefill_steering_config_hash == hash_prefill_old

        # After invalidation they should reflect the new config
        req.invalidate_steering_hashes()

        hash_prefill_new = req.prefill_steering_config_hash
        hash_decode_new = req.decode_steering_config_hash
        assert hash_prefill_new != hash_prefill_old
        assert hash_decode_new != hash_decode_old

    def test_noop_when_not_cached(self):
        """Calling invalidate before hashes are ever accessed must not
        raise."""
        req = _make_request(steering_vectors=STEERING_A)
        # Should not raise even though nothing is cached
        req.invalidate_steering_hashes()

        # Hashes should still be computable afterwards
        h = req.prefill_steering_config_hash
        assert h != 0

    def test_noop_when_no_steering(self):
        """Invalidation on a request with no steering vectors is a no-op."""
        req = _make_request()  # no steering
        req.invalidate_steering_hashes()
        assert req.prefill_steering_config_hash == 0
        assert req.decode_steering_config_hash == 0

    def test_phase_specific_invalidation(self):
        """Prefill-only and decode-only vectors should each recompute
        correctly after invalidation."""
        req = _make_request(
            prefill_steering_vectors=STEERING_A,
            decode_steering_vectors=STEERING_B,
        )

        old_prefill = req.prefill_steering_config_hash
        old_decode = req.decode_steering_config_hash
        assert old_prefill != 0
        assert old_decode != 0
        # prefill and decode should differ because vectors differ
        assert old_prefill != old_decode

        # Swap: move STEERING_B to prefill, STEERING_A to decode
        req.sampling_params = SamplingParams(
            max_tokens=16,
            prefill_steering_vectors=STEERING_B,
            decode_steering_vectors=STEERING_A,
        )
        req.invalidate_steering_hashes()

        new_prefill = req.prefill_steering_config_hash
        new_decode = req.decode_steering_config_hash
        # The hashes should have swapped roles
        assert new_prefill == old_decode
        assert new_decode == old_prefill

    def test_block_hash_steering_override_recomputes_hashes(self):
        """Switching block-hash steering overrides must recompute APC keys."""
        req = Request(
            request_id="req-block-hash",
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(
                max_tokens=4,
                prefill_steering_vectors=STEERING_A,
            ),
            pooling_params=None,
            resumable=False,
            block_hasher=get_request_block_hasher(2, sha256_cbor),
        )

        original_hashes = list(req.block_hashes)
        assert original_hashes

        req.set_block_hash_steering_overrides(prefill_hash=0)
        fallback_hashes = list(req.block_hashes)
        assert fallback_hashes != original_hashes

        req.set_block_hash_steering_overrides()
        assert req.block_hashes == original_hashes


# ---------------------------------------------------------------------------
# Tests: scheduler _update_request_as_session ordering
# ---------------------------------------------------------------------------


class TestSchedulerSteeringHashOrdering(unittest.TestCase):
    """Verify that _update_request_as_session applies sampling_params and
    invalidates steering hashes *before* update_block_hashes, so that new
    block hashes are computed with the correct steering config."""

    def test_session_update_reflects_new_steering(self):
        """After _update_request_as_session, the request's steering hashes
        must match the *new* sampling_params, not the old ones."""
        scheduler = _create_scheduler()

        session = _make_request(
            request_id="session",
            steering_vectors=STEERING_A,
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)

        # Record old hashes
        old_prefill_hash = session.prefill_steering_config_hash
        old_decode_hash = session.decode_steering_config_hash

        # Build an update with different steering vectors
        new_sp = SamplingParams(
            max_tokens=16,
            steering_vectors=STEERING_B,
        )
        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[4, 5, 6],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=new_sp,
        )

        scheduler._update_request_as_session(session, update)

        # Hashes must reflect STEERING_B, not STEERING_A
        assert session.prefill_steering_config_hash != old_prefill_hash
        assert session.decode_steering_config_hash != old_decode_hash

        # Compute expected hash from a fresh request with STEERING_B
        ref = _make_request(steering_vectors=STEERING_B)
        assert session.prefill_steering_config_hash == ref.prefill_steering_config_hash
        assert session.decode_steering_config_hash == ref.decode_steering_config_hash

    def test_session_update_no_steering_to_steering(self):
        """Transitioning from no steering to steering must produce
        non-zero hashes after session update."""
        scheduler = _create_scheduler()

        session = _make_request(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)
        assert session.prefill_steering_config_hash == 0

        new_sp = SamplingParams(
            max_tokens=16,
            steering_vectors=STEERING_A,
        )
        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[4, 5],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=new_sp,
        )

        scheduler._update_request_as_session(session, update)

        assert session.prefill_steering_config_hash != 0
        assert session.decode_steering_config_hash != 0

    def test_session_update_steering_to_no_steering(self):
        """Transitioning from steering to no steering must produce
        zero hashes after session update."""
        scheduler = _create_scheduler()

        session = _make_request(
            request_id="session",
            steering_vectors=STEERING_A,
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)
        assert session.prefill_steering_config_hash != 0

        new_sp = SamplingParams(max_tokens=16)
        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[4, 5],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=new_sp,
        )

        scheduler._update_request_as_session(session, update)

        assert session.prefill_steering_config_hash == 0
        assert session.decode_steering_config_hash == 0

    def test_session_updates_num_prompt_tokens_before_rehash(self):
        """Streaming updates must bump num_prompt_tokens before
        update_block_hashes runs so appended prompt tokens are hashed in
        prompt phase, not decode phase."""
        scheduler = _create_scheduler()

        session = _make_request(
            request_id="session",
            steering_vectors=STEERING_A,
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)

        observed_prompt_lens: list[int] = []
        original_update_block_hashes = session.update_block_hashes

        def wrapped_update_block_hashes():
            observed_prompt_lens.append(session.num_prompt_tokens)
            return original_update_block_hashes()

        session.update_block_hashes = wrapped_update_block_hashes  # type: ignore[method-assign]

        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[4, 5],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=SamplingParams(
                max_tokens=16,
                steering_vectors=STEERING_B,
            ),
        )

        scheduler._update_request_as_session(session, update)

        assert observed_prompt_lens == [5]
        assert session.num_prompt_tokens == 5

    def test_session_update_refreshes_block_hash_steering_overrides(self):
        """Streaming updates must refresh the APC-facing steering override
        hashes, not just the cached_property values."""
        scheduler = _create_scheduler()

        session = _make_request(
            request_id="session",
            steering_vectors=STEERING_A,
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)

        old_block_prefill = session.block_hash_prefill_steering_config_hash
        old_block_decode = session.block_hash_decode_steering_config_hash

        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[4, 5],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=SamplingParams(
                max_tokens=16,
                steering_vectors=STEERING_B,
            ),
        )

        scheduler._update_request_as_session(session, update)

        assert (
            session.block_hash_prefill_steering_config_hash
            == session.prefill_steering_config_hash
        )
        assert (
            session.block_hash_decode_steering_config_hash
            == session.decode_steering_config_hash
        )
        assert session.block_hash_prefill_steering_config_hash != old_block_prefill
        assert session.block_hash_decode_steering_config_hash != old_block_decode

    def test_session_update_rebuilds_existing_block_hash_chain(self):
        """Streaming continuation must rebuild existing APC hashes because
        previously generated tokens become prompt tokens in the next turn."""
        scheduler = _create_scheduler()
        block_hasher = get_request_block_hasher(2, sha256_cbor)

        session = _make_request(
            request_id="session",
            steering_vectors=STEERING_A,
            prompt_token_ids=[1, 2],
        )
        session._block_hasher = block_hasher
        session.block_hashes.clear()
        session.update_block_hashes()
        session.append_output_token_ids([3, 4])
        session.num_computed_tokens = len(session.all_token_ids)

        old_block_hashes = list(session.block_hashes)

        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[5, 6],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=SamplingParams(
                max_tokens=16,
                steering_vectors=STEERING_B,
            ),
        )

        scheduler._update_request_as_session(session, update)

        ref = _make_request(
            request_id="ref",
            steering_vectors=STEERING_B,
            prompt_token_ids=[1, 2, 3, 4, 5, 6],
        )
        ref._block_hasher = block_hasher
        ref.block_hashes.clear()
        ref.update_block_hashes()

        assert session.block_hashes == ref.block_hashes
        assert session.block_hashes != old_block_hashes

    def test_session_update_refreshes_skip_reading_prefix_cache(self):
        """Streaming updates must refresh the request-level APC read policy
        after swapping sampling params."""
        scheduler = _create_scheduler()

        session = _make_request(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)
        assert session.skip_reading_prefix_cache is False

        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[4, 5],
            max_tokens=16,
            arrival_time=1.0,
            sampling_params=SamplingParams(
                max_tokens=16,
                skip_reading_prefix_cache=True,
            ),
        )

        scheduler._update_request_as_session(session, update)

        assert session.skip_reading_prefix_cache is True
