# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for GPUModelRunner._update_streaming_request function."""

from unittest.mock import Mock

import pytest

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def mock_model_runner_with_input_batch():
    """Create a mock GPUModelRunner with a real InputBatch for e2e testing."""

    runner = Mock(spec=GPUModelRunner)
    runner.uses_mrope = False
    runner.requests = {}
    runner.max_num_reqs = 10
    runner.max_model_len = 1024

    # Create a real InputBatch for e2e testing
    runner.input_batch = InputBatch(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device="cpu",
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        logitsprocs=None,
        is_pooling_model=False,
    )
    runner.late_interaction_runner = Mock()
    return runner


def test_e2e_streaming_request_update_basic_flow(mock_model_runner_with_input_batch):
    """Test that streaming session are updated correctly.

    This test validates that when a streaming session is updated with new prompt tokens:
    1. The request is removed from InputBatch before updating (avoids duplication)
    2. Request state fields are updated correctly
    3. output_token_ids is cleared (intermediate outputs are now in prompt_token_ids)
    """
    runner = mock_model_runner_with_input_batch
    req_id = "streaming_req_0"

    # Step 1: Create initial request state with some computed tokens
    initial_req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.5),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[10, 11],  # Generated 2 tokens
    )
    runner.requests[req_id] = initial_req_state

    # Add request to InputBatch
    runner.input_batch.add_request(initial_req_state)
    assert req_id in runner.input_batch.req_id_to_index

    # Step 2: Create new request data with extended prompt
    # The scheduler has already set prompt_token_ids to the full sequence
    # (original prompt + intermediate outputs + new prompt)
    new_req_data = Mock()
    new_req_data.prompt_token_ids = [
        1,
        2,
        3,
        10,
        4,
        5,
    ]  # Full sequence with intermediate output (10)
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 4  # 3 original prompt + 1 intermediate output

    # Step 3: Update the request
    updated_req_state = GPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    # Step 4: Verify the request state was updated correctly
    assert updated_req_state.prompt_token_ids == [1, 2, 3, 10, 4, 5]
    assert updated_req_state.num_computed_tokens == 4
    assert updated_req_state.sampling_params.temperature == 0.8
    assert updated_req_state.sampling_params.max_tokens == 50
    assert updated_req_state.block_ids == ([0, 1],)

    # Verify output_token_ids were cleared
    # (intermediate outputs are now in prompt_token_ids)
    assert updated_req_state.output_token_ids == []

    # Verify the same object is returned
    assert runner.requests[req_id] is updated_req_state

    # Verify request was removed from InputBatch during update (avoids duplication)
    assert req_id not in runner.input_batch.req_id_to_index


def test_e2e_streaming_with_multimodal_features(mock_model_runner_with_input_batch):
    """Test that streaming session with multimodal features are updated correctly.

    This test validates that when a streaming session with mm features is updated:
    1. The request is removed from InputBatch before updating (avoids duplication)
    2. Multimodal features from both requests are preserved and merged correctly
    3. New prompt tokens (including intermediate outputs) are appended correctly
    4. output_token_ids is cleared (intermediate outputs are now in prompt_token_ids)
    """
    runner = mock_model_runner_with_input_batch
    req_id = "streaming_mm_req_0"

    # Step 1: Create initial request state with one multimodal feature
    mm_feature_1 = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        modality="audio",
        identifier="audio_1",
        mm_position=PlaceholderRange(offset=2, length=10),
    )

    initial_req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3, 4],  # 2 + 10 (mm) + 2 = 14 tokens
        mm_features=[mm_feature_1],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=14,
        output_token_ids=[100],  # Generated 1 token
    )
    runner.requests[req_id] = initial_req_state

    # Add request to InputBatch
    runner.input_batch.add_request(initial_req_state)
    assert req_id in runner.input_batch.req_id_to_index

    # Step 2: Create new request data with additional multimodal feature
    # The scheduler has already set prompt_token_ids to the full sequence
    # (original prompt + intermediate outputs + new prompt with new multimodal feature)
    mm_feature_2 = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        modality="audio",
        identifier="audio_2",
        mm_position=PlaceholderRange(offset=15, length=5),
    )

    new_req_data = Mock()
    # Full sequence: [1, 2] + [0]*10 + [3, 4] + [100] + [0]*5 + [5] = 21 tokens
    new_req_data.prompt_token_ids = [1, 2] + [0] * 10 + [3, 4, 100] + [0] * 5 + [5]
    new_req_data.mm_features = [mm_feature_1, mm_feature_2]
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 14  # 14 tokens from initial request

    # Step 3: Update the request
    updated_req_state = GPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    # Step 4: Verify the request state was updated correctly
    # Verify multimodal features are preserved
    assert len(updated_req_state.mm_features) == 2
    assert updated_req_state.mm_features[0] == mm_feature_1
    assert updated_req_state.mm_features[1] == mm_feature_2

    # Verify prompt tokens include intermediate output (100) and new tokens
    # Initial: 2 + 10 (mm1) + 2 = 14 tokens
    # New: 2 + 10 (mm1) + 2 + 1 (output 100) + 5 (mm2) + 1 = 21 tokens
    assert len(updated_req_state.prompt_token_ids) == 21
    assert updated_req_state.prompt_token_ids == [1, 2] + [0] * 10 + [3, 4, 100] + [
        0
    ] * 5 + [5]

    # Verify output_token_ids were cleared
    # (intermediate outputs are now in prompt_token_ids)
    assert updated_req_state.output_token_ids == []

    # Verify other parameters were updated
    assert updated_req_state.num_computed_tokens == 14
    assert updated_req_state.sampling_params.temperature == 0.7
    assert updated_req_state.sampling_params.max_tokens == 30
    assert updated_req_state.block_ids == ([0, 1],)

    # Verify the same object is returned
    assert runner.requests[req_id] is updated_req_state

    # Verify request was removed from InputBatch during update (avoids duplication)
    assert req_id not in runner.input_batch.req_id_to_index


# ---------------------------------------------------------------------------
# Helpers for steering tests
# ---------------------------------------------------------------------------


def _make_steering_runner():
    """Create a mock GPUModelRunner with steering infrastructure."""
    runner = Mock(spec=GPUModelRunner)
    runner.uses_mrope = False
    runner.requests = {}
    runner.max_num_reqs = 10
    runner.max_model_len = 1024

    runner.input_batch = InputBatch(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device="cpu",
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        is_spec_decode=False,
        logitsprocs=None,
        is_pooling_model=False,
    )

    runner.late_interaction_runner = Mock()

    runner._steering_manager = Mock()
    runner._steering_manager.register_config = Mock()
    runner._steering_manager.release_config = Mock()
    runner._req_steering_phase = {}
    return runner


def _make_req_state(
    req_id,
    prefill_hash=0,
    decode_hash=0,
):
    """Create a minimal CachedRequestState for steering tests."""
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.5),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[10, 11],
        prefill_steering_config_hash=prefill_hash,
        decode_steering_config_hash=decode_hash,
    )


def _make_new_req_data(
    req_id,
    prefill_hash=0,
    decode_hash=0,
    effective_prefill=None,
    effective_decode=None,
):
    """Create a mock NewRequestData with steering hashes."""
    new_req_data = Mock()
    new_req_data.req_id = req_id
    new_req_data.prompt_token_ids = [1, 2, 3, 10, 4, 5]
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 4
    new_req_data.prefill_steering_config_hash = prefill_hash
    new_req_data.decode_steering_config_hash = decode_hash

    sp = Mock()
    sp.effective_prefill_steering = effective_prefill
    sp.effective_decode_steering = effective_decode
    new_req_data.sampling_params = sp
    return new_req_data


# ---------------------------------------------------------------------------
# Steering refresh tests
# ---------------------------------------------------------------------------


def test_streaming_update_refreshes_steering_hashes():
    """Changing steering between turns updates hashes, releases old config,
    and registers the new prefill config."""
    runner = _make_steering_runner()
    req_id = "steer_req_0"

    req_state = _make_req_state(req_id, prefill_hash=111, decode_hash=222)
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)
    runner._req_steering_phase[req_id] = "decode"

    prefill_vectors = {"layer.0": {0: [0.1, 0.2]}}
    new_req_data = _make_new_req_data(
        req_id,
        prefill_hash=333,
        decode_hash=444,
        effective_prefill=prefill_vectors,
    )

    updated = GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    # Hashes must be refreshed on the state.
    assert updated.prefill_steering_config_hash == 333
    assert updated.decode_steering_config_hash == 444

    # Old decode config released (was in decode phase with hash 222).
    runner._steering_manager.release_config.assert_called_once_with(222, "decode")

    # New prefill config registered.
    runner._steering_manager.register_config.assert_called_once_with(
        333, prefill_vectors, phase="prefill"
    )

    # Phase tracking reset to prefill.
    assert runner._req_steering_phase[req_id] == "prefill"


def test_streaming_update_no_steering_to_steering():
    """A request that previously had no steering gains steering on re-add."""
    runner = _make_steering_runner()
    req_id = "steer_req_1"

    req_state = _make_req_state(req_id, prefill_hash=0, decode_hash=0)
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)
    # No phase entry for a request without steering.

    prefill_vectors = {"layer.0": {0: [0.5]}}
    new_req_data = _make_new_req_data(
        req_id,
        prefill_hash=111,
        decode_hash=222,
        effective_prefill=prefill_vectors,
    )

    updated = GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    assert updated.prefill_steering_config_hash == 111
    assert updated.decode_steering_config_hash == 222

    # No old config to release (hashes were 0).
    runner._steering_manager.release_config.assert_not_called()

    # New prefill config registered.
    runner._steering_manager.register_config.assert_called_once_with(
        111, prefill_vectors, phase="prefill"
    )

    assert runner._req_steering_phase[req_id] == "prefill"


def test_streaming_update_steering_to_no_steering():
    """A request that had steering clears it on re-add."""
    runner = _make_steering_runner()
    req_id = "steer_req_2"

    req_state = _make_req_state(req_id, prefill_hash=111, decode_hash=0)
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)
    runner._req_steering_phase[req_id] = "prefill"

    new_req_data = _make_new_req_data(
        req_id,
        prefill_hash=0,
        decode_hash=0,
    )

    updated = GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    assert updated.prefill_steering_config_hash == 0
    assert updated.decode_steering_config_hash == 0

    # Old prefill config released.
    runner._steering_manager.release_config.assert_called_once_with(111, "prefill")

    # No new config registered.
    runner._steering_manager.register_config.assert_not_called()

    # Phase tracking cleared.
    assert req_id not in runner._req_steering_phase


def test_streaming_update_steering_hashes_unchanged():
    """Same non-zero hashes: old released, new registered (net refcount
    neutral)."""
    runner = _make_steering_runner()
    req_id = "steer_req_3"

    req_state = _make_req_state(req_id, prefill_hash=111, decode_hash=222)
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)
    runner._req_steering_phase[req_id] = "prefill"

    prefill_vectors = {"layer.0": {0: [0.3]}}
    new_req_data = _make_new_req_data(
        req_id,
        prefill_hash=111,
        decode_hash=222,
        effective_prefill=prefill_vectors,
    )

    GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    # Old prefill config released (was in prefill phase).
    runner._steering_manager.release_config.assert_called_once_with(111, "prefill")

    # New prefill config registered (re-entering prefill).
    runner._steering_manager.register_config.assert_called_once_with(
        111, prefill_vectors, phase="prefill"
    )


def test_streaming_update_no_steering_manager():
    """Without a steering manager, hashes are still updated on state and no
    errors occur."""
    runner = _make_steering_runner()
    req_id = "steer_req_4"

    req_state = _make_req_state(req_id, prefill_hash=111, decode_hash=222)
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)

    # Remove the steering manager to simulate no-steering setup.
    del runner._steering_manager

    new_req_data = _make_new_req_data(
        req_id,
        prefill_hash=333,
        decode_hash=444,
    )

    updated = GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    # Hashes still updated on the state even without a manager.
    assert updated.prefill_steering_config_hash == 333
    assert updated.decode_steering_config_hash == 444


def test_streaming_update_capacity_error_propagates():
    """When register_config raises RuntimeError under the strict-capacity
    contract, the streaming refresh path must propagate the exception
    rather than silently deferring it."""
    runner = _make_steering_runner()
    req_id = "steer_req_5"

    req_state = _make_req_state(req_id, prefill_hash=0, decode_hash=0)
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)

    # Make register_config fail with RuntimeError (capacity full).
    runner._steering_manager.register_config.side_effect = RuntimeError("capacity full")

    prefill_vectors = {"layer.0": {0: [0.1]}}
    new_req_data = _make_new_req_data(
        req_id,
        prefill_hash=555,
        decode_hash=666,
        effective_prefill=prefill_vectors,
    )

    with pytest.raises(RuntimeError, match="capacity full"):
        GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    # Registration was attempted exactly once before the raise.
    runner._steering_manager.register_config.assert_called_once()


def test_streaming_update_decode_phase_no_hash_vector_mismatch():
    """Verify no hash/vector mismatch when a decode-phase request gets a
    streaming update with new steering vectors.

    Regression test: previously ``_reset_steering_for_resumption`` was called
    *after* ``req_state.sampling_params`` was updated to the NEW params but
    *before* the steering config hashes were updated.  Inside ``_reset``,
    it would read ``req_state.sampling_params`` (NEW) paired with
    ``req_state.prefill_steering_config_hash`` (still OLD), leading to a
    mismatch where the old hash was registered with the new vectors.

    With the fix the explicit block handles the streaming path correctly:
    it saves old hashes before mutation, releases the old decode config,
    then registers the new prefill config using data from ``new_req_data``.

    This test uses a *real* SteeringManager to assert that the final
    registration state is internally consistent (correct hash -> correct
    vectors, correct refcount).
    """
    from vllm.v1.worker.steering_manager import SteeringManager

    runner = Mock(spec=GPUModelRunner)
    runner.uses_mrope = False
    runner.requests = {}
    runner.max_num_reqs = 10
    runner.max_model_len = 1024

    runner.input_batch = InputBatch(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device="cpu",
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        is_spec_decode=False,
        logitsprocs=None,
        is_pooling_model=False,
    )
    runner.late_interaction_runner = Mock()

    # Use a real SteeringManager so we can inspect actual state.
    mgr = SteeringManager(max_steering_configs=4)
    runner._steering_manager = mgr
    runner._req_steering_phase = {}

    req_id = "hash_mismatch_req"

    # --- Set up OLD state: request in decode phase with old hashes ---
    old_prefill_hash = 1000
    old_decode_hash = 2000
    old_prefill_vectors = {"layer.0": {0: [0.1, 0.2, 0.3, 0.4]}}
    old_decode_vectors = {"layer.0": {0: [0.5, 0.6, 0.7, 0.8]}}

    # Register old configs in the manager as if they were added during
    # the request's initial prefill and prefill->decode transition.
    mgr.register_config(old_prefill_hash, old_prefill_vectors, phase="prefill")
    mgr.register_config(old_decode_hash, old_decode_vectors, phase="decode")
    # Release old prefill (transition to decode would have released it).
    mgr.release_config(old_prefill_hash, "prefill")

    req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.5),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=5,
        output_token_ids=[10, 11, 12],
        prefill_steering_config_hash=old_prefill_hash,
        decode_steering_config_hash=old_decode_hash,
    )
    runner.requests[req_id] = req_state
    runner.input_batch.add_request(req_state)
    runner._req_steering_phase[req_id] = "decode"

    # Verify old decode config is registered before the update.
    assert (old_decode_hash, "decode") in mgr.config_to_row
    assert mgr.config_refcounts[(old_decode_hash, "decode")] == 1

    # --- Build NEW request data with different steering vectors ---
    new_prefill_hash = 3000
    new_decode_hash = 4000
    new_prefill_vectors = {"layer.0": {0: [1.0, 2.0, 3.0, 4.0]}}

    new_req_data = Mock()
    new_req_data.req_id = req_id
    new_req_data.prompt_token_ids = [1, 2, 3, 4, 5, 10, 11, 12, 6, 7]
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 8  # re-entering prefill

    new_req_data.prefill_steering_config_hash = new_prefill_hash
    new_req_data.decode_steering_config_hash = new_decode_hash

    sp = Mock()
    sp.effective_prefill_steering = new_prefill_vectors
    sp.effective_decode_steering = {"layer.0": {0: [5.0, 6.0, 7.0, 8.0]}}
    new_req_data.sampling_params = sp

    # --- Execute the streaming update ---
    updated = GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    # --- Assertions ---

    # 1. Hashes on the state must be the NEW ones.
    assert updated.prefill_steering_config_hash == new_prefill_hash
    assert updated.decode_steering_config_hash == new_decode_hash

    # 2. Old decode config must have been released (refcount -> 0, freed).
    assert (old_decode_hash, "decode") not in mgr.config_to_row
    assert (old_decode_hash, "decode") not in mgr.config_refcounts

    # 3. Old prefill config must still be absent (was released earlier).
    assert (old_prefill_hash, "prefill") not in mgr.config_to_row

    # 4. The NEW prefill config must be registered with refcount 1.
    assert (new_prefill_hash, "prefill") in mgr.config_to_row
    assert mgr.config_refcounts[(new_prefill_hash, "prefill")] == 1

    # 5. The registered vectors must be the NEW prefill vectors (not old).
    stored = mgr.config_vectors[(new_prefill_hash, "prefill")]
    assert "layer.0" in stored
    assert 0 in stored["layer.0"]
    import torch

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    assert torch.allclose(stored["layer.0"][0].squeeze(), expected)

    # 6. The NEW decode config must NOT be registered yet (it will be
    #    registered when the request transitions from prefill to decode).
    assert (new_decode_hash, "decode") not in mgr.config_to_row

    # 7. Phase must be "prefill" (streaming re-adds start in prefill).
    assert runner._req_steering_phase[req_id] == "prefill"
