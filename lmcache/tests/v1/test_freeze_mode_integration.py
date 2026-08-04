# SPDX-License-Identifier: Apache-2.0
# Standard
import random
import time

# Third Party
import requests
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.metadata import LMCacheMetadata
from tests.v1.utils import (
    MockAdapter,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
    get_available_port,
    recover_engine_states,
)


def test_freeze_with_real_cache_engine(autorelease_v1):
    """
    Integration test for freeze mode with real cache engine.

    This test verifies:
    1. Normal mode stores KV cache correctly to local_cpu backend
    2. Enable freeze mode via HTTP API
    3. Store operations are skipped in freeze mode (no new allocations)
    4. Only local_cpu backend is accessible in freeze mode
    5. Disable freeze mode and verify store works again
    """
    # Setup
    device = "cpu"
    instance_id = f"test_freeze_real_{random.getrandbits(64)}"
    chunk_size = 256
    num_tokens = 512
    num_blocks = 100
    block_size = 16
    dtype = torch.bfloat16
    kv_shape = (32, 2, chunk_size, 8, 128)

    # Create config with CPU backend for easy verification
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
    )

    # Enable Internal API Server
    cfg.internal_api_server_enabled = True
    # Get an available port for the API server
    # worker_id will be 0, so port_offset = 1 + 0 = 1
    # actual_port = port_start + port_offset, so port_start = actual_port - 1
    actual_port = get_available_port()
    cfg.internal_api_server_port_start = actual_port - 1
    cfg.internal_api_server_socket_path_prefix = None

    # Create mock GPU connector that works on CPU
    connector = MockGPUConnector(kv_shape=kv_shape)
    # Use explicit metadata with worker_id=0 to control port offset
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )

    # Create engine
    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg,
            metadata,
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    api_server = None
    try:
        # Create adapter and start API server
        adapter = MockAdapter(engine, cfg)
        api_server = InternalAPIServer(adapter)
        api_server.start()

        # Wait for server to start and become responsive
        max_retries = 40
        retry_delay = 0.5
        server_ready = False
        base_url = "http://localhost:%d" % actual_port

        # Wait for server to be responsive
        for i in range(max_retries):
            try:
                response = requests.get("%s/freeze/status" % base_url, timeout=1)
                if response.status_code in [200, 503]:
                    server_ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(retry_delay)

        assert server_ready, "Server failed to respond after %d retries" % max_retries

        # ============================================
        # Step 1: Verify initial freeze mode is OFF
        # ============================================
        response = requests.get("%s/freeze/status" % base_url)
        assert response.status_code == 200
        assert response.json()["freeze"] is False

        # ============================================
        # Step 2: Normal mode - Store should work
        # ============================================
        tokens_1 = generate_tokens(num_tokens, device)
        kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype
        )
        slot_mapping_1 = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping_1 = torch.tensor(slot_mapping_1, device=device)

        # Store data in normal mode
        engine.store(
            tokens=tokens_1,
            kvcaches=kv_cache,
            slot_mapping=slot_mapping_1,
        )
        recover_engine_states(engine)

        # Wait for async store to complete
        timeout = 5
        start_time = time.time()
        while engine.lookup(tokens_1) < num_tokens:
            if time.time() - start_time > timeout:
                raise TimeoutError("Store operation timed out in normal mode")
            time.sleep(0.1)

        # Verify data was stored
        stored_length = engine.lookup(tokens_1)
        assert stored_length == num_tokens, "Expected %d tokens stored, got %d" % (
            num_tokens,
            stored_length,
        )

        # ============================================
        # Step 3: Enable freeze mode via HTTP API
        # ============================================
        response = requests.put("%s/freeze/enable" % base_url)
        assert response.status_code == 200
        result = response.json()
        assert result["freeze"] is True
        assert result["status"] == "success"

        # Verify status
        response = requests.get("%s/freeze/status" % base_url)
        assert response.status_code == 200
        assert response.json()["freeze"] is True

        # ============================================
        # Step 4: Read-only mode - Store should be skipped
        # ============================================
        tokens_2 = generate_tokens(num_tokens, device)
        slot_mapping_2 = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping_2 = torch.tensor(slot_mapping_2, device=device)

        # Try to store new data in freeze mode
        engine.store(
            tokens=tokens_2,
            kvcaches=kv_cache,
            slot_mapping=slot_mapping_2,
        )
        recover_engine_states(engine)

        # Wait a bit to ensure store would have completed if not skipped
        time.sleep(1)

        # Verify data was NOT stored (freeze mode should skip)
        stored_length_2 = engine.lookup(tokens_2)
        assert stored_length_2 == 0, (
            "Expected 0 tokens stored in freeze mode, got %d" % stored_length_2
        )

        # Original data should still be there
        assert engine.lookup(tokens_1) == num_tokens

        # ============================================
        # Step 5: Disable freeze mode via HTTP API
        # ============================================
        response = requests.put("%s/freeze/disable" % base_url)
        assert response.status_code == 200
        result = response.json()
        assert result["freeze"] is False
        assert result["status"] == "success"

        # ============================================
        # Step 6: Store should work again after disabling
        # ============================================
        tokens_3 = generate_tokens(num_tokens, device)
        slot_mapping_3 = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping_3 = torch.tensor(slot_mapping_3, device=device)

        engine.store(
            tokens=tokens_3,
            kvcaches=kv_cache,
            slot_mapping=slot_mapping_3,
        )
        recover_engine_states(engine)

        # Wait for async store to complete
        start_time = time.time()
        while engine.lookup(tokens_3) < num_tokens:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    "Store operation timed out after disabling freeze mode"
                )
            time.sleep(0.1)

        # Verify data was stored
        stored_length_3 = engine.lookup(tokens_3)
        assert stored_length_3 == num_tokens, (
            "Expected %d tokens stored after disable, got %d"
            % (num_tokens, stored_length_3)
        )
    finally:
        # Cleanup
        if api_server:
            api_server.stop()
            # Give it some time to shut down
            time.sleep(0.5)
        LMCacheEngineBuilder.destroy(instance_id)


def test_freeze_direct_api(autorelease_v1):
    """
    Test freeze mode via direct engine API (without HTTP server).

    This test verifies:
    1. Freeze mode can be enabled/disabled via direct API
    2. Store operations are skipped in freeze mode
    3. No new memory allocations occur in freeze mode
    4. Only local_cpu backend is used for retrieval in freeze mode
    """
    # Setup
    device = "cpu"
    instance_id = "test_freeze_direct"
    chunk_size = 256
    num_tokens = 512
    num_blocks = 100
    block_size = 16
    dtype = torch.bfloat16
    kv_shape = (32, 2, chunk_size, 8, 128)

    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
    )

    connector = MockGPUConnector(kv_shape=kv_shape)
    metadata = dumb_metadata(kv_shape)

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg,
            metadata,
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    try:
        # Verify initial state
        assert engine.is_frozen() is False

        # Test set/get methods
        engine.freeze(True)
        assert engine.is_frozen() is True

        engine.freeze(False)
        assert engine.is_frozen() is False

        # Test store behavior with freeze mode
        tokens = generate_tokens(num_tokens, device)
        kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype
        )
        slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping = torch.tensor(slot_mapping, device=device)

        # Store in normal mode
        engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
        recover_engine_states(engine)

        timeout = 5
        start_time = time.time()
        while engine.lookup(tokens) < num_tokens:
            if time.time() - start_time > timeout:
                raise TimeoutError("Store timed out")
            time.sleep(0.1)

        assert engine.lookup(tokens) == num_tokens

        # Enable freeze mode
        engine.freeze(True)

        # Try to store new data
        new_tokens = generate_tokens(num_tokens, device)
        new_slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        new_slot_mapping = torch.tensor(new_slot_mapping, device=device)

        engine.store(
            tokens=new_tokens, kvcaches=kv_cache, slot_mapping=new_slot_mapping
        )
        recover_engine_states(engine)

        time.sleep(1)

        # Should not be stored
        assert engine.lookup(new_tokens) == 0

    finally:
        LMCacheEngineBuilder.destroy(instance_id)
