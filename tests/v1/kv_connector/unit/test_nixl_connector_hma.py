# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NixlConnectorScheduler sw_sizes calculation with HMA."""

from unittest.mock import patch

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)

from .utils import (
    create_vllm_config,
    make_kv_cache_config,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "hma_enabled,expected_sw_sizes",
    [
        # HMA enabled: FullAttentionSpec (0) + SlidingWindowSpec (2048/16=128)
        (True, [0, 128]),
        # HMA disabled: only FullAttentionSpec (0)
        (False, [0]),
    ],
)
@patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.current_platform")
def test_sw_sizes(mock_platform, hma_enabled, expected_sw_sizes):
    """Test sw_sizes is correctly computed based on HMA enabled/disabled."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnectorScheduler,
    )

    mock_platform.device_type = "cpu"

    block_size = 16
    vllm_config = create_vllm_config(block_size=block_size)
    # SW 2048 tokens=>128 blocks
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, hma_enabled=hma_enabled, sw_size=2048
    )

    scheduler = NixlConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    # in number of blocks
    assert scheduler.sw_sizes == expected_sw_sizes, (
        f"Expected sw_sizes={expected_sw_sizes}, got {scheduler.sw_sizes}"
    )


@pytest.mark.cpu_test
def test_logical_to_kernel_block_ids_with_hma():
    """Test _logical_to_kernel_block_ids expands blocks when HMA is enabled.

    When HMA is enabled, the logical block size may differ from the kernel
    block size. Each logical block maps to multiple kernel blocks.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnectorWorker,
    )

    # Create a mock worker with just the required attributes
    # (use __new__ to skip __init__)
    worker = object.__new__(NixlConnectorWorker)

    # Simulate HMA scenario: logical block size = 32, kernel block size = 16
    # So each logical block maps to 2 kernel blocks eg [0]->[0,1]
    worker._physical_blocks_per_logical_kv_block = 2

    # Test conversion: FA + SW group
    logical_block_ids = [[0, 1, 2], [3, 4]]
    kernel_block_ids = worker._logical_to_kernel_block_ids(logical_block_ids)

    expected_kernel_block_ids = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert kernel_block_ids == expected_kernel_block_ids, (
        f"Expected {expected_kernel_block_ids}, got {kernel_block_ids}"
    )


@pytest.mark.parametrize("model_name, sw_size", [("google/gemma-3-1b-it", 512)])
def test_fewer_blocks_with_hma(monkeypatch, model_name, sw_size):
    """Test that a prefill instance returns fewer "remote blocks" for the SWA groups
    when sequence exceeds the sliding window.
    """
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    block_size = 16
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.5,
        "kv_transfer_config": kv_transfer_config,
        "max_model_len": 2048,
        # NOTE: Make sure HMA is enabled
        "disable_hybrid_kv_cache_manager": False,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": False,
        "block_size": block_size,
    }

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def run_hma_test(llm: LLM):
        remote_prefill_opts = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        # Simulate sidecar request
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"kv_transfer_params": remote_prefill_opts},
        )
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        kv_managers = scheduler.kv_cache_manager.coordinator.single_type_managers
        # HMA enabled with FA + SWA groups
        assert len(kv_managers) > 2
        for kv_manager in kv_managers:
            assert isinstance(kv_manager, (SlidingWindowManager, FullAttentionManager))
        req_to_blocks = kv_managers[0].req_to_blocks
        assert len(req_to_blocks) == 0

        # Process some request with length exceeding the sliding window
        outputs = llm.generate(["hi" * 1401], sampling_params)
        kv_params = outputs[0].kv_transfer_params
        print("kv_params", kv_params)

        expected_num_remote_blocks = sw_size // block_size
        remote_block_ids = kv_params["remote_block_ids"]
        assert (
            len(remote_block_ids[0])
            == expected_num_remote_blocks
            < len(remote_block_ids[-1])
        )
        for group_block_ids in remote_block_ids[:-1]:
            assert len(group_block_ids) == expected_num_remote_blocks

    def run_test_and_cleanup():
        llm = LLM(**llm_kwargs)
        try:
            run_hma_test(llm)
        finally:
            llm.llm_engine.engine_core.shutdown()

    run_test_and_cleanup()


@pytest.mark.cpu_test
def test_nixl_metadata_hma_block_ids_structure():
    """
    Test that NixlConnectorMetadata correctly stores block IDs for multiple
    KV cache groups when HMA is enabled.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnectorMetadata,
    )

    metadata = NixlConnectorMetadata()

    # Add request with block IDs for 2 groups (FA + SW)
    fa_blocks = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 blocks for FA
    sw_blocks = [8, 9, 10, 11]  # 4 blocks for SW (clipped)

    metadata.add_new_req_to_recv(
        request_id="test-req-hma",
        local_block_ids=(fa_blocks, sw_blocks),
        kv_transfer_params={
            "remote_block_ids": ([10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-test-req-hma",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )

    assert "test-req-hma" in metadata.reqs_to_recv
    req_meta = metadata.reqs_to_recv["test-req-hma"]

    # Verify local block IDs structure
    assert len(req_meta.local_block_ids) == 2
    assert list(req_meta.local_block_ids[0]) == fa_blocks
    assert list(req_meta.local_block_ids[1]) == sw_blocks

    # Verify remote block IDs structure
    assert req_meta.remote is not None
    assert len(req_meta.remote.block_ids) == 2
    assert list(req_meta.remote.block_ids[0]) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert list(req_meta.remote.block_ids[1]) == [18, 19, 20, 21]
