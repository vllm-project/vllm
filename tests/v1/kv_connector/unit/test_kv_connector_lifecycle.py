# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm import device_allocator
from vllm.device_allocator import (
    AllocationData,
    run_after_map_hooks,
    run_before_unmap_hooks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnector,
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# Importing utils registers TestExampleConnector with the factory
from .utils import create_vllm_config


def _make_empty_scheduler_output():
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=ExampleConnectorMetadata(),
    )


def _init_worker_connector():
    vllm_config = create_vllm_config(
        kv_connector="TestExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"name": "unit"},
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    # Initialize the global connector instance.
    # kv_transfer init now syncs engine_id across TP, so unit tests need
    # a minimal mocked TP group.
    mock_tp_group = MagicMock()
    mock_tp_group.broadcast_object.side_effect = lambda value, src=0: value

    with patch(
        "vllm.distributed.parallel_state.get_tp_group",
        return_value=mock_tp_group,
    ):
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    return vllm_config


def test_kv_connector_mixin_clears_metadata():
    vllm_config = _init_worker_connector()
    try:
        # Minimal scheduler output with empty metadata; mixin should still
        # bind/clear metadata even if no loads happen
        scheduler_output = _make_empty_scheduler_output()

        # Invoke the no-forward path which uses the mixin context manager
        KVConnectorModelRunnerMixin.kv_connector_no_forward(
            scheduler_output, vllm_config
        )

        # Verify clear_connector_metadata was called on the connector
        connector = get_kv_transfer_group()
        assert connector._connector_metadata is None
        # Test connector wrapper records method calls
        assert connector.call_record.get("bind_connector_metadata", 0) == 1
        assert connector.call_record.get("wait_for_save", 0) == 1
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        ensure_kv_transfer_shutdown()


@pytest.mark.parametrize("supported", [False, True], ids=["unsupported", "supported"])
def test_worker_connector_follows_the_kv_cache_mapping_iff_it_supports_sleep_mode(
    monkeypatch, supported
):
    """Only a connector that supports sleep mode is released before the KV
    cache is unmapped and restored after it is mapped again."""
    monkeypatch.setattr(device_allocator, "_tag_hooks", {})
    monkeypatch.setattr(
        ExampleConnector, "supports_sleep_mode", classmethod(lambda cls, c: supported)
    )
    monkeypatch.setattr(ExampleConnector, "release_kv_caches", lambda self: None)
    monkeypatch.setattr(ExampleConnector, "restore_kv_caches", lambda self: None)
    kv_cache = AllocationData(handle=(0, 0, 0, 0), tag="kv_cache")

    def unmap_and_map() -> None:
        run_before_unmap_hooks([kv_cache])
        kv_cache.is_asleep = True
        run_after_map_hooks([kv_cache])  # still unmapped
        kv_cache.is_asleep = False
        run_after_map_hooks([kv_cache])

    _init_worker_connector()
    connector = get_kv_transfer_group()
    try:
        unmap_and_map()
    finally:
        ensure_kv_transfer_shutdown()
    unmap_and_map()  # a shut down connector is not called

    calls = connector.call_record
    assert (calls["release_kv_caches"], calls["restore_kv_caches"]) == (
        (1, 1) if supported else (0, 0)
    )
