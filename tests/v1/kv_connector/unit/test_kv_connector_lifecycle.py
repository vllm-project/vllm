# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
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


def test_kv_connector_mixin_clears_metadata():
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
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        ensure_kv_transfer_shutdown()


def _make_active_connector(vllm_config):
    mock_kv_connector = MagicMock()
    with (
        patch(
            "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
            return_value=mock_kv_connector,
        ),
        patch(
            "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
            return_value=True,
        ),
    ):
        connector = ActiveKVConnector(vllm_config, {})
    return connector, mock_kv_connector


def _called_names(mock_kv_connector):
    return [name for name, _args, _kwargs in mock_kv_connector.mock_calls]


def test_active_connector_start_loads_is_sole_load_entry():
    """pre_forward and post_forward never start loads; start_loads is the
    only entry into start_load_kv."""
    vllm_config = create_vllm_config()
    connector, mock_kv_connector = _make_active_connector(vllm_config)

    scheduler_output = _make_empty_scheduler_output()
    connector.pre_forward(scheduler_output)
    mock_kv_connector.start_load_kv.assert_not_called()

    connector.start_loads()
    assert mock_kv_connector.start_load_kv.call_count == 1

    connector.post_forward(set())
    # post_forward finalizes the step but must not start another load.
    assert mock_kv_connector.start_load_kv.call_count == 1
    mock_kv_connector.clear_connector_metadata.assert_called_once()


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_active_connector_no_forward_starts_loads_once(has_sync_kv_loads):
    """no_forward drives the full step lifecycle with exactly one load start,
    ordered before finish_forward for sync loads and after for async."""
    vllm_config = create_vllm_config()
    connector, mock_kv_connector = _make_active_connector(vllm_config)

    scheduler_output = _make_empty_scheduler_output()
    scheduler_output.has_sync_kv_loads = has_sync_kv_loads
    connector.no_forward(scheduler_output)

    names = _called_names(mock_kv_connector)
    assert names.count("start_load_kv") == 1
    if has_sync_kv_loads:
        assert names.index("start_load_kv") < names.index("finish_forward")
    else:
        assert names.index("finish_forward") < names.index("start_load_kv")
    assert "clear_connector_metadata" in names


def test_active_connector_start_loads_respects_disabled():
    vllm_config = create_vllm_config()
    with patch("vllm.v1.worker.gpu.kv_connector.kv_transfer_state"):
        connector, mock_kv_connector = _make_active_connector(vllm_config)
        connector.set_disabled(True)
        connector.start_loads()
        mock_kv_connector.start_load_kv.assert_not_called()
