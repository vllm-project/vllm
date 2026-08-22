# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lifecycle import (
    KV_CONNECTOR_LIFECYCLE_LOG_PREFIX,
    KVConnectorLifecycleEvent,
    LoggingKVConnectorLifecycleSink,
    NoOpKVConnectorLifecycleSink,
    create_kv_connector_lifecycle_sink,
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


def _logging_sink(sample_rate: float = 1.0) -> LoggingKVConnectorLifecycleSink:
    return LoggingKVConnectorLifecycleSink(
        connector="NixlConnector",
        transfer_mode="pull",
        component="worker",
        sample_rate=sample_rate,
    )


def test_lifecycle_sink_is_disabled_by_default():
    sink = create_kv_connector_lifecycle_sink(
        create_vllm_config(), transfer_mode="pull", component="scheduler"
    )

    assert isinstance(sink, NoOpKVConnectorLifecycleSink)
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.lifecycle.logger.info"
    ) as log:
        sink.emit(
            KVConnectorLifecycleEvent.TRANSFER_STAGED,
            "private-request-id",
            role="consumer",
        )
    log.assert_not_called()


def test_lifecycle_log_is_structured_and_privacy_safe():
    request_id = "request-private-aaaaaaaa"
    remote_request_id = "request-private-bbbbbbbb"

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.lifecycle.logger.info"
    ) as log:
        _logging_sink().emit(
            KVConnectorLifecycleEvent.TRANSFER_STARTED,
            request_id,
            role="consumer",
            remote_request_id=remote_request_id,
            block_count=7,
        )

    log.assert_called_once()
    fmt, prefix, payload = log.call_args.args
    assert fmt == "%s%s"
    assert prefix == KV_CONNECTOR_LIFECYCLE_LOG_PREFIX
    assert request_id not in payload
    assert remote_request_id not in payload

    record = msgspec.json.decode(payload)
    assert record["schema"] == "vllm-kv-connector-v1"
    assert record["event"] == "transfer_started"
    assert record["connector"] == "NixlConnector"
    assert record["transfer_mode"] == "pull"
    assert record["component"] == "worker"
    assert record["role"] == "consumer"
    assert record["block_count"] == 7
    assert record["request_id_hash"] != record["remote_request_id_hash"]
    assert isinstance(record["wall_ns"], int)
    assert isinstance(record["monotonic_ns"], int)


def test_related_request_ids_share_sampling_and_group_hash():
    first_id = "cmpl-shared-request-aaaaaaaa"
    second_id = "cmpl-shared-request-bbbbbbbb"

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.lifecycle.logger.info"
    ) as log:
        sink = _logging_sink(sample_rate=0.5)
        sink.emit(
            KVConnectorLifecycleEvent.TRANSFER_STAGED,
            first_id,
            role="consumer",
        )
        sink.emit(
            KVConnectorLifecycleEvent.TRANSFER_STAGED,
            second_id,
            role="consumer",
        )

    # Sampling is based on the shared ID after stripping the random suffix.
    assert log.call_count in (0, 2)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.lifecycle.logger.info"
    ) as log:
        sink = _logging_sink()
        sink.emit(
            KVConnectorLifecycleEvent.TRANSFER_STAGED,
            first_id,
            role="consumer",
        )
        sink.emit(
            KVConnectorLifecycleEvent.TRANSFER_STAGED,
            second_id,
            role="consumer",
        )

    first = msgspec.json.decode(log.call_args_list[0].args[2])
    second = msgspec.json.decode(log.call_args_list[1].args[2])
    assert first["request_id_hash"] != second["request_id_hash"]
    assert first["request_group_id_hash"] == second["request_group_id_hash"]


@pytest.mark.parametrize("sample_rate", [-0.1, 1.1, float("nan")])
def test_lifecycle_sink_rejects_invalid_sample_rate(sample_rate: float):
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        _logging_sink(sample_rate=sample_rate)


def test_factory_enables_configured_sink():
    config = create_vllm_config(
        kv_connector_extra_config={
            "kv_connector_lifecycle_trace_sample_rate": 0.25,
        }
    )
    sink = create_kv_connector_lifecycle_sink(
        config, transfer_mode="pull", component="scheduler"
    )

    assert isinstance(sink, LoggingKVConnectorLifecycleSink)
