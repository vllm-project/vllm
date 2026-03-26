# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
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
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "TestExampleConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["name"] = "unit"

    # Initialize the global connector instance
    ensure_kv_transfer_initialized(vllm_config)

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
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()


@pytest.mark.parametrize("has_step_work", [False, True])
def test_active_kv_connector_pre_forward_work(has_step_work: bool):
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "TestExampleConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["name"] = (
        "nonempty" if has_step_work else "empty"
    )

    ensure_kv_transfer_initialized(vllm_config)

    try:
        connector = ActiveKVConnector(vllm_config, {})
        scheduler_output = _make_empty_scheduler_output()
        if has_step_work:
            scheduler_output.kv_connector_metadata.add_request(
                token_ids=list(range(vllm_config.cache_config.block_size + 1)),
                block_ids=[0],
                block_size=vllm_config.cache_config.block_size,
                is_store=False,
                mm_hashes=[],
            )

        connector.pre_forward(scheduler_output)
        connector.post_forward(scheduler_output)

        wrapped = get_kv_transfer_group()
        expected = 1 if has_step_work else 0
        assert wrapped.call_record.get("bind_connector_metadata", 0) == expected
        assert wrapped.call_record.get("handle_preemptions", 0) == expected
        assert wrapped.call_record.get("start_load_kv", 0) == expected
        assert wrapped.call_record.get("wait_for_save", 0) == expected
        assert wrapped.call_record.get("get_finished", 0) == 1
        assert wrapped.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()
