# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnectorMetadata)
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

from .utils import (assert_scheduler_empty, create_request, create_scheduler,
                    create_vllm_config)


def test_basic_interface():
    """Unit test for basic NixlConnector interface functionality."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)

    assert len(kv_connector_metadata.requests) == 1
    assert request_id in kv_connector_metadata.requests
    req_meta = kv_connector_metadata.requests[request_id]

    for block_id, block in zip(
            req_meta.local_block_ids, scheduler.kv_cache_manager.coordinator.
            single_type_managers[0].req_to_blocks[request_id]):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """
    Test that we can handle case where prompt is < block.

    In this case, the P worker will send empty remote_block_ids.
    The D worker should not schedule an async read in this case,
    since there is nothing to pull.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 0 remote blocks.
    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True,
                             num_remote_blocks=0)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request should not have to read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)
    assert len(kv_connector_metadata.requests) == 0

    # This request should be scheduled regularly.
    assert len(scheduler_output.scheduled_new_reqs) == 1


@pytest.mark.asyncio
async def test_timeout():
    model_name = "Qwen/Qwen3-0.6B"
    # Start a single P instance
    args = [
        "--enforce-eager", "--gpu_memory_utilization", "0.5",
        "--kv-transfer-config",
        "'{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}'"
    ]
    # Options for remote prefilling.
    remote_prefill_opts = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "stream": False,
        "max_tokens": 1
    }
    with RemoteOpenAIServer("Qwen/Qwen3-0.6B", args) as remote_server:
        client = remote_server.get_async_client()
        result = await client.chat.completions.create(
            model=model_name,
            extra_body={"kv_transfer_params": remote_prefill_opts},
            temperature=0.0)
        print(result)
    # Run generation - this should trigger saving KV cache
    # _ = llm.generate(["The capital of Portugal is"])
    # assert len(scheduler.pending_kv_free_req_ids) > 0


def test(monkeypatch):
    model_name = "Qwen/Qwen3-0.6B"
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
    )
    remote_prefill_opts = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        # "stream": False,
        # "max_tokens": 1
    }
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        # other sampling parameters...
        extra_args={"kv_transfer_params": remote_prefill_opts})
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    # Run generation - this should trigger saving KV cache
    out = llm.generate(["What is the capital of Japan?"], sampling_params)
    # TODO Check request was NOT freed
    import time
    time.sleep(2)
    # Some other request
    _ = llm.generate(["What is the capital of Italy?"], sampling_params)
    assert_scheduler_empty(scheduler)
    assert len(scheduler.pending_kv_free_req_ids) > 0
