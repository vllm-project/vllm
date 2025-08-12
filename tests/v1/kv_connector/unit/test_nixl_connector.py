# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import textwrap
import time
from unittest.mock import patch

import pytest
import ray

from vllm import LLM
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    KVConnectorRole, NixlAgentMetadata, NixlConnector, NixlConnectorMetadata,
    NixlConnectorWorker)
from vllm.forward_context import ForwardContext
from vllm.mocks.mock_nixl_connector import FakeNixlWrapper
from vllm.sampling_params import SamplingParams

from .utils import create_request, create_scheduler, create_vllm_config


def _make_stub_pkg() -> str:
    """Return a directory that makes
       `from nixl._api import nixl_agent` resolve to our FakeNixlWrapper."""
    td = tempfile.mkdtemp()
    pkg_root = os.path.join(td, "nixl", "_api")
    os.makedirs(pkg_root, exist_ok=True)

    stub = textwrap.dedent("""\
        # Forward the real FakeNixlWrapper that the driver already defined.
        print("In fake package")
        from vllm.mocks.mock_nixl_connector import FakeNixlWrapper as nixl_agent
    """)
    with open(os.path.join(pkg_root, "__init__.py"), "w") as f:
        f.write(stub)

    # touch parent package
    open(os.path.join(td, "nixl", "__init__.py"), "w").close()
    return td


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

    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv
    req_meta = kv_connector_metadata.reqs_to_recv[request_id]

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
    assert len(kv_connector_metadata.reqs_to_recv) == 0

    # This request should be scheduled regularly.
    assert len(scheduler_output.scheduled_new_reqs) == 1


class FakeNixlConnectorWorker(NixlConnectorWorker):

    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(self, *args, hand_shake_latency: float = 1.8, **kwargs):
        super().__init__(*args, **kwargs)
        self._hand_shake_latency = hand_shake_latency

    def _nixl_handshake(self, host: str, port: int, remote_tp_size: int,
                        expected_engine_id: str) -> dict[int, str]:
        # Mimic slow _nixl_handshake, as well as bypass zmq communication.
        time.sleep(self._hand_shake_latency)
        # These should've been done in register_kv_caches(), called by
        # gpu_model_runner. Here we just hardcode some dummy values.
        self.slot_size_bytes = 4096
        self.block_len = self.slot_size_bytes * self.block_size
        self.num_blocks = 1
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        assert expected_engine_id == self.REMOTE_ENGINE_ID

        remote_agent_name = self.add_remote_agent(
            NixlAgentMetadata(
                engine_id=self.REMOTE_ENGINE_ID,
                agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                num_blocks=1,
                block_len=self.block_len,
                attn_backend_name=self.backend_name,
            ),
            remote_tp_size=remote_tp_size)
        return {0: remote_agent_name}


class TestNixlHandshake:

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    def test_multi_xfer_one_engine(
        self,
        # dist_init is a fixture that initializes the distributed environment.
        dist_init):
        """Test case where multiple xfers are initiated to the same engine.
        
        This test triggers the connector to load remote KV for the same
        `request_id`. The transfer is not done immediately due to
        `set_cycles_before_xfer_done`, so there is a state where there are
        multiple transfer states for the same `request_id`, and `get_finished`
        should handle it correctly (wait for all transfers to be done).
        """
        vllm_config = create_vllm_config()

        request_id = "req_id"

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id, hand_shake_latency=0)
        assert isinstance(connector.connector_worker.nixl_wrapper,
                          FakeNixlWrapper)
        connector.connector_worker.nixl_wrapper.set_cycles_before_xfer_done(3)
        num_xfers = 4
        while True:
            # For the same request_id, initiate multiple xfers across different
            # round of `execute_model` calls.
            metadata = NixlConnectorMetadata()
            if num_xfers > 0:
                num_xfers -= 1
                metadata.add_new_req(
                    request_id=request_id,
                    local_block_ids=[
                        num_xfers + 1, num_xfers + 2, num_xfers + 3
                    ],
                    kv_transfer_params={
                        "remote_block_ids":
                        [num_xfers + 4, num_xfers + 5, num_xfers + 6],
                        "remote_engine_id":
                        FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                        "remote_host":
                        "localhost",
                        "remote_port":
                        1234,
                        "remote_tp_size":
                        1,
                    })
            connector.bind_connector_metadata(metadata)

            # Mimic maybe_setup_kv_connector in gpu_model_runner.
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, "start_load_kv took " \
                f"{_after_load - _before_load} seconds"

            # Mimic get_finished_kv_transfers in gpu_model_runner.
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                assert request_id in done_recving
                break

            connector.clear_connector_metadata()

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    @pytest.mark.parametrize("decode_tp_size, prefill_tp_size", [
        (1, 1),
        (2, 1),
        (4, 2),
        (4, 4),
    ])
    def test_async_load_kv(
            self,
            # Fixture that initializes the distributed environment.
            dist_init,
            # Simulate consumer-producer TP sizes.
            decode_tp_size,
            prefill_tp_size):
        """Test that NixlConnector's start_load_kv should be non-blocking."""

        vllm_config = create_vllm_config()
        vllm_config.parallel_config.tensor_parallel_size = decode_tp_size

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id)
        metadata = NixlConnectorMetadata()
        metadata.add_new_req(request_id="id",
                             local_block_ids=[1, 2, 3],
                             kv_transfer_params={
                                 "remote_block_ids": [4, 5, 6],
                                 "remote_engine_id":
                                 FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                                 "remote_host": "localhost",
                                 "remote_port": 1234,
                                 "remote_tp_size": prefill_tp_size,
                             })
        connector.bind_connector_metadata(metadata)

        timeout = 2.5
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, "start_load_kv took " \
                f"{_after_load - _before_load} seconds"
            time.sleep(0.5)  # backoff for the async handshake to complete.
            connector.bind_connector_metadata(NixlConnectorMetadata())
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                return
        raise TimeoutError("Took too long to complete async handshake.")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper)
    def test_concurrent_load_kv(
        self,
        # dist_init is a fixture that initializes the distributed environment.
        dist_init):
        """Test that multiple start_load_kv calls should occur concurrently."""

        vllm_config = create_vllm_config()

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id)
        metadata = NixlConnectorMetadata()
        total_reqs = 5
        for i in range(total_reqs):
            metadata.add_new_req(request_id=f"id_{i}",
                                 local_block_ids=[1, 2, 3],
                                 kv_transfer_params={
                                     "remote_block_ids": [4, 5, 6],
                                     "remote_engine_id":
                                     FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                                     "remote_host": "localhost",
                                     "remote_port": 1234,
                                     "remote_tp_size": 1,
                                 })
        connector.bind_connector_metadata(metadata)

        timeout = 2.5 * total_reqs
        cnt_finished_reqs = 0
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, "start_load_kv took " \
                f"{_after_load - _before_load} seconds"
            time.sleep(0.5)  # backoff for the async handshake to complete.
            connector.bind_connector_metadata(NixlConnectorMetadata())
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                cnt_finished_reqs += len(done_recving)
                if cnt_finished_reqs == total_reqs:
                    return
        raise TimeoutError("Took too long to complete async handshake.")


# NOTE: resource cleanup in mp backend is a bit finicky, so the order in which
# we put here is important. First run ray, it will clean up the resources, then
# the rest of the tests.
@pytest.mark.parametrize("distributed_executor_backend", ["ray", None])
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper)
def test_abort_timeout_on_prefiller(monkeypatch, distributed_executor_backend):
    """
    Test lifecycle of an aborted Remote Prefill request hitting the timeout.
    -----> P 
            |  {process request}
     <-/--- |  {result is NOT delivered, eg proxy is down}
            |
            |
            |  {eventually free blocks}
    """
    model_name = "Qwen/Qwen3-0.6B"
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    timeout = 6
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", str(timeout))

    # Build runtime_env only if we’re using Ray
    if distributed_executor_backend == "ray":
        runtime_env = {
            "working_dir": _make_stub_pkg(),  # ship stub package
            "env_vars": {
                "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": str(timeout),
            },
        }
        ray.init(runtime_env=runtime_env)

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
        distributed_executor_backend=distributed_executor_backend,
    )
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
        extra_args={"kv_transfer_params": remote_prefill_opts})
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0].req_to_blocks

    padding = "Just making this request a little longer so that we're sure "
    "we're not hitting the small-request lower bound beneath which we don't "
    "actually trigger the whole kv transfer, but rather just recompute the "
    "blocks on D."
    _ = llm.generate([f"What is the capital of Japan? {padding}"],
                     sampling_params)

    # Request finished but not freed
    assert '0' in scheduler.finished_req_ids and '0' in req_to_blocks
    # Some other request, 0 still not freed
    _ = llm.generate([f"What is the capital of Italy? {padding}"],
                     sampling_params)
    assert '0' in req_to_blocks
    assert '1' in scheduler.finished_req_ids and '1' in req_to_blocks

    # Wait for timeout and trigger another scheduler loop
    time.sleep(timeout)
    _ = llm.generate([f"What is the capital of France? {padding}"],
                     sampling_params)
    # Request-0 times out and is cleared!
    assert '0' not in req_to_blocks
