# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
import os
import tempfile
import textwrap
import time
import uuid
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import ray
import torch

from vllm import LLM
from vllm.config import KVTransferConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import (
    KVOutputAggregator,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import nixl_connector
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    KVConnectorRole,
    NixlAgentMetadata,
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorScheduler,
    NixlConnectorWorker,
    NixlHandshakePayload,
    NixlKVConnectorStats,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_shutdown,
    has_kv_transfer_group,
)
from vllm.forward_context import ForwardContext
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheTensor
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.utils import AttentionGroup

from .utils import create_request, create_scheduler, create_vllm_config


@pytest.fixture(scope="module", autouse=True)
def clear_kv_transfer():
    """
    The test cases in this file use `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
    causing the global variable `_KV_CONNECTOR_AGENT`
    to be assigned but never deleted.

    Since the current pytest process does not terminate and instead
    continues running tests from other files,
    this global variable remains in memory and interferes
    with test cases in other modules.

    So we use this fixture to ensure that the global variable
    `_KV_CONNECTOR_AGENT` is properly cleaned up after each test.
    """
    yield
    if has_kv_transfer_group():
        ensure_kv_transfer_shutdown()


def get_default_xfer_telemetry(
    xferDurationS: float = 1,
    postDurationS: float = 1,
    totalBytes: int = 1,
    descCount: int = 1,
) -> dict:
    class AttributeDict(dict):
        __slots__ = ()
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__  # type: ignore[assignment]

    # We can't instantiate nixlXferTelemetry because it's read only and
    # ray env does not have NIXL, so we must fake it
    return AttributeDict(
        xferDuration=xferDurationS * 1e6,  # in us
        postDuration=postDurationS * 1e6,  # in us
        totalBytes=totalBytes,
        descCount=descCount,
    )


class FakeNixlWrapper:
    """Mock implementation of NixlWrapper for testing.

    We don't inherit from nixl._api.nixl_agent because nixl may not be
    installed.

    Note: The complete source of this class is also used in the
    `_make_fake_nixl_pkg` function to create a fake nixl package
    for Ray workers.
    """

    AGENT_METADATA = b"fake_agent_metadata"
    REMOTE_AGENT_NAME = "remote_agent"

    def __init__(self, agent_name: str, *args, **kwargs):
        self._cycles_before_xfer_done = 0
        self._check_xfer_state_cycles: defaultdict[int, int] = defaultdict(lambda: 0)

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in caches_data]

    def register_memory(self, descs, backends) -> None:
        pass

    def deregister_memory(self, descs) -> None:
        pass

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in blocks_data]

    def prep_xfer_dlist(self, agent_name: str, descs: list) -> int:
        return uuid.uuid4().int

    def get_agent_metadata(self) -> bytes:
        return self.AGENT_METADATA

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return self.REMOTE_AGENT_NAME

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        # Used to collect done_sending, which we don't test yet.
        return {}

    def check_xfer_state(self, handle: int) -> str:
        if self._check_xfer_state_cycles[handle] >= self._cycles_before_xfer_done:
            return "DONE"
        self._check_xfer_state_cycles[handle] += 1
        return "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def release_dlist_handle(self, handle: int) -> None:
        pass

    def remove_remote_agent(self, agent: str) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def make_prepped_xfer(
        self,
        xfer_type: str,
        local_xfer_side_handle: int,
        local_block_descs_ids: list[int],
        remote_xfer_side_handle: int,
        remote_block_descs_ids: list[int],
        notif_msg: bytes | None = None,
    ) -> int:
        return uuid.uuid4().int

    def transfer(self, handle: int) -> str:
        return "PROC"

    def get_xfer_telemetry(self, handle: int) -> dict:
        return get_default_xfer_telemetry()

    ############################################################
    # Follow are for changing the behavior during testing.
    ############################################################

    def set_cycles_before_xfer_done(self, cycles: int):
        """Set the number of cycles before a transfer is considered done."""


@contextlib.contextmanager
def _make_fake_nixl_pkg():
    """Context manager that creates a temporary package making
       `from nixl._api import nixl_agent` resolve to our FakeNixlWrapper.
       Also creates rixl package for ROCm compatibility.

    Automatically cleans up the temporary directory when done.
    """
    with tempfile.TemporaryDirectory() as td:
        # Create both nixl and rixl packages for cross-platform compatibility
        for pkg_name in ["nixl", "rixl"]:
            pkg_root = os.path.join(td, pkg_name, "_api")
            os.makedirs(pkg_root, exist_ok=True)

            # Get the source code of FakeNixlWrapper class and dedent it
            fake_nixl_source = inspect.getsource(FakeNixlWrapper)
            fake_nixl_source = textwrap.dedent(fake_nixl_source)

            stub = f"""\
# Copy of FakeNixlWrapper implementation for Ray workers
import uuid
from collections import defaultdict

{fake_nixl_source}

# Export as nixl_agent
nixl_agent = FakeNixlWrapper
"""
            with open(os.path.join(pkg_root, "__init__.py"), "w") as f:
                f.write(stub)

            # Mock nixlXferTelemetry class
            pkg_root2 = os.path.join(td, pkg_name, "_bindings")
            os.makedirs(pkg_root2, exist_ok=True)
            with open(os.path.join(pkg_root2, "__init__.py"), "w") as f:
                f.write("class nixlXferTelemetry: pass")
            # touch parent package
            open(os.path.join(td, pkg_name, "__init__.py"), "w").close()

        yield td


def test_basic_interface():
    """Unit test for basic NixlConnector interface functionality."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )
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
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """
    Test that we can handle case where prompt is < block.

    In this case, the P worker will still send remote_block_ids of the
    partial block. The D worker should schedule an async read
    in this case.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_kv_transfer_handshake(dist_init):
    """Unit test for basic NixlConnector interface functionality."""
    from vllm.config import set_current_vllm_config

    # Test setup, we creates a scheduler that contains a NixlConnector
    # of role SCHEDULER, and expect it to be serving NixlAgentMetadata from
    # all workers of the instance.
    vllm_config = create_vllm_config()
    # in case the test runs on non-GPU machine
    vllm_config.kv_transfer_config.kv_buffer_device = "cpu"
    scheduler = create_scheduler(vllm_config)

    with set_current_vllm_config(vllm_config):
        # Create two NixlConnector of role WORKER, one is the worker of
        # the scheduler (prefill), the other is a worker of decode instance.

        # Prefill connector will register KV cache to populate proper handshake
        # metadata.
        prefill_connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
        )
        shared_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        unique_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        kv_caches = {
            "layer0": shared_tensor,
            "layer1": unique_tensor,
            "layer2": shared_tensor,
        }
        prefill_connector.register_kv_caches(kv_caches)

        # Simulate EngineCore initialization that would gather connector
        # metadata from all workers
        metadata = prefill_connector.get_handshake_metadata()

        # metadata is a NixlHandshakePayload, decode it to get NixlAgentMetadata
        decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
        expected_agent_metadata = decoder.decode(metadata.agent_metadata_bytes)

        # The scheduler connector expects metadata to be in
        # dict[int, KVConnectorHandshakeMetadata], where the first key is
        # the dp_rank, the second key is the tp_rank.
        scheduler_connector = scheduler.get_kv_connector()
        scheduler_connector.set_xfer_handshake_metadata({0: metadata})

        # Simulate a request that finishes prefill, which returns
        # corresponding NixlConnectorMetadata for decode instance.
        BLOCK_SIZE = vllm_config.cache_config.block_size
        NUM_EXTERNAL_FULL_BLOCKS = 2
        NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

        request = create_request(
            request_id=1,
            block_size=BLOCK_SIZE,
            num_tokens=NUM_TOKENS,
            do_remote_decode=True,
        )
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        delay, kv_connector_metadata = scheduler.get_kv_connector().request_finished(
            request, [0, 1, 2]
        )
        assert delay

        # Decode connector will be able to create handshake with the prefill connector.
        decode_connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        decode_connector.register_kv_caches(kv_caches)

        # Here we are testing the retrieval of NIXLAgentMetadata.
        # Knowing the implementation detail, we override the add_remote_agent
        # to validate the metadata received is the same as the one in prefill_connector.
        with patch.object(
            decode_connector.connector_worker, "add_remote_agent"
        ) as mock_add_remote_agent:
            mock_add_remote_agent.return_type = "remote_agent"

            decode_connector.connector_worker._nixl_handshake(
                kv_connector_metadata["remote_host"],
                kv_connector_metadata["remote_port"],
                kv_connector_metadata["tp_size"],
                kv_connector_metadata["remote_engine_id"],
            )

            received_metadata = mock_add_remote_agent.call_args.args
            assert received_metadata[0] == expected_agent_metadata
            assert received_metadata[1] == 0  # remote_tp_rank
            assert received_metadata[2] == 1  # remote_tp_size

        # Need to shutdown the background thread to release NIXL side channel port
        scheduler_connector.shutdown()


class FakeNixlConnectorWorker(NixlConnectorWorker):
    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(
        self, *args, hand_shake_latency: float = 1.8, kv_cache_layout="HND", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._hand_shake_latency = hand_shake_latency
        self.kv_cache_layout = kv_cache_layout
        # Mock register_kv_caches attribute needed for tests that do not call it.
        self.src_xfer_handles_by_block_size = {self.block_size: 1}
        test_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=self.attn_backend,
            tensor_shape=test_shape,
        )

        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.kv_topo.cross_layers_blocks
        )

    def _nixl_handshake(
        self, host: str, port: int, remote_tp_size: int, expected_engine_id: str
    ) -> dict[int, str]:
        # Mimic slow _nixl_handshake, as well as bypass zmq communication.
        time.sleep(self._hand_shake_latency)
        # These should've been done in register_kv_caches(), called by
        # gpu_model_runner. Here we just hardcode some dummy values.
        slot_size_bytes = 4096
        self.slot_size_per_layer = [slot_size_bytes]
        self.block_len_per_layer = [slot_size_bytes * self.block_size]
        self.num_blocks = 1
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        assert expected_engine_id == self.REMOTE_ENGINE_ID

        # Adjust remote block length metadata to satisfy heterogeneous TP
        # invariants enforced during handshake validation.
        remote_block_lens = list(self.block_len_per_layer)
        tp_ratio = self.kv_topo.tp_ratio(remote_tp_size)
        if remote_tp_size > self.world_size:
            # P TP > D TP case, block_len of remote is smaller
            remote_block_lens = [
                block_len // (-tp_ratio) for block_len in remote_block_lens
            ]
        elif remote_tp_size < self.world_size:
            remote_block_lens = [
                block_len * tp_ratio for block_len in remote_block_lens
            ]

        # When remote tp_size > local tp_size, handshake with multiple
        # remote ranks.
        num_hanshakes = 1 if tp_ratio > 0 else -tp_ratio
        remote_agents: dict[int, str] = {}
        for remote_tp_rank in range(num_hanshakes):
            remote_agent_name = self.add_remote_agent(
                NixlAgentMetadata(
                    engine_id=self.REMOTE_ENGINE_ID,
                    agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                    kv_caches_base_addr=[0],
                    device_id=remote_tp_rank,
                    num_blocks=1,
                    block_lens=remote_block_lens,
                    # `self.kv_cache_layout` is only forced to HND when vllm engine
                    # is started. We mock HND here.
                    kv_cache_layout="HND",
                    block_size=self.block_size,
                ),
                remote_tp_rank=remote_tp_rank,
                remote_tp_size=remote_tp_size,
            )
            remote_agents[remote_tp_rank] = remote_agent_name
        return remote_agents


class TestNixlHandshake:
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    def test_multi_xfer_one_engine(
        self,
        default_vllm_config,
        # dist_init is a fixture that initializes the distributed environment.
        dist_init,
    ):
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
            vllm_config, connector.engine_id, hand_shake_latency=0
        )
        assert isinstance(connector.connector_worker.nixl_wrapper, FakeNixlWrapper)
        worker = connector.connector_worker
        worker.nixl_wrapper.set_cycles_before_xfer_done(3)
        # simulate handshake
        worker.dst_xfer_side_handles = {
            FakeNixlConnectorWorker.REMOTE_ENGINE_ID: {0: 1}
        }
        worker.kv_cache_layout = "HND"
        num_xfers = 4
        while True:
            # For the same request_id, initiate multiple xfers across different
            # round of `execute_model` calls.
            metadata = NixlConnectorMetadata()
            if num_xfers > 0:
                num_xfers -= 1
                metadata.add_new_req_to_recv(
                    request_id=request_id,
                    local_block_ids=[num_xfers + 1, num_xfers + 2, num_xfers + 3],
                    kv_transfer_params={
                        "remote_block_ids": [
                            num_xfers + 4,
                            num_xfers + 5,
                            num_xfers + 6,
                        ],
                        "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                        "remote_request_id": f"prefill-{request_id}",
                        "remote_host": "localhost",
                        "remote_port": 1234,
                        "remote_tp_size": 1,
                    },
                )
            connector.bind_connector_metadata(metadata)

            # Mimic logic in KVConnectorModelRunnerMixin._get_kv_connector_output.
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
                slot_mapping={},
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, (
                f"start_load_kv took {_after_load - _before_load} seconds"
            )

            # Mimic logic in KVConnectorModelRunnerMixin._get_kv_connector_output.
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                assert request_id in done_recving
                break

            connector.clear_connector_metadata()

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    @pytest.mark.parametrize(
        "decode_tp_size, prefill_tp_size",
        [
            (1, 1),
            (2, 1),
            (4, 2),
            (4, 4),
        ],
    )
    def test_async_load_kv(
        self,
        default_vllm_config,
        # Fixture that initializes the distributed environment.
        dist_init,
        # Simulate consumer-producer TP sizes.
        decode_tp_size,
        prefill_tp_size,
    ):
        """Test that NixlConnector's start_load_kv should be non-blocking."""

        vllm_config = create_vllm_config()
        vllm_config.parallel_config.tensor_parallel_size = decode_tp_size

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id
        )
        metadata = NixlConnectorMetadata()
        metadata.add_new_req_to_recv(
            request_id="id",
            local_block_ids=[1, 2, 3],
            kv_transfer_params={
                "remote_block_ids": [4, 5, 6],
                "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                "remote_request_id": "prefill-id",
                "remote_host": "localhost",
                "remote_port": 1234,
                "remote_tp_size": prefill_tp_size,
            },
        )
        connector.bind_connector_metadata(metadata)

        timeout = 2.5
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
                slot_mapping={},
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, (
                f"start_load_kv took {_after_load - _before_load} seconds"
            )
            time.sleep(0.5)  # backoff for the async handshake to complete.
            connector.bind_connector_metadata(NixlConnectorMetadata())
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                return
        raise TimeoutError("Took too long to complete async handshake.")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    @pytest.mark.parametrize("local_tp_size", [1, 2])
    def test_prefill_tp_size_greater_than_decode_tp_size(
        self, local_tp_size: int, default_vllm_config, dist_init
    ):
        """
        Verify remote TP > local TP handshake succeeds with different
        remote configurations.
        """

        vllm_config = create_vllm_config()
        local_tp_size = 1
        vllm_config.parallel_config.tensor_parallel_size = local_tp_size

        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id, hand_shake_latency=0
        )
        worker = connector.connector_worker

        # Minimal local registration params used by add_remote_agent
        worker.slot_size_per_layer = [4096]
        worker.block_len_per_layer = [4096 * worker.block_size]
        worker.num_blocks = 1
        worker.dst_num_blocks[worker.engine_id] = worker.num_blocks
        worker.src_blocks_data = [(0, worker.block_len_per_layer[0], worker.tp_rank)]

        def check_handshake(remote_tp_size: int):
            tp_ratio = remote_tp_size // local_tp_size
            assert set(remote_agents.keys()) == set(range(tp_ratio))

            remote_engine_id = worker.REMOTE_ENGINE_ID
            assert worker._tp_size[remote_engine_id] == remote_tp_size
            assert -tp_ratio == worker.kv_topo.tp_ratio_from_engine_id(remote_engine_id)
            # ensure src_xfer_handles_by_tp_ratio is populated with tpratio chunks
            assert -tp_ratio in worker.src_xfer_handles_by_tp_ratio
            assert len(worker.src_xfer_handles_by_tp_ratio[-tp_ratio]) == tp_ratio
            assert remote_engine_id in worker.dst_xfer_side_handles
            assert set(worker.dst_xfer_side_handles[remote_engine_id].keys()) == set(
                range(tp_ratio)
            )

        remote_agents = worker._nixl_handshake(
            host="localhost",
            port=1234,
            remote_tp_size=2,
            expected_engine_id=worker.REMOTE_ENGINE_ID,
        )
        check_handshake(2)

        # NOTE flexiblity: a second remote with higher number of ranks is
        # discovered. This is not a scenario we actively support right now, but
        # the connector allows it.
        worker.REMOTE_ENGINE_ID = "remote_engine_2"
        remote_agents = worker._nixl_handshake(
            host="localhost",
            port=1234,
            remote_tp_size=6,
            expected_engine_id=worker.REMOTE_ENGINE_ID,
        )
        check_handshake(6)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    @pytest.mark.parametrize("local_tp_size", [1, 2])
    def test_prefill_tp_size_greater_than_decode_tp_size_mla(
        self, local_tp_size: int, default_vllm_config, dist_init
    ):
        """
        Verify remote TP > local TP handshake succeeds with different
        remote configurations for an MLA model.
        """
        vllm_config = create_vllm_config()
        d_tp_size = 1
        p_tp_size = 2

        # Build two separate connectors/workers to emulate P TP=2 ranks.
        conn_p0 = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        conn_p1 = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        conn_p0.connector_worker = FakeNixlConnectorWorker(
            vllm_config, conn_p0.engine_id, hand_shake_latency=0
        )
        conn_p1.connector_worker = FakeNixlConnectorWorker(
            vllm_config, conn_p1.engine_id, hand_shake_latency=0
        )

        # Force P world size to 2 for both workers and emulate distinct tp_ranks.
        # Also enable MLA path so that expected_finished_count is updated.
        for rank, worker in enumerate(
            (conn_p0.connector_worker, conn_p1.connector_worker)
        ):
            worker.world_size = p_tp_size
            worker.kv_topo.remote_tp_size = {worker.engine_id: p_tp_size}
            worker.tp_rank = rank
            worker.use_mla = True

        req_id = "req-ep-dp2-p0"
        now = time.perf_counter()
        # Register a request on P that is waiting for consumers to read
        # (both workers track it).
        conn_p0.connector_worker._reqs_to_send[req_id] = now + 10.0
        conn_p0.connector_worker._reqs_to_process.add(req_id)
        conn_p1.connector_worker._reqs_to_send[req_id] = now + 10.0
        conn_p1.connector_worker._reqs_to_process.add(req_id)

        # Simulate a read notification coming from D with (tp=1, dp=2).
        notif = f"{req_id}:{d_tp_size}".encode()
        # D0-0->P0 notif
        conn_p0.connector_worker.nixl_wrapper.get_new_notifs = lambda: {
            "agent": [notif]
        }  # type: ignore[method-assign]
        conn_p1.connector_worker.nixl_wrapper.get_new_notifs = lambda: {
            "agent": [notif]
        }  # type: ignore[method-assign]

        # Trigger notification processing via get_finished().
        done_sending0, _ = conn_p0.get_finished(finished_req_ids=set())
        done_sending1, _ = conn_p1.get_finished(finished_req_ids=set())
        assert req_id in done_sending0 and req_id in done_sending1

        # E2E aggregation: ensure the aggregated output marks the request
        # as finished using the connector's expected_finished_count.
        from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

        aggregator = KVOutputAggregator.from_connector(conn_p0, world_size=2)

        out0 = ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(
                finished_sending=done_sending0,
                finished_recving=None,
            ),
        )
        out1 = ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(
                finished_sending=done_sending1,
                finished_recving=None,
            ),
        )
        aggregated = aggregator.aggregate([out0, out1], output_rank=0)
        assert aggregated.kv_connector_output is not None
        assert aggregated.kv_connector_output.finished_sending == {req_id}

        # Producers cleaned up state for the finished request.
        assert req_id not in conn_p0.connector_worker._reqs_to_send
        assert req_id not in conn_p0.connector_worker._reqs_to_process
        assert req_id not in conn_p1.connector_worker._reqs_to_send
        assert req_id not in conn_p1.connector_worker._reqs_to_process

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    def test_concurrent_load_kv(
        self,
        default_vllm_config,
        # dist_init is a fixture that initializes the distributed environment.
        dist_init,
    ):
        """Test that multiple start_load_kv calls should occur concurrently."""

        vllm_config = create_vllm_config()

        # Test worker role in decode server.
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id
        )
        # Register (mocked) local xfer handler
        # worker = connector.connector_worker
        # worker.src_xfer_handles_by_block_size = {worker.block_size: 1}
        metadata = NixlConnectorMetadata()
        total_reqs = 5
        for i in range(total_reqs):
            metadata.add_new_req_to_recv(
                request_id=f"id_{i}",
                local_block_ids=[1, 2, 3],
                kv_transfer_params={
                    "remote_block_ids": [4, 5, 6],
                    "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                    "remote_request_id": f"prefill-id-{i}",
                    "remote_host": "localhost",
                    "remote_port": 1234,
                    "remote_tp_size": 1,
                },
            )
        connector.bind_connector_metadata(metadata)

        timeout = 2.5 * total_reqs
        cnt_finished_reqs = 0
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            dummy_ctx = ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                virtual_engine=0,
                slot_mapping={},
            )
            _before_load = time.perf_counter()
            connector.start_load_kv(dummy_ctx)
            _after_load = time.perf_counter()
            assert _after_load - _before_load < 0.1, (
                f"start_load_kv took {_after_load - _before_load} seconds"
            )
            time.sleep(0.5)  # backoff for the async handshake to complete.
            connector.bind_connector_metadata(NixlConnectorMetadata())
            _, done_recving = connector.get_finished(finished_req_ids=set())
            if len(done_recving) > 0:
                cnt_finished_reqs += len(done_recving)
                if cnt_finished_reqs == total_reqs:
                    return
        raise TimeoutError("Took too long to complete async handshake.")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    def test_handshake_fails_on_kv_cache_layout_mismatch(
        self, default_vllm_config, dist_init
    ):
        """
        Verify that adding a remote agent fails if kv_cache_layout differs.
        This test is only relevant for heterogeneous TP.
        """
        vllm_config = create_vllm_config()

        # Mock TP world size to 2 to force heterogeneous TP when
        # remote_tp_size=1
        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.get_tensor_model_parallel_world_size",  # noqa: E501
            return_value=2,
        ):
            # Initialize connector and worker (with fake NIXL wrapper)
            connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
            connector.connector_worker = FakeNixlConnectorWorker(
                vllm_config, connector.engine_id, hand_shake_latency=0
            )
            worker = connector.connector_worker

            # Minimal local registration params used by add_remote_agent
            worker.slot_size_per_layer = [4096]
            worker.block_len_per_layer = [4096 * worker.block_size]
            worker.num_blocks = 1
            worker.dst_num_blocks[worker.engine_id] = worker.num_blocks

            # Metadata with different kv_cache_layout than local worker
            mismatched_layout = "HND" if worker.kv_cache_layout != "HND" else "NHD"
            meta = NixlAgentMetadata(
                engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                device_id=0,
                num_blocks=1,
                block_lens=worker.block_len_per_layer,
                kv_cache_layout=mismatched_layout,
                block_size=worker.block_size,
            )

            with pytest.raises(RuntimeError):
                # mismatched layout is expected to fail
                worker.add_remote_agent(meta, remote_tp_size=2)
                worker.add_remote_agent(meta, remote_tp_size=1)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
        FakeNixlWrapper,
    )
    def test_handshake_succeed_on_kv_cache_layout_mismatch_with_experimental(
        self, default_vllm_config, dist_init
    ):
        """
        Verify that adding a remote agent fails if kv_cache_layout differs.
        This test is only relevant for heterogeneous TP.
        """
        vllm_config = create_vllm_config(enable_permute_local_kv=True)

        # Mock TP world size to 2 to force heterogeneous TP when
        # remote_tp_size=1
        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.get_tensor_model_parallel_world_size",  # noqa: E501
            return_value=2,
        ):
            # Initialize connector and worker (with fake NIXL wrapper)
            connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
            connector.connector_worker = FakeNixlConnectorWorker(
                vllm_config,
                connector.engine_id,
                hand_shake_latency=0,
                kv_cache_layout="NHD",
            )
            worker = connector.connector_worker

            # Minimal local registration params used by add_remote_agent
            worker.slot_size_per_layer = [2048]
            worker.block_len_per_layer = [2048 * worker.block_size]
            worker.num_blocks = 1
            worker.dst_num_blocks[worker.engine_id] = worker.num_blocks

            # Metadata with different kv_cache_layout than local worker
            meta = NixlAgentMetadata(
                engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                device_id=0,
                num_blocks=1,
                # prefill TP=1, decode TP=2, remote block_lens is double to local
                block_lens=[i * 2 for i in worker.block_len_per_layer],
                kv_cache_layout="HND",
                block_size=worker.block_size,
            )

            # We don't check layout for homogeneous TP and MLA for now, as the
            # whole block is moved.
            worker.add_remote_agent(meta, remote_tp_size=1)


# NOTE: resource cleanup in mp backend is a bit finicky, so the order in which
# we put here is important. First run ray, it will clean up the resources, then
# the rest of the tests.
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_kv_connector_stats(default_vllm_config, dist_init):
    """Test that KV transfer stats are properly recorded and retrieved."""
    vllm_config = create_vllm_config()

    # Test worker role in decode server.
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config, connector.engine_id, hand_shake_latency=0
    )

    # Verify that xfer_stats starts empty
    initial_stats = connector.get_kv_connector_stats()
    assert initial_stats is None

    # Create transfer metadata
    request_id = "test_req_for_stats"
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=[1, 2, 3],
        kv_transfer_params={
            "remote_block_ids": [4, 5, 6],
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"prefill-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )
    connector.bind_connector_metadata(metadata)

    # Start the transfer
    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
        slot_mapping={},
    )
    connector.start_load_kv(dummy_ctx)

    # Verify stats are recorded after transfer is complete
    max_iterations = 2
    # Clear metadata before start_load_kv to prevent reprocessing same request
    connector.bind_connector_metadata(NixlConnectorMetadata())
    for _ in range(max_iterations):
        # Need to call start_load_kv to process completed handshakes
        connector.start_load_kv(dummy_ctx)
        _, done_recving = connector.get_finished(finished_req_ids=set())
        if len(done_recving) > 0 and request_id in done_recving:
            break
        time.sleep(0.1)  # Small delay to allow background handshake to complete
    else:
        assert "Transfer did not complete within expected iterations"

    # Now check that stats were recorded
    stats_after_transfer = connector.get_kv_connector_stats()
    assert isinstance(stats_after_transfer, NixlKVConnectorStats)

    # Verify stats values are recorded
    assert not stats_after_transfer.is_empty()
    assert stats_after_transfer.num_successful_transfers == 1

    # Verify stats are reset after retrieval
    stats_after_reset = connector.get_kv_connector_stats()
    assert stats_after_reset is None


def test_kv_connector_stats_aggregation():
    """
    Test KV transfer stats aggregation across TP ranks using
    KVOutputAggregator (used by MultiprocExecutor).
    """

    # Create KVOutputAggregator for 3 workers (simulating TP=3), same thing
    # done in MultiprocExecutor.execute_model
    aggregator = KVOutputAggregator(expected_finished_count=3)

    # Create stats for multiple workers with different transfer patterns
    worker1_stats = NixlKVConnectorStats()
    worker2_stats = NixlKVConnectorStats()
    worker3_stats = NixlKVConnectorStats()

    # Record different transfers on each worker
    # Worker 1: 2 transfers
    stats = get_default_xfer_telemetry()
    worker1_stats.record_transfer(stats)
    worker1_stats.record_transfer(stats)

    # Worker 2: 1 transfer
    worker2_stats.record_transfer(stats)

    # Worker 3: 3 transfers
    stats = get_default_xfer_telemetry(
        xferDurationS=2, postDurationS=2, totalBytes=2, descCount=2
    )
    worker3_stats.record_transfer(stats)
    worker3_stats.record_transfer(stats)
    worker3_stats.record_transfer(stats)

    # Create ModelRunnerOutput instances for each worker
    worker_outputs = []
    for i, worker_stats in enumerate([worker1_stats, worker2_stats, worker3_stats]):
        output = ModelRunnerOutput(
            req_ids=[f"req_{i}"],
            req_id_to_index={f"req_{i}": 0},
            sampled_token_ids=[[123]],  # dummy token
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(
                finished_sending=set([f"req_{i}_send"])
                if i < 2
                else None,  # Workers 0,1 finished sending
                finished_recving=set([f"req_{i}_recv"])
                if i > 0
                else None,  # Workers 1,2 finished receiving
                kv_connector_stats=worker_stats,
            ),
        )
        worker_outputs.append(output)

    # Use the real aggregation mechanism (like MultiprocExecutor.execute_model)
    aggregated_output = aggregator.aggregate(worker_outputs, output_rank=0)
    kv_connector_stats = aggregated_output.kv_connector_output.kv_connector_stats
    assert isinstance(kv_connector_stats, NixlKVConnectorStats)
    # Number of total transfers across all workers.
    assert kv_connector_stats.num_successful_transfers == 6
    # Logging proc, call reduce() to get CLI-friendly stats.
    cli_stats = kv_connector_stats.reduce()
    assert cli_stats["Avg xfer time (ms)"] == 1500.0
    assert cli_stats["Avg post time (ms)"] == 1500.0
    assert cli_stats["Avg number of descriptors"] == 1.5


def test_multi_kv_connector_stats_aggregation():
    """
    Test MultiKVConnectorStats aggregation across TP ranks using
    KVOutputAggregator (used by MultiprocExecutor).
    """

    aggregator = KVOutputAggregator(expected_finished_count=3)

    from dataclasses import dataclass

    # Mock a KVConnectorStats class for testing aggregation over connectors.
    @dataclass
    class FooKVConnectorStats(KVConnectorStats):
        def reset(self):
            self.data = {"num_foo_transfers": 0}

        def record_transfer(self):
            if "num_foo_transfers" not in self.data:
                self.data["num_foo_transfers"] = 0
            self.data["num_foo_transfers"] += 1

        def is_empty(self) -> bool:
            return self.data["num_foo_transfers"] == 0

        def aggregate(self, other: "FooKVConnectorStats") -> "FooKVConnectorStats":
            if not other.is_empty():
                self.data["num_foo_transfers"] += other.data["num_foo_transfers"]
            return self

    def make_multi_stats(nixl_count: int, foo_count: int) -> MultiKVConnectorStats:
        data: dict[str, KVConnectorStats] = {}
        if nixl_count > 0:
            nixl_stats = NixlKVConnectorStats()
            for _ in range(nixl_count):
                nixl_stats.record_transfer(get_default_xfer_telemetry())
            data["NixlConnector"] = nixl_stats
        if foo_count > 0:
            foo_stats = FooKVConnectorStats()
            for _ in range(foo_count):
                foo_stats.record_transfer()
            data["FooConnector"] = foo_stats
        return MultiKVConnectorStats(data=data)

    # Create heterogeneous stats across 3 workers
    worker_patterns = [(2, 1), (3, 0), (0, 5)]  # (Nixl, Foo)

    worker_outputs: list[ModelRunnerOutput] = []
    for i, (nixl, foo) in enumerate(worker_patterns):
        stats = make_multi_stats(nixl, foo)
        output = ModelRunnerOutput(
            req_ids=[f"req_{i}"],
            req_id_to_index={f"req_{i}": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(
                finished_sending=set([f"req_{i}_send"]) if i < 2 else None,
                finished_recving=set([f"req_{i}_recv"]) if i > 0 else None,
                kv_connector_stats=stats,
            ),
        )
        worker_outputs.append(output)

    aggregated_output = aggregator.aggregate(worker_outputs, output_rank=0)
    kv_connector_stats = aggregated_output.kv_connector_output.kv_connector_stats
    assert isinstance(kv_connector_stats, MultiKVConnectorStats)

    # Validate per-connector totals across workers
    assert isinstance(kv_connector_stats["NixlConnector"], NixlKVConnectorStats)
    assert kv_connector_stats["NixlConnector"].num_successful_transfers == 5
    assert isinstance(kv_connector_stats["FooConnector"], FooKVConnectorStats)
    assert kv_connector_stats["FooConnector"].data["num_foo_transfers"] == 6


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_scheduler_kv_connector_stats_aggregation():
    """Test scheduler and worker KV connector stats aggregation."""
    from vllm.v1.core.sched.output import SchedulerOutput

    scheduler = create_scheduler(create_vllm_config())

    # Worker stats with transfer metrics
    worker_stats = NixlKVConnectorStats()
    worker_stats.record_transfer(get_default_xfer_telemetry())
    worker_stats.data["remote_tokens"] = []

    # Scheduler stats with custom metric (needs dummy transfer to avoid being skipped)
    scheduler_stats = NixlKVConnectorStats()
    scheduler_stats.data.update(
        {  # dummy transfer just for testing, to bypass is_empty() check
            "transfer_duration": [0],
            "post_duration": [0],
            "bytes_transferred": [0],
            "num_descriptors": [0],
            "remote_tokens": [128],
        }
    )

    # Mock the scheduler connector's stats method
    scheduler.connector.get_kv_connector_stats = lambda: MultiKVConnectorStats(
        data={"NixlConnector": scheduler_stats}
    )

    model_output = ModelRunnerOutput(
        req_ids=["req_0"],
        req_id_to_index={"req_0": 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None],
        kv_connector_output=KVConnectorOutput(
            kv_connector_stats=MultiKVConnectorStats(
                data={"NixlConnector": worker_stats}
            )
        ),
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={"req_0": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)

    final_stats = next(
        iter(engine_core_outputs.values())
    ).scheduler_stats.kv_connector_stats
    nixl_stats = final_stats["NixlConnector"]
    assert nixl_stats.num_successful_transfers == 2
    assert nixl_stats.data["remote_tokens"] == [128]


@pytest.mark.parametrize("distributed_executor_backend", ["ray", None])
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
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
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.5,
        "kv_transfer_config": kv_transfer_config,
        "distributed_executor_backend": distributed_executor_backend,
    }

    timeout = 6
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", str(timeout))

    def run_test_and_cleanup():
        llm = LLM(**llm_kwargs)
        try:
            _run_abort_timeout_test(llm, timeout)
        finally:
            llm.llm_engine.engine_core.shutdown()

    # Build runtime_env only if we're using Ray
    if distributed_executor_backend == "ray":
        with _make_fake_nixl_pkg() as working_dir:
            runtime_env = {
                "working_dir": working_dir,  # ship fake nixl package
                "env_vars": {
                    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": str(timeout),
                    # TODO: for ray to carry over, remove once we set
                    "NIXL_TELEMETRY_ENABLE": "1",
                },
            }
            ray.init(runtime_env=runtime_env)
            try:
                run_test_and_cleanup()
            finally:
                ray.shutdown()
    else:
        run_test_and_cleanup()


class RequestIdMapper:
    """Helper class to map external request IDs to internal request IDs."""

    def __init__(self, output_processor: OutputProcessor):
        self.req_id_mapping: dict[str, str] = {}
        self.original_add_request = output_processor.add_request
        output_processor.add_request = self._add_request

    def _add_request(self, request: EngineCoreRequest, *args, **kwargs):
        self.req_id_mapping[request.external_req_id] = request.request_id
        return self.original_add_request(request, *args, **kwargs)

    def __call__(self, external_req_id: str) -> str:
        return self.req_id_mapping[external_req_id]


def _run_abort_timeout_test(llm: LLM, timeout: int):
    """Helper function to run the abort timeout test logic."""
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
    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks

    id_mapper = RequestIdMapper(llm.llm_engine.output_processor)

    def req_id(outputs: list[RequestOutput]) -> str:
        assert len(outputs) == 1
        return id_mapper(outputs[0].request_id)

    padding = "Just making this request a little longer so that we're sure "
    "we're not hitting the small-request lower bound beneath which we don't "
    "actually trigger the whole kv transfer, but rather just recompute the "
    "blocks on D."
    req0_id = req_id(
        llm.generate([f"What is the capital of Japan? {padding}"], sampling_params)
    )

    # Request finished but not freed
    assert req0_id in scheduler.finished_req_ids and req0_id in req_to_blocks
    # Some other request, 0 still not freed
    req1_id = req_id(
        llm.generate([f"What is the capital of Italy? {padding}"], sampling_params)
    )
    assert req0_id in req_to_blocks
    assert req1_id in scheduler.finished_req_ids and req1_id in req_to_blocks

    # Wait for timeout and trigger another scheduler loop
    time.sleep(timeout)
    _ = llm.generate([f"What is the capital of France? {padding}"], sampling_params)
    # Request-0 times out and is cleared!
    assert req0_id not in req_to_blocks
    # Need to shutdown the background thread to release NIXL side channel port
    llm.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize("enable_cross_layers", [False, True])
@pytest.mark.parametrize(
    "attn_backend",
    [
        pytest.param(
            "FLASH_ATTN",
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Attention backend FLASH_ATTN is not supported on ROCm",
            ),
        ),
        pytest.param(
            "ROCM_ATTN",
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(),
                reason="Attention backend ROCM_ATTN is only supported on ROCm",
            ),
        ),
        "TRITON_ATTN",
        "FLASHINFER",
    ],
)
def test_register_kv_caches(
    default_vllm_config, dist_init, attn_backend, enable_cross_layers
):
    """
    Test that register_kv_caches() properly calls nixl_wrapper methods with
    correct data.

    This test verifies:
    1. nixl_wrapper.get_reg_descs() is called with caches_data containing
       tensor metadata
    2. nixl_wrapper.get_xfer_descs() is called with blocks_data containing
       block layout info
    """

    vllm_config = create_vllm_config(attention_backend=attn_backend)

    # Enable cross layers blocks
    vllm_config.kv_transfer_config.kv_connector_extra_config[
        "enable_cross_layers_blocks"
    ] = enable_cross_layers

    # Import the appropriate backend based on the parameter
    if attn_backend == "FLASH_ATTN":
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        backend_cls = FlashAttentionBackend
    elif attn_backend == "ROCM_ATTN":
        from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

        backend_cls = RocmAttentionBackend
    else:  # TRITON
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

        backend_cls = TritonAttentionBackend

    nixl_module = "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"
    with (
        patch(f"{nixl_module}.NixlWrapper") as mock_nixl_wrapper,
        patch(f"{nixl_module}.threading.Event"),
        patch(f"{nixl_module}.threading.Thread") as mock_thread,
        patch(f"{nixl_module}.get_current_attn_backend") as mock_get_attn_backend,
    ):
        # Ensure get_attn_backend returns the correct value due to
        # _cached_get_attn_backend returning the backend from previous
        # test run if not mocking.
        mock_get_attn_backend.return_value = backend_cls

        # Create connector
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config, connector.engine_id, hand_shake_latency=0
        )

        # Get the mock instance
        mock_wrapper_instance = mock_nixl_wrapper.return_value
        connector.connector_worker.nixl_wrapper = mock_wrapper_instance

        # Appease NixlHandshakePayload encoding with some bytes
        mock_wrapper_instance.get_agent_metadata.return_value = b"fake_agent_metadata"

        # Reassure the shutdown() check that the thread is terminated
        mock_thread.return_value.is_alive.return_value = False

        expected_tensor_size: int
        expected_base_addrs: list[int]
        expected_num_entries: int
        kv_caches: dict[str, torch.Tensor]
        if connector.prefer_cross_layer_blocks:
            num_layers = 32
            block_size = 16
            num_blocks = 8
            kv_cache_spec = AttentionSpec(
                block_size=block_size,
                num_kv_heads=4,
                head_size=64,
                dtype=torch.bfloat16,
            )
            kv_cache_config = KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=kv_cache_spec.page_size_bytes * num_blocks,
                        shared_by=["dummy-layer"],
                    )
                    for i in range(num_layers)
                ],
                # allocate_uniform_kv_caches does not use this
                kv_cache_groups=[],
            )

            with set_current_vllm_config(vllm_config):
                _, cross_layers_kv_cache, _ = (
                    KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
                        kv_cache_config=kv_cache_config,
                        attn_groups=[
                            [
                                AttentionGroup(
                                    backend=backend_cls,
                                    layer_names=[],
                                    kv_cache_spec=kv_cache_spec,
                                    kv_cache_group_id=0,
                                )
                            ]
                        ],
                        cache_dtype=torch.bfloat16,
                        device=torch.cuda.current_device(),
                        kernel_block_sizes=[block_size],
                    )
                )
            # Store tensor info for validation
            expected_tensor_size = (
                cross_layers_kv_cache.element_size() * cross_layers_kv_cache.numel()
            )
            expected_base_addrs = [
                cross_layers_kv_cache.data_ptr(),
            ]
            expected_num_entries = 1

            expected_blocks_count = 8

            kv_caches = {"all-layers": cross_layers_kv_cache}

        else:
            # Create test kv cache tensors using proper backend shape
            kv_cache_shape = backend_cls.get_kv_cache_shape(
                num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
            )
            shared_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
            unique_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
            kv_caches = {
                "layer0": shared_tensor,
                "layer1": unique_tensor,
                "layer2": shared_tensor,
            }

            # Store tensor info for validation

            test_shape = backend_cls.get_kv_cache_shape(
                num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
            )
            is_blocks_first = len(test_shape) == 5 and test_shape[0] == 1

            if is_blocks_first:
                expected_tensor_size = (
                    shared_tensor.element_size() * shared_tensor.numel()
                )
                expected_base_addrs = [
                    shared_tensor.data_ptr(),
                    unique_tensor.data_ptr(),
                ]
                expected_num_entries = 2
            else:
                expected_tensor_size = (
                    shared_tensor[0].element_size() * shared_tensor[0].numel()
                )
                expected_base_addrs = [
                    shared_tensor[0].data_ptr(),
                    shared_tensor[1].data_ptr(),
                    unique_tensor[0].data_ptr(),
                    unique_tensor[1].data_ptr(),
                ]
                expected_num_entries = 4
            expected_blocks_count = 8

        # Execute register_kv_caches
        connector.register_kv_caches(kv_caches)

        # Verify get_reg_descs was called with caches_data
        assert mock_wrapper_instance.get_reg_descs.called
        caches_data, _ = mock_wrapper_instance.get_reg_descs.call_args[0]
        assert len(caches_data) == expected_num_entries

        for i, cache_entry in enumerate(caches_data):
            base_addr, size, _tp_rank, _ = cache_entry
            assert size == expected_tensor_size, (
                f"Entry {i}: Expected tensor size {expected_tensor_size}, got {size}"
            )
            assert base_addr == expected_base_addrs[i], (
                f"Entry {i}: Expected base address {expected_base_addrs[i]}, "
                f"got {base_addr}"
            )

        # Verify get_xfer_descs was called with blocks_data
        assert mock_wrapper_instance.get_xfer_descs.called
        blocks_data, _ = mock_wrapper_instance.get_xfer_descs.call_args[0]

        # Validate blocks_data structure and size
        assert len(blocks_data) == expected_blocks_count, (
            f"Expected {expected_blocks_count} blocks, got {len(blocks_data)}"
        )

        if connector.prefer_cross_layer_blocks:
            num_blocks = 8
            expected_block_len = expected_tensor_size // num_blocks
        else:
            num_blocks = 2
            if is_blocks_first:
                expected_block_len = expected_tensor_size // num_blocks // 2
            else:
                expected_block_len = expected_tensor_size // num_blocks

        for i, block_entry in enumerate(blocks_data):
            block_start_addr, block_len, tp_rank = block_entry
            assert block_len == expected_block_len, (
                f"Block entry {i}: Expected block len {expected_block_len}, "
                f"got {block_len}"
            )


class FakePlatform(Platform):
    device_type: str = "oot"

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported
        kv_buffer_device for nixl.
        """
        return {"oot": ("oot",)}

    @classmethod
    def get_nixl_memory_type(cls) -> str | None:
        """
        Returns the nixl memory type for the current platform.
        """
        return "VRAM"


@pytest.mark.parametrize(
    "kv_buffer_device, nixl_memory_type",
    [
        ("oot", "VRAM"),
    ],
)
def test_kv_buffer_to_nixl_memory_types(
    default_vllm_config, dist_init, kv_buffer_device, nixl_memory_type
):
    """
    Test that register_kv_caches() passes the correct memory types from the
    config to the nixl_wrapper.
    """
    vllm_config = create_vllm_config()
    # Override the default memory types in the config
    vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        _NIXL_SUPPORTED_DEVICE,
    )

    _NIXL_SUPPORTED_DEVICE.update(FakePlatform.get_nixl_supported_devices())

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.threading.Thread"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.current_platform",
            FakePlatform,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector._NIXL_SUPPORTED_DEVICE",
            _NIXL_SUPPORTED_DEVICE,
        ),
    ):  # noqa: E501
        # Create connector and replace its worker with a fake one for isolation
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)

        # Verify get_reg_descs was called with the correct memory_type
        assert connector.connector_worker.kv_buffer_device == kv_buffer_device
        assert connector.connector_worker.nixl_memory_type == nixl_memory_type


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_shutdown_cleans_up_resources(default_vllm_config, dist_init):
    """Test that shutdown() properly cleans up all resources."""
    vllm_config = create_vllm_config()

    scheduler = NixlConnectorScheduler(
        vllm_config, vllm_config.kv_transfer_config.engine_id
    )
    worker = NixlConnectorWorker(vllm_config, vllm_config.kv_transfer_config.engine_id)
    nixl_wrapper = worker.nixl_wrapper

    with (
        patch.object(worker, "_handshake_initiation_executor") as mock_exec,
        patch.object(scheduler, "_nixl_handshake_listener_t") as mock_listener,
        patch.object(nixl_wrapper, "release_xfer_handle") as mock_rel_xfer,
        patch.object(nixl_wrapper, "release_dlist_handle") as mock_rel_dlist,
        patch.object(nixl_wrapper, "remove_remote_agent") as mock_rem_agent,
        patch.object(nixl_wrapper, "deregister_memory") as mock_dereg,
    ):
        worker._recving_transfers = {"req1": [123]}
        # Mock register_kv_cache which registers local handle
        worker.src_xfer_handles_by_block_size = {worker.block_size: 455}
        # P TP = 2 * D TP case, we should register 2 local handles
        worker.src_xfer_handles_by_tp_ratio = {-2: [456, 457]}
        worker.dst_xfer_side_handles = {"engine1": {0: 789}}
        worker._remote_agents = {"engine1": {0: "agent1"}}
        worker._registered_descs = ["desc1", "desc2"]

        mock_listener.is_alive.return_value = False

        worker.shutdown()

        # Test idempotency
        worker.shutdown()
        worker.shutdown()

        mock_exec.shutdown.assert_called_with(wait=False)

        # Same sequence on scheduler.shutdown()
        scheduler.shutdown()
        scheduler.shutdown()
        scheduler.shutdown()
        mock_listener.join.assert_called_once()

        mock_rel_xfer.assert_called_once_with(123)
        assert mock_rel_dlist.call_count == 4
        mock_rel_dlist.assert_any_call(455)  # src handle (whole region)
        mock_rel_dlist.assert_any_call(456)  # src handle (1st chunk)
        mock_rel_dlist.assert_any_call(457)  # src handle (2nd chunk)
        mock_rel_dlist.assert_any_call(789)  # dst handle
        mock_rem_agent.assert_called_once_with("agent1")
        assert mock_dereg.call_count == 2
        mock_dereg.assert_any_call("desc1")
        mock_dereg.assert_any_call("desc2")


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_aborted_request_removed_from_worker_in_batch(default_vllm_config, dist_init):
    """
    Create and schedule a request so that P adds it to in-batch tracking via
    the real scheduler, then simulate an abort (request not in next scheduler
    iteration) and verify the worker no longer tracks it as in-batch.
    """
    vllm_config = create_vllm_config()

    scheduler = create_scheduler(vllm_config)
    # KVConnector Worker in P
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config, connector.engine_id, hand_shake_latency=0
    )

    # Create a request that triggers do_remote_decode so that
    # the scheduler adds it to reqs_in_batch
    req = create_request(request_id=1, do_remote_decode=True, max_tokens=1)
    scheduler.add_request(req)

    # First scheduling pass - examinate build_connector_meta output
    sched_out = scheduler.schedule()
    kv_meta = sched_out.kv_connector_metadata
    assert kv_meta is not None
    assert isinstance(kv_meta, NixlConnectorMetadata)
    assert req.request_id in kv_meta.reqs_in_batch

    #### Model Runner start ####
    # Bind scheduler-produced metadata and start worker processing.
    connector.bind_connector_metadata(kv_meta)

    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
        slot_mapping={},
    )
    connector.start_load_kv(dummy_ctx)

    # Ensure it was tracked by the worker
    assert req.request_id in connector.connector_worker._reqs_to_process

    #### Model Runner end ####

    # Abort request - request_finished call in connector scheduler
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    # Second scheduling pass - build metadata with aborted request
    sched_out2 = scheduler.schedule()
    kv_meta2 = sched_out2.kv_connector_metadata
    assert kv_meta2 is not None
    assert isinstance(kv_meta2, NixlConnectorMetadata)
    assert req.request_id not in kv_meta2.reqs_in_batch

    # Bind empty/abort metadata and run worker step
    #### Model Runner start ####
    connector.bind_connector_metadata(kv_meta2)
    connector.start_load_kv(dummy_ctx)

    # After abort, the worker should not keep tracking it as "in-batch"
    assert req.request_id not in connector.connector_worker._reqs_to_process
    #### Model Runner end ####


class FailingNixlWrapper(FakeNixlWrapper):
    """Mock NixlWrapper that fails on specific operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_handshake = False
        self.fail_transfer_setup = False
        self.fail_send_notif = False
        self.fail_transfer_state = False  # Returns "ERR" state
        self.fail_transfer_exception = False  # Raises exception in check_xfer_state

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        if self.fail_handshake:
            from zmq.error import Again

            raise Again("Simulated timeout failure")
        return super().add_remote_agent(agent_metadata)

    def make_prepped_xfer(
        self,
        xfer_type: str,
        local_xfer_side_handle: int,
        local_block_descs_ids: list[int],
        remote_xfer_side_handle: int,
        remote_block_descs_ids: list[int],
        notif_msg: bytes | None = None,
    ) -> int:
        if self.fail_transfer_setup:
            # classic RuntimeError to simulate failure
            raise RuntimeError("BAD STATUS")
        return super().make_prepped_xfer(
            xfer_type,
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg,
        )

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        if self.fail_send_notif:
            raise RuntimeError("Simulated send_notif failure")
        return super().send_notif(agent_name, notif_msg)

    def check_xfer_state(self, handle: int) -> str:
        if self.fail_transfer_exception:
            raise RuntimeError("Simulated check_xfer_state exception")
        if self.fail_transfer_state:
            return "ERR"  # Bad transfer state
        return super().check_xfer_state(handle)


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FailingNixlWrapper,
)
@pytest.mark.parametrize(
    "failure_type,wrapper_config,needs_get_finished",
    [
        ("transfer_setup_failed", {"fail_transfer_setup": True}, False),
        ("handshake_failed", {"fail_handshake": True}, False),
        ("notification_failed", {"fail_send_notif": True}, False),
        ("transfer_failed", {"fail_transfer_state": True}, True),
        ("transfer_exception", {"fail_transfer_exception": True}, True),
    ],
)
def test_transfer_failure_logging(
    default_vllm_config,
    dist_init,
    failure_type,
    wrapper_config,
    needs_get_finished,
):
    """Test that transfer failures are logged with structured context.

    Run with `pytest -sv` to see the log output.

    Covers failure types:
    - transfer_setup_failed: make_prepped_xfer fails
    - handshake_failed: add_remote_agent fails during request handshake
    - notification_failed: send_notif fails
    - transfer_failed: check_xfer_state returns bad state (e.g., "ERR")
    - transfer_exception: check_xfer_state raises exception
    """
    import logging

    vllm_config = create_vllm_config()

    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config, connector.engine_id, hand_shake_latency=0.0
    )

    # Configure FailingNixlWrapper to fail in the specified way
    for key, value in wrapper_config.items():
        setattr(connector.connector_worker.nixl_wrapper, key, value)

    request_id = f"test_{failure_type}_req"

    # For notification_failed, we need empty local blocks
    # (full cache hit path to trigger send_notif)
    local_blocks = [] if failure_type == "notification_failed" else [10, 11, 12]
    remote_blocks = [20, 21, 22]

    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=local_blocks,
        kv_transfer_params={
            "remote_block_ids": remote_blocks,
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"prefill-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )
    connector.bind_connector_metadata(metadata)

    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
        slot_mapping={},
    )

    # Capture logs from the nixl_connector logger specifically
    # vLLM loggers have propagate=False, so we need to capture directly
    nixl_logger = logging.getLogger(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"
    )
    captured_logs: list[logging.LogRecord] = []

    class LogCapture(logging.Handler):
        def emit(self, record):
            captured_logs.append(record)

    handler = LogCapture()
    handler.setLevel(logging.ERROR)
    nixl_logger.addHandler(handler)

    try:
        connector.start_load_kv(dummy_ctx)
        # Process the ready_requests queue (for async handshake)
        connector.bind_connector_metadata(NixlConnectorMetadata())
        # Wait for async handshake to complete
        time.sleep(0.2)
        connector.start_load_kv(dummy_ctx)

        # For transfer_failed/transfer_exception, the error happens in
        # get_finished() when checking transfer state
        if needs_get_finished:
            connector.get_finished(finished_req_ids=set())
    finally:
        nixl_logger.removeHandler(handler)

    # Print logs for manual comparison between commits
    error_logs = [r for r in captured_logs if r.levelno >= logging.ERROR]
    print("\n" + "=" * 60)
    print(f"CAPTURED ERROR LOGS for {failure_type}:")
    print("=" * 60)
    for i, record in enumerate(error_logs):
        print(f"\n--- Log {i + 1} ---")
        print(f"Message: {record.message}")
    print("=" * 60 + "\n")

    assert len(error_logs) >= 1, f"Expected at least one error log for {failure_type}"

    # Verify structured logging output (new format)
    # Check that at least one log matches the expected format
    all_messages = [r.message for r in error_logs]
    combined_logs = "\n".join(all_messages)

    assert any("NIXL transfer failure" in msg for msg in all_messages), (
        f"Expected structured log format with 'NIXL transfer failure' prefix "
        f"for {failure_type}. Got: {all_messages}"
    )
    assert any("failure_type" in msg for msg in all_messages), (
        f"Expected 'failure_type' in logs. Got: {all_messages}"
    )
    assert any("Context:" in msg for msg in all_messages), (
        f"Expected 'Context:' in logs. Got: {all_messages}"
    )
    # Check that the expected failure_type appears in at least one log
    # Note: handshake_failed also triggers handshake_setup_failed
    assert failure_type in combined_logs or (
        failure_type == "handshake_failed" and "handshake_setup_failed" in combined_logs
    ), f"Expected '{failure_type}' in logs. Got: {all_messages}"


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FailingNixlWrapper,
)
def test_handshake_failure_returns_finished(default_vllm_config, dist_init):
    """Test that handshake failures mark blocks invalid and return via get_finished."""
    vllm_config = create_vllm_config()

    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config, connector.engine_id, hand_shake_latency=0.1
    )
    connector.connector_worker.nixl_wrapper.fail_handshake = True

    request_id = "test_handshake_fail"
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=[1, 2, 3],
        kv_transfer_params={
            "remote_block_ids": [4, 5, 6],
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"prefill-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )
    connector.bind_connector_metadata(metadata)

    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
        slot_mapping={},
    )
    connector.start_load_kv(dummy_ctx)

    # Wait for handshake to fail
    time.sleep(0.3)

    # Check that blocks were marked invalid
    invalid_blocks = connector.get_block_ids_with_load_errors()
    assert invalid_blocks == {1, 2, 3}

    # Check that request appears in get_finished
    _, done_recving = connector.get_finished(finished_req_ids=set())
    assert request_id in done_recving


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FailingNixlWrapper,
)
def test_transfer_setup_failure_returns_finished(default_vllm_config, dist_init):
    """Test that transfer setup failures mark blocks invalid
    and return via get_finished."""
    vllm_config = create_vllm_config()

    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config, connector.engine_id, hand_shake_latency=0
    )
    connector.connector_worker.nixl_wrapper.fail_transfer_setup = True

    request_id = "test_transfer_fail"
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=[7, 8, 9],
        kv_transfer_params={
            "remote_block_ids": [10, 11, 12],
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"prefill-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )
    connector.bind_connector_metadata(metadata)

    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
        slot_mapping={},
    )
    connector.start_load_kv(dummy_ctx)

    # Wait for handshake to complete and process ready_requests
    connector.bind_connector_metadata(NixlConnectorMetadata())
    time.sleep(0.1)
    connector.start_load_kv(dummy_ctx)

    # check that blocks were marked invalid
    invalid_blocks = connector.get_block_ids_with_load_errors()
    assert invalid_blocks == {7, 8, 9}

    # ensure request appears in get_finished
    _, done_recving = connector.get_finished(finished_req_ids=set())
    assert request_id in done_recving


@pytest.mark.parametrize(
    "mismatch_type,config_overrides,version_override,should_fail,enforce_handshake_compat",
    [
        ("vllm_version", {}, {"vllm_version": "0.6.1"}, True, True),
        ("nixl_connector_version", {}, {"connector_version": 37}, True, True),
        ("model_name", {"model": "facebook/opt-350m"}, {}, True, True),
        ("dtype", {"dtype": "bfloat16"}, {}, True, True),
        ("cache_dtype", {"cache_dtype": "fp8"}, {}, True, True),
        ("num_kv_heads", {"hf_overrides": {"num_key_value_heads": 8}}, {}, True, True),
        (
            "num_hidden_layers",
            {"hf_overrides": {"num_hidden_layers": 24}},
            {},
            True,
            True,
        ),
        ("hidden_size", {"hf_overrides": {"hidden_size": 1536}}, {}, True, True),
        ("block_size", {"block_size": 8}, {}, False, True),
        ("matching_config", {}, {}, False, True),
        ("escape_hatch", {"model": "facebook/opt-350m"}, {}, False, False),
    ],
)
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_compatibility_hash_validation(
    default_vllm_config,
    dist_init,
    mismatch_type,
    config_overrides,
    version_override,
    should_fail,
    enforce_handshake_compat,
):
    """
    Test NIXL compatibility hash validation during handshake.

    Parameters:
        mismatch_type: description of what is being tested
        config_overrides: dict of config to override for the remote instance
        version_override: version dict e.g. {"vllm_version": "0.6.1"}
        should_fail: whether the handshake should fail
        enforce_handshake_compat: whether to enforce compatibility checking
    """
    local_vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        block_size=16,
        kv_connector_extra_config={
            "enforce_handshake_compat": enforce_handshake_compat
        },
    )
    decode_connector = NixlConnector(local_vllm_config, KVConnectorRole.WORKER)
    decode_worker = decode_connector.connector_worker
    kv_cache_shape = decode_worker.attn_backend.get_kv_cache_shape(
        num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
    )
    shared_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    unique_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    kv_caches = {
        "layer0": shared_tensor,
        "layer1": unique_tensor,
        "layer2": shared_tensor,
    }
    decode_connector.register_kv_caches(kv_caches)

    remote_config_params: dict[str, Any] = {
        "model": "facebook/opt-125m",
        "block_size": 16,
        **config_overrides,
    }
    remote_vllm_config = create_vllm_config(**remote_config_params)

    with contextlib.ExitStack() as stack:
        if "vllm_version" in version_override:
            stack.enter_context(
                patch("vllm.__version__", version_override["vllm_version"])
            )
        elif "connector_version" in version_override:
            stack.enter_context(
                patch.object(
                    nixl_connector,
                    "NIXL_CONNECTOR_VERSION",
                    version_override["connector_version"],
                )
            )
        remote_hash = compute_nixl_compatibility_hash(
            remote_vllm_config,
            decode_worker.backend_name,
            decode_worker.kv_topo.cross_layers_blocks,
        )

    prefill_block_size = config_overrides.get("block_size", 16)
    prefill_metadata = NixlAgentMetadata(
        engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
        agent_metadata=FakeNixlWrapper.AGENT_METADATA,
        kv_caches_base_addr=[0],
        device_id=0,
        num_blocks=1,
        block_lens=[4096 * prefill_block_size],  # slot_size * block_size
        kv_cache_layout="HND",
        block_size=prefill_block_size,
    )
    handshake_payload = NixlHandshakePayload(
        compatibility_hash=remote_hash,
        agent_metadata_bytes=msgspec.msgpack.encode(prefill_metadata),
    )

    # Mock ZMQ socket to return our handshake payload
    mock_socket = MagicMock()
    mock_socket.recv.return_value = msgspec.msgpack.encode(handshake_payload)

    # Mock add_remote_agent to avoid actual NIXL operations
    # Patch zmq_ctx to return our mock socket
    with (
        patch.object(decode_worker, "add_remote_agent", return_value="fake_agent"),
        patch.object(nixl_connector, "zmq_ctx") as mock_zmq_ctx,
    ):
        mock_zmq_ctx.return_value.__enter__.return_value = mock_socket

        if should_fail:
            with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
                decode_worker._nixl_handshake(
                    host="localhost",
                    port=1234,
                    remote_tp_size=1,
                    expected_engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                )
        else:
            result = decode_worker._nixl_handshake(
                host="localhost",
                port=1234,
                remote_tp_size=1,
                expected_engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            )
            # Verify handshake returned agent mapping
            assert isinstance(result, dict)
            assert len(result) == 1


@pytest.mark.parametrize(
    "error_scenario",
    [
        "handshake_decode_error",
        "handshake_validation_error",
        "metadata_decode_error",
        "metadata_validation_error",
    ],
)
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_handshake_decode_errors(default_vllm_config, dist_init, error_scenario):
    """
    Test that msgspec decode errors are properly handled during handshake.

    Tests both DecodeError and ValidationError for both decoders:
    - NixlHandshakePayload decoder
    - NixlAgentMetadata decoder
    """
    local_vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        block_size=16,
    )
    decode_connector = NixlConnector(local_vllm_config, KVConnectorRole.WORKER)
    decode_worker = decode_connector.connector_worker

    backend = get_current_attn_backend(local_vllm_config)
    test_shape = backend.get_kv_cache_shape(
        num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
    )
    decode_worker.kv_topo = TpKVTopology(
        tp_rank=decode_worker.tp_rank,
        engine_id=decode_worker.engine_id,
        remote_tp_size=decode_worker._tp_size,  # shared state
        remote_block_size=decode_worker._block_size,  # shared state
        is_mla=decode_worker.use_mla,
        total_num_kv_heads=decode_worker.model_config.get_total_num_kv_heads(),
        attn_backend=backend,
        tensor_shape=test_shape,
    )

    decode_worker.compat_hash = compute_nixl_compatibility_hash(
        decode_worker.vllm_config,
        decode_worker.backend_name,
        decode_worker.kv_topo.cross_layers_blocks,
    )

    if error_scenario == "handshake_decode_error":
        msg_bytes = b"this is not valid msgpack data"
    elif error_scenario == "handshake_validation_error":
        msg_bytes = msgspec.msgpack.encode({"wrong_field": "value"})
    elif error_scenario == "metadata_decode_error":
        valid_handshake = NixlHandshakePayload(
            compatibility_hash=decode_worker.compat_hash,
            agent_metadata_bytes=b"invalid msgpack for metadata",
        )
        msg_bytes = msgspec.msgpack.encode(valid_handshake)

    elif error_scenario == "metadata_validation_error":
        valid_handshake = NixlHandshakePayload(
            compatibility_hash=decode_worker.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.encode({"missing": "fields"}),
        )
        msg_bytes = msgspec.msgpack.encode(valid_handshake)
    else:
        raise AssertionError(f"{error_scenario} not a valid scenario")

    mock_socket = MagicMock()
    mock_socket.recv.return_value = msg_bytes
    with (
        patch.object(decode_worker, "add_remote_agent", return_value="fake_agent"),
        patch.object(nixl_connector, "zmq_ctx") as mock_zmq_ctx,
    ):
        mock_zmq_ctx.return_value.__enter__.return_value = mock_socket

        with pytest.raises(RuntimeError):
            decode_worker._nixl_handshake(
                host="localhost",
                port=1234,
                remote_tp_size=1,
                expected_engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            )
