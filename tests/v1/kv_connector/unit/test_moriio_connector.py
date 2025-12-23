import pytest

from vllm.platforms import current_platform
import contextlib
import inspect
import os
import tempfile
import textwrap
import time
import uuid
from collections import defaultdict
from unittest.mock import patch
from unittest.mock import MagicMock
import pytest
import ray
import torch
import msgspec
from vllm import LLM
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    KVConnectorRole,
    )
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import MoRIIOConstants
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import MoRIIOAgentMetadata

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import MoRIIOConnectorMetadata

from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_shutdown,
    has_kv_transfer_group,
)
from vllm.forward_context import ForwardContext
from vllm.platforms.interface import Platform
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_request, create_scheduler
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)

class FakeMorIIOWrapper():

    def __init__(self, *args, **kwargs):
        pass


    def set_moriio_engine(self, moriio_engine):
        pass

    def set_backend_type(self, backend_type):
        pass

    def get_agent_metadata(self):
        pass

    def register_remote_engine(self, remote_packed_engine_metadata):
        pass

    def register_local_tensor(self, tensor: torch.Tensor):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        try:
            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(
                tensor
            )
            assert self.local_memory_metadata is not None, (
                "register_torch_tensor returned None"
            )
            local_memory_metadata_packed = self.local_memory_metadata.pack()
        except Exception as e:
            raise MoRIIOError(f"Failed to register local memory: {e}") from e
        self.local_memory_registered = True
        return local_memory_metadata_packed

    def get_unpack_memory_metadata(self, packed_memory_metadata):
        pass

    def build_session(self, local_memory_metadata, remote_memory_metadata):
        pass

    def read_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):  
        pass

    def write_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):  
        pass


    def write_remote_data_single(
        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0
    ):  
        pass

    def waiting_for_transfer_complete(self):
        pass

    def async_wait_reqid(self):
        pass

    def _handle_message(self, msg: bytes):
        pass

    def _handle_structured_message(self, data: dict):
        pass

    def _handle_completion_message(self, msg: str):
        pass

    def send_notify(self, req_ids, remote_ip, remote_port):
        pass

    def pop_finished_req_ids(self):
        pass

    def pop_finished_write_req_ids(self):
        pass

    def shutdown(self):
        pass


class FakeMoriIOConnectorWorker(MoRIIOConnectorWorker):
    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(
        self, *args, hand_shake_latency: float = 1.8, kv_cache_layout="HND", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._hand_shake_latency = hand_shake_latency
        self.kv_cache_layout = kv_cache_layout


from vllm.v1.core.sched.scheduler import Scheduler, SchedulerOutput

def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
    role="kv_consumer"
    # role="kv_producer"

) -> VllmConfig:
    """Initialize VllmConfig For Testing."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="MoRIIOConnector",
        kv_role=role,
        enable_permute_local_kv=enable_permute_local_kv,
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size, 1, 1, torch.float32, False)
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
    )

@pytest.fixture
def moriio_read_mode():
    """Set VLLM_MORIIO_CONNECTOR_READ_MODE environment variable for all tests."""
    os.environ['VLLM_MORIIO_CONNECTOR_READ_MODE'] = 'True'
    yield
    # Cleanup after test
    os.environ.pop('VLLM_MORIIO_CONNECTOR_READ_MODE', None)

def test_write_mode_basic_interface():
    """Unit test for basic MoriioConnector interface functionality."""
  
    # Test Prefill wirte metadata
    vllm_config = create_vllm_config(role="kv_consumer")
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
        do_remote_prefill=False
    )
    request_id = request.request_id

    scheduler.add_request(request)
    
    # Fake
    request.kv_transfer_params['remote_notify_port']=4789
    request.kv_transfer_params['remote_block_ids']=None
    request.kv_transfer_params["remote_host"]="127.0.0.1"
    request.kv_transfer_params["remote_port"]=4789
    request.kv_transfer_params["remote_handshake_port"]=4789
    request.kv_transfer_params["remote_engine_id"]="test_engine"
    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MoRIIOConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_save) == 1
    assert len(kv_connector_metadata.reqs_to_recv) == 0
    assert len(kv_connector_metadata.reqs_to_send) == 0
    assert request_id in kv_connector_metadata.reqs_to_save
    req_meta = kv_connector_metadata.reqs_to_save[request_id]

    for block_id, block in zip(
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_write_mode_chunk_prefill():
    """Unit test for basic MoriioConnector interface functionality."""
    MAX_NUM_BATCHED_TOKENS=64

    NUM_TOKENS = MAX_NUM_BATCHED_TOKENS*2+MAX_NUM_BATCHED_TOKENS//2

    # Test Prefill wirte metadata
    vllm_config = create_vllm_config(max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, role="kv_consumer")
    BLOCK_SIZE = vllm_config.cache_config.block_size

    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
  
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
        do_remote_prefill=False
    )
    request_id = request.request_id

    scheduler.add_request(request)
    
    # Fake
    request.kv_transfer_params['remote_notify_port']=4789
    request.kv_transfer_params['remote_block_ids']=None
    request.kv_transfer_params["remote_host"]="127.0.0.1"
    request.kv_transfer_params["remote_port"]=4789
    request.kv_transfer_params["remote_handshake_port"]=4789
    request.kv_transfer_params["remote_engine_id"]="test_engine"
    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MoRIIOConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_save) == 1
    assert len(kv_connector_metadata.reqs_to_recv) == 0
    assert len(kv_connector_metadata.reqs_to_send) == 0
    assert request_id in kv_connector_metadata.reqs_to_save
    req_meta = kv_connector_metadata.reqs_to_save[request_id]

    for block_id, block in zip(
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id

def test_read_mode_basic_interface(moriio_read_mode):
    
    # test decode read
    vllm_config = create_vllm_config(role="kv_consumer")
    scheduler = create_scheduler(vllm_config)
    #
    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=False,
        do_remote_prefill=True
    )
    request_id = request.request_id

    scheduler.add_request(request)
    block_list = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id]
    # Fake
    request.kv_transfer_params['remote_notify_port']=4789
    request.kv_transfer_params['remote_block_ids']=block_list
    request.kv_transfer_params["remote_host"]="127.0.0.1"
    request.kv_transfer_params["remote_port"]=4789
    request.kv_transfer_params["remote_handshake_port"]=4789
    request.kv_transfer_params["remote_engine_id"]="test_engine"
    # Remote Prefill, triggers MorIIOConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MoRIIOConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_save) == 0
    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert len(kv_connector_metadata.reqs_to_send) == 0
    # assert len(kv_connector_metadata.reqs_to_save) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv
    req_meta = kv_connector_metadata.reqs_to_recv[request_id]

    for block_id, block in zip(
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_register_kv_caches():
    from vllm.utils.network_utils import get_ip

    ROLE="kv_consumer"
    IP=get_ip()
    DEFAULT_PORT=6301
    vllm_config = create_vllm_config(role=ROLE)
    TP_RANK=0
    DP_RANK=0
    from  vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
    backend_cls = AiterFlashAttentionBackend

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
    expected_tensor_size = shared_tensor[0].element_size() * shared_tensor[0].numel()
    expected_base_addrs = [
        shared_tensor[0].data_ptr(),
        unique_tensor[1].data_ptr(),
        shared_tensor[0].data_ptr(),

    ]
    mock_group = MagicMock()
    mock_group.rank = TP_RANK  # 设置 rank
    mock_group.local_rank = TP_RANK 
    mock_group.world_size = 1  # 设置 world_size
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.MoRIIOWrapper"
        ) as mock_moriio_wrapper,
        patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common.get_tensor_model_parallel_rank",
        return_value=0
        ),
          patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common.get_tensor_model_parallel_world_size",
        return_value=0
        ),
         patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_tensor_model_parallel_world_size",
        return_value=0
        ),
        patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_world_group",
        return_value=mock_group
        ),
         patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_tp_group",
        return_value=mock_group
        ),
        patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.MoRIIOWrapper",
        FakeMorIIOWrapper,
        )
      
    ):
        # Create connector
        vllm_config.kv_transfer_config.kv_connector_extra_config.update({
            "proxy_ip": "127.0.0.1",
            "proxy_ping_port": 12345,
            "http_port": 12346,
        })
      
        connector = MoRIIOConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeMoriIOConnectorWorker(
            vllm_config, connector.engine_id, hand_shake_latency=0
        )

        # Get the mock instance
        mock_wrapper_instance = mock_moriio_wrapper.return_value
        # connector.connector_worker.moriio_wrapper = mock_wrapper_instance

        # Reassure the shutdown() check that the thread is terminated
        # mock_thread.return_value.is_alive.return_value = False
        from mori.io import (
            EngineDesc,
            IOEngine,
            MemoryDesc,
            PollCqMode,
            RdmaBackendConfig,
        )
        # Execute register_kv_caches
        connector.register_kv_caches(kv_caches)
        shared_tensor[0].data_ptr()
        unique_tensor[1].data_ptr()
        shared_tensor[0].data_ptr()

        assert         shared_tensor.data_ptr()==MemoryDesc.unpack(connector.connector_worker.layer_name_to_local_kv_cache_metadata["layer0"][0]).data
        assert         unique_tensor.data_ptr()==MemoryDesc.unpack(connector.connector_worker.layer_name_to_local_kv_cache_metadata["layer1"][0]).data
        assert         shared_tensor.data_ptr()==MemoryDesc.unpack(connector.connector_worker.layer_name_to_local_kv_cache_metadata["layer2"][0]).data
        expected_engine_key = f"{ROLE[3:]}:{IP}:{DEFAULT_PORT}:tp{TP_RANK}:dp{DP_RANK}"

        assert MemoryDesc.unpack(connector.connector_worker.layer_name_to_local_kv_cache_metadata["layer0"][0]).engine_key ==expected_engine_key

def test_moriio_handshake():
    from vllm.utils.network_utils import get_ip

    ROLE="kv_consumer"
    IP=get_ip()
    DEFAULT_PORT=6301
    vllm_config = create_vllm_config(role=ROLE)
    TP_RANK=0
    DP_RANK=0
    from  vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
    backend_cls = AiterFlashAttentionBackend

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
    expected_tensor_size = shared_tensor[0].element_size() * shared_tensor[0].numel()
    expected_base_addrs = [
        shared_tensor[0].data_ptr(),
        unique_tensor[1].data_ptr(),
        shared_tensor[0].data_ptr(),

    ]
    mock_group = MagicMock()
    mock_group.rank = TP_RANK 
    mock_group.local_rank = TP_RANK 
    mock_group.world_size = 1 
    with (
        patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common.get_tensor_model_parallel_rank",
        return_value=0
        ),
          patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common.get_tensor_model_parallel_world_size",
        return_value=0
        ),
         patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_tensor_model_parallel_world_size",
        return_value=0
        ),
        patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_world_group",
        return_value=mock_group
        ),
         patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_tp_group",
        return_value=mock_group
        ),
        patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.MoRIIOWrapper",
        FakeMorIIOWrapper,
        )
      
    ):
        # Create connector
        vllm_config.kv_transfer_config.kv_connector_extra_config.update({
            "proxy_ip": "127.0.0.1",
            "proxy_ping_port": 12345,
            "http_port": 12346,
            "handshake_port":5670
        })
        

            
        connector = MoRIIOConnector(vllm_config, KVConnectorRole.WORKER)
   
        from vllm.utils.network_utils import (
            get_ip,
            make_zmq_path,
            make_zmq_socket,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import zmq_ctx
        import zmq


        # Reassure the shutdown() check that the thread is terminated
        # mock_thread.return_value.is_alive.return_value = False
        from mori.io import (
            EngineDesc,
            IOEngine,
            MemoryDesc,
            PollCqMode,
            RdmaBackendConfig,
        )
        # Execute register_kv_caches
        connector.register_kv_caches(kv_caches)
        # connector.layer_name_to_local_kv_cache_metadata["layer0"]  expected_base_addrs = [
            
        path = make_zmq_path("tcp", "127.0.0.1", 5670)
        with zmq_ctx(zmq.DEALER, path) as sock:
            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = sock.recv_multipart()

            if len(received_frame) != 2 or received_frame[0] != b"":
                raise HandshakeError(f"Unexpected frame! {received_frame = }")

            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            assert isinstance(metadata, MoRIIOAgentMetadata)
          
     