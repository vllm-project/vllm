# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer engine backends.

Unit tests for engine classes (parsing, validation, registry).
Integration tests for NCCL and IPC weight transfer between processes using Ray.
"""

from unittest.mock import MagicMock

import pytest
import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_open_port


def create_mock_parallel_config(
    rank: int = 0,
    world_size: int = 1,
    dp_rank: int = 0,
) -> ParallelConfig:
    """Create a mock ParallelConfig for testing."""
    config = MagicMock(spec=ParallelConfig)
    config.rank = rank
    config.world_size = world_size
    config.data_parallel_rank = dp_rank
    return config


# --- Unit Tests: NCCLWeightTransferUpdateInfo Validation ---


class TestNCCLWeightTransferUpdateInfoValidation:
    """Test NCCLWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        """Test creating valid NCCLWeightTransferUpdateInfo."""
        info = NCCLWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["float32", "float32"],
            shapes=[[10, 10], [10]],
        )
        assert info.names == ["layer.weight", "layer.bias"]
        assert info.dtype_names == ["float32", "float32"]
        assert info.shapes == [[10, 10], [10]]

    def test_mismatched_dtype_names_raises(self):
        """Test that mismatched dtype_names length raises ValueError."""
        with pytest.raises(ValueError, match="dtype_names"):
            NCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],  # Only one dtype
                shapes=[[10, 10], [10]],
            )

    def test_mismatched_shapes_raises(self):
        """Test that mismatched shapes length raises ValueError."""
        with pytest.raises(ValueError, match="shapes"):
            NCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10]],  # Only one shape
            )

    def test_empty_lists_valid(self):
        """Test that empty lists are valid."""
        info = NCCLWeightTransferUpdateInfo(
            names=[],
            dtype_names=[],
            shapes=[],
        )
        assert len(info.names) == 0


# --- Unit Tests: Engine Parsing ---


class TestNCCLEngineParsing:
    """Test NCCLWeightTransferEngine parsing methods."""

    def test_parse_init_info_valid(self):
        """Test parsing valid init info dict."""
        config = WeightTransferConfig(backend="nccl")
        parallel_config = create_mock_parallel_config()
        engine = NCCLWeightTransferEngine(config, parallel_config)

        init_info = engine.parse_init_info(
            {
                "master_address": "127.0.0.1",
                "master_port": 12345,
                "rank_offset": 1,
                "world_size": 3,
            }
        )

        assert isinstance(init_info, NCCLWeightTransferInitInfo)
        assert init_info.master_address == "127.0.0.1"
        assert init_info.master_port == 12345
        assert init_info.rank_offset == 1
        assert init_info.world_size == 3

    def test_parse_init_info_missing_field_raises(self):
        """Test parsing init info with missing required field."""
        config = WeightTransferConfig(backend="nccl")
        parallel_config = create_mock_parallel_config()
        engine = NCCLWeightTransferEngine(config, parallel_config)

        with pytest.raises(ValueError, match="Invalid init_info"):
            engine.parse_init_info(
                {
                    "master_address": "127.0.0.1",
                    # Missing master_port, rank_offset, world_size
                }
            )

    def test_parse_update_info_valid(self):
        """Test parsing valid update info dict."""
        config = WeightTransferConfig(backend="nccl")
        parallel_config = create_mock_parallel_config()
        engine = NCCLWeightTransferEngine(config, parallel_config)

        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
            }
        )

        assert isinstance(update_info, NCCLWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert update_info.dtype_names == ["float32", "bfloat16"]
        assert update_info.shapes == [[100, 100], [50]]


# --- Unit Tests: Engine Registry ---


class TestEngineRegistry:
    """Test weight transfer engine registry."""

    def test_create_engine_nccl(self):
        """Test factory creates NCCL engine."""
        config = WeightTransferConfig(backend="nccl")
        parallel_config = create_mock_parallel_config()
        engine = WeightTransferEngineFactory.create_engine(config, parallel_config)
        assert isinstance(engine, NCCLWeightTransferEngine)

    def test_create_engine_ipc(self):
        """Test factory creates IPC engine."""
        config = WeightTransferConfig(backend="ipc")
        parallel_config = create_mock_parallel_config()
        engine = WeightTransferEngineFactory.create_engine(config, parallel_config)
        assert isinstance(engine, IPCWeightTransferEngine)

    def test_create_engine_invalid_backend(self):
        """Test factory raises for invalid backend."""
        # Pydantic validates Literal types at construction, so we can't create
        # a config with an invalid backend. Instead, we test by directly
        # accessing the registry or using model_construct to bypass validation.
        from pydantic import ValidationError

        # Test that Pydantic prevents invalid backend at construction
        with pytest.raises(ValidationError):
            WeightTransferConfig(backend="invalid")

        # Test factory error by creating a config with valid backend but
        # then manually modifying the backend attribute (bypassing validation)
        config = WeightTransferConfig(backend="nccl")
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(config, "backend", "invalid")
        parallel_config = create_mock_parallel_config()
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferEngineFactory.create_engine(config, parallel_config)

    def test_register_duplicate_raises(self):
        """Test registering duplicate engine name raises."""
        with pytest.raises(ValueError, match="already registered"):
            WeightTransferEngineFactory.register_engine(
                "nccl", NCCLWeightTransferEngine
            )


# --- Test receive_weights without init raises ---


def test_nccl_receive_weights_without_init_raises():
    """Test that receive_weights raises if init_transfer_engine wasn't called."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="nccl")
    parallel_config = create_mock_parallel_config()
    engine = NCCLWeightTransferEngine(config, parallel_config)

    update_info = NCCLWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10]],
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(update_info, lambda x: None)


# --- Integration Test: NCCL Weight Transfer Between Ray Tasks ---


@ray.remote(num_gpus=1)
def trainer_broadcast_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> bool:
    """Trainer task that broadcasts a tensor via NCCL."""
    import torch

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    # Create process group as rank 0 (trainer)
    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
    )
    # Ray sets CUDA_VISIBLE_DEVICES, so device 0 is the assigned GPU
    comm = PyNcclCommunicator(pg, device=0)

    # Create and broadcast the tensor
    dtype = getattr(torch, tensor_dtype)
    tensor_to_send = torch.ones(tensor_shape, dtype=dtype, device="cuda:0")
    comm.broadcast(tensor_to_send, src=0, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    return True


@ray.remote(num_gpus=1)
def inference_receive_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> dict:
    """Inference task that receives tensor via NCCLWeightTransferEngine."""
    from unittest.mock import MagicMock

    import torch

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferEngine,
        NCCLWeightTransferInitInfo,
        NCCLWeightTransferUpdateInfo,
    )

    # Create engine with mock parallel config
    config = WeightTransferConfig(backend="nccl")
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0

    engine = NCCLWeightTransferEngine(config, parallel_config)

    # Initialize the engine (joins as rank 1)
    init_info = NCCLWeightTransferInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,  # Trainer is rank 0, we become rank 1
        world_size=world_size,
    )
    engine.init_transfer_engine(init_info)

    # Receive weights with a no-op load_weights that captures the tensor
    received_tensors = []

    def noop_load_weights(weights: list[tuple[str, torch.Tensor]]):
        for name, tensor in weights:
            # Clone tensor to keep it after engine cleans up
            received_tensors.append((name, tensor.clone()))

    update_info = NCCLWeightTransferUpdateInfo(
        names=["test.weight"],
        dtype_names=[tensor_dtype],
        shapes=[tensor_shape],
    )
    engine.receive_weights(update_info, noop_load_weights)
    torch.cuda.synchronize()

    # Verify we received the tensor
    success = False
    received_shape = None
    received_sum = None

    if len(received_tensors) == 1:
        name, tensor = received_tensors[0]
        received_shape = list(tensor.shape)
        received_sum = tensor.sum().item()
        # Check shape matches and values are all 1s (trainer sends ones)
        if received_shape == tensor_shape:
            expected_sum = 1.0 * torch.tensor(tensor_shape).prod().item()
            if abs(received_sum - expected_sum) < 0.01:
                success = True

    engine.shutdown()

    return {
        "success": success,
        "received_shape": received_shape,
        "received_sum": received_sum,
    }


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL weight transfer test.",
)
def test_nccl_weight_transfer_between_processes():
    """Test NCCL weight transfer from trainer to inference process using Ray.

    This test verifies that the NCCLWeightTransferEngine can receive
    tensors broadcast by a trainer process via NCCL.
    """
    ray.init(ignore_reinit_error=True)

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2  # 1 trainer + 1 inference worker

    # Tensor to transfer: 100x100 ones
    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    # Start both tasks concurrently - Ray assigns GPUs automatically
    inference_future = inference_receive_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )
    trainer_future = trainer_broadcast_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )

    # Wait for both to complete
    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )


# --- Unit Tests: IPCWeightTransferUpdateInfo Validation ---


class TestIPCWeightTransferUpdateInfoValidation:
    """Test IPCWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        """Test creating valid IPCWeightTransferUpdateInfo."""
        if torch.cuda.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        # Create a dummy tensor and IPC handle
        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        info = IPCWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            ipc_handles=ipc_handles,
        )
        assert info.names == ["layer.weight"]
        assert info.dtype_names == ["float32"]
        assert info.shapes == [[10, 10]]
        assert len(info.ipc_handles) == 1

    def test_mismatched_dtype_names_raises(self):
        """Test that mismatched dtype_names length raises ValueError."""
        if torch.cuda.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}, {gpu_uuid: ipc_handle}]

        with pytest.raises(ValueError, match="dtype_names"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],  # Only one dtype
                shapes=[[10, 10], [10]],
                ipc_handles=ipc_handles,
            )

    def test_mismatched_shapes_raises(self):
        """Test that mismatched shapes length raises ValueError."""
        if torch.cuda.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}, {gpu_uuid: ipc_handle}]

        with pytest.raises(ValueError, match="shapes"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10]],  # Only one shape
                ipc_handles=ipc_handles,
            )

    def test_mismatched_ipc_handles_raises(self):
        """Test that mismatched ipc_handles length raises ValueError."""
        if torch.cuda.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]  # Only one handle

        with pytest.raises(ValueError, match="ipc_handles"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10], [10]],
                ipc_handles=ipc_handles,
            )

    def test_empty_lists_valid(self):
        """Test that empty lists are valid."""
        info = IPCWeightTransferUpdateInfo(
            names=[],
            dtype_names=[],
            shapes=[],
            ipc_handles=[],
        )
        assert len(info.names) == 0


# --- Unit Tests: IPC Engine Parsing ---


class TestIPCEngineParsing:
    """Test IPCWeightTransferEngine parsing methods."""

    def test_parse_update_info_valid(self):
        """Test parsing valid update info dict."""
        if torch.cuda.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        config = WeightTransferConfig(backend="ipc")
        parallel_config = create_mock_parallel_config()
        engine = IPCWeightTransferEngine(config, parallel_config)

        # Create dummy IPC handles
        dummy_tensor1 = torch.ones(100, 100, device="cuda:0")
        dummy_tensor2 = torch.ones(50, device="cuda:0")
        ipc_handle1 = reduce_tensor(dummy_tensor1)
        ipc_handle2 = reduce_tensor(dummy_tensor2)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle1}, {gpu_uuid: ipc_handle2}]

        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
                "ipc_handles": ipc_handles,
            }
        )

        assert isinstance(update_info, IPCWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert update_info.dtype_names == ["float32", "bfloat16"]
        assert update_info.shapes == [[100, 100], [50]]
        assert len(update_info.ipc_handles) == 2


# --- Integration Test: IPC Weight Transfer Between Ray Tasks ---


def get_physical_gpu_id(device_index: int = 0) -> str:
    """Get physical GPU UUID for a device."""
    props = torch.cuda.get_device_properties(device_index)
    return str(props.uuid)


@ray.remote(num_gpus=0.5)
class TrainerActor:
    """Trainer actor that creates and holds CUDA IPC handles."""

    def __init__(self, tensor_shape: list[int], tensor_dtype: str):
        import torch

        # Create tensor on GPU and keep it alive
        dtype = getattr(torch, tensor_dtype)
        self.tensor = torch.ones(tensor_shape, dtype=dtype, device="cuda:0")
        self.tensor.fill_(42.0)  # Fill with 42 to verify correct transfer

        # Create IPC handle (tensor must stay alive for IPC to work)
        ipc_handle = reduce_tensor(self.tensor)
        gpu_uuid = get_physical_gpu_id(0)

        torch.cuda.synchronize()

        self.ipc_handle_dict = {
            "ipc_handle": ipc_handle,
            "gpu_uuid": gpu_uuid,
            "shape": tensor_shape,
            "dtype": tensor_dtype,
        }

    def get_ipc_handle_dict(self) -> dict:
        """Return IPC handle dict. Tensor stays alive in this actor."""
        return self.ipc_handle_dict


@ray.remote(num_gpus=0.5)
def inference_receive_ipc_tensor(
    ipc_handle_dict: dict,
) -> dict:
    """Inference task that receives tensor via IPCWeightTransferEngine."""
    from unittest.mock import MagicMock

    import torch

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCWeightTransferEngine,
        IPCWeightTransferUpdateInfo,
    )

    # Create engine with mock parallel config
    config = WeightTransferConfig(backend="ipc")
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0

    engine = IPCWeightTransferEngine(config, parallel_config)

    # Initialize the engine (no-op for IPC)
    init_info = IPCWeightTransferInitInfo()
    engine.init_transfer_engine(init_info)

    # Receive weights with a no-op load_weights that captures the tensor
    received_tensors = []

    def noop_load_weights(weights: list[tuple[str, torch.Tensor]]):
        for name, tensor in weights:
            # Clone tensor to keep it after engine cleans up
            received_tensors.append((name, tensor.clone()))

    # Create update info with IPC handle
    update_info = IPCWeightTransferUpdateInfo(
        names=["test.weight"],
        dtype_names=[ipc_handle_dict["dtype"]],
        shapes=[ipc_handle_dict["shape"]],
        ipc_handles=[{ipc_handle_dict["gpu_uuid"]: ipc_handle_dict["ipc_handle"]}],
    )
    engine.receive_weights(update_info, noop_load_weights)
    torch.cuda.synchronize()

    # Verify we received the tensor
    success = False
    received_shape = None
    received_sum = None

    if len(received_tensors) == 1:
        name, tensor = received_tensors[0]
        received_shape = list(tensor.shape)
        received_sum = tensor.sum().item()
        # Check shape matches and values are all 42s (trainer sends 42s)
        if received_shape == ipc_handle_dict["shape"]:
            expected_sum = 42.0 * torch.tensor(ipc_handle_dict["shape"]).prod().item()
            if abs(received_sum - expected_sum) < 0.01:
                success = True

    engine.shutdown()

    return {
        "success": success,
        "received_shape": received_shape,
        "received_sum": received_sum,
    }


@pytest.mark.skipif(
    torch.cuda.device_count() < 1,
    reason="Need at least 1 GPU to run IPC weight transfer test.",
)
def test_ipc_weight_transfer_between_processes():
    """Test IPC weight transfer from trainer to inference process using Ray.

    This test verifies that the IPCWeightTransferEngine can receive
    tensors via CUDA IPC handles from a trainer process.
    Note: IPC requires processes to be on the same GPU/node.
    We use a placement group to ensure both processes are on the same GPU.
    """
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init(ignore_reinit_error=True)

    # Create a placement group to ensure both processes are on the same GPU
    # Use fractional GPUs so both tasks can share the same GPU bundle
    pg = placement_group([{"GPU": 1, "CPU": 2}])
    ray.get(pg.ready())

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
    )

    # Tensor to transfer: 100x100 filled with 42s
    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    # Create trainer actor that holds the tensor and IPC handle (stays alive)
    trainer_actor = TrainerActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=scheduling_strategy
    ).remote(tensor_shape, tensor_dtype)

    # Get IPC handle dict (tensor stays alive in trainer actor)
    ipc_handle_dict = ray.get(trainer_actor.get_ipc_handle_dict.remote())

    # Receive tensor in inference process using IPC handles (on same GPU)
    # Trainer actor stays alive during this operation
    inference_result = ray.get(
        inference_receive_ipc_tensor.options(
            scheduling_strategy=scheduling_strategy
        ).remote(ipc_handle_dict)
    )

    assert inference_result["success"], (
        f"IPC weight transfer failed. "
        f"Received shape: {inference_result['received_shape']}, "
        f"Received sum: {inference_result['received_sum']}"
    )


def test_ipc_receive_weights_missing_gpu_uuid_raises():
    """Test that receive_weights raises if GPU UUID not found in IPC handles."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="ipc")
    parallel_config = create_mock_parallel_config()
    engine = IPCWeightTransferEngine(config, parallel_config)

    # Create IPC handle with wrong GPU UUID
    dummy_tensor = torch.ones(10, 10, device="cuda:0")
    ipc_handle = reduce_tensor(dummy_tensor)
    wrong_uuid = "wrong-uuid-12345"
    ipc_handles = [{wrong_uuid: ipc_handle}]

    update_info = IPCWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10, 10]],
        ipc_handles=ipc_handles,
    )

    with pytest.raises(ValueError, match="IPC handle not found"):
        engine.receive_weights(update_info, lambda x: None)
