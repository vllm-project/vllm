# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer engine backends.

Unit tests for engine classes (parsing, validation, registry).
Integration tests for NCCL and IPC weight transfer between processes using Ray.
"""

import pickle
from dataclasses import asdict
from unittest.mock import MagicMock

import pybase64 as base64
import pytest
import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    RayVLLMWeightSyncClient,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
    layerwise_groups,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    assign_producer_indices,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseNCCLTrainerWeightTransferEngine,
    SparseNCCLWeightTransferEngine,
    SparseNCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _init_ray_for_weight_transfer() -> None:
    if ray.is_initialized():
        return
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            }
        },
    )


def _get_ray_assigned_device() -> torch.device:
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{int(gpu_ids[0])}")


def _set_ray_assigned_device() -> torch.device:
    device = _get_ray_assigned_device()
    current_platform.set_device(device)
    return device


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
    config.data_parallel_index = dp_rank
    return config


def create_mock_vllm_config(
    rank: int = 0,
    world_size: int = 1,
    dp_rank: int = 0,
) -> MagicMock:
    """Create a mock VllmConfig exposing parallel_config and model_config."""
    vllm_config = MagicMock()
    vllm_config.parallel_config = create_mock_parallel_config(rank, world_size, dp_rank)
    vllm_config.model_config = MagicMock()
    return vllm_config


# --- Unit Tests: NCCLWeightTransferUpdateInfo Validation ---


class TestNCCLWeightTransferUpdateInfoValidation:
    """Test NCCLWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        info = NCCLWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["float32", "float32"],
            shapes=[[10, 10], [10]],
        )
        assert info.names == ["layer.weight", "layer.bias"]
        assert info.dtype_names == ["float32", "float32"]
        assert info.shapes == [[10, 10], [10]]

    def test_mismatched_dtype_names_raises(self):
        with pytest.raises(ValueError, match="dtype_names"):
            NCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],  # Only one dtype
                shapes=[[10, 10], [10]],
            )

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="shapes"):
            NCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10]],  # Only one shape
            )

    def test_empty_lists_valid(self):
        info = NCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[])
        assert len(info.names) == 0


# --- Unit Tests: SparseNCCLWeightTransferUpdateInfo Validation ---


class TestSparseNCCLWeightTransferUpdateInfoValidation:
    """Test SparseNCCLWeightTransferUpdateInfo dataclass validation."""

    def test_valid_sparse_update_info(self):
        info = SparseNCCLWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["float32", "bfloat16"],
            shapes=[[10, 10], [10]],
            num_updates_list=[4, 2],
        )
        assert info.num_updates_list == [4, 2]

    def test_mismatched_dtype_names_raises(self):
        with pytest.raises(ValueError, match="dtype_names"):
            SparseNCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],
                shapes=[[10, 10], [10]],
                num_updates_list=[4, 2],
            )

    def test_rejects_empty_num_updates_list(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            SparseNCCLWeightTransferUpdateInfo(
                names=[],
                dtype_names=[],
                shapes=[],
                num_updates_list=[],
            )

    def test_rejects_mismatched_num_updates(self):
        with pytest.raises(ValueError, match="`num_updates_list`"):
            SparseNCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10], [10]],
                num_updates_list=[3],
            )

    def test_rejects_negative_num_updates(self):
        with pytest.raises(ValueError, match="non-negative"):
            SparseNCCLWeightTransferUpdateInfo(
                names=["layer.weight"],
                dtype_names=["float32"],
                shapes=[[10, 10]],
                num_updates_list=[-1],
            )


# --- Unit Tests: Engine Parsing ---


class TestNCCLEngineParsing:
    """Test NCCLWeightTransferEngine parsing methods."""

    def _make_engine(self):
        config = WeightTransferConfig(backend="nccl")
        return NCCLWeightTransferEngine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )

    def test_parse_init_info_valid(self):
        engine = self._make_engine()
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
        engine = self._make_engine()
        with pytest.raises(ValueError, match="Invalid init_info"):
            engine.parse_init_info({"master_address": "127.0.0.1"})

    def test_parse_update_info_valid(self):
        engine = self._make_engine()
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
        config = WeightTransferConfig(backend="nccl")
        engine = WeightTransferEngineFactory.create_engine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        assert isinstance(engine, NCCLWeightTransferEngine)

    def test_create_engine_ipc(self):
        config = WeightTransferConfig(backend="ipc")
        engine = WeightTransferEngineFactory.create_engine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        assert isinstance(engine, IPCWeightTransferEngine)

    def test_create_engine_sparse_nccl(self):
        config = WeightTransferConfig(backend="sparse_nccl")
        engine = WeightTransferEngineFactory.create_engine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        assert isinstance(engine, SparseNCCLWeightTransferEngine)

    def test_create_engine_invalid_backend(self):
        config = WeightTransferConfig(backend="invalid")
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferEngineFactory.create_engine(
                config,
                create_mock_vllm_config(),
                torch.device("cuda"),
                MagicMock(spec=torch.nn.Module),
            )

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            WeightTransferEngineFactory.register_engine(
                "nccl", NCCLWeightTransferEngine
            )


# --- Unit Tests: Sparse patch application (CPU) ---


class TestSparseNCCLPatchApplication:
    """Test SparseNCCLWeightTransferEngine._apply_patch on a real param."""

    def _make_engine(self, model):
        config = WeightTransferConfig(backend="sparse_nccl")
        return SparseNCCLWeightTransferEngine(
            config, create_mock_vllm_config(), torch.device("cpu"), model
        )

    def _make_model(self, numel: int = 8):
        model = torch.nn.Module()
        model.register_parameter(
            "w", torch.nn.Parameter(torch.zeros(numel), requires_grad=False)
        )

        def get_parameter(name):
            assert name == "w"
            return model.w

        model.get_parameter = get_parameter
        return model

    def test_apply_patch_updates_only_selected_entries(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        engine._apply_patch(
            SparseWeightPatch(
                name="w",
                indices=torch.tensor([1, 3], dtype=torch.int32),
                values=torch.tensor([5.0, 7.0], dtype=torch.float32),
            )
        )
        expected = torch.zeros(8)
        expected[1] = 5.0
        expected[3] = 7.0
        assert torch.equal(model.w.data, expected)

    def test_apply_patch_rejects_mismatched_lengths(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        with pytest.raises(ValueError, match="matching lengths"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1, 3], dtype=torch.int32),
                    values=torch.tensor([5.0], dtype=torch.float32),
                )
            )

    def test_apply_patch_rejects_non_int32_indices(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        with pytest.raises(ValueError, match="int32 indices"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1], dtype=torch.int64),
                    values=torch.tensor([5.0], dtype=torch.float32),
                )
            )

    def test_apply_patch_rejects_dtype_mismatch(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        with pytest.raises(ValueError, match="does not match"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1], dtype=torch.int32),
                    values=torch.tensor([5.0], dtype=torch.bfloat16),
                )
            )

    def test_apply_patch_rejects_non_contiguous_param(self):
        model = torch.nn.Module()
        model.register_parameter(
            "w",
            torch.nn.Parameter(
                torch.arange(12, dtype=torch.float32).view(3, 4).t(),
                requires_grad=False,
            ),
        )
        model.get_parameter = lambda name: model.w
        engine = self._make_engine(model)
        with pytest.raises(NotImplementedError, match="contiguous params"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                )
            )


# --- Test receive_weights without init raises ---


def test_nccl_receive_weights_without_init_raises():
    """Test that receive_weights raises if init_transfer_engine wasn't called."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="nccl")
    engine = NCCLWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda"),
        MagicMock(spec=torch.nn.Module),
    )

    update_info = NCCLWeightTransferUpdateInfo(
        names=["w"], dtype_names=["float32"], shapes=[[10]]
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(update_info)


def test_sparse_nccl_receive_weights_without_init_raises():
    """Test that sparse receive raises if init_transfer_engine wasn't called."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="sparse_nccl")
    engine = SparseNCCLWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda"),
        MagicMock(spec=torch.nn.Module),
    )

    update_info = SparseNCCLWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10]],
        num_updates_list=[2],
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(update_info)


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

    device = _set_ray_assigned_device()

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    # Create process group as rank 0 (trainer)
    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
    )
    comm = PyNcclCommunicator(pg, device=device.index)

    # Create and broadcast the tensor
    dtype = getattr(torch, tensor_dtype)
    tensor_to_send = torch.ones(tensor_shape, dtype=dtype, device=device)
    comm.broadcast(tensor_to_send, src=0, stream=torch.cuda.current_stream())
    torch.accelerator.synchronize()

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
    import contextlib
    from unittest.mock import MagicMock

    import torch

    _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferEngine,
        NCCLWeightTransferInitInfo,
        NCCLWeightTransferUpdateInfo,
    )

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    config = WeightTransferConfig(backend="nccl")
    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    recorder = Recorder()
    engine = NCCLWeightTransferEngine(
        config, vllm_config, torch.device("cuda"), recorder
    )
    # Transport-only test: bypass the set_current_vllm_config context that
    # receive_weights enters, since vllm_config here is a mock.
    import vllm.config as _vllm_config_mod

    _vllm_config_mod.set_current_vllm_config = lambda cfg: contextlib.nullcontext()

    # Initialize the engine (joins as rank 1)
    # Trainer broadcasts a single tensor unpacked, so the worker must not
    # expect the packed wire format (packed is a must-agree wire param shipped
    # on the init info).
    init_info = NCCLWeightTransferInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,  # Trainer is rank 0, we become rank 1
        world_size=world_size,
        packed=False,
    )
    engine.init_transfer_engine(init_info)

    update_info = NCCLWeightTransferUpdateInfo(
        names=["test.weight"],
        dtype_names=[tensor_dtype],
        shapes=[tensor_shape],
    )
    engine.receive_weights(update_info)
    torch.accelerator.synchronize()

    # Verify we received the tensor
    success = False
    received_shape = None
    received_sum = None

    if len(recorder.received) == 1:
        name, tensor = recorder.received[0]
        received_shape = list(tensor.shape)
        received_sum = tensor.sum().item()
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
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL weight transfer test.",
)
def test_nccl_weight_transfer_between_processes():
    """Test NCCL weight transfer from trainer to inference process using Ray.

    This test verifies that the NCCLWeightTransferEngine can receive
    tensors broadcast by a trainer process via NCCL.
    """
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2  # 1 trainer + 1 inference worker

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    inference_future = inference_receive_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )
    trainer_future = trainer_broadcast_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )

    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )


@ray.remote(num_gpus=1)
def trainer_broadcast_sparse_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
) -> bool:
    """Trainer task that broadcasts sparse patches via the trainer engine.

    The worker task drives its own init/receive directly (it is not an RPC
    endpoint), so the engine gets a no-op control-plane client; the NCCL
    rendezvous and the patch broadcasts are the real thing.
    """
    import torch

    device = _set_ray_assigned_device()

    from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.sparse_nccl_engine import (
        SparseNCCLTrainerInitInfo,
        SparseWeightPatch,
    )

    class NoopClient:
        def init_weight_transfer_engine(self, init_info):
            pass

        def start_weight_update(self):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self):
            pass

    patch = SparseWeightPatch(
        name="test.weight",
        indices=torch.tensor([1, 7, 25], dtype=torch.int32, device=device),
        values=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device=device),
        full_shape=(10, 10),
    )
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=SparseNCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            rank=0,
        ),
        client=NoopClient(),
    )
    engine.send_weights([patch])
    torch.accelerator.synchronize()
    engine.shutdown()
    return True


@ray.remote(num_gpus=1)
def inference_receive_sparse_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
) -> dict:
    """Inference task that receives sparse patches via the sparse engine."""
    from unittest.mock import MagicMock

    import torch

    device = _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.sparse_nccl_engine import (
        SparseNCCLWeightTransferEngine,
        SparseNCCLWeightTransferUpdateInfo,
    )

    config = WeightTransferConfig(backend="sparse_nccl")
    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    # Real module holding the target parameter the patch will modify.
    model = torch.nn.Module()
    model.register_parameter(
        "w", torch.nn.Parameter(torch.zeros(30, device="cuda"), requires_grad=False)
    )
    model.get_parameter = lambda name: model.w

    update_info = SparseNCCLWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[30]],
        num_updates_list=[3],
    )

    engine = SparseNCCLWeightTransferEngine(
        config, vllm_config, torch.device("cuda"), model
    )
    from vllm.distributed.weight_transfer.nccl_common import (
        NCCLWeightTransferInitInfo,
    )

    engine.init_transfer_engine(
        NCCLWeightTransferInitInfo(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
    )
    engine.receive_weights(update_info)
    torch.accelerator.synchronize()

    expected = torch.zeros(30, dtype=torch.float32, device=device)
    expected[[1, 7, 25]] = torch.tensor(
        [10.0, 20.0, 30.0], dtype=torch.float32, device=device
    )
    success = torch.equal(model.w.data, expected)
    engine.shutdown()
    return {
        "success": success,
        "selected_values": model.w.data[[1, 7, 25]].cpu().tolist(),
    }


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL sparse weight transfer test.",
)
def test_nccl_sparse_weight_transfer_between_processes():
    """Test NCCL sparse weight transfer from trainer to inference process."""
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2

    inference_future = inference_receive_sparse_tensor.remote(
        master_address, master_port, world_size
    )
    trainer_future = trainer_broadcast_sparse_tensor.remote(
        master_address, master_port, world_size
    )

    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        "Sparse weight transfer failed. "
        f"Received selected values: {result['selected_values']}"
    )


# --- Unit Tests: IPCWeightTransferUpdateInfo Validation ---


class TestIPCWeightTransferUpdateInfoValidation:
    """Test IPCWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
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
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
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
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
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
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]  # Only one handle

        with pytest.raises(ValueError, match="ipc_handles"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10], [10]],
                ipc_handles=ipc_handles,
            )

    def test_valid_update_info_from_pickled(self, monkeypatch):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        info = IPCWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            ipc_handles_pickled=pickled,
        )
        assert info.ipc_handles == ipc_handles
        assert info.ipc_handles_pickled is None

    def test_pickled_requires_insecure_serialization_flag(self, monkeypatch):
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")

        with pytest.raises(ValueError, match="VLLM_ALLOW_INSECURE_SERIALIZATION=1"):
            IPCWeightTransferUpdateInfo(
                names=[],
                dtype_names=[],
                shapes=[],
                ipc_handles_pickled=base64.b64encode(pickle.dumps([])).decode("utf-8"),
            )

    def test_both_handles_and_pickled_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        with pytest.raises(ValueError, match="Cannot specify both"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight"],
                dtype_names=["float32"],
                shapes=[[10, 10]],
                ipc_handles=ipc_handles,
                ipc_handles_pickled=pickled,
            )

    def test_neither_handles_nor_pickled_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight"],
                dtype_names=["float32"],
                shapes=[[10, 10]],
            )

    def test_empty_lists_valid(self):
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

    def _make_engine(self):
        config = WeightTransferConfig(backend="ipc")
        return IPCWeightTransferEngine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )

    def test_parse_update_info_valid(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        engine = self._make_engine()

        dummy_tensor1 = torch.ones(100, 100, device="cuda:0")
        dummy_tensor2 = torch.ones(50, device="cuda:0")
        _, ipc_args1 = reduce_tensor(dummy_tensor1)
        _, ipc_args2 = reduce_tensor(dummy_tensor2)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_args1}, {gpu_uuid: ipc_args2}]

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

    def test_parse_update_info_pickled(self, monkeypatch):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        engine = self._make_engine()

        dummy_tensor1 = torch.ones(100, 100, device="cuda:0")
        dummy_tensor2 = torch.ones(50, device="cuda:0")
        _, ipc_args1 = reduce_tensor(dummy_tensor1)
        _, ipc_args2 = reduce_tensor(dummy_tensor2)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_args1}, {gpu_uuid: ipc_args2}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
                "ipc_handles_pickled": pickled,
            }
        )

        assert isinstance(update_info, IPCWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert len(update_info.ipc_handles) == 2
        assert gpu_uuid in update_info.ipc_handles[0]
        assert gpu_uuid in update_info.ipc_handles[1]

    def test_parse_update_info_ignores_none_pickled_handles(self):
        engine = self._make_engine()
        ipc_handles = [{"gpu-uuid": ("ipc-args",)}]

        update_info = engine.parse_update_info(
            {
                "names": ["w1"],
                "dtype_names": ["float32"],
                "shapes": [[1]],
                "ipc_handles": ipc_handles,
                "ipc_handles_pickled": None,
            }
        )

        assert isinstance(update_info, IPCWeightTransferUpdateInfo)
        assert update_info.ipc_handles == ipc_handles

    def test_parse_update_info_both_handles_and_pickled_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        engine = self._make_engine()

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        with pytest.raises(ValueError, match="Cannot specify both"):
            engine.parse_update_info(
                {
                    "names": ["layer.weight"],
                    "dtype_names": ["float32"],
                    "shapes": [[10, 10]],
                    "ipc_handles": ipc_handles,
                    "ipc_handles_pickled": pickled,
                }
            )


# --- Integration Test: IPC Weight Transfer Between Ray Tasks ---


def get_physical_gpu_id(device_index: int = 0) -> str:
    """Get physical GPU UUID for a device."""
    props = torch.cuda.get_device_properties(device_index)
    return str(props.uuid)


@ray.remote(num_gpus=0.5)
class TrainerActor:
    """Trainer actor that creates and holds CUDA IPC handles."""

    def __init__(self, tensor_shape: list[int], tensor_dtype: str):
        device = _set_ray_assigned_device()

        # Create tensor on GPU and keep it alive
        dtype = getattr(torch, tensor_dtype)
        self.tensor = torch.ones(tensor_shape, dtype=dtype, device=device)
        self.tensor.fill_(42.0)  # Fill with 42 to verify correct transfer

        _, ipc_args = reduce_tensor(self.tensor)
        gpu_uuid = get_physical_gpu_id(device.index)

        torch.accelerator.synchronize()

        self.ipc_handle_dict = {
            "ipc_handle": ipc_args,
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
    mode: str = "ray",
) -> dict:
    """Inference task that receives tensor via IPCWeightTransferEngine."""
    import contextlib
    import os

    # Worker-side: ipc_handles_pickled is deserialized via pickle.
    if mode == "http":
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from unittest.mock import MagicMock

    import torch

    device = _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCWeightTransferEngine,
    )

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    # Trainer sends unpacked IPC handles; the worker learns packed=False from
    # the init handshake below (IPCWeightTransferInitInfo defaults to False).
    config = WeightTransferConfig(backend="ipc")
    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    recorder = Recorder()
    engine = IPCWeightTransferEngine(config, vllm_config, device, recorder)
    # Transport-only test: bypass the set_current_vllm_config context that
    # receive_weights enters, since vllm_config here is a mock.
    import vllm.config as _vllm_config_mod

    _vllm_config_mod.set_current_vllm_config = lambda cfg: contextlib.nullcontext()

    init_info = IPCWeightTransferInitInfo()
    engine.init_transfer_engine(init_info)

    ipc_handles = [{ipc_handle_dict["gpu_uuid"]: ipc_handle_dict["ipc_handle"]}]

    if mode == "ray":
        update_dict: dict = {
            "names": ["test.weight"],
            "dtype_names": [ipc_handle_dict["dtype"]],
            "shapes": [ipc_handle_dict["shape"]],
            "ipc_handles": ipc_handles,
        }
    elif mode == "http":
        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        update_dict = {
            "names": ["test.weight"],
            "dtype_names": [ipc_handle_dict["dtype"]],
            "shapes": [ipc_handle_dict["shape"]],
            "ipc_handles_pickled": pickled,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    update_info = engine.parse_update_info(update_dict)
    engine.receive_weights(update_info)
    torch.accelerator.synchronize()

    success = False
    received_shape = None
    received_sum = None

    if len(recorder.received) == 1:
        name, tensor = recorder.received[0]
        received_shape = list(tensor.shape)
        received_sum = tensor.sum().item()
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
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU to run IPC weight transfer test.",
)
@pytest.mark.parametrize("mode", ["ray", "http"])
def test_ipc_weight_transfer_between_processes(mode: str):
    """Test IPC weight transfer from trainer to inference process using Ray."""
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    _init_ray_for_weight_transfer()

    pg = placement_group([{"GPU": 1, "CPU": 2}])
    ray.get(pg.ready())

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
    )

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    trainer_actor = TrainerActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=scheduling_strategy
    ).remote(tensor_shape, tensor_dtype)

    ipc_handle_dict = ray.get(trainer_actor.get_ipc_handle_dict.remote())

    inference_result = ray.get(
        inference_receive_ipc_tensor.options(
            scheduling_strategy=scheduling_strategy
        ).remote(ipc_handle_dict, mode=mode)
    )

    assert inference_result["success"], (
        f"IPC weight transfer failed (mode={mode}). "
        f"Received shape: {inference_result['received_shape']}, "
        f"Received sum: {inference_result['received_sum']}"
    )


def test_ipc_receive_weights_missing_gpu_uuid_raises():
    """Test that receive_weights raises if GPU UUID not found in IPC handles."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="ipc")
    engine = IPCWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda:0"),
        MagicMock(spec=torch.nn.Module),
    )
    # No init handshake here, so the engine keeps its default packed=False.

    dummy_tensor = torch.ones(10, 10, device="cuda:0")
    _, ipc_handle = reduce_tensor(dummy_tensor)
    wrong_uuid = "wrong-uuid-12345"
    ipc_handles = [{wrong_uuid: ipc_handle}]

    update_info = IPCWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10, 10]],
        ipc_handles=ipc_handles,
    )

    with pytest.raises(ValueError, match="IPC handle not found"):
        engine.receive_weights(update_info)


class RecordingClient:
    """A fake VLLMWeightSyncClient that records the order of calls."""

    def __init__(self):
        self.order: list[str] = []
        self.last_init_info: dict | None = None
        self.last_update_info: dict | None = None

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        self.order.append("init")
        self.last_init_info = init_info

    def start_weight_update(self) -> None:
        self.order.append("start")

    def update_weights(self, update_info: dict) -> None:
        self.order.append("update")
        self.last_update_info = update_info

    def finish_weight_update(self) -> None:
        self.order.append("finish")


def _module_with(*pairs):
    """A tiny nn.Module exposing the given (name, tensor) pairs as parameters,
    so trainer tests can build a ModuleSource without a real model."""
    module = torch.nn.Module()
    for name, tensor in pairs:
        module.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    return module


class _DummyTrainerEngine(TrainerWeightTransferEngine):
    """Minimal concrete trainer engine to exercise base-class + factory."""

    @classmethod
    def trainer_init(cls, init_info, *, client, source):
        return cls(client=client, source=source)

    def send_weights(self):
        pass


class TestTrainerClients:
    """Structural protocol conformance for the built-in clients."""

    def test_recording_client_is_protocol(self):
        assert isinstance(RecordingClient(), VLLMWeightSyncClient)

    def test_http_client_is_protocol(self):
        assert isinstance(
            HTTPVLLMWeightSyncClient("http://localhost:8000"), VLLMWeightSyncClient
        )

    def test_ray_client_is_protocol(self):
        assert isinstance(RayVLLMWeightSyncClient(MagicMock()), VLLMWeightSyncClient)

    def test_ray_client_sends_typed_requests(self, monkeypatch):
        """Ray client must hand the actor typed Request objects, not raw dicts."""
        import ray

        monkeypatch.setattr(ray, "get", lambda refs: None)
        handle = MagicMock()
        client = RayVLLMWeightSyncClient(handle)

        client.init_weight_transfer_engine({"master_addr": "x"})
        (init_req,), _ = handle.init_weight_transfer_engine.remote.call_args
        assert isinstance(init_req, WeightTransferInitRequest)
        assert init_req.init_info == {"master_addr": "x"}

        client.update_weights({"names": ["w"]})
        (update_req,), _ = handle.update_weights.remote.call_args
        assert isinstance(update_req, WeightTransferUpdateRequest)
        assert update_req.update_info == {"names": ["w"]}

    def test_http_client_pickles_ipc_handles_for_json(self, monkeypatch):
        """HTTP update_weights must encode raw ipc_handles as a base64 pickle."""
        captured = {}

        def fake_post(self, path, json=None):
            captured["path"] = path
            captured["json"] = json

        monkeypatch.setattr(HTTPVLLMWeightSyncClient, "_post", fake_post)
        client = HTTPVLLMWeightSyncClient("http://localhost:8000")
        client.update_weights({"names": ["w"], "ipc_handles": [{"gpu": ("args",)}]})
        sent = captured["json"]["update_info"]
        assert "ipc_handles" not in sent
        assert "ipc_handles_pickled" in sent
        assert pickle.loads(base64.b64decode(sent["ipc_handles_pickled"])) == [
            {"gpu": ("args",)}
        ]

    def test_http_client_passes_through_nccl_update_info(self, monkeypatch):
        """NCCL update_info has only JSON-native fields and passes unchanged."""
        captured = {}

        def fake_post(self, path, json=None):
            captured["json"] = json

        monkeypatch.setattr(HTTPVLLMWeightSyncClient, "_post", fake_post)
        client = HTTPVLLMWeightSyncClient("http://localhost:8000")
        update_info = {"names": ["w"], "dtype_names": ["float32"], "shapes": [[4]]}
        client.update_weights(update_info)
        assert captured["json"]["update_info"] == update_info


class TestModuleSource:
    """`ModuleSource` metadata vs. materialized iteration (dense, no GPU)."""

    def test_metadata_reads_shape_and_dtype(self):
        source = ModuleSource(
            _module_with(("w", torch.zeros(2, 3)), ("b", torch.zeros(3)))
        )
        meta = source.metadata()
        assert [m.name for m in meta] == ["w", "b"]
        assert [m.shape for m in meta] == [(2, 3), (3,)]
        assert all(m.dtype == torch.float32 for m in meta)

    def test_iteration_yields_materialized_tensors(self):
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        source = ModuleSource(_module_with(("w", w)))
        pairs = list(source)
        assert [name for name, _ in pairs] == ["w"]
        assert torch.equal(pairs[0][1], w)

    def test_source_is_reiterable(self):
        source = ModuleSource(_module_with(("w", torch.zeros(2))))
        assert [n for n, _ in source] == [n for n, _ in source] == ["w"]


class TestWeightSourceGroupContract:
    """`groups()` / `iter_groups()` on the WeightSource ABC. Group indices are
    what `owned_groups()` names and what backends gather and free by, so the
    default must agree with `layerwise_groups` over `metadata()`."""

    class _Source(WeightSource):
        """Minimal source over an ordered (name, tensor) list, optionally owning
        only some groups (in which case it iterates only those, per contract)."""

        def __init__(self, names, owned=None, reverse=False):
            self._pairs = [(n, torch.full((2,), float(i))) for i, n in enumerate(names)]
            self._owned = owned
            self._reverse = reverse

        def metadata(self):
            return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._pairs]

        def owned_groups(self):
            return self._owned

        def __iter__(self):
            pairs = self._pairs
            if self._owned is not None:
                all_groups = layerwise_groups([n for n, _ in pairs])
                keep = {n for i in self._owned for n in all_groups[i]}
                pairs = [(n, t) for n, t in pairs if n in keep]
            return iter(list(reversed(pairs)) if self._reverse else pairs)

    def _source(self, names, owned=None, reverse=False):
        return self._Source(names, owned, reverse)

    def test_groups_defaults_to_the_layerwise_partition(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a", "norm.w"]
        assert self._source(names).groups() == layerwise_groups(names)

    def test_groups_is_restricted_to_owned_groups(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a", "norm.w"]
        assert self._source(names, owned=[1, 2]).groups() == [
            ["model.layers.0.a"],
            ["model.layers.1.a"],
        ]

    def test_iter_groups_batches_the_stream_per_group(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.0.b", "norm.w"]
        batches = list(self._source(names).iter_groups())
        assert [ns for ns, _ in batches] == [
            ["embed.w"],
            ["model.layers.0.a", "model.layers.0.b"],
            ["norm.w"],
        ]
        assert all(len(ns) == len(ts) for ns, ts in batches)

    def test_iter_groups_yields_the_tensors_iteration_produced(self):
        names = ["model.layers.0.a", "model.layers.0.b"]
        (batch,) = list(self._source(names).iter_groups())
        _names, tensors = batch
        assert [float(t[0]) for t in tensors] == [0.0, 1.0]

    def test_iter_groups_yields_only_owned_groups(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a"]
        batches = list(self._source(names, owned=[2]).iter_groups())
        assert [ns for ns, _ in batches] == [["model.layers.1.a"]]

    def test_out_of_order_iteration_raises(self):
        """Materializing is usually a collective, so a rank that iterates out of
        order deadlocks its peers -- fail loudly instead."""
        source = self._source(["model.layers.0.a", "model.layers.0.b"], reverse=True)
        with pytest.raises(RuntimeError, match="iteration order must match"):
            list(source.iter_groups())

    def test_a_source_may_override_iter_groups(self):
        """The extension point: materialize a whole group in one step instead of
        one generator resume per tensor."""
        calls = []
        base = self._Source

        class _Batched(base):
            def iter_groups(self):
                for group in self.groups():
                    calls.append(len(group))
                    yield group, [torch.zeros(2) for _ in group]

        source = _Batched(["model.layers.0.a", "model.layers.0.b"])
        assert [ns for ns, _ in source.iter_groups()] == [
            ["model.layers.0.a", "model.layers.0.b"]
        ]
        assert calls == [2]


class TestDeferredProcessingContract:
    """`defers_processing` and `drain_pending` are two halves of one contract: a
    caller that takes over the update tail (running its own
    `finalize_layerwise_reload` instead of going through `finish_weight_update`)
    reads the flag and calls the method. Both must be answerable on any engine, or
    that caller ends up reaching through a getattr."""

    def _engines(self):
        return {
            name: loader()
            for name, loader in WeightTransferEngineFactory._registry.items()
        }

    def test_every_engine_declares_whether_it_defers(self):
        for name, cls in self._engines().items():
            assert isinstance(cls.defers_processing, bool), name

    def test_every_engine_can_be_drained(self):
        """The default is a no-op, so a caller never has to check whether the
        method exists before calling it."""
        for name, cls in self._engines().items():
            assert callable(cls.drain_pending), name

    def test_the_default_is_not_to_defer(self):
        assert WeightTransferEngine.defers_processing is False

    def test_a_synchronous_engine_drains_as_a_no_op(self):
        engine = object.__new__(WeightTransferEngineFactory._registry["nccl"]())
        engine.drain_pending()  # must not raise, and must not need any state

    def test_the_rdt_engine_defers_and_overrides_the_drain(self):
        """The one engine the contract exists for."""
        cls = WeightTransferEngineFactory._registry["sharded_rdt"]()
        assert cls.defers_processing is True
        assert cls.drain_pending is not WeightTransferEngine.drain_pending


class TestTrainerFactory:
    """WeightTransferTrainerFactory registry mechanics."""

    def test_registry_has_all_backends(self):
        assert "nccl" in WeightTransferTrainerFactory._registry
        assert "ipc" in WeightTransferTrainerFactory._registry
        assert "sparse_nccl" in WeightTransferTrainerFactory._registry

    def test_register_and_dispatch(self):
        saved = dict(WeightTransferTrainerFactory._registry)
        try:
            WeightTransferTrainerFactory.register_engine("dummy", _DummyTrainerEngine)
            engine = WeightTransferTrainerFactory.trainer_init(
                MagicMock(backend="dummy"),  # backend read from the init info
                client=RecordingClient(),
                source=ModuleSource(_module_with(("w", torch.zeros(2)))),
            )
            assert isinstance(engine, _DummyTrainerEngine)
            with pytest.raises(ValueError, match="already registered"):
                WeightTransferTrainerFactory.register_engine(
                    "dummy", _DummyTrainerEngine
                )
        finally:
            WeightTransferTrainerFactory._registry = saved

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferTrainerFactory.trainer_init(
                MagicMock(backend="nope"),
                client=RecordingClient(),
                source=ModuleSource(_module_with(("w", torch.zeros(2)))),
            )

    def test_ipc_init_info_declares_backend(self):
        assert IPCTrainerInitInfo.backend == "ipc"

    def test_nccl_init_info_declares_backend(self):
        assert NCCLTrainerInitInfo.backend == "nccl"

    def test_sparse_nccl_init_info_declares_backend(self):
        assert SparseNCCLTrainerInitInfo.backend == "sparse_nccl"

    def test_trainer_init_info_subclass_must_set_backend(self):
        with pytest.raises(TypeError, match="class-level `backend`"):

            class _NoBackend(TrainerInitInfo):
                pass


class TestTrainerEngineBase:
    """Base-class construction (no GPU)."""

    def test_source_stored_and_sender_by_default(self):
        engine = _DummyTrainerEngine(
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
        )
        assert engine.is_sender is True
        assert [name for name, _ in engine.source] == ["w"]

    def test_shutdown_default_is_noop(self):
        engine = _DummyTrainerEngine(
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
            is_sender=False,
        )
        assert engine.is_sender is False
        engine.shutdown()  # must not raise


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (CUDA IPC handles).",
)
def test_ipc_trainer_send_weights_drives_client_in_order():
    """send_weights issues start -> update -> finish and ships per-round metadata;
    the packed wire param rides the init info, not the per-round update_info."""
    client = RecordingClient()
    engine = IPCTrainerWeightTransferEngine(
        client=client,
        source=ModuleSource(_module_with(("w", torch.ones(4, device="cuda")))),
        packed=False,
    )

    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4]]
    assert "packed" not in client.last_update_info


def test_ipc_trainer_init_ships_packed_to_worker():
    """trainer_init drives the inference-side init handshake and propagates the
    must-agree `packed` flag to the worker."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU (CUDA IPC handles).")

    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=IPCTrainerInitInfo(rank=0, packed=True),  # backend from init info
        client=client,
        source=ModuleSource(_module_with(("w", torch.ones(4, device="cuda")))),
    )

    assert isinstance(engine, IPCTrainerWeightTransferEngine)
    assert engine.is_sender is True
    assert engine.packed is True
    assert client.order == ["init"]
    assert client.last_init_info == {"packed": True}


def test_nccl_trainer_init_ships_worker_init_info(monkeypatch):
    """The sender's trainer_init drives the inference-side init handshake with
    the worker-shaped init info (rank_offset=1) while opening its own endpoint,
    and propagates the must-agree wire params to the worker."""
    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine_mod

    # Bypass the real NCCL rendezvous.
    monkeypatch.setattr(nccl_engine_mod, "trainer_init", lambda info: MagicMock())

    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=NCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=3,
            rank=0,
            packed=True,
            packed_buffer_size_bytes=1024,
            packed_num_buffers=3,
        ),
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4)))),
    )

    assert isinstance(engine, NCCLTrainerWeightTransferEngine)
    assert engine.is_sender is True
    assert engine.packed is True
    assert client.order == ["init"]
    assert client.last_init_info == {
        "master_address": "127.0.0.1",
        "master_port": 29500,
        "rank_offset": 1,
        "world_size": 3,
        "packed": True,
        "packed_buffer_size_bytes": 1024,
        "packed_num_buffers": 3,
    }


def test_nccl_worker_learns_wire_params_from_init_handshake(monkeypatch):
    """The worker engine reads packed + buffer geometry from the
    trainer-supplied init info at the handshake, not from the config or the
    per-round update info."""
    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine_mod

    monkeypatch.setattr(
        nccl_engine_mod, "worker_init_process_group", lambda info, pc: MagicMock()
    )

    engine = NCCLWeightTransferEngine(
        WeightTransferConfig(backend="nccl"),
        create_mock_vllm_config(),
        torch.device("cuda:0"),
        MagicMock(spec=torch.nn.Module),
    )
    assert engine.packed is False  # pre-handshake default (legacy unpacked)
    engine.init_transfer_engine(
        NCCLWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            rank_offset=1,
            world_size=2,
            packed=True,
            packed_buffer_size_bytes=2048,
            packed_num_buffers=4,
        )
    )

    assert engine.packed is True
    assert engine.packed_buffer_size_bytes == 2048
    assert engine.packed_num_buffers == 4


def test_nccl_trainer_init_non_sender_skips_rendezvous_and_client():
    """Non-sender trainer ranks build an engine without opening an endpoint or
    touching the client; they only join the collectives in send_weights."""
    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=NCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=3,
            rank=1,
        ),
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4)))),
    )

    assert engine.is_sender is False
    assert engine.model_update_group is None
    assert client.order == []

    # send_weights on a non-sender only iterates the source (packed mode needs
    # no CUDA stream on non-senders), never the client.
    engine.send_weights()
    assert client.order == []


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (NCCL broadcast / CUDA stream).",
)
def test_nccl_trainer_send_weights_drives_client_in_order():
    """send_weights issues start -> update -> finish and ships per-round
    metadata; the packed wire params ride the init handshake, not the
    per-round update_info."""
    client = RecordingClient()
    engine = NCCLTrainerWeightTransferEngine(
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4, device="cuda")))),
        packed=False,
    )
    # Bypass the real NCCL rendezvous; broadcast is a no-op.
    engine.model_update_group = MagicMock()

    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4]]
    assert "packed" not in client.last_update_info


def _sparse_patch(device: str = "cpu") -> SparseWeightPatch:
    return SparseWeightPatch(
        name="w",
        indices=torch.tensor([1, 3], dtype=torch.int32, device=device),
        values=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
        full_shape=(4, 4),
    )


def test_sparse_nccl_trainer_init_ships_worker_init_info(monkeypatch):
    """The sender's trainer_init drives the init handshake with the
    worker-shaped init info; sparse ships no packed wire params, so the worker
    keeps its unpacked defaults. Sparse takes no `source`."""
    import vllm.distributed.weight_transfer.sparse_nccl_engine as sparse_mod

    monkeypatch.setattr(sparse_mod, "trainer_init", lambda info: MagicMock())

    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=SparseNCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=2,
            rank=0,
        ),
        client=client,
    )

    assert isinstance(engine, SparseNCCLTrainerWeightTransferEngine)
    assert client.order == ["init"]
    assert client.last_init_info == {
        "master_address": "127.0.0.1",
        "master_port": 29500,
        "rank_offset": 1,
        "world_size": 2,
        "packed": False,
        "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the sparse patch path moves tensors to the device",
)
def test_sparse_nccl_trainer_send_weights_drives_client_in_order():
    """send_weights takes the round's patches and ships per-patch metadata
    (names / shapes / num_updates_list) + broadcasts indices + values each."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()

    engine.send_weights([_sparse_patch()])

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4, 4]]
    assert client.last_update_info["num_updates_list"] == [2]
    # One broadcast for indices + one for values per patch.
    assert engine.model_update_group.broadcast.call_count == 2


def test_sparse_nccl_trainer_send_weights_empty_round_is_noop():
    """A round with no patches must not touch the client (an empty sparse
    update info is invalid by construction)."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()

    engine.send_weights([])
    engine.send_weights()  # no argument is also a no-op round

    assert client.order == []


def test_sparse_nccl_trainer_send_weights_requires_full_shape():
    patch = _sparse_patch()
    patch.full_shape = None
    engine = SparseNCCLTrainerWeightTransferEngine(client=RecordingClient())
    engine.model_update_group = MagicMock()

    with pytest.raises(ValueError, match="full_shape"):
        engine.send_weights([patch])


def test_sparse_nccl_trainer_non_sender_skips_client():
    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=SparseNCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=2,
            rank=1,
        ),
        client=client,
    )

    assert engine.is_sender is False
    assert engine.model_update_group is None
    assert isinstance(engine, SparseNCCLTrainerWeightTransferEngine)
    engine.send_weights([_sparse_patch()])
    assert client.order == []


# ---------------------------------------------------------------------------
# Sharded RDT trainer engine
# ---------------------------------------------------------------------------


class _ListSource(WeightSource):
    """A WeightSource over an explicit ordered (name, cpu-tensor) list, so the
    sharded-RDT group/order logic can be tested without a real model."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def metadata(self):
        return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._pairs]

    def __iter__(self):
        return iter(self._pairs)


class _FakeProducerServer:
    """In-process stand-in for the _RDTProducerServer Ray actor. Records the
    engine->server call sequence and, by default, frees each group as soon as
    it is published (simulating the consumer's free_gather back-edge) so the
    gather loop's backpressure never blocks."""

    def __init__(self, auto_free=True):
        self.order: list[str] = []
        self.published: list[tuple] = []
        self.free_targets: list[int] = []
        self.inflight: list[tuple] = []
        self.auto_free = auto_free
        self.free_counts: dict[tuple, int] = {}
        self._pending_freed: list[tuple] = []

    def begin_sync(self):
        self.order.append("begin")

    def publish_group(self, key, entries, free_target):
        self.order.append("publish")
        self.published.append(key)
        self.free_targets.append(free_target)
        self.inflight.append(key)
        freed: list[tuple] = self._pending_freed
        self._pending_freed = []
        if self.auto_free or self.free_counts.get(key, 0) >= free_target:
            self.inflight.remove(key)
            freed = freed + [key]
        return freed

    def free_gather(self, names):
        """Consumer back-edge; may arrive before the group's publish."""
        key = tuple(names)
        self.free_counts[key] = self.free_counts.get(key, 0) + 1
        if key in self.inflight:
            self.inflight.remove(key)
            self._pending_freed.append(key)

    def free_one(self):
        """Manually free the oldest in-flight group (backpressure test)."""
        key = self.inflight.pop(0)
        self._pending_freed.append(key)

    def end_sync(self):
        self.order.append("end")
        freed = self._pending_freed
        self._pending_freed = []
        return freed

    def set_gather_error(self, message):
        self.order.append("error")


def _rdt_engine_with_fake_server(
    source, *, is_sender, client, server, monkeypatch, fleet_owned=None
):
    """Build a ShardedRDTTrainerWeightTransferEngine wired to an in-process fake
    server (no Ray, no CUDA IPC): bypass trainer_init's spawn, set the
    group-major metadata, and route _rpc to the fake."""
    import vllm.distributed.weight_transfer.sharded_rdt_trainer as mod

    # reduce_tensor needs CUDA; the fake server never rebuilds, so stub it.
    monkeypatch.setattr(mod, "reduce_tensor", lambda t: (None, ("fake",)))

    init_info = ShardedRDTTrainerInitInfo(num_consumers=1, rank=0 if is_sender else 1)
    engine = ShardedRDTTrainerWeightTransferEngine(
        client=client, source=source, is_sender=is_sender, init_info=init_info
    )
    engine._meta = list(source.metadata())
    names = [m.name for m in engine._meta]
    engine._groups = layerwise_groups(names)
    engine._server = server
    engine._rpc = lambda method, *args: getattr(server, method)(*args)
    # What trainer_init resolves from the source's ownership + the fleet's
    # all-gather. ``fleet_owned`` stands in for that all-gather so a partial-
    # ownership rank can be tested without a real process group; the fleet must
    # cover every group or the router rejects it (nothing would serve the rest).
    if fleet_owned is None:
        engine._build_router(1, 0)
    else:
        # Stands in for the (metadata digest, owned groups) all-gather; every
        # rank reports THIS rank's digest, i.e. agreeing metadata.
        monkeypatch.setattr(
            engine,
            "_all_gather_owned",
            lambda w, mine: [(mine[0], o) for o in fleet_owned],
        )
        engine._build_router(len(fleet_owned), 0)
    return engine


def _rdt_source_two_layers():
    return _ListSource(
        [
            ("embed.weight", torch.zeros(2)),
            ("model.layers.0.w", torch.zeros(2)),
            ("model.layers.1.w", torch.zeros(2)),
            ("norm.weight", torch.zeros(2)),
        ]
    )


class TestShardedRDTTrainerInitInfo:
    def test_declares_backend(self):
        assert ShardedRDTTrainerInitInfo.backend == "sharded_rdt"

    def test_rank_is_keyword_only_and_drives_is_sender(self):
        assert ShardedRDTTrainerInitInfo(num_consumers=4, rank=0).is_sender is True
        assert ShardedRDTTrainerInitInfo(num_consumers=4, rank=1).is_sender is False
        with pytest.raises(TypeError):
            # rank is keyword-only.
            ShardedRDTTrainerInitInfo(4, 0)  # type: ignore[misc]

    def test_registered_in_trainer_factory(self):
        cls = WeightTransferTrainerFactory._registry["sharded_rdt"]()
        assert cls is ShardedRDTTrainerWeightTransferEngine


def test_sharded_rdt_trainer_init_requires_source():
    with pytest.raises(ValueError, match="requires a WeightSource"):
        ShardedRDTTrainerWeightTransferEngine.trainer_init(
            ShardedRDTTrainerInitInfo(num_consumers=1, rank=0),
            client=RecordingClient(),
            source=None,
        )


def test_sharded_rdt_worker_init_info_is_group_major(monkeypatch):
    source = _rdt_source_two_layers()
    engine = _rdt_engine_with_fake_server(
        source,
        is_sender=True,
        client=RecordingClient(),
        server=_FakeProducerServer(),
        monkeypatch=monkeypatch,
    )
    worker_init = engine._build_worker_init_info(["srv_rk0"])
    # 4 params -> pre / layer0 / layer1 / post = 4 groups of length 1.
    assert worker_init.names == [
        "embed.weight",
        "model.layers.0.w",
        "model.layers.1.w",
        "norm.weight",
    ]
    assert worker_init.group_lens == [1, 1, 1, 1]
    assert worker_init.trainer_actor_names == ["srv_rk0"]
    assert worker_init.produce_method_name == "rdt_produce_weights_batched"
    assert sum(worker_init.group_lens) == len(worker_init.names)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_send_weights_drives_client_in_order(monkeypatch):
    server = _FakeProducerServer(auto_free=True)
    client = RecordingClient()
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=True,
        client=client,
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    # begin, one publish per group (4), end.
    assert server.order == ["begin", "publish", "publish", "publish", "publish", "end"]
    assert len(server.published) == 4
    # every group freed -> no engine-held refs remain.
    assert engine._inflight == {}


def test_sharded_rdt_send_weights_group_order_mismatch_raises(monkeypatch):
    # Source whose iteration order disagrees with its metadata order.
    class _BadSource(_ListSource):
        def __iter__(self):
            reordered = list(self._pairs)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            return iter(reordered)

    server = _FakeProducerServer()
    engine = _rdt_engine_with_fake_server(
        _BadSource(_rdt_source_two_layers()._pairs),
        is_sender=True,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(RuntimeError, match="iteration order must match"):
        engine.send_weights()
    assert "error" in server.order  # gather error propagated to the server


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_non_sender_skips_client(monkeypatch):
    class _RaisingClient(RecordingClient):
        def start_weight_update(self):
            raise AssertionError("non-sender must not touch the client")

        def update_weights(self, update_info):
            raise AssertionError("non-sender must not touch the client")

        def finish_weight_update(self):
            raise AssertionError("non-sender must not touch the client")

    server = _FakeProducerServer(auto_free=True)
    client = _RaisingClient()
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=False,
        client=client,
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()  # gathers only; must not raise
    assert client.order == []
    assert server.order == ["begin", "publish", "publish", "publish", "publish", "end"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_send_weights_surfaces_update_error(monkeypatch):
    class _FailingUpdateClient(RecordingClient):
        def update_weights(self, update_info):
            self.order.append("update")
            raise RuntimeError("inference side rejected update")

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=True,
        client=_FailingUpdateClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(RuntimeError, match="inference side rejected update"):
        engine.send_weights()


class TestRdtRouter:
    """Who serves and frees each gather group.

    A wrong answer here is not a wrong number but a hang: a consumer pulling
    from a producer that never gathered a group waits forever, and a published
    group nobody frees stalls the producer's end_sync. So every case checks the
    conservation law that makes the credit loop terminate — for each group, the
    per-producer free targets sum to the consumer count.
    """

    @staticmethod
    def _conserved(router, num_groups):
        return all(
            sum(router.free_target(p, g) for p in router.owners(g))
            == router.num_consumers
            for g in range(num_groups)
        )

    def test_identity_when_fleets_match(self):
        r = RdtRouter(8, 8, None, num_groups=6)
        assert [r.bound_producers(c) for c in range(8)] == [[c] for c in range(8)]
        assert all(r.producer_for(3, g) == 3 for g in range(6))
        assert self._conserved(r, 6)

    def test_gather_to_all_keeps_the_historical_binding(self):
        # 16 producers / 8 consumers: same producers per consumer as the
        # pre-router block rule, but each group is pulled from ONE of them.
        r = RdtRouter(16, 8, None, num_groups=95)
        for c in range(8):
            assert r.bound_producers(c) == assign_producer_indices(16, 8, c)
        assert [r.producer_for(0, g) for g in range(4)] == [0, 1, 0, 1]
        # No producer is left publishing groups nobody pulls from it.
        served = {r.producer_for(c, g) for c in range(8) for g in range(95)}
        assert served == set(range(16))
        assert self._conserved(r, 95)

    def test_fan_in_shares_one_producer(self):
        r = RdtRouter(2, 8, None, num_groups=5)
        assert [r.bound_producers(c) for c in range(8)] == [[c // 4] for c in range(8)]
        assert r.free_target(0, 0) == 4 and r.free_target(1, 0) == 4
        assert self._conserved(r, 5)

    def test_pipeline_stages_route_to_the_owning_stage(self):
        # 2 stages x 8 ranks; groups 0-2 on stage 0, 3-5 on stage 1.
        owners = [list(range(8))] * 3 + [list(range(8, 16))] * 3
        r = RdtRouter(16, 8, owners)
        for c in range(8):
            assert r.bound_producers(c) == [c, c + 8]
            assert [r.producer_for(c, g) for g in range(6)] == [c] * 3 + [c + 8] * 3
        assert r.owned_groups(0) == [0, 1, 2]
        assert r.owned_groups(8) == [3, 4, 5]
        # A rank owning a group serves exactly one consumer; non-owners serve none.
        assert r.free_target(0, 0) == 1 and r.free_target(0, 3) == 0
        assert self._conserved(r, 6)
        r.validate()

    def test_owner_without_a_consumer_gets_a_zero_target(self):
        # Fewer consumers than a stage has ranks: some owners serve nothing and
        # must not publish (the trainer skips those groups).
        owners = [list(range(8))] * 4
        r = RdtRouter(8, 2, owners)
        r.validate()
        assert self._conserved(r, 4)
        assert any(r.free_target(p, g) == 0 for p in range(8) for g in range(4)), (
            "expected some (producer, group) pairs to serve no consumer"
        )

    def test_validate_rejects_an_unowned_group(self):
        with pytest.raises(ValueError, match="no owner"):
            RdtRouter(4, 2, [[0, 1], [], [2, 3]]).validate()

    def test_validate_rejects_an_out_of_range_owner(self):
        with pytest.raises(ValueError, match="out of range"):
            RdtRouter(2, 2, [[0, 5]]).validate()


class _OwnedSource(_ListSource):
    """A source that gathers only some groups, like a pipeline-parallel rank."""

    def __init__(self, pairs, owned_group_idx):
        super().__init__(pairs)
        self._owned = list(owned_group_idx)
        groups = layerwise_groups([n for n, _ in pairs])
        self._owned_names = [n for gi in self._owned for n in groups[gi]]

    def owned_groups(self):
        return list(self._owned)

    def __iter__(self):
        by_name = dict(self._pairs)
        return iter([(n, by_name[n]) for n in self._owned_names])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_publishes_only_owned_groups(monkeypatch):
    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _OwnedSource(_rdt_source_two_layers()._pairs, [1, 2]),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        fleet_owned=[[1, 2], [0, 3]],  # this rank holds the layers, rank 1 the rest
    )
    engine.send_weights()

    assert server.published == [("model.layers.0.w",), ("model.layers.1.w",)]
    assert server.order == ["begin", "publish", "publish", "end"]
    assert engine._inflight == {}
    assert engine._group_owners == [[1], [0], [0], [1]]


def test_sharded_rdt_owned_group_order_mismatch_raises(monkeypatch):
    class _MisorderedOwned(_OwnedSource):
        def __iter__(self):
            return iter(list(super().__iter__())[::-1])

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _MisorderedOwned(_rdt_source_two_layers()._pairs, [1, 2]),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        fleet_owned=[[1, 2], [0, 3]],
    )
    with pytest.raises(RuntimeError, match="iteration order must match"):
        engine.send_weights()
    assert "error" in server.order


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_skips_publishing_a_group_no_consumer_pulls(monkeypatch):
    """A group gathered but routed to nobody must not be published: it would
    hold a backpressure slot that no free ever releases."""
    server = _FakeProducerServer(auto_free=True)
    source = _rdt_source_two_layers()
    engine = _rdt_engine_with_fake_server(
        source,
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    engine._free_targets[1] = 0  # group 1 serves no consumer from this rank
    engine.send_weights()

    assert ("model.layers.0.w",) not in server.published
    assert len(server.published) == 3
    assert server.order.count("publish") == 3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_publish_carries_the_group_free_target(monkeypatch):
    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()
    assert server.free_targets == [1, 1, 1, 1]


def test_sharded_rdt_worker_init_info_carries_group_owners(monkeypatch):
    import json

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _OwnedSource(_rdt_source_two_layers()._pairs, [0, 1]),
        is_sender=True,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        fleet_owned=[[0, 1], [2, 3]],
    )
    worker_init = engine._build_worker_init_info(["srv_rk0", "srv_rk1"])
    assert worker_init.group_owners == [[0], [0], [1], [1]]
    assert len(worker_init.group_owners) == len(worker_init.group_lens)
    # The payload crosses the control plane as JSON.
    assert json.loads(json.dumps(asdict(worker_init)))["group_owners"] == [
        [0],
        [0],
        [1],
        [1],
    ]


def test_weight_source_owns_every_group_by_default():
    """The contract's default: a source that says nothing owns the whole model."""
    src = _rdt_source_two_layers()
    assert src.owned_groups() is None
    assert ModuleSource(torch.nn.Linear(2, 2)).owned_groups() is None


def test_sharded_rdt_rejects_out_of_range_owned_group(monkeypatch):
    class _BadOwned(_OwnedSource):
        def owned_groups(self):
            return [0, 99]

    with pytest.raises(ValueError, match="out of range"):
        _rdt_engine_with_fake_server(
            _BadOwned(_rdt_source_two_layers()._pairs, [0]),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
        )


def test_sharded_rdt_rejects_empty_owned_groups(monkeypatch):
    class _OwnsNothing(_ListSource):
        def owned_groups(self):
            return []

    with pytest.raises(ValueError, match="empty"):
        _rdt_engine_with_fake_server(
            _OwnsNothing(_rdt_source_two_layers()._pairs),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
        )


def test_sharded_rdt_rejects_metadata_disagreement_across_ranks(monkeypatch):
    """Only the sender's metadata reaches the consumers, so a rank describing
    just its own share must fail loudly rather than silently drop the rest."""
    engine = _rdt_engine_with_fake_server(
        _OwnedSource(_rdt_source_two_layers()._pairs, [1, 2]),
        is_sender=False,
        client=RecordingClient(),
        server=_FakeProducerServer(auto_free=True),
        monkeypatch=monkeypatch,
        fleet_owned=[[1, 2], [0, 3]],  # a covering fleet, so construction succeeds
    )
    # Now rank 1 reports a DIFFERENT metadata digest for the same model.
    monkeypatch.setattr(
        engine,
        "_all_gather_owned",
        lambda w, mine: [mine, ("deadbeefdeadbeef", [0, 3])],
    )
    with pytest.raises(ValueError, match="disagrees across trainer ranks"):
        engine._build_router(2, 0)


def _serve_ring_server(src_name, src):
    """A producer server with one cached tensor and a pre-seeded serve ring, so a
    pull needs no Ray and no NIXL registration. Returns (server, serve) where
    ``serve(chain)`` packs one spec into the SAME ring slot every time — which is
    what puts the destination-view cache, and only it, under test."""
    from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
        _RDTProducerServer,
    )

    srv = _RDTProducerServer(
        num_rdt_buffers=2,
        arena_presize_gb=0.0,
        nosync=False,
        pack_check=False,
        gather_lookahead=2,
    )
    srv._cache[src_name] = src
    srv._serve_rings[0] = [
        torch.empty(1 << 16, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]

    def serve(chain):
        srv._serve_idx[0] = 0
        return srv.rdt_produce_weights_batched([(src_name, chain)], consumer_id=0)[0]

    return srv, serve


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the producer server needs a CUDA device"
)
def test_serve_does_not_reuse_packed_views_of_another_shape():
    """Two requests can share a name yet pack different slices of it.

    The producer caches the destination views it carves into a serve ring slot.
    Keyed by name alone, the second request is packed through the first's views.
    Reachable with layerwise_split > 1, when one name's copies split across chunks.
    """
    name = "model.layers.0.w"
    src = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(8, 8)
    _srv, serve = _serve_ring_server(name, src)

    serve((("narrow", (0, 0, 2), ()),))  # 2 rows
    wide = src.narrow(0, 0, 6)  # 6 rows, same name, same slot
    blob = serve((("narrow", (0, 0, 6), ()),))

    got = blob[: wide.numel() * wide.element_size()].view(wide.dtype).reshape(6, 8)
    assert torch.equal(got, wide)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the producer server needs a CUDA device"
)
def test_serve_does_not_reuse_packed_views_of_another_dtype():
    """The SILENT case of the same cache: two requests whose slices have the same
    name and the same shape but different dtypes pack at identical offsets, so
    reusing the stale views raises nothing — ``copy_`` just casts, and the blob
    carries the wrong bytes with no check downstream."""
    name = "model.layers.0.w"
    src = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(8, 8)
    _srv, serve = _serve_ring_server(name, src)

    serve((("to", (torch.float16,), ()),))  # same shape, fp16
    blob = serve(())  # same name and shape, bf16

    got = blob[: src.numel() * src.element_size()].view(src.dtype).reshape(8, 8)
    assert torch.equal(got, src), "packed through a cached view of the wrong dtype"
