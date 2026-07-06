# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer engine backends.

Unit tests for engine classes (parsing, validation, registry).
Integration tests for NCCL and IPC weight transfer between processes using Ray.
"""

import pickle
from unittest.mock import MagicMock

import pybase64 as base64
import pytest
import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import (
    IPCWeightTransferConfig,
    NCCLWeightTransferConfig,
    WeightTransferConfig,
)
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    RayVLLMWeightSyncClient,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
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
    from vllm.config.weight_transfer import NCCLWeightTransferConfig
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

    # Trainer broadcasts a single tensor unpacked, so the worker must not expect
    # the packed wire format.
    config = NCCLWeightTransferConfig(packed=False)
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
    init_info = NCCLWeightTransferInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,  # Trainer is rank 0, we become rank 1
        world_size=world_size,
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
    """Trainer task that broadcasts sparse patches via NCCL."""
    import torch

    device = _set_ray_assigned_device()

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.weight_transfer.sparse_nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        SparseNCCLWeightTransferEngine,
        SparseWeightPatch,
    )

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
    )
    comm = PyNcclCommunicator(pg, device=device.index)

    patch = SparseWeightPatch(
        name="test.weight",
        indices=torch.tensor([1, 7, 25], dtype=torch.int32, device=device),
        values=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device=device),
    )
    SparseNCCLWeightTransferEngine.trainer_send_weights(
        iter([patch]),
        NCCLTrainerSendWeightsArgs(group=comm),
    )
    torch.accelerator.synchronize()
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
    from vllm.config.weight_transfer import IPCWeightTransferConfig
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

    # Trainer sends unpacked IPC handles, so the worker reads packed=False.
    config = IPCWeightTransferConfig(packed=False)
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

    config = IPCWeightTransferConfig(packed=False)
    engine = IPCWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda:0"),
        MagicMock(spec=torch.nn.Module),
    )

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


# --- Unit Tests: Trainer-side engines + clients ---


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


class TestTrainerFactory:
    """WeightTransferTrainerFactory registry."""

    def test_registry_has_nccl_and_ipc(self):
        assert "nccl" in WeightTransferTrainerFactory._registry
        assert "ipc" in WeightTransferTrainerFactory._registry

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferTrainerFactory.trainer_init(
                "nope",
                WeightTransferConfig(backend="nope"),
                NCCLWeightTransferInitInfo(
                    master_address="x", master_port=1, rank_offset=1, world_size=2
                ),
                client=RecordingClient(),
                source=ModuleSource(torch.nn.Module()),
            )


def _module_with(*pairs):
    """A tiny nn.Module exposing the given (name, tensor) pairs as parameters,
    so trainer tests can build a ModuleSource without a real model."""
    module = torch.nn.Module()
    for name, tensor in pairs:
        module.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    return module


class _DummyTrainerEngine(TrainerWeightTransferEngine):
    """Minimal concrete trainer engine to exercise base-class construction."""

    @classmethod
    def trainer_init(cls, config, init_info, *, client, source):
        return cls(config, client=client, source=source)

    def send_weights(self):
        pass


class TestTrainerEngineBase:
    """Base-class construction (no GPU)."""

    def test_source_is_stored_and_iterable(self):
        engine = _DummyTrainerEngine(
            WeightTransferConfig(backend="nccl"),
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
        )
        assert engine.is_sender is True
        assert [name for name, _ in engine.source] == ["w"]


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (NCCL broadcast / CUDA stream).",
)
def test_nccl_trainer_send_weights_drives_client_in_order():
    """send_weights issues start -> update -> finish and ships metadata."""
    client = RecordingClient()
    engine = NCCLTrainerWeightTransferEngine(
        NCCLWeightTransferConfig(packed=False),
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4, device="cuda")))),
    )
    # Bypass the real NCCL rendezvous; broadcast is a no-op.
    engine.model_update_group = MagicMock()

    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4]]
    # packed wire params no longer ride on the per-round update_info.
    assert "packed" not in client.last_update_info


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (CUDA IPC handles).",
)
def test_ipc_trainer_send_weights_drives_client_in_order():
    client = RecordingClient()
    engine = IPCTrainerWeightTransferEngine(
        IPCWeightTransferConfig(packed=False),
        client=client,
        source=ModuleSource(_module_with(("w", torch.ones(4, device="cuda")))),
    )

    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
