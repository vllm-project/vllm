# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer engine backends.

Unit tests for engine classes (parsing, validation, registry).
Integration tests for NCCL and IPC weight transfer between processes using Ray.
"""

import builtins
import importlib.util
import pickle
import runpy
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
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
    ParamMeta,
    RayVLLMWeightSyncClient,
    Replicate,
    Shard,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    WeightTransferEngine,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
    iter_owned,
    local_shape,
    mesh_coords,
    owned_names,
    owner_groups,
    supports_iter_full,
    supports_iter_local,
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
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseNCCLTrainerWeightTransferEngine,
    SparseNCCLWeightTransferEngine,
    SparseNCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)
from vllm.platforms import current_platform
from vllm.utils.nccl import _nccl_has_no_cache
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


@pytest.mark.parametrize(
    "engine_type",
    [
        NCCLWeightTransferEngine,
        NCCLTrainerWeightTransferEngine,
        SparseNCCLWeightTransferEngine,
        SparseNCCLTrainerWeightTransferEngine,
    ],
)
def test_nccl_shutdown_destroys_communicator_once(engine_type):
    engine = object.__new__(engine_type)
    group = MagicMock()
    engine.model_update_group = group

    engine.shutdown()
    engine.shutdown()

    group.destroy.assert_called_once_with()
    assert engine.model_update_group is None


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

    def test_modelexpress_registration_is_native_and_lazy(self, monkeypatch):
        from vllm.distributed.weight_transfer import factory

        imported = []
        module = MagicMock()
        module.ModelExpressWeightTransferEngine.__name__ = (
            "ModelExpressWeightTransferEngine"
        )

        def import_module(name):
            imported.append(name)
            return module

        monkeypatch.setattr(factory.importlib, "import_module", import_module)
        config = WeightTransferConfig(backend="modelexpress")
        vllm_config = create_mock_vllm_config()
        device = torch.device("cpu")
        model = MagicMock(spec=torch.nn.Module)

        assert imported == []
        engine = WeightTransferEngineFactory.create_engine(
            config, vllm_config, device, model
        )

        assert imported == ["vllm.distributed.weight_transfer.modelexpress_engine"]
        module.ModelExpressWeightTransferEngine.assert_called_once_with(
            config, vllm_config, device, model
        )
        assert engine is module.ModelExpressWeightTransferEngine.return_value

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


# max_calls=1: a batch-invariant run leaves NCCL pins in the worker process.
@ray.remote(num_gpus=1, max_calls=1)
def inference_receive_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
    batch_invariant: bool = False,
) -> dict:
    """Inference task that receives tensor via NCCLWeightTransferEngine."""
    import contextlib
    from unittest.mock import MagicMock

    import torch

    _set_ray_assigned_device()
    if batch_invariant:
        from vllm.model_executor.determinism.batch_invariant import (
            override_envs_for_invariance,
        )

        override_envs_for_invariance()
        # Like vLLM's own groups, a pinned communicator makes NCCL read the
        # pins before the transfer group exists.
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{get_open_port()}",
            rank=0,
            world_size=1,
        )
        torch.distributed.all_reduce(torch.ones(1, device="cuda"))

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
@pytest.mark.parametrize(
    "batch_invariant",
    [
        pytest.param(False, id="default"),
        pytest.param(
            True,
            id="batch-invariant-worker",
            marks=pytest.mark.skipif(
                not _nccl_has_no_cache(), reason="Needs CUDA NCCL >= 2.29.7."
            ),
        ),
    ],
)
def test_nccl_weight_transfer_between_processes(batch_invariant):
    """Test NCCL weight transfer from trainer to inference process using Ray.

    This test verifies that the NCCLWeightTransferEngine can receive
    tensors broadcast by a trainer process via NCCL, including when the
    worker runs with batch-invariance NCCL pins the trainer does not have.
    """
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2  # 1 trainer + 1 inference worker

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    inference_future = inference_receive_tensor.remote(
        master_address,
        master_port,
        world_size,
        tensor_shape,
        tensor_dtype,
        batch_invariant,
    )
    trainer_future = trainer_broadcast_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )

    # A mismatched NCCL config deadlocks instead of failing.
    trainer_result, result = ray.get([trainer_future, inference_future], timeout=300)

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )


def test_sparse_nccl_checkpoint_chunks_to_ep_local_experts_cpu(monkeypatch):
    """Replay global expert patches through two EP-local loaders on CPU."""

    class CaptureClient:
        def __init__(self):
            self.order: list[str] = []
            self.update_infos: list[dict] = []

        def start_weight_update(self):
            self.order.append("start")

        def update_weights(self, update_info):
            self.order.append("update")
            self.update_infos.append(update_info)

        def finish_weight_update(self):
            self.order.append("finish")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_, **__: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda *_: nullcontext())

    client = CaptureClient()
    sender = SparseNCCLTrainerWeightTransferEngine(client=client)
    sender.model_update_group = MagicMock()
    sender.model_update_group.device = torch.device("cpu")
    wire_payloads = []
    sender.model_update_group.broadcast.side_effect = lambda tensor, **_: (
        wire_payloads.append(tensor.clone())
    )

    expert_names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
    ]
    client.start_weight_update()
    for name, index, value in zip(
        expert_names,
        (0, 3),
        (5.0, 7.0),
        strict=True,
    ):
        sender.send_weight_chunk(
            [
                SparseWeightPatch(
                    name=name,
                    indices=torch.tensor([index], dtype=torch.int32),
                    values=torch.tensor([value]),
                    full_shape=(2, 2),
                )
            ]
        )
    client.finish_weight_update()

    assert client.order == ["start", "update", "update", "finish"]
    assert [info["names"] for info in client.update_infos] == [
        [expert_names[0]],
        [expert_names[1]],
    ]
    assert [info["shapes"] for info in client.update_infos] == [
        [[2, 2]],
        [[2, 2]],
    ]
    assert [info["num_updates_list"] for info in client.update_infos] == [
        [1],
        [1],
    ]

    expected_gates = (
        [[5.0, -1.0], [-1.0, -1.0]],
        [[-1.0, -1.0], [-1.0, 7.0]],
    )
    for ep_rank, expected_gate in enumerate(expected_gates):
        model = torch.nn.Module()
        model.register_parameter(
            "w13_weight",
            torch.nn.Parameter(torch.full((1, 4, 2), -1.0), requires_grad=False),
        )
        local_name = expert_names[ep_rank]
        seen_names: list[str] = []
        loaded_names: set[str] = set()
        copy_count = [0]

        def load_weights(
            weights,
            local_name=local_name,
            target=model,
            seen=seen_names,
            loaded=loaded_names,
            count=copy_count,
        ):
            for name, checkpoint_weight in weights:
                seen.append(name)
                if name != local_name:
                    continue
                target.w13_weight.data[0, :2].copy_(checkpoint_weight)
                loaded.add(name)
                count[0] += 1
            return loaded

        model.load_weights = load_weights
        receiver = SparseNCCLWeightTransferEngine(
            WeightTransferConfig(backend="sparse_nccl"),
            create_mock_vllm_config(rank=ep_rank, world_size=2),
            torch.device("cpu"),
            model,
        )
        receiver.model_update_group = MagicMock()
        receiver.model_update_group.device = torch.device("cpu")
        payloads = iter(wire_payloads)
        receiver.model_update_group.broadcast.side_effect = (
            lambda tensor, payloads=payloads, **_: tensor.copy_(next(payloads))
        )

        receiver.start_weight_update()
        for update_info in client.update_infos:
            receiver.receive_weights(SparseNCCLWeightTransferUpdateInfo(**update_info))
        receiver.finish_weight_update()

        assert seen_names == expert_names
        assert loaded_names == {local_name}
        assert copy_count == [1]
        assert model.w13_weight[0, :2].tolist() == expected_gate
        assert model.w13_weight[0, 2:].eq(-1).all()
        foreign_value = 7.0 if ep_rank == 0 else 5.0
        assert not model.w13_weight.eq(foreign_value).any()
        receiver.shutdown()

    sender.shutdown()


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

    def finish_weight_update(self, weight_version: str | None = None) -> None:
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

        client.finish_weight_update("step-42")
        handle.finish_weight_update.remote.assert_called_once_with()
        handle.update_weight_version.remote.assert_called_once_with("step-42")

        client.shutdown_weight_transfer_engine()
        handle.shutdown_weight_transfer_engine.remote.assert_called_once_with()

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

        client.update_weights([{"ipc_handles": {"gpu": ("args",)}}, {"names": []}])
        sent = captured["json"]["update_info"]
        assert sent[1] == {"names": []}
        assert pickle.loads(base64.b64decode(sent[0]["ipc_handles_pickled"])) == {
            "gpu": ("args",)
        }

    def test_http_client_passes_through_nccl_update_info(self, monkeypatch):
        """NCCL update_info has only JSON-native fields and passes unchanged."""
        captured = {}

        def fake_post(self, path, json=None):
            captured["path"] = path
            captured["json"] = json

        monkeypatch.setattr(HTTPVLLMWeightSyncClient, "_post", fake_post)
        client = HTTPVLLMWeightSyncClient("http://localhost:8000")
        update_info = {"names": ["w"], "dtype_names": ["float32"], "shapes": [[4]]}
        client.update_weights(update_info)
        assert captured["json"]["update_info"] == update_info

        client.finish_weight_update("step-42")
        assert captured["json"] == {"weight_version": "step-42"}

        client.shutdown_weight_transfer_engine()
        assert captured == {"path": "shutdown_weight_transfer_engine", "json": None}


class TestModuleSource:
    """`ModuleSource` metadata vs. its iteration modes (dense, no GPU)."""

    def test_metadata_reads_shape_and_dtype(self):
        source = ModuleSource(
            _module_with(("w", torch.zeros(2, 3)), ("b", torch.zeros(3)))
        )
        meta = source.metadata()
        assert [m.name for m in meta] == ["w", "b"]
        assert [m.shape for m in meta] == [(2, 3), (3,)]
        assert all(m.dtype == torch.float32 for m in meta)
        assert all(m.owners is None for m in meta)

    def test_iter_full_yields_materialized_tensors(self):
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        source = ModuleSource(_module_with(("w", w)))
        pairs = [pair for batch in source.iter_full() for pair in batch]
        assert [name for name, _ in pairs] == ["w"]
        assert torch.equal(pairs[0][1], w)

    def test_source_is_reiterable(self):
        source = ModuleSource(_module_with(("w", torch.zeros(2))))
        assert [b[0][0] for b in source.iter_full()] == ["w"]
        assert [b[0][0] for b in source.iter_full()] == ["w"]

    def test_metadata_agrees_with_both_modes(self):
        """Engines declare the round from `metadata()` and send what a mode
        yields, so both modes must pass the same check."""
        source = ModuleSource(
            _module_with(("w", torch.zeros(2, 3)), ("b", torch.zeros(3)))
        )
        meta = source.metadata()
        for local in (False, True):
            batches = list(iter_owned(source, meta, 0, local=local))
            assert [i for batch in batches for i, _ in batch] == [0, 1]


@pytest.fixture
def fake_world():
    """A 4-rank process group, as rank 1, without other processes: enough to
    build DTensors and read their layouts."""
    from torch.testing._internal.distributed.fake_pg import FakeStore

    torch.distributed.init_process_group(
        "fake", store=FakeStore(), rank=1, world_size=4
    )
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


class TestModuleSourceDTensor:
    """`ModuleSource` declares a DTensor's mesh ranks as its owners."""

    @staticmethod
    def _dtensor_module(placements_of):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_tensor

        mesh = init_device_mesh("cpu", (2, 2))
        module = torch.nn.Module()
        for name, placements in placements_of.items():
            module.register_parameter(
                name,
                torch.nn.Parameter(
                    distribute_tensor(torch.zeros(6, 4), mesh, placements),
                    requires_grad=False,
                ),
            )
        return module

    def test_metadata_declares_mesh_and_placements(self, fake_world):
        from torch.distributed.tensor import Replicate as TR
        from torch.distributed.tensor import Shard as TS

        source = ModuleSource(
            self._dtensor_module({"w": [TS(0), TS(1)], "r": [TR(), TR()]})
        )
        w, r = source.metadata()
        assert w.owners == (0, 1, 2, 3)
        assert w.mesh_shape == (2, 2)
        assert w.placements == (Shard(0), Shard(1))
        assert r.owners == (0, 1, 2, 3)
        assert r.placements is None

    def test_iter_local_yields_the_declared_local_shape(self, fake_world):
        from torch.distributed.tensor import Shard as TS

        source = ModuleSource(self._dtensor_module({"w": [TS(0), TS(1)]}))
        meta = source.metadata()
        ((index, local),) = next(iter_owned(source, meta, 1, local=True))
        assert index == 0
        assert tuple(local.shape) == local_shape(meta[0], 1) == (3, 2)

    def test_partial_placement_is_declared_replicated(self, fake_world):
        """`Partial` has no `Shard`/`Replicate` spelling, so the source declares
        the parameter replicated and `iter_local` must gather it."""
        from torch.distributed.tensor import Partial

        source = ModuleSource(self._dtensor_module({"p": [Partial(), Partial()]}))
        (meta,) = source.metadata()
        assert meta.owners == (0, 1, 2, 3)
        assert meta.placements is None


class TestParamMetaOwnership:
    """Ownership and layout fields on `ParamMeta`, and the helpers derived
    from them. Every engine plans from these, so they must agree with DTensor
    on who holds what."""

    def test_owners_default_to_a_flat_mesh(self):
        meta = ParamMeta("w", torch.float32, (4,), owners=[2, 3])
        assert meta.owners == (2, 3)
        assert meta.mesh_shape == (2,)
        assert meta.placements is None

    def test_all_replicate_placements_normalize_to_none(self):
        meta = ParamMeta(
            "w", torch.float32, (4,), owners=(0, 1), placements=(Replicate(),)
        )
        assert meta.placements is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"placements": (Shard(0),)}, "without owners"),
            ({"owners": (0, 0)}, "distinct"),
            ({"owners": (0, 1, 2), "mesh_shape": (2, 2)}, "does not cover"),
            ({"owners": (0, 1), "placements": (Shard(0), Shard(0))}, "1-D mesh"),
            ({"owners": (0, 1), "placements": (Shard(1),)}, "shards dim 1"),
        ],
    )
    def test_inconsistent_layouts_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ParamMeta("w", torch.float32, (4,), **kwargs)

    def test_mesh_coords_are_row_major_over_owners(self):
        meta = ParamMeta(
            "w", torch.float32, (4,), owners=(4, 5, 6, 7), mesh_shape=(2, 2)
        )
        assert [mesh_coords(meta, r) for r in (4, 5, 6, 7)] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

    def test_local_shape_follows_torch_chunk_outer_axis_first(self):
        """Two axes sharding one uneven dim: 10 -> (5, 5) -> (3, 2), as DTensor
        computes it."""
        meta = ParamMeta(
            "w",
            torch.float32,
            (10, 6),
            owners=(0, 1, 2, 3),
            mesh_shape=(2, 2),
            placements=(Shard(0), Shard(0)),
        )
        assert [local_shape(meta, r) for r in range(4)] == [
            (3, 6),
            (2, 6),
            (3, 6),
            (2, 6),
        ]

    def test_owned_names_and_owner_groups(self):
        """A PP=2 x TP=2 layout: each stage owns its own layers."""
        metas = [
            ParamMeta("embed", torch.float32, (4,), owners=(0, 1)),
            ParamMeta("layers.0.w", torch.float32, (4,), owners=(0, 1)),
            ParamMeta("layers.1.w", torch.float32, (4,), owners=(2, 3)),
            ParamMeta("norm", torch.float32, (4,)),
        ]
        assert owned_names(metas, 0) == ["embed", "layers.0.w", "norm"]
        assert owned_names(metas, 3) == ["layers.1.w", "norm"]
        assert owner_groups(metas, 4) == [(0, 1), (2, 3), (0, 1, 2, 3)]
        with pytest.raises(ValueError, match="num_ranks"):
            owner_groups(metas)


class _BatchSource(WeightSource):
    """Yields scripted batches from whichever modes it is given."""

    def __init__(self, meta, full=None, local=None):
        self._meta = meta
        self._full = full
        self._local = local

    def metadata(self):
        return list(self._meta)

    def iter_full(self):
        return iter(self._full)

    def iter_local(self):
        return iter(self._local)


class TestIterOwned:
    """`iter_owned` is the one check between a source's stream and its
    metadata that every engine relies on."""

    META = [
        ParamMeta("a", torch.float32, (4,), owners=(0, 1), placements=(Shard(0),)),
        ParamMeta("b", torch.float32, (2,), owners=(1,)),
        ParamMeta("c", torch.float32, (3,)),
    ]

    def test_drops_none_and_keeps_the_sources_batches(self):
        source = _BatchSource(
            self.META,
            full=[
                [("a", torch.zeros(4)), ("b", None)],
                [("c", torch.zeros(3))],
            ],
        )
        batches = list(iter_owned(source, self.META, 0, local=False))
        assert [[i for i, _ in b] for b in batches] == [[0], [2]]

    def test_local_mode_checks_the_local_shape(self):
        ok = _BatchSource(
            self.META, local=[[("a", torch.zeros(2))], [("c", torch.zeros(3))]]
        )
        assert len(list(iter_owned(ok, self.META, 0, local=True))) == 2
        bad = _BatchSource(self.META, local=[[("a", torch.zeros(4))]])
        with pytest.raises(ValueError, match=r"implies torch.float32 \(2,\)"):
            list(iter_owned(bad, self.META, 0, local=True))

    def test_a_name_this_rank_does_not_own_is_rejected(self):
        source = _BatchSource(self.META, full=[[("b", torch.zeros(2))]])
        with pytest.raises(ValueError, match="yielded 'b' where rank 0 owns 'a'"):
            list(iter_owned(source, self.META, 0, local=False))

    def test_a_truncated_stream_names_the_first_missing(self):
        source = _BatchSource(self.META, full=[[("a", torch.zeros(4))]])
        with pytest.raises(ValueError, match="first missing is 'c'"):
            list(iter_owned(source, self.META, 0, local=False))

    def test_modes_are_detected_from_overrides(self):
        class _FullOnly(WeightSource):
            def metadata(self):
                return []

            def iter_full(self):
                return iter(())

        class _LocalOnly(WeightSource):
            def metadata(self):
                return []

            def iter_local(self):
                return iter(())

        assert supports_iter_full(_FullOnly()) and not supports_iter_local(_FullOnly())
        assert supports_iter_local(_LocalOnly())
        assert not supports_iter_full(_LocalOnly())


class TestDeferredProcessingContract:
    """`defers_processing` and `drain_pending` are two halves of one contract: a
    caller that takes over the update tail (running its own
    `finalize_layerwise_reload` instead of going through `finish_weight_update`)
    reads the flag and calls the method. Both must be answerable on any engine, or
    that caller ends up reaching through a getattr."""

    def _engines(self):
        registry = dict(WeightTransferEngineFactory._registry)
        # ModelExpress is an optional, separately installed package, but its
        # backend is always registered. Skip it when the package is missing.
        if importlib.util.find_spec("modelexpress") is None:
            registry.pop("modelexpress")
        return {name: loader() for name, loader in registry.items()}

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
        assert [b[0][0] for b in engine.source.iter_full()] == ["w"]

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
    monkeypatch.setattr(
        nccl_engine_mod, "open_trainer_endpoint", lambda info: MagicMock()
    )

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


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (NCCL broadcast / CUDA stream).",
)
def test_nccl_trainer_sends_metadata_only_when_changed():
    """An unchanged parameter set ships empty update info; a changed one ships
    the full metadata again."""
    client = RecordingClient()
    engine = NCCLTrainerWeightTransferEngine(
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4, device="cuda")))),
        packed=False,
    )
    engine.model_update_group = MagicMock()

    engine.send_weights()
    assert client.last_update_info is not None
    assert client.last_update_info["shapes"] == [[4]]
    engine.send_weights()
    assert client.last_update_info == {"names": [], "dtype_names": [], "shapes": []}
    engine.source = ModuleSource(_module_with(("w", torch.zeros(8, device="cuda"))))
    engine.send_weights()
    assert client.last_update_info["shapes"] == [[8]]


def test_nccl_worker_reuses_metadata_per_model(monkeypatch):
    """Empty update info reuses the previous call's metadata for the same model;
    target and draft keep separate copies, and re-init clears them."""
    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine_mod

    monkeypatch.setattr(
        nccl_engine_mod, "worker_init_process_group", lambda info, pc: MagicMock()
    )
    target, draft = MagicMock(spec=torch.nn.Module), MagicMock(spec=torch.nn.Module)
    received: list = []
    for model in (target, draft):
        model.load_weights = lambda w, m=model: received.append((m, w[0][0]))
    engine = NCCLWeightTransferEngine(
        WeightTransferConfig(backend="nccl"),
        create_mock_vllm_config(),
        torch.device("cpu"),
        target,
    )
    init_info = NCCLWeightTransferInitInfo(
        master_address="127.0.0.1", master_port=29500, rank_offset=1, world_size=2
    )
    engine.init_transfer_engine(init_info)
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())

    def update(model, *names):
        engine.set_weight_update_target(model, None)
        engine.receive_weights(
            NCCLWeightTransferUpdateInfo(
                names=list(names),
                dtype_names=["float32"] * len(names),
                shapes=[[2]] * len(names),
            )
        )

    update(target, "t")
    update(draft, "d")
    update(target)
    update(draft)
    engine.init_transfer_engine(init_info)
    update(target)

    assert received == [(target, "t"), (draft, "d"), (target, "t"), (draft, "d")]


class _ScriptedSource(WeightSource):
    """Declares `meta` but yields whatever `pairs` says, one per batch — used to
    drive the metadata/iteration agreement checks."""

    def __init__(self, meta, pairs):
        self._meta = meta
        self._pairs = pairs

    def metadata(self):
        return list(self._meta)

    def iter_full(self):
        for pair in self._pairs:
            yield [pair]


def _mock_group_engine(source, monkeypatch, **kwargs):
    """Unpacked trainer engine with a mocked group and stream (no GPU needed)."""
    engine = NCCLTrainerWeightTransferEngine(
        client=RecordingClient(), source=source, packed=False, **kwargs
    )
    engine.model_update_group = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    return engine


def test_nccl_trainer_init_requires_source():
    """NCCL is a full-resync backend: it cannot run without a WeightSource."""
    with pytest.raises(ValueError, match="requires a WeightSource"):
        NCCLTrainerWeightTransferEngine.trainer_init(
            NCCLTrainerInitInfo(
                master_address="127.0.0.1", master_port=29500, world_size=2, rank=0
            ),
            client=RecordingClient(),
        )


def test_nccl_trainer_send_weights_rejects_reordered_source(monkeypatch):
    """The worker sizes its buffers (and cuts packed chunks) from metadata(), so
    iteration disagreeing with it must raise rather than corrupt the stream."""
    meta = [
        ParamMeta("w", torch.float32, (4,)),
        ParamMeta("b", torch.float32, (2,)),
    ]
    reordered = [("b", torch.zeros(2)), ("w", torch.zeros(4))]
    engine = _mock_group_engine(_ScriptedSource(meta, reordered), monkeypatch)

    with pytest.raises(ValueError, match="yielded 'b' where rank 0 owns 'w'"):
        engine.send_weights()


def test_nccl_trainer_send_weights_rejects_dtype_disagreement(monkeypatch):
    """A source that declares one wire dtype and materializes another would make
    the two sides disagree on every byte offset."""
    meta = [ParamMeta("w", torch.float32, (4,))]
    engine = _mock_group_engine(
        _ScriptedSource(meta, [("w", torch.zeros(4, dtype=torch.bfloat16))]),
        monkeypatch,
    )

    with pytest.raises(ValueError, match="metadata implies torch.float32"):
        engine.send_weights()


def test_nccl_trainer_send_weights_rejects_truncated_source(monkeypatch):
    """Yielding fewer parameters than declared leaves the worker waiting."""
    meta = [
        ParamMeta("w", torch.float32, (4,)),
        ParamMeta("b", torch.float32, (2,)),
    ]
    engine = _mock_group_engine(
        _ScriptedSource(meta, [("w", torch.zeros(4))]), monkeypatch
    )

    with pytest.raises(ValueError, match="yielded 1 of the 2 names"):
        engine.send_weights()


def test_nccl_trainer_rejects_a_second_sender(monkeypatch):
    """Each parameter is sent by its first owner and broadcast has one sender,
    so a pipeline stage that does not start at rank 0 cannot be sent; every
    rank sees the same metadata and raises alike."""
    meta = [
        ParamMeta("w", torch.float32, (4,), owners=(0, 1)),
        ParamMeta("b", torch.float32, (2,), owners=(2, 3)),
    ]
    source = _ScriptedSource(meta, [])
    for is_sender, rank in ((True, 0), (False, 2)):
        engine = _mock_group_engine(source, monkeypatch, is_sender=is_sender, rank=rank)
        with pytest.raises(ValueError, match="'b' is sent by its first owner, rank 2"):
            engine.send_weights()


def test_nccl_trainer_non_sender_owner_drives_its_gathers(monkeypatch):
    """A non-sender owner (an FSDP peer) still iterates what it owns, since
    the sender's gathers need it; it never touches the client or the group."""
    meta = [ParamMeta("w", torch.float32, (4,), owners=(0, 1))]
    yielded: list[str] = []

    class _Recording(_ScriptedSource):
        def iter_full(self):
            for batch in super().iter_full():
                yielded.extend(name for name, _ in batch)
                yield batch

    engine = _mock_group_engine(
        _Recording(meta, [("w", torch.zeros(4))]),
        monkeypatch,
        is_sender=False,
        rank=1,
    )
    engine.send_weights()
    assert yielded == ["w"]
    assert engine.client.order == []
    engine.model_update_group.broadcast.assert_not_called()


def test_nccl_trainer_send_weights_broadcasts_contiguous(monkeypatch):
    """NCCL sends numel elements from data_ptr(), so a non-contiguous view must
    be linearized first or the worker receives unrelated memory."""
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    view = base.t()  # non-contiguous
    meta = [ParamMeta("w", torch.float32, tuple(view.shape))]
    engine = _mock_group_engine(_ScriptedSource(meta, [("w", view)]), monkeypatch)

    engine.send_weights()

    sent = engine.model_update_group.broadcast.call_args.args[0]
    assert sent.is_contiguous()
    assert torch.equal(sent, view)


def test_nccl_trainer_send_weights_raises_instead_of_hanging(monkeypatch):
    """A failed broadcast must surface even while the inference-side
    update_weights is still blocked in its matching NCCL call.

    Joining the RPC thread there would deadlock: the worker only returns once
    the broadcast it is waiting for arrives, which never happens.
    """
    rpc_entered = threading.Event()
    release_rpc = threading.Event()

    class _BlockingClient(RecordingClient):
        def update_weights(self, update_info):
            rpc_entered.set()
            release_rpc.wait(timeout=30)  # stands in for a wedged NCCL recv
            super().update_weights(update_info)

    class _FailingSource(WeightSource):
        def metadata(self):
            return [ParamMeta("w", torch.float32, (4,))]

        def iter_full(self):
            # Fail only once the RPC is provably in flight.
            rpc_entered.wait(timeout=60)
            raise RuntimeError("broadcast blew up")

    engine = NCCLTrainerWeightTransferEngine(
        client=_BlockingClient(), source=_FailingSource(), packed=False
    )
    engine.model_update_group = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="broadcast blew up"):
            engine.send_weights()
        elapsed = time.perf_counter() - started
        assert rpc_entered.is_set(), "the RPC was never in flight"
        assert not release_rpc.is_set(), "send_weights waited for the wedged RPC"
        # Regression guard: joining the RPC thread would park here until the
        # client's own timeout expires instead of raising immediately.
        assert elapsed < 5.0, f"send_weights blocked {elapsed:.1f}s on the RPC"
    finally:
        release_rpc.set()


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

    monkeypatch.setattr(sparse_mod, "open_trainer_endpoint", lambda info: MagicMock())

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


def test_sparse_nccl_trainer_send_weights_drives_client_in_order(monkeypatch):
    """One-shot send drives start, update, and finish in order."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()
    engine.model_update_group.device = torch.device("cpu")
    # The group is a mock, so the stream is just a handle it is handed (and a
    # handle _post_send_sync can synchronize).
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())

    engine.send_weights([_sparse_patch()])

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4, 4]]
    assert client.last_update_info["num_updates_list"] == [2]
    # One broadcast for indices + one for values per patch.
    assert engine.model_update_group.broadcast.call_count == 2
    engine.shutdown()


def test_sparse_nccl_trainer_send_weights_empty_round_is_noop():
    """A round with no patches must not touch the client (an empty sparse
    update info is invalid by construction)."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()

    engine.send_weights([])
    engine.send_weights()  # no argument is also a no-op round
    engine.send_weight_chunk([])
    engine.send_weight_chunk()

    assert client.order == []


def test_sparse_nccl_trainer_send_weights_requires_full_shape():
    patch = _sparse_patch()
    patch.full_shape = None
    engine = SparseNCCLTrainerWeightTransferEngine(client=RecordingClient())
    engine.model_update_group = MagicMock()
    engine.model_update_group.device = torch.device("cpu")

    with pytest.raises(ValueError, match="full_shape"):
        engine.send_weights([patch])


def test_sparse_nccl_trainer_rejects_source():
    """Sparse is a delta backend; a WeightSource would silently never be sent."""
    with pytest.raises(ValueError, match="takes no WeightSource"):
        SparseNCCLTrainerWeightTransferEngine(
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
        )


def test_sparse_nccl_trainer_validates_patch_before_any_rpc():
    """Malformed patches must fail on the trainer, before start_weight_update:
    the worker's own checks only run once the broadcasts are already in flight,
    where a size mismatch wedges both sides instead of raising."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()
    engine.model_update_group.device = torch.device("cpu")

    mismatched = SparseWeightPatch(
        name="w",
        indices=torch.tensor([1, 3], dtype=torch.int32),
        values=torch.tensor([1.0], dtype=torch.float32),
        full_shape=(4, 4),
    )
    with pytest.raises(ValueError, match="matching lengths"):
        engine.send_weights([mismatched])

    wrong_index_dtype = SparseWeightPatch(
        name="w",
        indices=torch.tensor([1, 3], dtype=torch.int64),
        values=torch.tensor([1.0, 2.0], dtype=torch.float32),
        full_shape=(4, 4),
    )
    with pytest.raises(ValueError, match="int32 indices"):
        engine.send_weights([wrong_index_dtype])

    assert client.order == []


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
    engine.send_weight_chunk([_sparse_patch()])
    assert client.order == []


# --- Optional ModelExpress Client Lifecycle ---


def _import_modelexpress_shim():
    spec = importlib.util.find_spec(
        "vllm.distributed.weight_transfer.modelexpress_engine"
    )
    assert spec is not None and spec.origin is not None
    runpy.run_path(spec.origin, run_name="_test_modelexpress_shim")


@pytest.mark.parametrize(
    "missing_module",
    [
        "modelexpress",
        "modelexpress_rl",
        "modelexpress_rl.inference",
        "modelexpress_rl.inference.engines",
        "modelexpress_rl.inference.engines.vllm",
        "modelexpress_rl.inference.engines.vllm.weight_transfer_engine",
    ],
)
def test_modelexpress_missing_backend_explains_source_install(
    monkeypatch, missing_module
):
    original_import = builtins.__import__
    error = ModuleNotFoundError(name=missing_module)

    def import_without_backend(name, *args, **kwargs):
        if name == "modelexpress":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_backend)
    with pytest.raises(ImportError, match="modelexpress_client/python") as exc_info:
        _import_modelexpress_shim()
    assert exc_info.value.__cause__ is error
    assert "uv pip install" in str(exc_info.value)


@pytest.mark.parametrize("missing_module", ["grpc", "modelexpress.internal_dependency"])
def test_modelexpress_preserves_transitive_import_errors(monkeypatch, missing_module):
    original_import = builtins.__import__
    error = ModuleNotFoundError(name=missing_module)

    def import_without_dependency(name, *args, **kwargs):
        if name == "modelexpress":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        _import_modelexpress_shim()
    assert exc_info.value is error


@pytest.fixture
def mx_client(monkeypatch):
    client_module = pytest.importorskip("modelexpress_rl.inference.client")
    client = MagicMock()
    monkeypatch.setattr(
        client_module.ModelExpressGeneratorClient,
        "initialize",
        MagicMock(return_value=client),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", MagicMock())
    return client


def _make_modelexpress_engine():
    return WeightTransferEngineFactory.create_engine(
        WeightTransferConfig(backend="modelexpress"),
        SimpleNamespace(
            parallel_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="test/model"),
        ),
        torch.device("cpu"),
        torch.nn.Linear(2, 2),
    )


@pytest.fixture
def mx_worker(mx_client, monkeypatch):
    from vllm.v1.worker import gpu_worker

    monkeypatch.setattr(gpu_worker, "set_current_vllm_config", lambda _: nullcontext())
    worker = object.__new__(gpu_worker.Worker)
    worker.weight_transfer_engine = _make_modelexpress_engine()
    worker.vllm_config = worker.weight_transfer_engine.vllm_config
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.model_runner = MagicMock()
    return worker


def test_modelexpress_worker_rejects_updates_before_initialization(
    mx_worker, mx_client
):
    with pytest.raises(RuntimeError, match="not initialized"):
        mx_worker.start_weight_update()
    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        mx_worker.update_weights({"version_id": "version-a"})
    with pytest.raises(RuntimeError, match="without a matching"):
        mx_worker.finish_weight_update()
    mx_client.stage_weight.assert_not_called()


@pytest.mark.parametrize(
    "second_update, error_type",
    [
        ({"version_id": "version-a"}, RuntimeError),
        ({"version_id": "version-b"}, RuntimeError),
        ({"version_id": ""}, ValueError),
        ({"version_id": None}, ValueError),
        ({}, ValueError),
    ],
)
def test_modelexpress_worker_recovers_after_rejected_update(
    mx_worker, mx_client, second_update, error_type
):
    """A rejected payload must release A so a fresh session can actually install B."""
    mx_worker.weight_transfer_engine.init_transfer_engine(
        mx_worker.weight_transfer_engine.parse_init_info({})
    )
    staged_a = SimpleNamespace(version_id="version-a", metrics={}, release=MagicMock())
    staged_b = SimpleNamespace(version_id="version-b", metrics={}, release=MagicMock())
    mx_client.stage_weight.side_effect = [staged_a, staged_b]
    mx_client.apply_weight.return_value = {}
    mx_worker.start_weight_update()
    mx_worker.update_weights({"version_id": "version-a"})

    with pytest.raises(error_type):
        mx_worker.update_weights(second_update)
    staged_a.release.assert_called_once_with()
    mx_client.apply_weight.assert_called_once_with(staged_a)

    mx_worker.start_weight_update()
    mx_worker.update_weights({"version_id": "version-b"})
    mx_worker.finish_weight_update()

    assert [
        call.kwargs["version"].version_id
        for call in mx_client.stage_weight.call_args_list
    ] == ["version-a", "version-b"]
    mx_client.apply_weight.assert_called_with(staged_b)
    staged_b.release.assert_called_once_with()


def test_modelexpress_worker_can_retry_failed_finish(mx_worker, mx_client):
    mx_worker.weight_transfer_engine.init_transfer_engine(
        mx_worker.weight_transfer_engine.parse_init_info({})
    )
    staged = mx_client.stage_weight.return_value
    staged.release.side_effect = [RuntimeError("release failed"), None]
    mx_worker.start_weight_update()
    mx_worker.update_weights({"version_id": "version-a"})

    with pytest.raises(RuntimeError, match="release failed"):
        mx_worker.finish_weight_update()
    mx_worker.finish_weight_update()

    assert staged.release.call_count == 2
    assert not mx_worker._weight_update_active
    mx_worker.start_weight_update()


def test_native_engine_applies_and_releases_exact_versions(mx_client):
    from modelexpress_rl.inference.engines.vllm import (
        weight_transfer_engine as mx_engine,
    )

    from vllm.distributed.weight_transfer import modelexpress_engine

    engine = _make_modelexpress_engine()
    assert type(engine) is mx_engine.ModelExpressWeightTransferEngine
    assert (
        modelexpress_engine.ModelExpressWeightTransferInitInfo
        is mx_engine.ModelExpressWeightTransferInitInfo
    )
    assert (
        modelexpress_engine.ModelExpressWeightTransferUpdateInfo
        is mx_engine.ModelExpressWeightTransferUpdateInfo
    )
    assert not engine.supports_draft_weight_update
    engine.init_transfer_engine(engine.parse_init_info({}))

    for version_id in ("version-a", "version-b"):
        staged = SimpleNamespace(version_id=version_id, metrics={}, release=MagicMock())
        mx_client.stage_weight.return_value = staged
        mx_client.apply_weight.return_value = {}
        engine.start_weight_update()
        engine.update_weights({"version_id": version_id})

        assert (
            mx_client.stage_weight.call_args.kwargs["version"].version_id == version_id
        )
        mx_client.apply_weight.assert_called_with(staged)
        torch.accelerator.synchronize.assert_called()
        staged.release.assert_not_called()
        engine.finish_weight_update()
        staged.release.assert_called_once_with()

    engine.shutdown()
    engine.shutdown()
    mx_client.close.assert_called_once_with()


def test_vime_init_config_reaches_mx_client(mx_client):
    from modelexpress_rl.inference.client import ModelExpressGeneratorClient

    engine = _make_modelexpress_engine()
    ModelExpressGeneratorClient.initialize.assert_not_called()
    engine.init_transfer_engine(
        engine.parse_init_info(
            {
                "model_name": "policy",
                "server_url": "mx:8001",
                "initial_serving_version_id": "serving-a",
                "object_storage_type": "S3",
                "initial_base_version_id": "base-a",
                "seed_checkpoint_path": "/models/launch",
                "refit_checkpoint_dir": "/mxdelta/receiver",
                "refit_checkpoint_max_size_gb": 200,
                "object_storage_endpoint_url": "http://minio:9000",
                "object_storage_region_name": "us-west-2",
                "registration_ttl_seconds": 90,
                "lease_ttl_seconds": 60,
                "max_transfer_attempts": 4,
                "max_replay_chain_length": 17,
                "rpc_timeout_seconds": 12.5,
            }
        )
    )
    config = ModelExpressGeneratorClient.initialize.call_args.args[0]
    assert config.engine_context.model is engine.model
    assert config.engine_context.vllm_config is engine.vllm_config
    assert config.model_name == "policy"
    assert config.server_url == "mx:8001"
    assert config.initial_serving_version_id == "serving-a"
    assert config.registration_ttl_seconds == 90
    assert config.lease_ttl_seconds == 60
    assert config.max_transfer_attempts == 4
    assert config.max_replay_chain_length == 17
    assert config.rpc_timeout_seconds == 12.5
    storage = config.object_storage
    assert storage.storage_type.value == "S3"
    assert storage.initial_base_version_id == "base-a"
    assert storage.seed_checkpoint_path == "/models/launch"
    assert storage.refit_checkpoint_dir == "/mxdelta/receiver"
    assert storage.refit_checkpoint_max_size_gb == 200
    assert storage.endpoint_url == "http://minio:9000"
    assert storage.region_name == "us-west-2"


@pytest.mark.parametrize(
    "key",
    [
        "object_storage_type",
        "initial_base_version_id",
        "seed_checkpoint_path",
        "refit_checkpoint_dir",
        "object_storage_endpoint_url",
        "object_storage_region_name",
    ],
)
def test_incomplete_storage_config_rejected_before_init(mx_client, key):
    from modelexpress_rl.inference.client import ModelExpressGeneratorClient

    engine = _make_modelexpress_engine()
    with pytest.raises(ValueError, match="object storage requires"):
        engine.init_transfer_engine(engine.parse_init_info({key: "value"}))
    ModelExpressGeneratorClient.initialize.assert_not_called()


@pytest.mark.parametrize("version_id", ["", "   ", None, 1])
def test_invalid_version_rejected_before_staging(mx_client, version_id):
    engine = _make_modelexpress_engine()
    engine.init_transfer_engine(engine.parse_init_info({}))
    engine.start_weight_update()
    with pytest.raises(ValueError, match="version_id is required"):
        engine.update_weights({"version_id": version_id})
    mx_client.stage_weight.assert_not_called()


@pytest.mark.parametrize("failure", ["stage", "apply", "release"])
def test_update_failure_preserves_error_and_releases_handle(mx_client, failure):
    engine = _make_modelexpress_engine()
    engine.init_transfer_engine(engine.parse_init_info({}))
    staged = SimpleNamespace(version_id="version-a", metrics={}, release=MagicMock())
    mx_client.stage_weight.return_value = staged
    if failure == "stage":
        mx_client.stage_weight.side_effect = RuntimeError("stage failed")
    else:
        mx_client.apply_weight.side_effect = RuntimeError("apply failed")
        if failure == "release":
            staged.release.side_effect = RuntimeError("release failed")
    engine.start_weight_update()
    with pytest.raises(RuntimeError, match="stage failed|apply failed"):
        engine.update_weights({"version_id": "version-a"})
    assert staged.release.call_count == (0 if failure == "stage" else 1)
    torch.accelerator.synchronize.assert_not_called()
    engine.shutdown()
    mx_client.close.assert_called_once_with()


def test_plugin_does_not_replace_native_backend(mx_client):
    from modelexpress.engines.vllm.registration import (
        register_plugin_weight_transfer_engine,
    )
    from modelexpress_rl.inference.engines.vllm.weight_transfer_engine import (
        ModelExpressWeightTransferEngine,
    )

    native_loader = WeightTransferEngineFactory._registry["modelexpress"]
    register_plugin_weight_transfer_engine()
    assert WeightTransferEngineFactory._registry["modelexpress"] is native_loader
    assert type(_make_modelexpress_engine()) is ModelExpressWeightTransferEngine
