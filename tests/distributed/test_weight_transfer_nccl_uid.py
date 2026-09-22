# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the pre-shared ncclUniqueId weight-transfer rendezvous.

The rendezvous lets a peer that cannot join a TCPStore (e.g. a torch-free JAX
trainer) initialize NCCL weight transfer from a unique id minted out of band.

* Unit tests for the mutually-exclusive rendezvous-mode validation on the NCCL
  init infos.
* Two-GPU integration tests driving the real worker engine against a peer: one
  via vLLM's own ``PyNcclCommunicator``, one via a genuinely torch-free trainer
  (cupy buffer + raw ``ncclBroadcast``).

The shared Ray helpers come from the sibling module (unchanged); the worker task
is local so the sibling's TCP test and its worker stay untouched.
"""

from unittest.mock import MagicMock

import pybase64 as base64
import pytest
import ray
import torch

from tests.distributed.test_weight_transfer import (
    _init_ray_for_weight_transfer,
    _set_ray_assigned_device,
    create_mock_vllm_config,
)
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
)

# Standard base64 of exactly 128 zero bytes: a well-formed (dummy) ncclUniqueId.
VALID_UID_B64 = base64.b64encode(b"\x00" * 128).decode()


class TestNCCLRendezvousValidation:
    """Test the xor-of-rendezvous-modes validation on the NCCL init infos."""

    def test_worker_init_info_is_keyword_only(self):
        # kw_only so a stale positional call fails loudly rather than silently
        # swapping fields once the rendezvous fields became optional.
        with pytest.raises(TypeError):
            NCCLWeightTransferInitInfo(1, 3, "127.0.0.1", 12345)  # type: ignore[call-arg]

    def test_trainer_init_info_accepts_positional_fields(self):
        # NCCLTrainerInitInfo keeps positional subclass fields; only `rank`
        # (from TrainerInitInfo) is keyword-only.
        trainer_positional = NCCLTrainerInitInfo("127.0.0.1", 12345, 3, rank=0)
        trainer_keyword = NCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=12345,
            world_size=3,
            rank=0,
        )
        assert trainer_positional == trainer_keyword

    def test_tcp_mode_valid(self):
        info = NCCLWeightTransferInitInfo(
            rank_offset=1,
            world_size=3,
            master_address="127.0.0.1",
            master_port=12345,
        )
        assert info.master_address == "127.0.0.1"
        assert info.nccl_unique_id_b64 is None

    def test_uid_mode_valid(self):
        info = NCCLWeightTransferInitInfo(
            rank_offset=1,
            world_size=3,
            nccl_unique_id_b64=VALID_UID_B64,
        )
        assert info.nccl_unique_id_b64 == VALID_UID_B64
        assert info.master_address is None
        assert info.master_port is None
        assert VALID_UID_B64 not in repr(info)

    def test_both_modes_raises(self):
        with pytest.raises(ValueError, match="not both"):
            NCCLWeightTransferInitInfo(
                rank_offset=1,
                world_size=3,
                master_address="127.0.0.1",
                master_port=12345,
                nccl_unique_id_b64=VALID_UID_B64,
            )

    def test_neither_mode_raises(self):
        with pytest.raises(ValueError, match="need nccl_unique_id_b64"):
            NCCLWeightTransferInitInfo(rank_offset=1, world_size=3)

    def test_half_specified_tcp_raises(self):
        with pytest.raises(ValueError, match="master_port"):
            NCCLWeightTransferInitInfo(
                rank_offset=1,
                world_size=3,
                master_address="127.0.0.1",
            )

    def test_bad_base64_uid_raises(self):
        with pytest.raises(ValueError, match="not valid standard base64"):
            NCCLWeightTransferInitInfo(
                rank_offset=1,
                world_size=3,
                nccl_unique_id_b64="!!!!",
            )

    def test_wrong_length_uid_raises(self):
        short_uid = base64.b64encode(b"\x00" * 64).decode()
        with pytest.raises(ValueError, match="128"):
            NCCLWeightTransferInitInfo(
                rank_offset=1,
                world_size=3,
                nccl_unique_id_b64=short_uid,
            )

    def test_parse_init_info_uid_valid(self):
        engine = NCCLWeightTransferEngine(
            WeightTransferConfig(backend="nccl"),
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        init_info = engine.parse_init_info(
            {
                "nccl_unique_id_b64": VALID_UID_B64,
                "rank_offset": 1,
                "world_size": 3,
            }
        )
        assert isinstance(init_info, NCCLWeightTransferInitInfo)
        assert init_info.nccl_unique_id_b64 == VALID_UID_B64
        assert init_info.master_address is None

    def test_parse_init_info_both_modes_raises(self):
        engine = NCCLWeightTransferEngine(
            WeightTransferConfig(backend="nccl"),
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        with pytest.raises(ValueError, match="not both"):
            engine.parse_init_info(
                {
                    "master_address": "127.0.0.1",
                    "master_port": 12345,
                    "nccl_unique_id_b64": VALID_UID_B64,
                    "rank_offset": 1,
                    "world_size": 3,
                }
            )

    @pytest.mark.parametrize(
        ("unique_id_bytes", "rank", "world_size", "match"),
        [
            (b"\x00" * 64, 0, 2, "128-byte"),
            (b"\x00" * 128, -1, 2, "out of range"),
            (b"\x00" * 128, 2, 2, "out of range"),
        ],
    )
    def test_from_unique_id_bytes_validation(
        self, unique_id_bytes, rank, world_size, match
    ):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        with pytest.raises(ValueError, match=match):
            PyNcclCommunicator.from_unique_id_bytes(
                unique_id_bytes,
                rank=rank,
                world_size=world_size,
                device=0,
            )


@ray.remote(num_gpus=1)
def inference_receive_tensor(
    init_info_dict: dict,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> dict:
    """Worker that joins via the given init dict and records the tensor it gets.

    Local to this module (rather than the sibling's TCP-only worker) so it can
    take a UID init payload -- the same dict boundary an external trainer POSTs.
    """
    import contextlib

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

    try:
        engine.init_transfer_engine(NCCLWeightTransferInitInfo(**init_info_dict))
        engine.receive_weights(
            NCCLWeightTransferUpdateInfo(
                names=["test.weight"],
                dtype_names=[tensor_dtype],
                shapes=[tensor_shape],
            )
        )
        torch.accelerator.synchronize()

        received_shape = None
        received_sum = None
        if len(recorder.received) == 1:
            _name, tensor = recorder.received[0]
            received_shape = list(tensor.shape)
            received_sum = tensor.sum().item()

        return {
            "success": received_shape == tensor_shape
            and abs(received_sum - torch.tensor(tensor_shape).prod().item()) < 0.01,
            "received_shape": received_shape,
            "received_sum": received_sum,
        }
    finally:
        engine.shutdown()


@ray.remote(num_gpus=1)
def trainer_broadcast_tensor_uid(
    uid_b64: str,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> dict:
    """Drive rank 0 via vLLM's own communicator from a pre-shared unique id.

    A real torch-free trainer would mint the id and drive rank 0 itself; here we
    reuse the real send protocol by injecting the communicator into a sender
    engine, exercising the worker-side UID path end to end.
    """
    import pybase64 as base64
    import torch

    _set_ray_assigned_device()

    from vllm.distributed.weight_transfer import ModuleSource
    from vllm.distributed.weight_transfer.nccl_common import uid_init_process_group
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerWeightTransferEngine,
    )

    class NoopClient:
        def init_weight_transfer_engine(self, init_info):
            pass

        def start_weight_update(self):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self, weight_version=None):
            pass

    dtype = getattr(torch, tensor_dtype)
    model = torch.nn.Module()
    test_module = torch.nn.Module()
    test_module.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.ones(tensor_shape, dtype=dtype, device="cuda"), requires_grad=False
        ),
    )
    model.add_module("test", test_module)

    engine = NCCLTrainerWeightTransferEngine(
        client=NoopClient(),
        source=ModuleSource(model),
        is_sender=True,
        packed=False,
    )
    engine.model_update_group = uid_init_process_group(
        base64.b64decode(uid_b64),
        rank=0,
        world_size=world_size,
        device=torch.accelerator.current_device_index(),
    )
    try:
        engine.send_weights()
        torch.accelerator.synchronize()
        return {"success": True}
    finally:
        engine.shutdown()


@ray.remote(num_gpus=1)
def trainer_broadcast_tensor_torch_free(
    uid_b64: str,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> dict:
    """Trainer peer that uses no torch and no PyNcclCommunicator (a JAX stand-in).

    Drives NCCL through the raw ctypes wrapper with a cupy send buffer, so it
    exercises interop with a foreign NCCL peer rather than vLLM against itself.
    ``PyNcclCommunicator.__init__`` runs a one-element warm-up all_reduce right
    after ``ncclCommInitRank``; that is a collective, so this peer must issue the
    matching all_reduce before the broadcast or both ranks deadlock.
    """
    import cupy as cp
    import numpy as np
    import pybase64 as base64

    device = _set_ray_assigned_device()
    cp.cuda.Device(device.index).use()

    from vllm.distributed.device_communicators.pynccl_wrapper import (
        NCCLLibrary,
        buffer_type,
        cudaStream_t,
        ncclDataTypeEnum,
        ncclRedOpTypeEnum,
    )

    assert tensor_dtype == "float32", "harness only wires float32"
    # Fill on the host and copy up (a plain H2D memcpy) so we neither use torch
    # nor make cupy JIT-compile a fill kernel (which needs NVRTC). All-ones lets
    # the worker verify by sum == numel.
    sendbuff = cp.asarray(np.ones(tensor_shape, dtype=np.float32))
    warmup = cp.asarray(np.zeros(1, dtype=np.float32))
    numel = int(sendbuff.size)
    # The copies above run on the default stream; drain them before the NCCL
    # collectives read the buffers on our own stream.
    cp.cuda.Device(device.index).synchronize()
    stream = cp.cuda.Stream()

    nccl = NCCLLibrary()
    comm = nccl.ncclCommInitRank(
        world_size, nccl.unique_id_from_bytes(base64.b64decode(uid_b64)), 0
    )
    try:
        # Mirror the worker's init-time warm-up all_reduce (rank 0 must join it).
        nccl.ncclAllReduce(
            buffer_type(warmup.data.ptr),
            buffer_type(warmup.data.ptr),
            1,
            ncclDataTypeEnum.ncclFloat32,
            ncclRedOpTypeEnum.ncclSum,
            comm,
            cudaStream_t(stream.ptr),
        )
        nccl.ncclBroadcast(
            buffer_type(sendbuff.data.ptr),
            buffer_type(sendbuff.data.ptr),
            numel,
            ncclDataTypeEnum.ncclFloat32,
            0,  # root
            comm,
            cudaStream_t(stream.ptr),
        )
        stream.synchronize()
        return {"success": True}
    finally:
        # Abort (not the collective destroy) for an uncoordinated teardown.
        nccl.ncclCommAbort(comm)


def _run_uid_transfer(trainer_task) -> None:
    """Mint a unique id, run the worker and the given trainer peer concurrently,
    and assert the tensor round-trips. Both ranks must enter init together --
    there is no store barrier on the UID path."""
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    _init_ray_for_weight_transfer()

    nccl = NCCLLibrary()
    uid_b64 = base64.b64encode(bytes(nccl.ncclGetUniqueId().internal)).decode()

    world_size = 2  # 1 trainer + 1 inference worker
    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    inference_future = inference_receive_tensor.remote(
        {
            "rank_offset": 1,
            "world_size": world_size,
            "nccl_unique_id_b64": uid_b64,
            "packed": False,
        },
        tensor_shape,
        tensor_dtype,
    )
    trainer_future = trainer_task.remote(
        uid_b64, world_size, tensor_shape, tensor_dtype
    )

    try:
        trainer_result, result = ray.get(
            [trainer_future, inference_future], timeout=120
        )
    finally:
        ray.shutdown()

    assert trainer_result["success"], "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL weight transfer test.",
)
def test_nccl_weight_transfer_between_processes_uid():
    """Weight transfer over a pre-shared ncclUniqueId (no TCPStore), with both
    ranks on vLLM's ``PyNcclCommunicator``."""
    _run_uid_transfer(trainer_broadcast_tensor_uid)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL weight transfer test.",
)
def test_nccl_weight_transfer_torch_free_trainer():
    """UID weight transfer against a foreign peer that uses no torch and no vLLM
    communicator, only a cupy buffer and raw ``ncclBroadcast``."""
    pytest.importorskip("cupy")
    _run_uid_transfer(trainer_broadcast_tensor_torch_free)
