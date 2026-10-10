# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding and the wire-type
validation that keeps a bad plan from reaching a collective, since a mismatch
there is a hang rather than an error. The transfer itself needs the m2n runtime
and multiple GPUs, so it is exercised separately.
"""

import logging
from functools import wraps
from types import MethodType
from unittest.mock import Mock

import pybase64 as base64
import pytest
import ray
import torch

from vllm.distributed.weight_transfer import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    REPLICATED,
    M2NMesh,
    M2NParamMeta,
    check_placements,
    check_transferable,
    publish_destination_placements,
    resolve_layout,
    validate_layout,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    M2NWeightTransferEngine,
    M2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_layout import (
    M2NDestination,
    resolve_parameter_destinations,
)
from vllm.distributed.weight_transfer.m2n_source import (
    mesh_from_tensor,
    placements_from_tensor,
)
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

VALID_UID_B64 = base64.b64encode(b"\x00" * 128).decode()


class TestLayout:
    def test_replicated_splits_nothing(self):
        """A replicated tensor imposes no divisibility constraint. The shape
        dims are deliberately coprime with the mesh size: a layout that actually
        split the tensor would reject them."""
        mesh = M2NMesh((2, 2), start_rank=1)
        indivisible_shape = (7, 13)  # neither dim divisible by any mesh axis

        resolved_mesh, placements = resolve_layout(mesh, REPLICATED)
        validate_layout(resolved_mesh, placements, indivisible_shape, "destination")

    def test_replicated_keeps_the_same_ranks(self):
        """Replication re-factors the mesh to get a size-1 axis for its no-op
        shard. That is only sound if it still covers exactly the same GPUs."""
        mesh = M2NMesh((2, 3), start_rank=4)
        resolved_mesh, _ = resolve_layout(mesh, REPLICATED)
        assert resolved_mesh.size == mesh.size
        assert resolved_mesh.start_rank == mesh.start_rank

    def test_sharded_keeps_its_own_factorization(self):
        """Rank order decides who owns which shard, so a sharded tensor must
        not be re-factored the way a replicated one is."""
        mesh = M2NMesh((2, 3), start_rank=4)
        resolved_mesh, placements = resolve_layout(mesh, (REPLICATE, 0))
        assert resolved_mesh == mesh
        assert placements == (REPLICATE, 0)

    def test_two_shard_axes_rejected(self):
        """One axis has to replicate; a 2-D mesh that shards both is not
        something a single reshard can express."""
        with pytest.raises(ValueError, match="shards both"):
            check_placements((0, 1))

    def test_placement_below_replicate_rejected(self):
        with pytest.raises(ValueError, match=r"parameter 'w'.*-2"):
            check_placements((-2, REPLICATE), "parameter 'w' source placements")

    def test_shard_dim_must_exist(self):
        with pytest.raises(ValueError, match="rank 2"):
            validate_layout(M2NMesh((1, 2), 0), (REPLICATE, 2), (8, 16), "source")

    def test_shard_must_divide_evenly(self):
        with pytest.raises(ValueError, match="does not divide evenly"):
            validate_layout(M2NMesh((1, 3), 0), (REPLICATE, 0), (8, 16), "source")


class TestSourceLayout:
    def test_plain_tensor_is_replicated_across_trainer_ranks(self):
        """A tensor with no DTensor metadata is the same on every trainer rank,
        so the source describes it as replicated over all of them."""
        assert placements_from_tensor(torch.zeros(4)) is REPLICATED
        assert mesh_from_tensor(torch.zeros(4), 4) == M2NMesh((4, 1), 0)


class TestTransferable:
    def test_unsupported_dtype_names_the_parameter(self):
        with pytest.raises(ValueError, match="'w'"):
            check_transferable("w", torch.complex64, (4,))

    def test_rank_four_rejected(self):
        with pytest.raises(ValueError, match="rank 4"):
            check_transferable("w", torch.bfloat16, (2, 2, 2, 2))


class TestWireTypes:
    def _init_info(self, **overrides):
        fields = dict(
            master_address="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=3,
            src_mesh_dims=[1, 1],
            dst_mesh_dims=[2, 1],
            names=["w"],
            dtype_names=["bfloat16"],
            shapes=[[16, 16]],
            src_placements=[None],
        )
        fields.update(overrides)
        return M2NWeightTransferInitInfo(**fields)

    def test_accepts_a_consistent_plan(self):
        assert self._init_info().names == ["w"]

    def test_accepts_pre_shared_nccl_unique_id(self):
        info = self._init_info(
            master_address=None,
            master_port=None,
            nccl_unique_id_b64=VALID_UID_B64,
        )
        assert info.nccl_unique_id_bytes == b"\x00" * 128
        assert VALID_UID_B64 not in repr(info)

    def test_destination_plan_uses_uid_communicator_without_group(self):
        comm = Mock(rank=0, device=torch.device("cpu"), group=None)

        def receive_plan(tensor, src):
            assert src == 1
            tensor.copy_(torch.tensor([[-1, 0], [-2, -2]], dtype=torch.int8))

        comm.broadcast.side_effect = receive_plan

        placements = publish_destination_placements(comm, 1, None, 2)

        assert placements == [(REPLICATE, 0), REPLICATED]

    def test_rejects_both_rendezvous_modes(self):
        with pytest.raises(ValueError, match="not both"):
            self._init_info(nccl_unique_id_b64=VALID_UID_B64)

    def test_trainer_accepts_pre_shared_nccl_unique_id(self):
        info = M2NTrainerInitInfo(
            nccl_unique_id_b64=VALID_UID_B64,
            world_size=3,
            num_trainer_ranks=1,
            rank=0,
        )
        assert info.nccl_unique_id_bytes == b"\x00" * 128
        assert VALID_UID_B64 not in repr(info)

    def test_ragged_plan_rejected(self):
        with pytest.raises(ValueError, match="`shapes`"):
            self._init_info(shapes=[])

    def test_destination_mesh_must_cover_the_workers(self):
        """The trainer declares the inference mesh, so one that does not cover
        the workers is a config error — and it has to fail the init RPC, since
        a mismatched mesh would otherwise surface as a hung collective."""
        with pytest.raises(ValueError, match="dst_mesh_dims"):
            self._init_info(dst_mesh_dims=[3, 1])  # 3 != the 2 workers

    def test_world_must_hold_a_trainer_and_a_worker(self):
        with pytest.raises(ValueError, match="rank_offset"):
            self._init_info(rank_offset=3, world_size=3)

    @pytest.mark.parametrize("dtype_name", ["not_a_dtype", "Tensor"])
    def test_invalid_dtype_name_names_parameter(self, monkeypatch, dtype_name):
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n",
            lambda: object(),
        )
        engine = object.__new__(M2NWeightTransferEngine)

        with pytest.raises(ValueError, match=r"parameter 'w'.*dtype"):
            engine.init_transfer_engine(self._init_info(dtype_names=[dtype_name]))

    def _init_32_to_4_plan(self, monkeypatch, destination):
        m2n = Mock()
        m2n.Handle.create.return_value = object()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", lambda: m2n
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "resolve_parameter_destinations",
            lambda *args, **kwargs: [destination],
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_process_group",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "publish_destination_placements",
            lambda *args: args[2],
        )

        engine = object.__new__(M2NWeightTransferEngine)
        engine.parallel_config = Mock(pipeline_parallel_size=1)
        engine.model_config = Mock(quantization=None)
        engine.model = torch.nn.Module()
        engine.init_transfer_engine(
            self._init_info(
                rank_offset=32,
                world_size=36,
                src_mesh_dims=[1, 32],
                dst_mesh_dims=[1, 4],
                shapes=[[32, 8]],
                src_placements=[[REPLICATE, 0]],
            )
        )

    def test_limits_use_resolved_sharded_destination(self, monkeypatch):
        destination = M2NDestination("w", (REPLICATE, 0), torch.empty(8, 8))

        self._init_32_to_4_plan(monkeypatch, destination)

    def test_limits_still_reject_replicated_destination(self, monkeypatch):
        destination = M2NDestination("w", REPLICATED, None)

        with pytest.raises(ValueError, match=r"32 source shards.*MAX_SOURCES=16"):
            self._init_32_to_4_plan(monkeypatch, destination)

    def test_update_preflights_all_names_before_reshard(self):
        engine = object.__new__(M2NWeightTransferEngine)
        engine._handle = object()
        engine.model_update_group = object()
        engine._index = {"valid": 0}
        engine._metas = [M2NParamMeta("valid", torch.float32, (4,), REPLICATED)]
        engine._reshard = Mock()

        with pytest.raises(ValueError, match=r"parameter 'unknown'"):
            engine.receive_weights(
                M2NWeightTransferUpdateInfo(names=["valid", "unknown"])
            )

        engine._reshard.assert_not_called()

    def test_fallback_tensors_share_one_load_weights_invocation(self, monkeypatch):
        class CoupledModel:
            def __init__(self):
                self.calls = 0
                self.loaded = []

            def load_weights(self, weights):
                self.calls += 1
                received = list(weights)
                if [name for name, _ in received] == ["pair.weight", "pair.scale"]:
                    self.loaded = [(name, tensor.clone()) for name, tensor in received]

        model = CoupledModel()
        direct = torch.zeros(1)
        engine = object.__new__(M2NWeightTransferEngine)
        engine._handle = object()
        engine.model_update_group = object()
        engine.device = torch.device("cpu")
        engine.model = model
        engine._index = {"pair.weight": 0, "direct": 1, "pair.scale": 2}
        engine._metas = [
            M2NParamMeta(name, torch.float32, (1,), REPLICATED)
            for name in engine._index
        ]
        engine._parameter_destinations = [
            M2NDestination("pair.weight", REPLICATED, None),
            M2NDestination("direct", REPLICATED, direct),
            M2NDestination("pair.scale", REPLICATED, None),
        ]
        reshard_order = []

        def reshard(comm, stream, meta, placements, buffer):
            reshard_order.append(meta.name)
            buffer.fill_(len(reshard_order))

        engine._reshard = reshard
        stream = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.comm_ptr", lambda group: 0
        )
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

        engine.receive_weights(
            M2NWeightTransferUpdateInfo(names=["pair.weight", "direct", "pair.scale"])
        )

        assert model.calls == 1
        assert [name for name, _ in model.loaded] == ["pair.weight", "pair.scale"]
        assert [tensor.item() for _, tensor in model.loaded] == [1, 3]
        assert direct.item() == 2
        assert reshard_order == ["pair.weight", "direct", "pair.scale"]
        assert stream.synchronize.call_count == 2

    def test_trainer_rank_must_be_a_trainer_rank(self):
        """A trainer rank must fall within the trainer portion of the group."""
        with pytest.raises(ValueError, match="num_trainer_ranks"):
            M2NTrainerInitInfo(
                master_address="127.0.0.1",
                master_port=1234,
                world_size=4,
                num_trainer_ranks=2,
                rank=2,
            )

    def test_trainer_destination_mesh_must_cover_the_workers(self):
        """The destination mesh must include every inference worker."""
        with pytest.raises(ValueError, match="dst_mesh_dims"):
            M2NTrainerInitInfo(
                master_address="127.0.0.1",
                master_port=1234,
                world_size=6,
                num_trainer_ranks=2,
                dst_mesh_dims=(3, 1),  # 3 != the 4 inference workers
                rank=0,
            )

    def test_destination_mesh_defaults_to_flat(self):
        """A replicated destination does not care how the mesh is factored, so
        callers that do not shard it need not supply one."""
        info = M2NTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=1234,
            world_size=6,
            num_trainer_ranks=2,
            rank=0,
        )
        assert info.destination_mesh_dims == (4, 1)

    def test_sender_is_trainer_rank_zero(self):
        """Trainer rank 0 drives the inference control plane."""
        info = M2NTrainerInitInfo(
            master_address="127.0.0.1", master_port=1234, world_size=4, rank=0
        )
        assert info.is_sender


class TestRegistration:
    def test_both_registries_expose_the_backend(self):
        """Both worker and trainer factories register nccl_m2n."""
        assert "nccl_m2n" in WeightTransferEngineFactory._registry
        assert "nccl_m2n" in WeightTransferTrainerFactory._registry

    def test_init_info_dispatches_to_the_backend(self):
        """Trainer init info selects the nccl_m2n factory entry."""
        assert M2NTrainerInitInfo.backend == "nccl_m2n"


# ---------------------------------------------------------------------------
# End-to-end transfer
#
# Transport-only, in the style of the NCCL/sparse tests in
# test_weight_transfer.py: two Ray tasks with one GPU each drive the two engines
# directly, so no HTTP server or LLM instance is involved. The worker is not an
# RPC endpoint here, so the trainer engine gets a no-op control-plane client --
# the NCCL rendezvous and the reshard itself are the real thing.
# ---------------------------------------------------------------------------

SHAPE = [64, 32]
DTYPE = "float32"


def _init_ray() -> None:
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


def _assigned_device() -> "torch.device":
    gpu_ids = ray.get_gpu_ids()
    device = torch.device(f"cuda:{int(gpu_ids[0])}" if gpu_ids else "cuda:0")
    current_platform.set_device(device)
    return device


@ray.remote(num_gpus=1)
def _m2n_trainer_send(nccl_unique_id_b64: str, world_size: int) -> bool:
    """Send one parameter through the real trainer engine."""
    device = _assigned_device()

    from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.m2n_source import DTensorModuleSource
    from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

    class NoopClient:
        def init_weight_transfer_engine(self, init_info):
            pass

        def start_weight_update(self):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self, weight_version=None):
            pass

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(
                    SHAPE[0] * SHAPE[1], dtype=torch.float32, device=device
                ).reshape(SHAPE)
            )

    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=M2NTrainerInitInfo(
            nccl_unique_id_b64=nccl_unique_id_b64,
            world_size=world_size,
            num_trainer_ranks=1,
            dst_mesh_dims=(1, world_size - 1),
            rank=0,
        ),
        client=NoopClient(),
        source=DTensorModuleSource(Tiny(), num_trainer_ranks=1),
    )
    engine.send_weights()
    torch.accelerator.synchronize()
    engine.shutdown()
    return True


@ray.remote(num_gpus=1)
def _m2n_worker_receive(
    nccl_unique_id_b64: str,
    world_size: int,
    worker_rank: int = 0,
) -> dict:
    """Receive that parameter through the real worker engine."""
    import contextlib
    from unittest.mock import MagicMock

    device = _assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.m2n_engine import (
        M2NWeightTransferEngine,
        M2NWeightTransferInitInfo,
        M2NWeightTransferUpdateInfo,
    )

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []
            if world_size > 2:
                self.weight = torch.nn.Parameter(
                    torch.zeros(
                        SHAPE[0] // (world_size - 1),
                        SHAPE[1],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.weight.output_dim = 0
                self.weight.weight_loader = MethodType(
                    ColumnParallelLinear.weight_loader,
                    Mock(tp_size=world_size - 1),
                )

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = worker_rank
    parallel_config.world_size = world_size - 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = world_size - 1
    parallel_config.pipeline_parallel_size = 1
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.quantization = None

    recorder = Recorder()
    engine = M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"), vllm_config, device, recorder
    )
    # Transport-only: receive_weights enters set_current_vllm_config, and
    # vllm_config here is a mock.
    import vllm.config as _vllm_config_mod

    _vllm_config_mod.set_current_vllm_config = lambda cfg: contextlib.nullcontext()

    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            nccl_unique_id_b64=nccl_unique_id_b64,
            rank_offset=1,  # trainer occupies rank 0
            world_size=world_size,
            src_mesh_dims=[1, 1],
            dst_mesh_dims=[1, world_size - 1],
            names=["weight"],
            dtype_names=[DTYPE],
            shapes=[SHAPE],
            src_placements=[None],  # replicated on the single trainer rank
        )
    )
    engine.receive_weights(M2NWeightTransferUpdateInfo(names=["weight"]))
    torch.accelerator.synchronize()

    full = torch.arange(
        SHAPE[0] * SHAPE[1], dtype=torch.float32, device=device
    ).reshape(SHAPE)
    direct = world_size > 2
    name: str | None
    got: torch.Tensor | None
    if direct:
        name, got = "weight", recorder.weight
        expected = full.chunk(world_size - 1, dim=0)[worker_rank]
    else:
        name, got = recorder.received[0] if recorder.received else (None, None)
        expected = full
    result = {
        "count": len(recorder.received),
        "name": name,
        "shape": list(got.shape) if got is not None else None,
        "exact": bool(torch.equal(got, expected)) if got is not None else False,
        "direct": direct,
    }
    engine.shutdown()
    return result


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs: one trainer rank and one inference worker.",
)
def test_m2n_weight_transfer_between_processes():
    """A parameter survives a real reshard from a trainer process to a worker.

    This is the only test here that moves bytes: it builds both engines, joins
    one NCCL communicator across two processes, and reshards. Everything else
    in this file checks how a transfer is *described*.
    """
    pytest.importorskip("nccl.m2n", reason="nccl_m2n backend needs the m2n runtime")
    _init_ray()

    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    nccl = NCCLLibrary()
    nccl_unique_id_b64 = base64.b64encode(
        bytes(nccl.ncclGetUniqueId().internal)
    ).decode()
    world_size = 2  # 1 trainer + 1 inference worker

    worker = _m2n_worker_receive.remote(nccl_unique_id_b64, world_size)
    trainer = _m2n_trainer_send.remote(nccl_unique_id_b64, world_size)
    trainer_ok, result = ray.get([trainer, worker])

    assert trainer_ok, "trainer engine did not complete"
    assert result["count"] == 1, f"expected one parameter, got {result['count']}"
    assert result["name"] == "weight"
    assert result["shape"] == SHAPE
    assert result["exact"], "resharded tensor does not match what the trainer sent"


@pytest.mark.skipif(
    torch.accelerator.device_count() < 3,
    reason="Need at least 3 GPUs: one trainer rank and two inference workers.",
)
def test_m2n_weight_transfer_to_tp2_shards():
    """Two inference workers receive their own TP shard in live model storage."""
    pytest.importorskip("nccl.m2n", reason="nccl_m2n backend needs the m2n runtime")
    _init_ray()

    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    nccl = NCCLLibrary()
    nccl_unique_id_b64 = base64.b64encode(
        bytes(nccl.ncclGetUniqueId().internal)
    ).decode()
    world_size = 3

    workers = [
        _m2n_worker_receive.remote(nccl_unique_id_b64, world_size, rank)
        for rank in range(2)
    ]
    trainer = _m2n_trainer_send.remote(nccl_unique_id_b64, world_size)
    trainer_ok, *results = ray.get([trainer, *workers])

    assert trainer_ok, "trainer engine did not complete"
    for rank, result in enumerate(results):
        assert result["count"] == 0, f"worker {rank} unexpectedly called load_weights"
        assert result["name"] == "weight"
        assert result["shape"] == [SHAPE[0] // 2, SHAPE[1]]
        assert result["direct"]
        assert result["exact"], f"worker {rank} received the wrong shard"


class _Model(torch.nn.Module):
    """Stand-in for a TP=2 shard of a tiny model.

    `column` is sharded on dim 0 and `row` on dim 1 (the two shapes a TP linear
    layer produces), `norm` is replicated, and `fused` has no checkpoint-name
    counterpart — the shape a fused parameter presents to the resolver.
    """

    def __init__(self) -> None:
        super().__init__()
        self.column = torch.nn.Parameter(torch.zeros(8, 16))
        self.column.output_dim = 0
        self.column.weight_loader = MethodType(
            ColumnParallelLinear.weight_loader, Mock(tp_size=2)
        )
        self.row = torch.nn.Parameter(torch.zeros(16, 8))
        self.row.input_dim = 1
        self.row.weight_loader = MethodType(
            RowParallelLinear.weight_loader, Mock(tp_size=2)
        )
        self.norm = torch.nn.Parameter(torch.zeros(16))
        self.norm.weight_loader = default_weight_loader
        self.fused = torch.nn.Parameter(torch.zeros(24, 16))


class _MixedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Module()
        self.proj.weight = torch.nn.Parameter(torch.zeros(16, 8))
        self.proj.weight.input_dim = 1
        self.proj.weight.weight_loader = MethodType(
            RowParallelLinear.weight_loader, Mock(tp_size=2)
        )
        self.proj.bias = torch.nn.Parameter(torch.zeros(16))


def _resolve(names, dtypes, shapes, **kwargs):
    defaults = dict(num_workers=2, shard_axis_size=2, allow_direct=True)
    defaults.update(kwargs)
    return resolve_parameter_destinations(_Model(), names, dtypes, shapes, **defaults)


class TestDestinationResolution:
    @pytest.mark.parametrize(
        "name,expected_dim",
        [("column", 0), ("row", 1)],
    )
    def test_sharded_parameter_resolves_to_its_tp_dim(self, name, expected_dim):
        [destination] = _resolve([name], [torch.float32], [(16, 16)])
        assert destination.direct
        assert destination.placements == (REPLICATE, expected_dim)

    def test_replicated_parameter_needs_no_placement(self):
        """A parameter every rank holds in full is REPLICATED, so it works
        whatever way the inference mesh happens to be factored."""
        [destination] = _resolve(["norm"], [torch.float32], [(16,)])
        assert destination.direct
        assert destination.placements is REPLICATED

    def test_direct_destination_is_the_live_parameter(self):
        model = _Model()
        [destination] = resolve_parameter_destinations(
            model,
            ["column"],
            [torch.float32],
            [(16, 16)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert destination.tensor.data_ptr() == model.column.data_ptr()

    def test_known_transforming_model_submodule_falls_back(self):
        class GPT2Model(_Model):
            pass

        GPT2Model.__module__ = "vllm.model_executor.models.gpt2"
        model = torch.nn.Module()
        model.transforming = GPT2Model()
        [destination] = resolve_parameter_destinations(
            model,
            ["transforming.row"],
            [torch.float32],
            [(16, 16)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )

        assert not destination.direct
        assert destination.placements is REPLICATED

    def test_unknown_name_falls_back(self):
        """Fused parameters reach the worker under checkpoint names that do not
        exist in the model; those must take the full-tensor path."""
        [destination] = _resolve(["mlp.gate_proj.weight"], [torch.float32], [(16, 16)])
        assert not destination.direct
        assert destination.placements is REPLICATED

    def test_fallback_parameter_demotes_direct_sibling_in_same_module(self):
        """Layerwise finalization must not overwrite a directly loaded weight
        when a sibling bias takes the fallback path."""
        destinations = resolve_parameter_destinations(
            _MixedModel(),
            ["proj.weight", "proj.bias"],
            [torch.float32, torch.float32],
            [(16, 16), (16,)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )

        assert not any(destination.direct for destination in destinations)

    def test_unknown_loader_falls_back_despite_matching_name_and_shape(self):
        model = _Model()
        model.norm.weight_loader = lambda param, weight: param.data.copy_(weight)
        [destination] = resolve_parameter_destinations(
            model,
            ["norm"],
            [torch.float32],
            [(16,)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert not destination.direct

    def test_missing_loader_is_not_assumed_to_be_a_plain_copy(self):
        model = _Model()
        del model.norm.weight_loader
        [destination] = resolve_parameter_destinations(
            model,
            ["norm"],
            [torch.float32],
            [(16,)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert not destination.direct

    def test_wrapped_known_loader_falls_back(self):
        model = _Model()

        @wraps(default_weight_loader)
        def wrapped_loader(param, weight):
            default_weight_loader(param, weight)

        model.norm.weight_loader = wrapped_loader
        [destination] = resolve_parameter_destinations(
            model,
            ["norm"],
            [torch.float32],
            [(16,)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert not destination.direct

    def test_replicated_loader_with_transform_metadata_falls_back(self):
        model = _Model()
        model.norm.is_transposed = True
        [destination] = resolve_parameter_destinations(
            model,
            ["norm"],
            [torch.float32],
            [(16,)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert not destination.direct

    def test_packed_dim_zero_falls_back(self):
        model = _Model()
        model.column.packed_dim = 0
        [destination] = resolve_parameter_destinations(
            model,
            ["column"],
            [torch.float32],
            [(16, 16)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert not destination.direct

    def test_loader_declared_dim_wins_over_shape_guessing(self):
        model = _Model()
        model.column.output_dim = 1
        [destination] = resolve_parameter_destinations(
            model,
            ["column"],
            [torch.float32],
            [(16, 16)],
            num_workers=2,
            shard_axis_size=2,
            allow_direct=True,
        )
        assert not destination.direct

    def test_shape_the_tp_factor_cannot_explain_falls_back(self):
        [destination] = _resolve(["fused"], [torch.float32], [(16, 16)])
        assert not destination.direct

    def test_dtype_mismatch_falls_back(self):
        """A parameter stored in a different dtype than the wire dtype means a
        quantized or otherwise transformed layout — never write into it."""
        [destination] = _resolve(["column"], [torch.bfloat16], [(16, 16)])
        assert not destination.direct

    def test_allow_direct_false_forces_every_parameter_to_fall_back(self):
        destinations = _resolve(
            ["column", "norm"],
            [torch.float32] * 2,
            [(16, 16), (16,)],
            allow_direct=False,
        )
        assert not any(d.direct for d in destinations)

    def test_plan_reports_byte_coverage(self, caplog_vllm):
        """A small direct parameter must not hide a large fallback tensor."""
        with caplog_vllm.at_level(
            logging.INFO,
            logger="vllm.distributed.weight_transfer.m2n_layout",
        ):
            _resolve(
                ["norm", "missing_fused_weight"],
                [torch.float32, torch.bfloat16],
                [(16,), (1024, 1024)],
            )

        assert (
            "1/2 parameters resharded directly into the model, 1 via "
            "full-tensor fallback; direct byte coverage: 64/2097216 bytes "
            "(0.0%)" in caplog_vllm.text
        )
