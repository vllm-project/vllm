# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding and the wire-type
validation that keeps a bad plan from reaching a collective, since a mismatch
there is a hang rather than an error. The transfer itself needs the m2n runtime
and multiple GPUs, so it is exercised separately.
"""

from unittest.mock import Mock

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
    resolve_layout,
    validate_layout,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    M2NWeightTransferEngine,
    M2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_source import (
    mesh_from_tensor,
    placements_from_tensor,
)
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


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

    def test_update_preflights_all_names_before_reshard(self):
        engine = object.__new__(M2NWeightTransferEngine)
        engine._handle = object()
        engine.model_update_group = object()
        engine._index = {"valid": 0}
        engine._metas = [
            M2NParamMeta("valid", torch.float32, (4,), REPLICATED)
        ]
        engine._reshard = Mock()

        with pytest.raises(ValueError, match=r"parameter 'unknown'"):
            engine.receive_weights(
                M2NWeightTransferUpdateInfo(names=["valid", "unknown"])
            )

        engine._reshard.assert_not_called()

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
def _m2n_trainer_send(master_address: str, master_port: int, world_size: int) -> bool:
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
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            num_trainer_ranks=1,
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
def _m2n_worker_receive(master_address: str, master_port: int, world_size: int) -> dict:
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

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = 1
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
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,  # trainer occupies rank 0
            world_size=world_size,
            src_mesh_dims=[1, 1],
            dst_mesh_dims=[1, 1],
            names=["weight"],
            dtype_names=[DTYPE],
            shapes=[SHAPE],
            src_placements=[None],  # replicated on the single trainer rank
        )
    )
    engine.receive_weights(M2NWeightTransferUpdateInfo(names=["weight"]))
    torch.accelerator.synchronize()

    expected = torch.arange(
        SHAPE[0] * SHAPE[1], dtype=torch.float32, device=device
    ).reshape(SHAPE)
    name, got = recorder.received[0] if recorder.received else (None, None)
    result = {
        "count": len(recorder.received),
        "name": name,
        "shape": list(got.shape) if got is not None else None,
        "exact": bool(torch.equal(got, expected)) if got is not None else False,
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

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2  # 1 trainer + 1 inference worker

    worker = _m2n_worker_receive.remote(master_address, master_port, world_size)
    trainer = _m2n_trainer_send.remote(master_address, master_port, world_size)
    trainer_ok, result = ray.get([trainer, worker])

    assert trainer_ok, "trainer engine did not complete"
    assert result["count"] == 1, f"expected one parameter, got {result['count']}"
    assert result["name"] == "weight"
    assert result["shape"] == SHAPE
    assert result["exact"], "resharded tensor does not match what the trainer sent"
