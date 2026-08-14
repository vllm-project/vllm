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
