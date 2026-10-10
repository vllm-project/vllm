# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding and the wire-type
validation that keeps a bad plan from reaching a collective, since a mismatch
there is a hang rather than an error. The transfer itself needs the m2n runtime
and multiple GPUs, so it is exercised separately.
"""

import logging
import socket
from functools import wraps
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pybase64 as base64
import pytest
import ray
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    REPLICATE,
    REPLICATED,
    M2NLayout,
    M2NMesh,
    M2NNcclRuntime,
    M2NParamMeta,
    M2NWireParam,
    check_data_plane_agreement,
    check_placements,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    check_transferable,
    publish_destination_placements,
    resolve_layout,
    source_plan_digest,
    validate_layout,
    validate_local_tensor,
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
    M2NManifestEntry,
    M2NWeightSource,
    ManifestM2NWeightSource,
    mesh_from_tensor,
    placements_from_tensor,
)
from vllm.distributed.weight_transfer.m2n_trainer import (
    M2NTrainerInitInfo,
    M2NTrainerWeightTransferEngine,
)
from vllm.distributed.weight_transfer.nccl_common import (
    stateless_init_metadata_group,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

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

    def test_manifest_preserves_order_and_materializes_lazily(self):
        calls: list[int] = []

        def provider(index, value):
            def get_tensor():
                calls.append(index)
                return value

            return get_tensor

        source = ManifestM2NWeightSource(
            [
                M2NManifestEntry(
                    "second",
                    torch.float32,
                    (4,),
                    M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                    provider(0, torch.ones(4)),
                ),
                M2NManifestEntry(
                    "first",
                    torch.float32,
                    (2,),
                    M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                    provider(1, torch.full((2,), 2.0)),
                ),
            ]
        )

        assert [meta.name for meta in source.metadata()] == ["second", "first"]
        assert calls == []
        assert [name for name, _ in source] == ["second", "first"]
        assert calls == [0, 1]

    def test_trainer_boundary_rejects_duplicate_manifest_names(self):
        entry = M2NManifestEntry(
            "w",
            torch.float32,
            (4,),
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            lambda: torch.zeros(4),
        )
        source = ManifestM2NWeightSource([entry, entry])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match="duplicate parameter 'w'"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    @pytest.mark.parametrize(
        ("tensor", "match"),
        [
            (torch.zeros(4, dtype=torch.float32), "dtype"),
            (torch.zeros(2, 3).T, "non-contiguous"),
            (torch.zeros(3), "local shape"),
        ],
    )
    def test_local_tensor_must_match_manifest(self, tensor, match):
        shape = (3, 2) if match == "non-contiguous" else (4,)
        dtype = torch.bfloat16 if match == "dtype" else torch.float32
        meta = M2NParamMeta(
            "w",
            dtype,
            shape,
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
        )
        with pytest.raises(ValueError, match=rf"parameter 'w'.*{match}"):
            validate_local_tensor(meta, tensor)


class TestTransferable:
    def test_unsupported_dtype_names_the_parameter(self):
        with pytest.raises(ValueError, match="'w'"):
            check_transferable("w", torch.complex64, (4,))

    def test_rank_four_rejected(self):
        with pytest.raises(ValueError, match="rank 4"):
            check_transferable("w", torch.bfloat16, (2, 2, 2, 2))


class TestPreflightAgreement:
    def test_source_error_is_reported_before_data_plane(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {
                "phase": "source",
                "digest": None,
                "declared_digest": None,
                "error": "ValueError: bad local manifest",
            },
            {
                "phase": "source",
                "digest": None,
                "declared_digest": None,
                "error": "ValueError: bad local manifest",
            },
        ]

        with pytest.raises(RuntimeError, match="bad local manifest"):
            check_source_plan_agreement(
                group, [], local_error="ValueError: bad local manifest"
            )

    def test_data_plane_identity_must_match(self):
        group = Mock(world_size=2)

        def gather(envelope):
            peer = dict(envelope)
            peer["mode"] = "uid"
            peer["uid_digest"] = "different"
            return [envelope, peer]

        group.all_gather_obj.side_effect = gather
        with pytest.raises(RuntimeError, match="data-plane identity disagrees"):
            check_data_plane_agreement(group, None, max_cta=None)

    def test_runtime_error_is_shared_before_pynccl(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {"phase": "runtime_ready", "error": None},
            {"phase": "runtime_ready", "error": "ImportError: no nccl_m2n"},
        ]

        with pytest.raises(RuntimeError, match="no nccl_m2n"):
            check_runtime_ready_agreement(group, None)


class TestWireTypes:
    @staticmethod
    def _param(
        name="w",
        dtype_name="bfloat16",
        shape=(16, 16),
        src_mesh_dims=(1, 1),
        src_placements=None,
    ):
        return M2NWireParam(
            name=name,
            dtype_name=dtype_name,
            shape=shape,
            src_mesh_dims=src_mesh_dims,
            src_placements=src_placements,
        )

    def _init_info(self, **overrides):
        param = self._param()
        fields = dict(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=3,
            dst_mesh_dims=[2, 1],
            source_digest=source_plan_digest([param]),
            params=[param.to_dict()],
        )
        fields.update(overrides)
        if "params" in overrides and "source_digest" not in overrides:
            params = [M2NWireParam.from_dict(value) for value in fields["params"]]
            fields["source_digest"] = source_plan_digest(params)
        return M2NWeightTransferInitInfo(**fields)

    def test_accepts_a_consistent_plan(self):
        assert [param.name for param in self._init_info().parse_wire_params()] == ["w"]

    def test_uid_selects_data_plane_without_removing_metadata_rendezvous(self):
        info = self._init_info(nccl_unique_id_b64=VALID_UID_B64)
        assert info.master_address == "127.0.0.1"
        assert info.master_port == 1234
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

    def test_metadata_rendezvous_is_required_with_uid(self):
        with pytest.raises(ValueError, match="master_address"):
            self._init_info(master_address="", nccl_unique_id_b64=VALID_UID_B64)

    def test_trainer_accepts_pre_shared_nccl_unique_id(self):
        info = M2NTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=1234,
            nccl_unique_id_b64=VALID_UID_B64,
            world_size=3,
            num_trainer_ranks=1,
            rank=0,
        )
        assert info.nccl_unique_id_bytes == b"\x00" * 128
        assert VALID_UID_B64 not in repr(info)

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda value: value.pop("shape"),
            lambda value: value.update(extra="field"),
            lambda value: value.update(src_mesh_dims=[1]),
            lambda value: value.update(src_placements=[REPLICATE, REPLICATE]),
        ],
    )
    def test_malformed_wire_parameter_rejected(self, mutation):
        value = self._param().to_dict()
        mutation(value)
        with pytest.raises((TypeError, ValueError)):
            M2NWireParam.from_dict(value)

    def test_source_digest_authenticates_ordered_plan(self):
        params = [self._param("a"), self._param("b")]
        info = self._init_info(
            params=[param.to_dict() for param in params],
            source_digest=source_plan_digest(list(reversed(params))),
        )
        with pytest.raises(ValueError, match="source digest"):
            info.parse_wire_params()

    def test_trainer_disagreement_is_rejected_before_pynccl(self, monkeypatch):
        group = Mock(rank=1, world_size=3)

        def gather(envelope):
            peer = dict(envelope)
            peer["digest"] = "different"
            return [envelope, peer, envelope]

        group.all_gather_obj.side_effect = gather
        init_metadata = Mock(return_value=group)
        make_pynccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", Mock()
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.pynccl_from_metadata_group",
            make_pynccl,
        )
        vllm_config = SimpleNamespace(
            parallel_config=Mock(), model_config=SimpleNamespace(quantization=None)
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )

        with pytest.raises(RuntimeError, match="source digest disagrees"):
            engine.init_transfer_engine(self._init_info())

        make_pynccl.assert_not_called()

    def test_worker_rank_offset_selects_deployment_global_ranks(self, monkeypatch):
        group = Mock(rank=2, world_size=3)
        group.all_gather_obj.side_effect = lambda envelope: [envelope] * 3
        init_metadata = Mock(return_value=group)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", Mock()
        )
        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            lambda engine, info: setattr(engine, "_parameter_destinations", []),
        )
        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 1, object())
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(runtime, 0, Mock())),
        )
        comm = Mock()
        make_pynccl = Mock(return_value=comm)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.pynccl_from_metadata_group",
            make_pynccl,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "validate_m2n_nccl_communicator",
            Mock(),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "publish_destination_placements",
            Mock(return_value=[]),
        )
        vllm_config = SimpleNamespace(
            parallel_config=Mock(), model_config=SimpleNamespace(quantization=None)
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )
        info = self._init_info(params=[], worker_rank_offset=2)

        engine.init_transfer_engine(info)

        assert init_metadata.call_args.args[0].rank_offset == 2
        assert info.rank_offset == 1
        make_pynccl.assert_called_once_with(group, 0, library_path=runtime.library_path)

    def test_uid_data_plane_still_uses_tcp_metadata(self, monkeypatch):
        group = Mock(rank=1, world_size=3)
        group.all_gather_obj.side_effect = lambda envelope: [envelope] * 3
        init_metadata = Mock(return_value=group)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", Mock()
        )
        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            lambda engine, info: setattr(engine, "_parameter_destinations", []),
        )
        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 1, object())
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(runtime, 0, Mock())),
        )
        uid_init = Mock(return_value=Mock())
        tcp_init = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.uid_init_process_group",
            uid_init,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.pynccl_from_metadata_group",
            tcp_init,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "validate_m2n_nccl_communicator",
            Mock(),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "publish_destination_placements",
            Mock(return_value=[]),
        )
        vllm_config = SimpleNamespace(
            parallel_config=Mock(), model_config=SimpleNamespace(quantization=None)
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )

        engine.init_transfer_engine(
            self._init_info(params=[], nccl_unique_id_b64=VALID_UID_B64)
        )

        init_metadata.assert_called_once()
        uid_init.assert_called_once_with(
            b"\x00" * 128,
            rank=1,
            world_size=3,
            device=0,
            library_path=runtime.library_path,
        )
        tcp_init.assert_not_called()

    def test_partial_worker_initialization_aborts_owned_resources(self):
        vllm_config = SimpleNamespace(
            parallel_config=Mock(), model_config=SimpleNamespace(quantization=None)
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )
        handle = Mock()
        group = Mock()

        def fail_after_allocation(this, init_info):
            this._handle = handle
            this.model_update_group = group
            raise RuntimeError("late init failure")

        engine._init_transfer_engine = MethodType(fail_after_allocation, engine)

        with pytest.raises(RuntimeError, match="late init failure"):
            engine.init_transfer_engine(self._init_info())

        handle.destroy.assert_called_once_with()
        group.destroy.assert_called_once_with()
        assert engine._handle is None
        assert engine.model_update_group is None

    @pytest.mark.parametrize("worker_rank_offset", [0, 3])
    def test_worker_rank_offset_must_stay_in_destination_interval(
        self, worker_rank_offset
    ):
        with pytest.raises(ValueError, match="worker_rank_offset"):
            self._init_info(worker_rank_offset=worker_rank_offset)

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
    def test_invalid_dtype_name_names_parameter(self, dtype_name):
        engine = object.__new__(M2NWeightTransferEngine)
        param = self._param(dtype_name=dtype_name)

        with pytest.raises(ValueError, match=r"parameter 'w'.*dtype"):
            engine._prepare_source_plan((param,), num_trainer_ranks=1)

    def _init_32_to_4_plan(self, monkeypatch, destination):
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "resolve_parameter_destinations",
            lambda *args, **kwargs: [destination],
        )

        engine = object.__new__(M2NWeightTransferEngine)
        engine.parallel_config = Mock(pipeline_parallel_size=1)
        engine.model_config = Mock(quantization=None)
        engine.model = torch.nn.Module()
        param = self._param(
            shape=(32, 8),
            src_mesh_dims=(1, 32),
            src_placements=(REPLICATE, 0),
        )
        engine._prepare_source_plan((param,), num_trainer_ranks=32)
        engine._prepare_destination_plan(
            self._init_info(
                rank_offset=32,
                world_size=36,
                dst_mesh_dims=[1, 4],
                params=[param.to_dict()],
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
        engine._metas = [
            M2NParamMeta(
                "valid",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
        ]
        engine._reshard = Mock()

        with pytest.raises(ValueError, match=r"parameter 'unknown'"):
            engine._receive_weights(
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
            M2NParamMeta(
                name,
                torch.float32,
                (1,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
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

        engine._receive_weights(
            M2NWeightTransferUpdateInfo(names=["pair.weight", "direct", "pair.scale"])
        )

        assert model.calls == 1
        assert [name for name, _ in model.loaded] == ["pair.weight", "pair.scale"]
        assert [tensor.item() for _, tensor in model.loaded] == [1, 3]
        assert direct.item() == 2
        assert reshard_order == ["pair.weight", "direct", "pair.scale"]
        assert stream.synchronize.call_count == 2

    def test_subset_can_be_repeated_from_the_initialization_plan(self, monkeypatch):
        engine = object.__new__(M2NWeightTransferEngine)
        engine._handle = object()
        engine.model_update_group = object()
        engine.device = torch.device("cpu")
        engine.model = Mock()
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        engine._metas = [
            M2NParamMeta("a", torch.float32, (1,), layout),
            M2NParamMeta("b", torch.float32, (1,), layout),
        ]
        engine._index = {meta.name: index for index, meta in enumerate(engine._metas)}
        engine._parameter_destinations = [
            M2NDestination("a", REPLICATED, torch.zeros(1)),
            M2NDestination("b", REPLICATED, torch.zeros(1)),
        ]
        engine._reshard = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.comm_ptr", lambda group: 0
        )
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())

        update = M2NWeightTransferUpdateInfo(names=["b"])
        engine._receive_weights(update)
        engine._receive_weights(update)

        assert engine._reshard.call_count == 2
        assert all(call.args[2].name == "b" for call in engine._reshard.call_args_list)

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

    def test_reserved_socket_is_not_sent_to_workers(self):
        source = ManifestM2NWeightSource([])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=True
        )
        with socket.socket() as reservation:
            info = M2NTrainerInitInfo(
                rank=0,
                master_address="127.0.0.1",
                master_port=1234,
                world_size=2,
                listen_socket=reservation,
            )
            payload = engine._worker_init_info(info)

        assert "listen_socket" not in payload


def test_metadata_group_receives_reserved_socket(monkeypatch):
    reservation = Mock()
    create = Mock(return_value=Mock())
    monkeypatch.setattr("vllm.distributed.utils.StatelessProcessGroup.create", create)

    stateless_init_metadata_group(
        "127.0.0.1",
        1234,
        rank=0,
        world_size=2,
        listen_socket=reservation,
    )

    create.assert_called_once_with(
        host="127.0.0.1",
        port=1234,
        rank=0,
        world_size=2,
        listen_socket=reservation,
    )


class _StaticM2NSource(M2NWeightSource):
    def __init__(self, metadata, values):
        self._metadata = metadata
        self._values = values

    def metadata(self):
        return list(self._metadata)

    def __iter__(self):
        return iter(self._values)


class TestTrainerSourceContract:
    @staticmethod
    def _meta(name="weight"):
        return M2NParamMeta(
            name,
            torch.float32,
            (4,),
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
        )

    @staticmethod
    def _engine(source):
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )
        engine._m2n = Mock()
        engine._handle = object()
        engine._metas = source.metadata()
        engine._dst_mesh = M2NMesh((1, 1), 1)
        engine._dst_placements = [REPLICATED] * len(engine._metas)
        engine.group = Mock(comm=1)
        return engine

    def test_partial_trainer_initialization_aborts_owned_resources(self):
        source = _StaticM2NSource([], [])
        engine = self._engine(source)
        handle = Mock()
        group = Mock()
        executor = Mock()
        engine._handle = handle
        engine.group = group
        engine._executor = executor

        engine._abort(RuntimeError("init failed"))

        group.destroy.assert_called_once_with()
        handle.destroy.assert_called_once_with()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        assert engine.group is None
        assert engine._handle is None

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("missing", "first missing: 'weight'"),
            ("extra", "first extra"),
            ("reordered", "yielded 'other'.*declared 'weight'"),
        ],
    )
    def test_source_cardinality_and_order(self, monkeypatch, case, match):
        tensor = Mock(spec=torch.Tensor)
        tensor.dtype = torch.float32
        tensor.shape = (4,)
        tensor.device = torch.device("cuda:0")
        tensor.is_contiguous.return_value = True
        tensor.detach.return_value = tensor
        tensor.record_stream = Mock()
        values = {
            "missing": [],
            "extra": [("weight", tensor), ("extra", tensor)],
            "reordered": [("other", tensor)],
        }[case]
        source = _StaticM2NSource([self._meta()], values)
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match=match):
            engine._send()

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("dtype", "dtype"),
            ("shape", "local shape"),
            ("device", "expected cuda:0"),
            ("contiguity", "non-contiguous"),
        ],
    )
    def test_materialized_tensor_is_validated_before_reshard(
        self, monkeypatch, case, match
    ):
        tensor = Mock(spec=torch.Tensor)
        tensor.dtype = torch.float32
        tensor.shape = (4,)
        tensor.device = torch.device("cuda:0")
        tensor.is_contiguous.return_value = True
        if case == "dtype":
            tensor.dtype = torch.float16
        elif case == "shape":
            tensor.shape = (2,)
        elif case == "device":
            tensor.device = torch.device("cpu")
        else:
            tensor.is_contiguous.return_value = False

        source = _StaticM2NSource([self._meta()], [("weight", tensor)])
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(ValueError, match=match):
            engine._send()

        engine._m2n.reshard.assert_not_called()


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
def _m2n_trainer_send(
    master_address: str,
    master_port: int,
    world_size: int,
    nccl_unique_id_b64: str | None = None,
) -> bool:
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
    master_address: str,
    master_port: int,
    world_size: int,
    worker_rank: int = 0,
    worker_rank_offset: int | None = None,
    nccl_unique_id_b64: str | None = None,
) -> dict:
    """Receive that parameter through the real worker engine."""
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
    param = M2NWireParam("weight", DTYPE, tuple(SHAPE), (1, 1), REPLICATED)
    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address=master_address,
            master_port=master_port,
            nccl_unique_id_b64=nccl_unique_id_b64,
            rank_offset=1,  # trainer occupies rank 0
            world_size=world_size,
            dst_mesh_dims=[1, world_size - 1],
            source_digest=source_plan_digest([param]),
            params=[param.to_dict()],
            worker_rank_offset=worker_rank_offset,
        )
    )
    engine.start_weight_update()
    engine.receive_weights(M2NWeightTransferUpdateInfo(names=["weight"]))
    engine.finish_weight_update()
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


@ray.remote(num_gpus=1)
def _mixed_layout_trainer_send(
    master_address: str,
    master_port: int,
    trainer_rank: int,
) -> bool:
    """Send two tensors that use different 2-D source mesh factorizations."""
    device = _assigned_device()

    class NoopClient:
        def init_weight_transfer_engine(self, init_info):
            pass

        def start_weight_update(self):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self, weight_version=None):
            pass

    row_full = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    column_full = row_full + 100
    row_local = row_full.chunk(2, dim=0)[trainer_rank].contiguous()
    column_local = column_full.chunk(2, dim=1)[trainer_rank].contiguous()
    source = ManifestM2NWeightSource(
        [
            M2NManifestEntry(
                "row_sharded",
                torch.float32,
                (4, 4),
                M2NLayout(M2NMesh((1, 2), 0), (REPLICATE, 0)),
                lambda: row_local,
            ),
            M2NManifestEntry(
                "column_sharded",
                torch.float32,
                (4, 4),
                M2NLayout(M2NMesh((2, 1), 0), (1, REPLICATE)),
                lambda: column_local,
            ),
        ]
    )
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=M2NTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=4,
            num_trainer_ranks=2,
            dst_mesh_dims=(2, 1),
            rank=trainer_rank,
        ),
        client=NoopClient(),
        source=source,
    )
    engine.send_weights()
    engine.shutdown()
    return True


@ray.remote(num_gpus=1)
def _mixed_layout_worker_receive(
    master_address: str,
    master_port: int,
    worker_rank: int,
) -> dict[str, list]:
    device = _assigned_device()
    from unittest.mock import MagicMock

    from vllm.config.parallel import ParallelConfig

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = {}

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received[name] = tensor.clone()

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = worker_rank
    parallel_config.world_size = 2
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = 2
    parallel_config.pipeline_parallel_size = 1
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = SimpleNamespace(quantization=None)
    recorder = Recorder()
    engine = M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"),
        vllm_config,
        device,
        recorder,
    )
    params = [
        M2NWireParam("row_sharded", "float32", (4, 4), (1, 2), (REPLICATE, 0)),
        M2NWireParam("column_sharded", "float32", (4, 4), (2, 1), (1, REPLICATE)),
    ]
    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address=master_address,
            master_port=master_port,
            rank_offset=2,
            world_size=4,
            dst_mesh_dims=[2, 1],
            source_digest=source_plan_digest(params),
            params=[param.to_dict() for param in params],
        )
    )
    engine.start_weight_update()
    engine.receive_weights(
        M2NWeightTransferUpdateInfo(names=["row_sharded", "column_sharded"])
    )
    engine.finish_weight_update()
    result = {name: tensor.cpu().tolist() for name, tensor in recorder.received.items()}
    engine.shutdown()
    return result


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs: one trainer rank and one inference worker.",
)
@pytest.mark.parametrize("data_plane", ["tcp", "uid"])
def test_m2n_weight_transfer_between_processes(data_plane):
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
    nccl_unique_id_b64 = None
    if data_plane == "uid":
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

        nccl_unique_id_b64 = base64.b64encode(
            bytes(NCCLLibrary().ncclGetUniqueId().internal)
        ).decode()

    worker = _m2n_worker_receive.remote(
        master_address,
        master_port,
        world_size,
        nccl_unique_id_b64=nccl_unique_id_b64,
    )
    trainer = _m2n_trainer_send.remote(
        master_address,
        master_port,
        world_size,
        nccl_unique_id_b64,
    )
    trainer_ok, result = ray.get([trainer, worker], timeout=300)

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

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 3

    workers = [
        _m2n_worker_receive.remote(master_address, master_port, world_size, rank)
        for rank in range(2)
    ]
    trainer = _m2n_trainer_send.remote(master_address, master_port, world_size)
    trainer_ok, *results = ray.get([trainer, *workers], timeout=300)

    assert trainer_ok, "trainer engine did not complete"
    for rank, result in enumerate(results):
        assert result["count"] == 0, f"worker {rank} unexpectedly called load_weights"
        assert result["name"] == "weight"
        assert result["shape"] == [SHAPE[0] // 2, SHAPE[1]]
        assert result["direct"]
        assert result["exact"], f"worker {rank} received the wrong shard"


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4,
    reason="Need 4 GPUs: two trainer ranks and two inference workers.",
)
def test_m2n_transfer_with_per_parameter_source_meshes():
    """One collective sequence accepts heterogeneous source factorizations."""
    pytest.importorskip("nccl.m2n", reason="nccl_m2n backend needs the m2n runtime")
    _init_ray()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    trainers = [
        _mixed_layout_trainer_send.remote(master_address, master_port, rank)
        for rank in range(2)
    ]
    workers = [
        _mixed_layout_worker_receive.remote(master_address, master_port, rank)
        for rank in range(2)
    ]
    results = ray.get(trainers + workers, timeout=300)
    trainer_results, worker_results = results[:2], results[2:]

    assert trainer_results == [True, True]
    expected_row = torch.arange(16, dtype=torch.float32).reshape(4, 4).tolist()
    expected_column = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100
    ).tolist()
    for result in worker_results:
        assert result == {
            "row_sharded": expected_row,
            "column_sharded": expected_column,
        }


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
