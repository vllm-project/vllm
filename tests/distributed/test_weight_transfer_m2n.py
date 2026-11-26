# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding, whole-plan
validation, and the vLLM-owned source contract. The transfer itself needs the
M2N runtime and multiple GPUs, so it is exercised separately.
"""

import logging
from contextlib import contextmanager
from functools import cached_property, wraps
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, call

import pybase64 as base64
import pytest
import ray
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    ModuleSource,
    ParamMeta,
    Replicate,
    Shard,
    WeightSource,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    REPLICATE,
    REPLICATED,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    check_complete_plan_agreement,
    check_data_plane_agreement,
    check_plan_limits,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    create_m2n_communicator,
    expected_local_shape,
    m2n_param_meta,
    resolve_layout,
    source_plan_digest,
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


@pytest.mark.parametrize(
    "engine_type,group_attribute",
    [
        (M2NWeightTransferEngine, "model_update_group"),
        (M2NTrainerWeightTransferEngine, "group"),
    ],
)
def test_shutdown_releases_resources_once(monkeypatch, engine_type, group_attribute):
    engine = object.__new__(engine_type)
    handle = Mock()
    engine._handle = handle
    group = Mock()
    setattr(engine, group_attribute, group)
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    if engine_type is M2NWeightTransferEngine:
        engine._parameter_destinations = [Mock()]
    else:
        engine._dst_placements = [REPLICATED]
        engine._metadata_group = Mock()

    engine.shutdown()
    engine.shutdown()

    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    group.abort.assert_called_once_with()
    group.destroy.assert_not_called()
    assert engine._handle is None
    assert getattr(engine, group_attribute) is None
    if engine_type is M2NWeightTransferEngine:
        assert engine._parameter_destinations == []
    else:
        assert engine._dst_placements == []
        assert engine._metadata_group is None


@pytest.mark.parametrize(
    "engine_type,group_attribute",
    [
        (M2NWeightTransferEngine, "model_update_group"),
        (M2NTrainerWeightTransferEngine, "group"),
    ],
)
@pytest.mark.parametrize("failure", ["synchronize", "handle"])
def test_shutdown_attempts_all_cleanup_after_failure(
    monkeypatch, engine_type, group_attribute, failure
):
    engine = object.__new__(engine_type)
    handle = Mock()
    group = Mock()
    engine._handle = handle
    setattr(engine, group_attribute, group)
    if engine_type is M2NTrainerWeightTransferEngine:
        engine._metadata_group = Mock()

    synchronize = Mock()
    if failure == "synchronize":
        synchronize.side_effect = RuntimeError("synchronize failed")
    else:
        handle.destroy.side_effect = RuntimeError("handle destroy failed")
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    with pytest.raises(RuntimeError, match=failure):
        engine.shutdown()

    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    group.abort.assert_called_once_with()
    assert engine._handle is None
    assert getattr(engine, group_attribute) is None
    if engine_type is M2NTrainerWeightTransferEngine:
        assert engine._metadata_group is None

    engine.shutdown()
    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    group.abort.assert_called_once_with()


class _Shard(torch.Tensor):
    """A CPU tensor that records the streams it was handed to, since the tests
    run `_send` against a mocked stream."""

    def record_stream(self, stream):
        self.__dict__.setdefault("recorded", []).append(stream)


def _shard(shape=(4,), dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype).as_subclass(_Shard)


class _StaticSource(WeightSource):
    def __init__(self, metadata, values):
        self._metadata = metadata
        self._values = values

    @cached_property
    def metadata(self):
        return list(self._metadata)

    def iter_local(self):
        yield from self._values


def _fake_m2n():
    return SimpleNamespace(
        Mesh=lambda dims, start_rank=0: (tuple(dims), start_rank),
        Replicate=lambda: REPLICATE,
        Shard=lambda dim: dim,
        reshard=Mock(),
    )


class TestLayout:
    def test_replicated_splits_nothing(self):
        """A replicated tensor imposes no divisibility constraint. The shape
        dims are deliberately coprime with the mesh size: a layout that actually
        split the tensor would reject them."""
        mesh = M2NMesh((2, 2), start_rank=1)

        resolved_mesh, placements = resolve_layout(mesh, REPLICATED)
        assert resolved_mesh.dims == (4, 1)
        assert placements == (REPLICATE, 0)

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

    @pytest.mark.parametrize(
        ("placements", "match"),
        [((REPLICATE, 1), "tensor dim 1"), ((REPLICATE, 0), "divide evenly")],
    )
    def test_layout_validation_always_returns_shape_or_raises(self, placements, match):
        meta = M2NParamMeta(
            "w",
            torch.float32,
            (5,),
            M2NLayout(M2NMesh((1, 2), 0), placements),
        )

        with pytest.raises(ValueError, match=match):
            expected_local_shape(meta)

    def test_cross_sharded_fanout_over_capacity_is_rejected(self):
        src = (M2NMesh((1, 2), 0), (REPLICATE, 0))
        dst = (M2NMesh((1, 65), 2), (REPLICATE, 1))

        with pytest.raises(ValueError, match=r"parameter 'w'.*MAX_TARGETS=64"):
            check_plan_limits(src, dst, "w", (2, 65))


class TestSourceLayout:
    def test_declared_layout_converts_to_an_m2n_layout(self):
        """A base `ParamMeta` layout gains m2n's size-1 second axis; an
        unowned parameter is replicated over every trainer rank."""
        meta = ParamMeta(
            "w", torch.float32, (4, 4), owners=(0, 1), placements=(Shard(1),)
        )
        assert m2n_param_meta(meta, 2).source_layout == M2NLayout(
            M2NMesh((2, 1), 0), (1, REPLICATE)
        )
        plain = ParamMeta("b", torch.float32, (4,))
        assert m2n_param_meta(plain, 2).source_layout == M2NLayout(
            M2NMesh((2, 1), 0), REPLICATED
        )

    def test_partitioned_owners_are_rejected(self):
        """One communicator spans every trainer rank, so a parameter owned by
        only some of them (a pipeline stage) has no source group yet."""
        meta = ParamMeta("w", torch.float32, (4,), owners=(1,))
        with pytest.raises(ValueError, match=r"owned by trainer ranks \(1,\)"):
            m2n_param_meta(meta, 2)

    def test_trainer_boundary_rejects_duplicate_names(self):
        meta = ParamMeta("w", torch.float32, (4,))
        source = _StaticSource([meta, meta], [])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match="duplicate parameter 'w'"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    @pytest.mark.parametrize(
        ("dtype", "shape"),
        [(torch.complex64, (4,)), (torch.float32, (2, 2, 2, 2))],
    )
    def test_transport_capabilities_rejected_during_initialization(self, dtype, shape):
        source = _StaticSource([ParamMeta("w", dtype, shape)], [])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match=r"parameter 'w'.*(dtype|rank)"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    def test_uneven_shards_rejected_during_initialization(self):
        meta = ParamMeta(
            "w", torch.float32, (5,), owners=(0, 1), placements=(Shard(0),)
        )
        source = _StaticSource([meta], [])
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match=r"parameter 'w'.*divide evenly"):
            engine._prepare_source_plan(source, num_trainer_ranks=2)


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
            check_data_plane_agreement(group, None)

    def test_runtime_error_is_shared_before_nccl4py(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {"phase": "runtime_ready", "error": None},
            {"phase": "runtime_ready", "error": "ImportError: no nccl_m2n"},
        ]

        with pytest.raises(RuntimeError, match="no nccl_m2n"):
            check_runtime_ready_agreement(group, None)

    def test_complete_plan_error_is_shared_before_first_reshard(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {"phase": "complete_plan", "digest": "same", "error": None},
            {
                "phase": "complete_plan",
                "digest": None,
                "error": "ValueError: bad destination",
            },
        ]

        with pytest.raises(RuntimeError, match="bad destination"):
            check_complete_plan_agreement(
                group,
                "source",
                M2NMesh((1, 1), 1),
                [REPLICATED],
            )

    def test_complete_plan_digest_must_match(self):
        group = Mock(world_size=2)

        def gather(envelope):
            peer = dict(envelope)
            peer["digest"] = "different"
            return [envelope, peer]

        group.all_gather_obj.side_effect = gather
        with pytest.raises(RuntimeError, match="complete-plan digest disagrees"):
            check_complete_plan_agreement(
                group,
                "source",
                M2NMesh((1, 1), 1),
                [REPLICATED],
            )


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

    def test_m2n_communicator_uses_nccl4py_uid_under_unpinned_env(self, monkeypatch):
        nccl = Mock()
        group = Mock(rank=1, world_size=3)
        unique_id = object()
        communicator = object()
        nccl.UniqueId.from_bytes.return_value = unique_id
        active = False

        @contextmanager
        def unpinned():
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def init(**kwargs):
            assert active
            assert kwargs == {"nranks": 3, "rank": 1, "unique_id": unique_id}
            return communicator

        nccl.Communicator.init.side_effect = init
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_common.unpinned_nccl_env",
            unpinned,
        )

        result = create_m2n_communicator(nccl, group, b"shared-id")

        assert result is communicator
        nccl.UniqueId.from_bytes.assert_called_once_with(b"shared-id")
        nccl.Communicator.init.assert_called_once_with(
            nranks=3, rank=1, unique_id=unique_id
        )
        group.broadcast_obj.assert_not_called()

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda value: value.pop("shape"),
            lambda value: value.update(extra="field"),
            lambda value: value.update(src_mesh_dims=[1]),
            lambda value: value.update(src_placements=[0, 1]),
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

    @pytest.fixture
    def worker_init_env(self, monkeypatch):
        metadata_group = Mock(rank=1, world_size=3)
        metadata_group.all_gather_obj.side_effect = lambda envelope: [envelope] * 3
        init_metadata = Mock(return_value=metadata_group)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", Mock()
        )
        handle = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(Mock(), 0, handle)),
        )
        communicator = Mock()
        make_communicator = Mock(return_value=communicator)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
        )
        publish = Mock(return_value=[])
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "publish_destination_placements",
            publish,
        )
        vllm_config = SimpleNamespace(
            parallel_config=Mock(pipeline_parallel_size=1),
            model_config=SimpleNamespace(quantization=None),
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )
        return SimpleNamespace(
            engine=engine,
            metadata_group=metadata_group,
            init_metadata=init_metadata,
            handle=handle,
            communicator=communicator,
            make_communicator=make_communicator,
            publish=publish,
        )

    def test_trainer_disagreement_is_rejected_before_nccl4py(self, worker_init_env):
        group = worker_init_env.metadata_group

        def gather(envelope):
            peer = dict(envelope)
            peer["digest"] = "different"
            return [envelope, peer, envelope]

        group.all_gather_obj.side_effect = gather
        with pytest.raises(RuntimeError, match="source digest disagrees"):
            worker_init_env.engine.init_transfer_engine(self._init_info())

        worker_init_env.make_communicator.assert_not_called()

    def test_worker_rejects_active_reinit_but_allows_init_after_shutdown(
        self, monkeypatch, worker_init_env
    ):
        engine = worker_init_env.engine
        old_handle = Mock()
        old_group = Mock()
        engine._handle = old_handle
        engine.model_update_group = old_group

        with pytest.raises(RuntimeError, match="already initialized"):
            engine.init_transfer_engine(self._init_info(params=[]))
        worker_init_env.init_metadata.assert_not_called()

        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        engine.shutdown()

        engine.init_transfer_engine(self._init_info(params=[]))

        assert engine._handle is worker_init_env.handle
        assert engine.model_update_group is worker_init_env.communicator

    def test_partial_worker_initialization_releases_transport_resources(
        self, worker_init_env
    ):
        worker_init_env.publish.side_effect = RuntimeError("late init failure")

        with pytest.raises(RuntimeError, match="late init failure"):
            worker_init_env.engine.init_transfer_engine(self._init_info(params=[]))

        worker_init_env.communicator.abort.assert_called_once_with()
        worker_init_env.communicator.destroy.assert_not_called()
        worker_init_env.handle.destroy.assert_called_once_with()
        assert worker_init_env.engine.model_update_group is None
        assert worker_init_env.engine._handle is None

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

    def test_sharded_destination_within_capacity_initializes(self, monkeypatch):
        destination = M2NDestination("w", (REPLICATE, 0), torch.empty(8, 8))

        self._init_32_to_4_plan(monkeypatch, destination)

    def test_cross_sharded_destination_over_capacity_rejected_at_init(
        self, monkeypatch
    ):
        destination = M2NDestination("w", (REPLICATE, 1), torch.empty(32, 2))

        with pytest.raises(ValueError, match=r"parameter 'w'.*MAX_SOURCES=16"):
            self._init_32_to_4_plan(monkeypatch, destination)

    def test_replicated_destination_over_capacity_rejected_at_init(self, monkeypatch):
        destination = M2NDestination("w", REPLICATED, None)

        with pytest.raises(ValueError, match=r"parameter 'w'.*MAX_SOURCES=16"):
            self._init_32_to_4_plan(monkeypatch, destination)

    def test_invalid_destination_placement_rejected_at_init(self, monkeypatch):
        destination = M2NDestination("w", (REPLICATE, 2), torch.empty(8, 8))
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "resolve_parameter_destinations",
            lambda *args, **kwargs: [destination],
        )
        engine = object.__new__(M2NWeightTransferEngine)
        engine.parallel_config = Mock(pipeline_parallel_size=1)
        engine.model_config = Mock(quantization=None)
        engine.model = torch.nn.Module()
        param = self._param(shape=(32, 8))
        engine._prepare_source_plan((param,), num_trainer_ranks=1)

        with pytest.raises(
            ValueError, match=r"parameter 'w'.*destination layout.*tensor dim 2"
        ):
            engine._prepare_destination_plan(
                self._init_info(
                    rank_offset=1,
                    world_size=5,
                    dst_mesh_dims=[1, 4],
                    params=[param.to_dict()],
                )
            )

    def test_later_invalid_parameter_rejects_plan_before_any_reshard(self, monkeypatch):
        destinations = [
            M2NDestination("valid", (REPLICATE, 0), torch.empty(8, 8)),
            M2NDestination("invalid", (REPLICATE, 1), torch.empty(32, 2)),
        ]
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "resolve_parameter_destinations",
            lambda *args, **kwargs: destinations,
        )
        engine = object.__new__(M2NWeightTransferEngine)
        engine.parallel_config = Mock(pipeline_parallel_size=1)
        engine.model_config = Mock(quantization=None)
        engine.model = torch.nn.Module()
        params = (
            self._param(
                name="valid",
                shape=(32, 8),
                src_mesh_dims=(8, 4),
                src_placements=(REPLICATE, 0),
            ),
            self._param(
                name="invalid",
                shape=(32, 8),
                src_mesh_dims=(1, 32),
                src_placements=(REPLICATE, 0),
            ),
        )
        engine._prepare_source_plan(params, num_trainer_ranks=32)
        engine._reshard = Mock()

        with pytest.raises(ValueError, match=r"parameter 'invalid'.*MAX_SOURCES=16"):
            engine._prepare_destination_plan(
                self._init_info(
                    rank_offset=32,
                    world_size=36,
                    dst_mesh_dims=[1, 4],
                    params=[param.to_dict() for param in params],
                )
            )

        engine._reshard.assert_not_called()

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


class TestTrainerSourceContract:
    @staticmethod
    def _meta(name="weight"):
        return ParamMeta(name, torch.float32, (4,))

    @staticmethod
    def _engine(source):
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )
        engine._m2n = _fake_m2n()
        engine._handle = object()
        engine._prepare_source_plan(source, num_trainer_ranks=1)
        engine._dst_mesh = M2NMesh((1, 1), 1)
        engine._dst_placements = [REPLICATED] * len(engine._metas)
        engine.group = Mock(comm=1)
        return engine

    def test_partial_trainer_initialization_releases_transport_resources(
        self, monkeypatch
    ):
        metadata_group = Mock(rank=0, world_size=2)
        metadata_group.all_gather_obj.side_effect = lambda envelope: [envelope] * 2
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer."
            "stateless_init_metadata_group",
            Mock(return_value=metadata_group),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.import_m2n", Mock()
        )
        handle = Mock()
        nccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.prepare_m2n_local_runtime",
            Mock(return_value=(nccl, 0, handle)),
        )
        communicator = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.create_m2n_communicator",
            Mock(return_value=communicator),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer."
            "publish_destination_placements",
            Mock(return_value=[]),
        )
        source = _StaticSource([], [])
        client = Mock()
        client.init_weight_transfer_engine.side_effect = RuntimeError(
            "worker init failed"
        )
        engine = M2NTrainerWeightTransferEngine(
            client=client, source=source, is_sender=True
        )
        init_info = M2NTrainerInitInfo(
            rank=0,
            master_address="127.0.0.1",
            master_port=1234,
            world_size=2,
        )

        with pytest.raises(RuntimeError, match="worker init failed"):
            engine._initialize(init_info, source)

        communicator.abort.assert_called_once_with()
        communicator.destroy.assert_not_called()
        handle.destroy.assert_called_once_with()
        assert engine.group is None
        assert engine._handle is None

    def test_non_sender_observes_rank_zero_finish_failure(self):
        outcomes: dict[int, dict[str, object]] = {}
        sender_group = Mock()
        sender_group.send_obj.side_effect = lambda outcome, rank: outcomes.__setitem__(
            rank, outcome
        )
        sender_group.recv_obj.return_value = {"phase": "round_outcome_consumed"}

        source = _StaticSource([], [])
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = sender_group
        sender._num_trainer_ranks = 2
        sender._send = Mock()
        sender.client = Mock()
        sender.client.finish_weight_update.side_effect = RuntimeError("finish failed")

        with pytest.raises(RuntimeError, match="finish failed"):
            sender.send_weights()

        sender._send.assert_called_once_with()
        sender_group.recv_obj.assert_called_once_with(1)
        receiver_group = Mock()
        receiver_group.recv_obj.side_effect = lambda _source_rank: outcomes.pop(1)
        receiver = self._engine(source)
        receiver._metadata_group = receiver_group
        receiver._num_trainer_ranks = 2
        receiver._send = Mock()

        with pytest.raises(
            RuntimeError,
            match=(
                "worker update failed on trainer rank zero: RuntimeError: finish failed"
            ),
        ):
            receiver.send_weights()

        receiver._send.assert_called_once_with()
        receiver_group.recv_obj.assert_called_once_with(0)
        receiver_group.send_obj.assert_called_once_with(
            {"phase": "round_outcome_consumed"}, 0
        )
        assert not outcomes

    def test_sender_rejects_invalid_round_outcome_acknowledgement(self):
        source = _StaticSource([], [])
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = Mock()
        sender._metadata_group.recv_obj.return_value = {"phase": "wrong"}
        sender._num_trainer_ranks = 2
        sender._send = Mock()
        sender.client = Mock()

        with pytest.raises(RuntimeError, match="invalid round-outcome acknowledgement"):
            sender.send_weights()

    def test_non_sender_consumes_outcome_after_local_failure(self):
        source = _StaticSource([], [])
        receiver = self._engine(source)
        receiver._send = Mock(side_effect=[ValueError("local send failed"), None])
        receiver._metadata_group = Mock()
        receiver._metadata_group.recv_obj.side_effect = [
            {"ok": False, "error": "RuntimeError: first round failed"},
            {"ok": True, "error": None},
        ]

        with pytest.raises(ValueError, match="local send failed"):
            receiver.send_weights()
        receiver.send_weights()

        assert receiver._send.call_count == 2
        assert receiver._metadata_group.recv_obj.call_count == 2
        assert receiver._metadata_group.send_obj.call_args_list == [
            call({"phase": "round_outcome_consumed"}, 0),
            call({"phase": "round_outcome_consumed"}, 0),
        ]

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("missing", "first missing: 'weight'"),
            ("extra", "first extra"),
            ("reordered", "yielded 'other'.*declared 'weight'"),
        ],
    )
    def test_source_cardinality_and_order(self, monkeypatch, case, match):
        shard = _shard()
        values = {
            "missing": [],
            "extra": [("weight", shard), ("extra", shard)],
            "reordered": [("other", shard)],
        }[case]
        source = _StaticSource([self._meta()], values)
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match=match):
            engine._send()

    @pytest.mark.parametrize(
        "shard", [_shard(dtype=torch.float16), _shard(shape=(2,))], ids=str
    )
    def test_shard_must_match_declared_metadata(self, monkeypatch, shard):
        source = _StaticSource([self._meta()], [("weight", shard)])
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(ValueError, match="layout implies"):
            engine._send()

        engine._m2n.reshard.assert_not_called()

    def test_shard_is_resharded_and_recorded_on_the_send_stream(self, monkeypatch):
        shard = _shard()
        source = _StaticSource([self._meta()], [("weight", shard)])
        engine = self._engine(source)
        stream = Mock()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        assert engine._m2n.reshard.call_args.args[0] is shard
        assert shard.recorded == [stream]
        stream.synchronize.assert_called_once_with()


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
    trainer_rank: int = 0,
    mixed_layout: bool = False,
) -> bool:
    """Send either the basic or mixed-layout manifest through the real engine."""
    device = _assigned_device()

    from functools import cached_property

    from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
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

    num_trainer_ranks = 2 if mixed_layout else 1
    if mixed_layout:
        row_full = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        column_full = row_full + 100
        row_local = row_full.chunk(2, dim=0)[trainer_rank].contiguous()
        column_local = column_full.chunk(2, dim=1)[trainer_rank].contiguous()

        class MixedLayoutSource(WeightSource):
            """Declares each layout on the base `ParamMeta`; both shards come
            from `iter_local`."""

            @cached_property
            def metadata(self):
                return [
                    ParamMeta(
                        "row_sharded",
                        torch.float32,
                        (4, 4),
                        owners=(0, 1),
                        mesh_shape=(1, 2),
                        placements=(Replicate(), Shard(0)),
                    ),
                    ParamMeta(
                        "column_sharded",
                        torch.float32,
                        (4, 4),
                        owners=(0, 1),
                        placements=(Shard(1),),
                    ),
                ]

            def iter_local(self):
                yield "row_sharded", row_local
                yield "column_sharded", column_local

        source = MixedLayoutSource()
        dst_mesh_dims = (2, 1)
    else:
        source = ModuleSource(Tiny())
        dst_mesh_dims = (1, world_size - 1)

    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=M2NTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            nccl_unique_id_b64=nccl_unique_id_b64,
            world_size=world_size,
            num_trainer_ranks=num_trainer_ranks,
            dst_mesh_dims=dst_mesh_dims,
            rank=trainer_rank,
        ),
        client=NoopClient(),
        source=source,
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
    nccl_unique_id_b64: str | None = None,
    mixed_layout: bool = False,
) -> dict:
    """Receive either the basic or mixed-layout manifest through the engine."""
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
            self.received: list[tuple[str, torch.Tensor]] = []
            self.received_by_name: dict[str, torch.Tensor] = {}
            if not mixed_layout and world_size > 2:
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
                if mixed_layout:
                    self.received_by_name[name] = tensor.clone()
                else:
                    self.received.append((name, tensor.clone()))

    num_trainer_ranks = 2 if mixed_layout else 1
    num_workers = world_size - num_trainer_ranks
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = worker_rank
    parallel_config.world_size = num_workers
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = num_workers
    parallel_config.pipeline_parallel_size = 1
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.quantization = None

    recorder = Recorder()
    engine = M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"), vllm_config, device, recorder
    )
    if mixed_layout:
        params = [
            M2NWireParam("row_sharded", "float32", (4, 4), (1, 2), (REPLICATE, 0)),
            M2NWireParam("column_sharded", "float32", (4, 4), (2, 1), (1, REPLICATE)),
        ]
        names = [param.name for param in params]
        dst_mesh_dims = [2, 1]
    else:
        params = [M2NWireParam("weight", DTYPE, tuple(SHAPE), (1, 1), REPLICATED)]
        names = ["weight"]
        dst_mesh_dims = [1, num_workers]
    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address=master_address,
            master_port=master_port,
            nccl_unique_id_b64=nccl_unique_id_b64,
            rank_offset=num_trainer_ranks,
            world_size=world_size,
            dst_mesh_dims=dst_mesh_dims,
            source_digest=source_plan_digest(params),
            params=[param.to_dict() for param in params],
        )
    )
    engine.start_weight_update()
    engine.receive_weights(M2NWeightTransferUpdateInfo(names=names))
    engine.finish_weight_update()
    torch.accelerator.synchronize()

    if mixed_layout:
        result = {
            name: tensor.cpu().tolist()
            for name, tensor in recorder.received_by_name.items()
        }
        engine.shutdown()
        return result

    full = torch.arange(
        SHAPE[0] * SHAPE[1], dtype=torch.float32, device=device
    ).reshape(SHAPE)
    direct = num_workers > 1
    name: str | None
    got: torch.Tensor | None
    if direct:
        name, got = "weight", recorder.weight
        expected = full.chunk(num_workers, dim=0)[worker_rank]
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
        import nccl.core as nccl

        nccl_unique_id_b64 = base64.b64encode(
            bytes(nccl.get_unique_id().as_bytes)
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
        _m2n_trainer_send.remote(
            master_address,
            master_port,
            4,
            trainer_rank=rank,
            mixed_layout=True,
        )
        for rank in range(2)
    ]
    workers = [
        _m2n_worker_receive.remote(
            master_address,
            master_port,
            4,
            worker_rank=rank,
            mixed_layout=True,
        )
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
