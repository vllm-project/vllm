# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding, the vLLM-owned
source contract, and collective-safe whole-plan validation. The transfer itself
needs the M2N runtime and multiple GPUs, so it is exercised separately.
"""

import gc
import json
import logging
import socket
import weakref
from contextlib import contextmanager
from functools import wraps
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, call

import pybase64 as base64
import pytest
import ray
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    RetryableWeightUpdateError,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    REPLICATE,
    REPLICATED,
    M2NDestinationMode,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NSourceTopology,
    M2NTransferRoute,
    M2NWireParam,
    build_m2n_param_routes,
    build_m2n_transfer_routes,
    build_m2n_transfer_waves,
    check_communicator_init_agreement,
    check_data_plane_agreement,
    check_multi_communicator_launch_order,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    create_m2n_communicator,
    m2n_load_group,
    resolve_layout,
    source_plan_digest,
    validate_local_buffer,
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
from vllm.distributed.weight_transfer.m2n_plan import (
    M2NWireDestination,
    agree_destination_plan,
)
from vllm.distributed.weight_transfer.m2n_source import (
    M2NManifestEntry,
    M2NWeightSource,
    ManifestM2NWeightSource,
    mesh_from_tensor,
    placements_from_tensor,
)
from vllm.distributed.weight_transfer.m2n_staging import (
    M2NStagingPool,
    M2NStagingRequirement,
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
from vllm.model_executor.model_loader.sharded_weight import (
    ShardedWeightSpec,
    ShardedWeightTarget,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

VALID_UID_B64 = base64.b64encode(b"\x00" * 128).decode()
SECOND_VALID_UID_B64 = base64.b64encode(b"\x01" * 128).decode()


def test_multi_communicator_launch_order_preflight(monkeypatch):
    monkeypatch.delenv("NCCL_LAUNCH_ORDER_IMPLICIT", raising=False)
    check_multi_communicator_launch_order(1)

    with pytest.raises(RuntimeError, match="NCCL_LAUNCH_ORDER_IMPLICIT=1"):
        check_multi_communicator_launch_order(2)

    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    check_multi_communicator_launch_order(2)


@pytest.mark.parametrize("num_routes", [0, -1, True, 1.5])
def test_multi_communicator_launch_order_rejects_invalid_route_count(num_routes):
    with pytest.raises(ValueError, match="at least one transfer route"):
        check_multi_communicator_launch_order(num_routes)


@pytest.mark.parametrize(
    "engine_type,groups_attribute,legacy_attribute",
    [
        (
            M2NWeightTransferEngine,
            "_model_update_groups",
            "model_update_group",
        ),
        (
            M2NTrainerWeightTransferEngine,
            "_route_communicators",
            "group",
        ),
    ],
)
def test_shutdown_releases_resources_once(
    monkeypatch,
    engine_type,
    groups_attribute,
    legacy_attribute,
):
    if engine_type is M2NWeightTransferEngine:
        vllm_config = SimpleNamespace(parallel_config=Mock(), model_config=Mock())
        engine = engine_type(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )
    else:
        engine = engine_type(client=Mock(), source=Mock(), is_sender=True)
    handle = Mock()
    engine._handle = handle
    groups = {0: Mock(), 1: Mock()}
    setattr(engine, groups_attribute, groups)
    setattr(engine, legacy_attribute, groups[0])
    stream = Mock()
    stream_ref = weakref.ref(stream)
    engine._route_streams = {0: stream}
    del stream
    engine._routes = (Mock(),)
    engine._param_routes = (0,)
    if engine_type is M2NTrainerWeightTransferEngine:
        engine._transfer_waves = (Mock(),)

    def assert_stream_alive():
        assert stream_ref() is not None

    synchronize = Mock(side_effect=assert_stream_alive)
    handle.destroy.side_effect = assert_stream_alive
    for group in groups.values():
        group.abort.side_effect = assert_stream_alive
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    engine.shutdown()
    engine.shutdown()

    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    for group in groups.values():
        group.abort.assert_called_once_with()
        group.destroy.assert_not_called()
    assert engine._handle is None
    assert getattr(engine, groups_attribute) == {}
    assert getattr(engine, legacy_attribute) is None
    assert engine._route_streams == {}
    assert engine._routes == ()
    assert engine._param_routes == ()
    if engine_type is M2NWeightTransferEngine:
        assert engine._source_topology == M2NSourceTopology((), ())
        assert engine._parameter_destinations == []
        assert engine._staged_targets == {}
    else:
        assert engine._source_topology is None
        assert engine._destination_plan == []
        assert engine._transfer_waves == ()
        assert engine._metadata_group is None


@pytest.mark.parametrize(
    "engine_type,groups_attribute,legacy_attribute",
    [
        (
            M2NWeightTransferEngine,
            "_model_update_groups",
            "model_update_group",
        ),
        (
            M2NTrainerWeightTransferEngine,
            "_route_communicators",
            "group",
        ),
    ],
)
@pytest.mark.parametrize("failure", ["synchronize", "handle", "communicator"])
def test_shutdown_attempts_all_cleanup_after_failure(
    monkeypatch,
    engine_type,
    groups_attribute,
    legacy_attribute,
    failure,
):
    if engine_type is M2NWeightTransferEngine:
        vllm_config = SimpleNamespace(parallel_config=Mock(), model_config=Mock())
        engine = engine_type(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )
    else:
        engine = engine_type(client=Mock(), source=Mock(), is_sender=True)
    handle = Mock()
    groups = {0: Mock(), 1: Mock()}
    engine._handle = handle
    setattr(engine, groups_attribute, groups)
    setattr(engine, legacy_attribute, groups[0])
    engine._route_streams = {0: Mock()}
    engine._routes = (Mock(),)
    engine._param_routes = (0,)
    if engine_type is M2NTrainerWeightTransferEngine:
        engine._transfer_waves = (Mock(),)

    synchronize = Mock()
    if failure == "synchronize":
        synchronize.side_effect = RuntimeError("synchronize failed")
    elif failure == "handle":
        handle.destroy.side_effect = RuntimeError("handle destroy failed")
    else:
        groups[0].abort.side_effect = RuntimeError("communicator abort failed")
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    with pytest.raises(RuntimeError, match=failure):
        engine.shutdown()

    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    for group in groups.values():
        group.abort.assert_called_once_with()
        group.destroy.assert_not_called()
    assert engine._handle is None
    assert getattr(engine, groups_attribute) == {}
    assert getattr(engine, legacy_attribute) is None
    assert engine._route_streams == {}
    assert engine._routes == ()
    assert engine._param_routes == ()
    if engine_type is M2NTrainerWeightTransferEngine:
        assert engine._transfer_waves == ()
        assert engine._source_topology is None
        assert engine._metadata_group is None
    else:
        assert engine._source_topology == M2NSourceTopology((), ())

    engine.shutdown()
    synchronize.assert_called_once_with()
    handle.destroy.assert_called_once_with()
    for group in groups.values():
        group.abort.assert_called_once_with()


class _BufferLike:
    """Minimal nccl4py buffer protocol without torch-only methods."""

    def __init__(self, dtype=torch.float32, shape=(4,)):
        self.dtype = dtype
        self.shape = shape

    def data_ptr(self):
        return 1234


class _AllocatorAwareBuffer(_BufferLike):
    def __init__(self):
        super().__init__()
        self.record_stream = Mock()


class _FakeDistTensor:
    def __init__(self, buffer, local_shape=None, dtype=None, **kwargs):
        del kwargs
        self.local_shape = (
            tuple(local_shape) if local_shape is not None else tuple(buffer.shape)
        )
        self.dtype = dtype if dtype is not None else buffer.dtype


def _fake_m2n():
    return SimpleNamespace(
        DistTensor=_FakeDistTensor,
        Mesh=lambda dims, start_rank=0: (tuple(dims), start_rank),
        Replicate=lambda: REPLICATE,
        Shard=lambda dim: dim,
        reshard=Mock(),
    )


@contextmanager
def _fake_cuda_stream_context(_stream):
    yield


def _set_pp1_worker_transport(
    engine,
    num_params,
    *,
    dst_mesh_dims=(1, 1),
    group=None,
    stream=None,
):
    topology = M2NSourceTopology.single_group(1, num_params)
    routes = build_m2n_transfer_routes(
        topology,
        num_trainer_ranks=1,
        world_size=1 + dst_mesh_dims[0] * dst_mesh_dims[1],
        dst_mesh_dims=dst_mesh_dims,
    )
    group = Mock() if group is None else group
    stream = Mock() if stream is None else stream
    engine._source_topology = topology
    engine._routes = routes
    engine._param_routes = build_m2n_param_routes(routes, num_params)
    engine._model_update_groups = {0: group}
    engine.model_update_group = group
    engine._route_streams = {0: stream}
    return group, stream


class TestSourceTopology:
    @staticmethod
    def _params() -> list[M2NWireParam]:
        return [
            M2NWireParam(name, "float32", (4,), (1, 2), (REPLICATE, 0))
            for name in ("first", "second", "third")
        ]

    def test_round_trip_and_digest_preserve_rank_order_and_parameter_ownership(self):
        params = self._params()
        topology = M2NSourceTopology(
            groups=((2, 0), (1, 3)),
            param_groups=(0, 1, 0),
        )

        topology.validate(num_trainer_ranks=4, num_params=3)

        assert M2NSourceTopology.from_dict(topology.to_dict()) == topology
        assert source_plan_digest(params, topology) != source_plan_digest(
            params,
            M2NSourceTopology(
                groups=((0, 2), (1, 3)),
                param_groups=(0, 1, 0),
            ),
        )
        assert source_plan_digest(params, topology) != source_plan_digest(
            params,
            M2NSourceTopology(
                groups=((2, 0), (1, 3)),
                param_groups=(1, 0, 1),
            ),
        )

    def test_routes_use_group_rank_order_and_global_parameter_order(self):
        topology = M2NSourceTopology(
            groups=((2, 0), (1, 3)),
            param_groups=(0, 1, 0),
        )

        routes = build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=4,
            world_size=6,
            dst_mesh_dims=(1, 2),
        )

        assert routes == (
            M2NTransferRoute(
                source_group_index=0,
                members=(2, 0, 4, 5),
                param_indices=(0, 2),
                dst_mesh=M2NMesh((1, 2), 2),
            ),
            M2NTransferRoute(
                source_group_index=1,
                members=(1, 3, 4, 5),
                param_indices=(1,),
                dst_mesh=M2NMesh((1, 2), 2),
            ),
        )

    def test_disjoint_groups_are_valid(self):
        topology = M2NSourceTopology(
            groups=((0, 1), (2, 3)),
            param_groups=(0, 1),
        )

        topology.validate(num_trainer_ranks=4, num_params=2)
        routes = build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=4,
            world_size=5,
            dst_mesh_dims=(1, 1),
        )

        assert [route.members for route in routes] == [
            (0, 1, 4),
            (2, 3, 4),
        ]

    @pytest.mark.parametrize(
        ("groups", "param_groups", "num_params"),
        [
            (((),), (0,), 1),
            (((0, 0),), (0,), 1),
            (((0, 4),), (0,), 1),
            (((0, 1), (0, 1)), (0, 1), 2),
            (((0, 1), (1, 2)), (0, 1), 2),
            (((0,), (1,)), (0, 0), 2),
            (((0,),), (0,), 2),
            (((0,),), (1, 0), 2),
        ],
    )
    def test_invalid_topology_is_rejected(self, groups, param_groups, num_params):
        with pytest.raises(ValueError):
            topology = M2NSourceTopology(
                groups=groups,
                param_groups=param_groups,
            )
            topology.validate(num_trainer_ranks=4, num_params=num_params)

    def test_pp1_normalizes_to_one_route(self):
        topology = M2NSourceTopology.single_group(
            num_trainer_ranks=2,
            num_params=3,
        )

        assert topology == M2NSourceTopology(
            groups=((0, 1),),
            param_groups=(0, 0, 0),
        )
        assert build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=2,
            world_size=4,
            dst_mesh_dims=(1, 2),
        ) == (
            M2NTransferRoute(
                source_group_index=0,
                members=(0, 1, 2, 3),
                param_indices=(0, 1, 2),
                dst_mesh=M2NMesh((1, 2), 2),
            ),
        )

    def test_transfer_waves_advance_stage_major_groups_together(self):
        topology = M2NSourceTopology(
            groups=((0, 1), (2, 3)),
            param_groups=(1, 1, 0, 0),
        )
        routes = build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=4,
            world_size=5,
            dst_mesh_dims=(1, 1),
        )

        waves = build_m2n_transfer_waves(
            routes,
            load_groups=(0, 1, 2, 3),
        )

        assert set(routes[0].source_ranks).isdisjoint(routes[1].source_ranks)
        assert [(wave.issue_indices, wave.consume_index) for wave in waves] == [
            ((2, 0), 0),
            ((), 2),
            ((3, 1), 1),
            ((), 3),
        ]
        owner = {
            index: route_index
            for route_index, route in enumerate(routes)
            for index in route.param_indices
        }
        pending: set[int] = set()
        for wave in waves:
            for index in wave.issue_indices:
                assert owner[index] not in pending
                pending.add(owner[index])
            if wave.consume_index is not None:
                pending.remove(owner[wave.consume_index])
        assert not pending

    def test_load_groups_preserve_only_required_model_load_order(self):
        assert m2n_load_group(M2NDestinationMode.IN_PLACE, None) is None
        assert m2n_load_group(M2NDestinationMode.FULL_FALLBACK, None) == -1
        assert m2n_load_group(M2NDestinationMode.SHARDED_STAGING, 3) == 3
        with pytest.raises(ValueError, match="integer staging group"):
            m2n_load_group(M2NDestinationMode.SHARDED_STAGING, None)

    def test_transfer_waves_preserve_cross_route_load_dependencies(self):
        topology = M2NSourceTopology(
            groups=((0, 1), (2, 3)),
            param_groups=(1, 1, 0, 0),
        )
        routes = build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=4,
            world_size=5,
            dst_mesh_dims=(1, 1),
        )

        waves = build_m2n_transfer_waves(routes, (-1, 0, 0, -1))

        assert [(wave.issue_indices, wave.consume_index) for wave in waves] == [
            ((2, 0), 0),
            ((1,), 1),
            ((), 2),
            ((3,), 3),
        ]

    def test_transfer_waves_require_increasing_active_indices(self):
        topology = M2NSourceTopology.single_group(1, 3)
        routes = build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=1,
            world_size=2,
            dst_mesh_dims=(1, 1),
        )

        with pytest.raises(
            ValueError, match="active parameter indices must be increasing"
        ):
            build_m2n_transfer_waves(
                routes,
                load_groups=(0, 0, 0),
                active_indices=(2, 1),
            )

    def test_route_mapping_and_schedule_do_not_use_source_group_as_route_id(self):
        routes = (
            M2NTransferRoute(
                source_group_index=7,
                members=(0, 2),
                param_indices=(1, 3),
                dst_mesh=M2NMesh((1, 1), 1),
            ),
            M2NTransferRoute(
                source_group_index=3,
                members=(1, 2),
                param_indices=(0, 2),
                dst_mesh=M2NMesh((1, 1), 1),
            ),
        )

        assert build_m2n_param_routes(routes, 4) == (1, 0, 1, 0)
        waves = build_m2n_transfer_waves(routes, (0, 1, 2, 3))
        assert [(wave.issue_indices, wave.consume_index) for wave in waves] == [
            ((1, 0), 0),
            ((), 1),
            ((3, 2), 2),
            ((), 3),
        ]


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

    def test_invalid_placement_is_rejected_during_initialization(self):
        layout = M2NLayout(M2NMesh((1, 2), 0), (REPLICATE, 3))
        source = ManifestM2NWeightSource(
            [
                M2NManifestEntry(
                    "w",
                    torch.float32,
                    (4,),
                    layout,
                    lambda: _BufferLike(),
                )
            ]
        )
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match="tensor dim 3"):
            engine._prepare_source_plan(source, num_trainer_ranks=2)


class TestSourceLayout:
    def test_plain_tensor_is_replicated_across_trainer_ranks(self):
        """A tensor with no DTensor metadata is the same on every trainer rank,
        so the source describes it as replicated over all of them."""
        assert placements_from_tensor(torch.zeros(4)) is REPLICATED
        assert mesh_from_tensor(torch.zeros(4), 4) == M2NMesh((4, 1), 0)

    def test_manifest_preserves_order_and_materializes_lazily(self):
        values = [torch.ones(4), torch.full((2,), 2.0)]
        calls: list[int] = []

        def provider(index):
            def get_tensor():
                calls.append(index)
                return values[index]

            return get_tensor

        source = ManifestM2NWeightSource(
            [
                M2NManifestEntry(
                    "second",
                    torch.float32,
                    (4,),
                    M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                    provider(0),
                ),
                M2NManifestEntry(
                    "first",
                    torch.float32,
                    (2,),
                    M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                    provider(1),
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
            client=Mock(),
            source=source,
            is_sender=False,
        )

        with pytest.raises(ValueError, match="duplicate parameter 'w'"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    @pytest.mark.parametrize(
        ("dtype", "shape"),
        [(torch.complex64, (4,)), (torch.float32, (2, 2, 2, 2))],
    )
    def test_unsupported_transport_capability_is_rejected_at_init(self, dtype, shape):
        source = ManifestM2NWeightSource(
            [
                M2NManifestEntry(
                    "w",
                    dtype,
                    shape,
                    M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                    lambda: _BufferLike(dtype, shape),
                )
            ]
        )
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match=r"parameter 'w'.*(support|rank)"):
            engine._prepare_source_plan(source, num_trainer_ranks=1)

    @pytest.mark.parametrize(
        ("placements", "match"),
        [((REPLICATE, 1), "tensor dim 1"), ((REPLICATE, 0), "divide evenly")],
    )
    def test_invalid_layout_geometry_is_rejected_at_init(self, placements, match):
        source = ManifestM2NWeightSource(
            [
                M2NManifestEntry(
                    "w",
                    torch.float32,
                    (5,),
                    M2NLayout(M2NMesh((1, 2), 0), placements),
                    lambda: _BufferLike(shape=(5,)),
                )
            ]
        )
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )

        with pytest.raises(ValueError, match=match):
            engine._prepare_source_plan(source, num_trainer_ranks=2)

    @pytest.mark.parametrize("case", ["dtype", "shape"])
    def test_local_buffer_must_match_declared_metadata(self, case):
        buffer = _BufferLike()
        if case == "dtype":
            buffer.dtype = torch.float16
        else:
            buffer.shape = (3,)
        meta = M2NParamMeta(
            "w",
            torch.float32,
            (4,),
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
        )
        match = "dtype" if case == "dtype" else "local shape"
        with pytest.raises(ValueError, match=rf"parameter 'w'.*{match}"):
            validate_local_buffer(_fake_m2n(), meta, buffer)


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
            peer["routes"] = list(reversed(envelope["routes"]))
            return [envelope, peer]

        group.all_gather_obj.side_effect = gather
        with pytest.raises(RuntimeError, match="data-plane identity disagrees"):
            check_data_plane_agreement(group, (b"first", b"second"))

    def test_runtime_error_is_shared_before_nccl4py(self):
        group = Mock(world_size=2)
        group.all_gather_obj.return_value = [
            {"phase": "runtime_ready", "error": None},
            {"phase": "runtime_ready", "error": "ImportError: no nccl_m2n"},
        ]

        with pytest.raises(RuntimeError, match="no nccl_m2n"):
            check_runtime_ready_agreement(group, None)

    def test_source_topology_disagreement_is_reported_before_nccl4py(self):
        params = [
            M2NWireParam("first", "float32", (4,), (1, 2), (REPLICATE, 0)),
            M2NWireParam("second", "float32", (4,), (1, 2), (REPLICATE, 0)),
        ]
        topology = M2NSourceTopology(
            groups=((0, 1), (2, 3)),
            param_groups=(0, 1),
        )
        peer_topology = M2NSourceTopology(
            groups=((1, 0), (2, 3)),
            param_groups=(0, 1),
        )
        peer_digest = source_plan_digest(params, peer_topology)
        group = Mock(world_size=2)

        def gather(envelope):
            peer = dict(envelope)
            peer["digest"] = peer_digest
            peer["declared_digest"] = peer_digest
            return [envelope, peer]

        group.all_gather_obj.side_effect = gather

        with pytest.raises(RuntimeError, match="source digest disagrees"):
            check_source_plan_agreement(group, params, topology=topology)

    @pytest.mark.parametrize(
        ("peer_errors", "match"),
        [
            ([None, "NCCL failed"], "communicator 1.*NCCL failed"),
            ([None], "reported 1 communicator results, expected 2"),
        ],
    )
    def test_one_shot_communicator_init_agreement_reaches_every_rank(
        self, peer_errors, match
    ):
        group = Mock(world_size=2)
        group.all_gather_obj.side_effect = lambda envelope: [
            envelope,
            {
                "phase": "communicator_init",
                "errors": peer_errors,
                "error": None,
            },
        ]

        with pytest.raises(RuntimeError, match=match):
            check_communicator_init_agreement(group, [None, None])

        group.all_gather_obj.assert_called_once()


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
        if {
            "params",
            "source_topology",
        } & overrides.keys() and "source_digest" not in overrides:
            wire_params = [M2NWireParam.from_dict(value) for value in fields["params"]]
            topology = fields.get("source_topology")
            fields["source_digest"] = source_plan_digest(
                wire_params,
                None if topology is None else M2NSourceTopology.from_dict(topology),
            )
        return M2NWeightTransferInitInfo(**fields)

    def test_accepts_a_consistent_plan(self):
        assert [param.name for param in self._init_info().parse_wire_params()] == ["w"]

    def test_parse_source_plan_preserves_per_parameter_groups(self):
        params = [
            self._param("first"),
            self._param("second"),
        ]
        topology = M2NSourceTopology(
            groups=((1,), (0,)),
            param_groups=(1, 0),
        )
        info = self._init_info(
            rank_offset=2,
            world_size=4,
            params=[param.to_dict() for param in params],
            source_topology=topology.to_dict(),
        )

        assert info.parse_source_plan() == (tuple(params), topology)

    def test_destination_and_schedule_policy_use_current_wire_schema(self):
        param = M2NWireParam(
            "w",
            "bfloat16",
            (16, 16),
            (1, 1),
            REPLICATED,
            allow_full_fallback=False,
        )

        assert M2N_WIRE_SCHEMA_VERSION == 4
        assert param.to_dict()["allow_full_fallback"] is False
        assert self._init_info(
            params=[param.to_dict()],
            source_digest=source_plan_digest([param]),
        ).parse_wire_params() == (param,)

    def test_uid_selects_data_plane_without_removing_metadata_rendezvous(self):
        info = self._init_info(nccl_unique_id_b64=VALID_UID_B64)

        assert info.master_address == "127.0.0.1"
        assert info.master_port == 1234
        assert info.nccl_unique_id_bytes == b"\x00" * 128
        assert VALID_UID_B64 not in repr(info)

    def test_plural_uids_preserve_transfer_route_order(self):
        encoded = [VALID_UID_B64, SECOND_VALID_UID_B64]
        expected = (b"\x00" * 128, b"\x01" * 128)
        worker_info = self._init_info(nccl_unique_ids_b64=encoded)
        trainer_info = M2NTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=1234,
            world_size=3,
            num_trainer_ranks=2,
            rank=0,
            nccl_unique_ids_b64=encoded,
        )

        assert worker_info.nccl_unique_id_bytes_by_route(2) == expected
        assert trainer_info.data_plane_unique_ids(2) == expected

    def test_plural_uid_count_must_match_transfer_routes(self):
        worker_info = self._init_info(nccl_unique_ids_b64=[VALID_UID_B64])
        trainer_info = M2NTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=1234,
            world_size=3,
            num_trainer_ranks=2,
            rank=0,
            nccl_unique_ids_b64=[VALID_UID_B64],
        )

        with pytest.raises(ValueError, match="exactly one UID per transfer"):
            worker_info.nccl_unique_id_bytes_by_route(2)
        with pytest.raises(ValueError, match="1 entries, expected 2"):
            trainer_info.data_plane_unique_ids(2)

    def test_singular_uid_is_rejected_for_multiple_transfer_routes(self):
        worker_info = self._init_info(nccl_unique_id_b64=VALID_UID_B64)
        trainer_info = M2NTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=1234,
            world_size=3,
            num_trainer_ranks=2,
            rank=0,
            nccl_unique_id_b64=VALID_UID_B64,
        )

        with pytest.raises(ValueError, match="single transfer"):
            worker_info.nccl_unique_id_bytes_by_route(2)
        with pytest.raises(ValueError, match="only for one transfer"):
            trainer_info.data_plane_unique_ids(2)

    def test_data_plane_preflight_rejects_duplicate_pre_shared_uids(self):
        group = Mock(world_size=2)
        group.all_gather_obj.side_effect = lambda envelope: [envelope, envelope]

        with pytest.raises(RuntimeError, match="distinct"):
            check_data_plane_agreement(group, [b"same", b"same"])

        group.all_gather_obj.assert_called_once()

    def test_m2n_communicator_uses_nccl4py_uid(self, monkeypatch):
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

    def test_m2n_communicator_generates_uid_on_rank_zero(self):
        nccl = Mock()
        group = Mock(rank=0, world_size=3)
        group.broadcast_obj.side_effect = lambda envelope, src: envelope
        generated = Mock(as_bytes=b"generated-id")
        unique_id = object()
        nccl.get_unique_id.return_value = generated
        nccl.UniqueId.from_bytes.return_value = unique_id

        create_m2n_communicator(nccl, group, None)

        group.broadcast_obj.assert_called_once_with(
            {"unique_id": b"generated-id", "error": None}, src=0
        )
        nccl.UniqueId.from_bytes.assert_called_once_with(b"generated-id")

    def test_m2n_communicator_maps_global_rank_to_ordered_route_rank(self, monkeypatch):
        nccl = Mock()
        metadata_group = Mock(rank=0, world_size=6)
        unique_id = object()
        communicator = object()
        nccl.UniqueId.from_bytes.return_value = unique_id
        nccl.Communicator.init.return_value = communicator

        @contextmanager
        def unpinned():
            yield

        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_common.unpinned_nccl_env",
            unpinned,
        )

        result = create_m2n_communicator(
            nccl,
            metadata_group,
            b"shared-id",
            members=(2, 0, 4, 5),
        )

        assert result is communicator
        nccl.Communicator.init.assert_called_once_with(
            nranks=4,
            rank=1,
            unique_id=unique_id,
        )

    def test_route_nonmember_joins_uid_broadcast_without_nccl_init(self):
        nccl = Mock()
        metadata_group = Mock(rank=0, world_size=6)
        metadata_group.broadcast_obj.return_value = {
            "unique_id": b"generated-id",
            "error": None,
        }

        result = create_m2n_communicator(
            nccl,
            metadata_group,
            None,
            members=(2, 3, 4, 5),
        )

        assert result is None
        metadata_group.broadcast_obj.assert_called_once_with(
            None,
            src=2,
        )
        nccl.get_unique_id.assert_not_called()
        nccl.Communicator.init.assert_not_called()
        nccl.UniqueId.from_bytes.assert_not_called()

    def test_metadata_rendezvous_is_required_with_uid(self):
        with pytest.raises(ValueError, match="master_address"):
            self._init_info(master_address="", nccl_unique_id_b64=VALID_UID_B64)

    def test_trainer_accepts_pre_shared_nccl_unique_id(self):
        trainer_info = M2NTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=1234,
            nccl_unique_id_b64=VALID_UID_B64,
            world_size=3,
            num_trainer_ranks=1,
            rank=0,
        )
        assert trainer_info.nccl_unique_id_bytes == b"\x00" * 128
        assert VALID_UID_B64 not in repr(trainer_info)

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda value: value.pop("shape"),
            lambda value: value.update(extra="field"),
            lambda value: value.update(src_mesh_dims=[1]),
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

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rank_offset", True),
            ("rank_offset", "1"),
            ("world_size", True),
            ("world_size", "3"),
            ("worker_rank_offset", True),
            ("worker_rank_offset", "1"),
        ],
    )
    def test_rank_fields_require_integers(self, field, value):
        with pytest.raises(ValueError, match=field):
            self._init_info(**{field: value})

    def test_init_validates_destination_mesh_coverage(self):
        with pytest.raises(ValueError, match="dst_mesh_dims"):
            self._init_info(dst_mesh_dims=[3, 1])

    def test_parse_requires_list_params_and_nonempty_digest(self):
        info = self._init_info()
        info.params = ()
        with pytest.raises(TypeError, match="params.*list"):
            info.parse_wire_params()

        info = self._init_info()
        info.source_digest = ""
        with pytest.raises(ValueError, match="source_digest.*non-empty"):
            info.parse_wire_params()

    @pytest.mark.parametrize(
        ("field", "value"), [("name", 1), ("dtype_name", torch.float32)]
    )
    def test_wire_parameter_names_and_dtypes_require_strings(self, field, value):
        with pytest.raises(ValueError):
            self._param(**{field: value})

    def test_trainer_disagreement_is_rejected_before_nccl4py(self, monkeypatch):
        group = Mock(rank=1, world_size=3)

        def gather(envelope):
            peer = dict(envelope)
            peer["digest"] = "different"
            return [envelope, peer, envelope]

        group.all_gather_obj.side_effect = gather
        init_metadata = Mock(return_value=group)
        make_communicator = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n", Mock()
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
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

        make_communicator.assert_not_called()

    def test_worker_rejects_active_reinitialization(self, monkeypatch):
        init_metadata = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
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
        engine._handle = object()
        engine._model_update_groups = {0: object()}

        with pytest.raises(RuntimeError, match="already initialized"):
            engine.init_transfer_engine(self._init_info(params=[]))

        init_metadata.assert_not_called()

    def test_worker_rank_offset_selects_deployment_global_ranks(self, monkeypatch):
        group = Mock(rank=2, world_size=3)
        group.all_gather_obj.side_effect = lambda envelope: [envelope] * 3
        init_metadata = Mock(return_value=group)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata,
        )
        m2n = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n",
            Mock(return_value=m2n),
        )
        param = self._param()
        digest = source_plan_digest([param])
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.check_source_plan_agreement",
            Mock(return_value=digest),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.agree_destination_plan",
            Mock(return_value=([], "m2n-plan:test")),
        )
        destination = M2NDestination(
            name="w",
            mode=M2NDestinationMode.FULL_FALLBACK,
            placements=REPLICATED,
            local_shape=(16, 16),
            tensor=None,
        )

        def prepare_destination(engine, **_kwargs):
            engine._destination_shard_index = 0
            engine._parameter_destinations = [destination]

        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            prepare_destination,
        )
        nccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(nccl, 0, Mock())),
        )
        comm = Mock()
        make_communicator = Mock(return_value=comm)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
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
        info = self._init_info(worker_rank_offset=2)

        engine.init_transfer_engine(info)

        assert init_metadata.call_args.args[0].rank_offset == 2
        assert info.rank_offset == 1
        make_communicator.assert_called_once_with(
            nccl,
            group,
            None,
            members=(0, 1, 2),
        )

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
            "vllm.distributed.weight_transfer.m2n_engine.agree_destination_plan",
            Mock(return_value=([], "m2n-plan:test")),
        )
        destination = M2NDestination(
            name="w",
            mode=M2NDestinationMode.FULL_FALLBACK,
            placements=REPLICATED,
            local_shape=(16, 16),
            tensor=None,
        )

        def prepare_destination(engine, **_kwargs):
            engine._destination_shard_index = 0
            engine._parameter_destinations = [destination]

        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            prepare_destination,
        )
        nccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(nccl, 0, Mock())),
        )
        make_communicator = Mock(return_value=Mock())
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
        )
        vllm_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.model_config = Mock()
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )
        info = self._init_info(nccl_unique_id_b64=VALID_UID_B64)

        engine.init_transfer_engine(info)

        init_metadata.assert_called_once()
        make_communicator.assert_called_once_with(
            nccl,
            group,
            b"\x00" * 128,
            members=(0, 1, 2),
        )

    def test_destination_mesh_must_cover_the_workers(self):
        """The trainer declares the inference mesh, so one that does not cover
        the workers is a preflight error rather than a mismatched collective."""
        engine = object.__new__(M2NWeightTransferEngine)
        engine._dst_mesh = M2NMesh((3, 1), 1)
        with pytest.raises(ValueError, match="dst_mesh_dims"):
            engine._prepare_destination_plan(
                metadata_rank=1,
                num_workers=2,
            )

    def test_source_mesh_must_cover_its_group_on_worker(self):
        engine = object.__new__(M2NWeightTransferEngine)
        param = self._param(src_mesh_dims=(2, 1))
        topology = M2NSourceTopology.single_group(1, 1)

        with pytest.raises(ValueError, match="covers 2 ranks.*group 0 contains 1"):
            engine._prepare_source_plan((param,), topology)

    def test_destination_planning_failure_precedes_nccl4py(self, monkeypatch):
        group = Mock(rank=1)
        make_communicator = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            Mock(return_value=group),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.check_source_plan_agreement",
            Mock(return_value=source_plan_digest([self._param()])),
        )
        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            Mock(side_effect=ValueError("bad destination plan")),
        )

        def reject_plan(*args, local_error, **kwargs):
            assert "bad destination plan" in local_error
            raise RuntimeError(local_error)

        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.agree_destination_plan",
            reject_plan,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
        )
        vllm_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.model_config = Mock()
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )

        with pytest.raises(RuntimeError, match="bad destination plan"):
            engine.init_transfer_engine(self._init_info())

        make_communicator.assert_not_called()

    def test_destination_agreement_failure_releases_staging_before_nccl(
        self, monkeypatch
    ):
        metadata_group = Mock(rank=1, world_size=3)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            Mock(return_value=metadata_group),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.check_source_plan_agreement",
            Mock(return_value=source_plan_digest([])),
        )

        def prepare_destination_plan(engine, **_kwargs):
            engine._parameter_destinations = []
            engine._destination_shard_index = 0
            engine._staging_pool = Mock()
            engine._uses_load_weights = True

        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            prepare_destination_plan,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.agree_destination_plan",
            Mock(side_effect=RuntimeError("destination agreement failed")),
        )
        prepare_runtime = Mock()
        make_communicator = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            prepare_runtime,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
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

        with pytest.raises(RuntimeError, match="destination agreement failed"):
            engine.init_transfer_engine(self._init_info(params=[]))

        prepare_runtime.assert_not_called()
        make_communicator.assert_not_called()
        assert engine._dst_mesh is None
        assert engine._parameter_destinations == []
        assert engine._destination_shard_index is None
        assert engine._staging_pool is None
        assert not engine._uses_load_weights

    def test_partial_worker_initialization_releases_transport_resources(
        self, monkeypatch
    ):
        metadata_group = Mock(rank=2, world_size=3)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            Mock(return_value=metadata_group),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.import_m2n",
            Mock(return_value=Mock()),
        )
        params = [self._param("first"), self._param("second")]
        topology = M2NSourceTopology(
            groups=((0,), (1,)),
            param_groups=(0, 1),
        )
        digest = source_plan_digest(params, topology)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.check_source_plan_agreement",
            Mock(return_value=digest),
        )

        def prepare_destination_plan(engine, **_kwargs):
            engine._parameter_destinations = [
                M2NDestination(
                    param.name,
                    M2NDestinationMode.FULL_FALLBACK,
                    REPLICATED,
                    param.shape,
                    None,
                )
                for param in params
            ]
            engine._destination_shard_index = 0
            engine._staging_pool = Mock()
            engine._uses_load_weights = True

        monkeypatch.setattr(
            M2NWeightTransferEngine,
            "_prepare_destination_plan",
            prepare_destination_plan,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.agree_destination_plan",
            Mock(return_value=([], "m2n-plan:test")),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.check_data_plane_agreement",
            Mock(),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.check_runtime_ready_agreement",
            Mock(),
        )

        def check_communicator_init(_group, local_errors):
            assert local_errors[0] is None
            assert "communicator init failed" in local_errors[1]
            raise RuntimeError(local_errors[1])

        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine."
            "check_communicator_init_agreement",
            check_communicator_init,
        )
        handle = Mock()
        nccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(nccl, 0, handle)),
        )
        first_communicator = Mock()
        make_communicator = Mock(
            side_effect=[
                first_communicator,
                RuntimeError("communicator init failed"),
            ]
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.create_m2n_communicator",
            make_communicator,
        )
        monkeypatch.setattr(torch.cuda, "Stream", Mock(return_value=Mock()))
        vllm_config = SimpleNamespace(
            parallel_config=Mock(), model_config=SimpleNamespace(quantization=None)
        )
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            torch.nn.Module(),
        )

        with pytest.raises(RuntimeError, match="communicator init failed"):
            engine.init_transfer_engine(
                self._init_info(
                    rank_offset=2,
                    world_size=3,
                    dst_mesh_dims=[1, 1],
                    params=[param.to_dict() for param in params],
                    source_topology=topology.to_dict(),
                )
            )

        assert make_communicator.call_args_list == [
            call(nccl, metadata_group, None, members=(0, 2)),
            call(nccl, metadata_group, None, members=(1, 2)),
        ]
        first_communicator.abort.assert_called_once_with()
        handle.destroy.assert_called_once_with()
        assert engine.model_update_group is None
        assert engine._model_update_groups == {}
        assert engine._route_streams == {}
        assert engine._routes == ()
        assert engine._handle is None
        assert engine._dst_mesh is None
        assert engine._parameter_destinations == []
        assert engine._destination_shard_index is None
        assert engine._staging_pool is None
        assert not engine._uses_load_weights

    def test_world_must_hold_a_trainer_and_a_worker(self):
        with pytest.raises(ValueError, match="rank_offset"):
            self._init_info(rank_offset=3, world_size=3)

    @pytest.mark.parametrize("worker_rank_offset", [0, 3])
    def test_worker_rank_offset_must_stay_in_destination_interval(
        self, worker_rank_offset
    ):
        with pytest.raises(ValueError, match="worker_rank_offset"):
            self._init_info(worker_rank_offset=worker_rank_offset)

    @pytest.mark.parametrize("dtype_name", ["not_a_dtype", "Tensor"])
    def test_invalid_dtype_name_names_parameter(self, dtype_name):
        engine = object.__new__(M2NWeightTransferEngine)
        param = self._param(dtype_name=dtype_name)

        with pytest.raises(ValueError, match=r"parameter 'w'.*dtype"):
            engine._prepare_source_plan(
                (param,),
                M2NSourceTopology.single_group(1, 1),
            )

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
        engine._prepare_source_plan(
            (param,),
            M2NSourceTopology.single_group(32, 1),
        )
        engine._dst_mesh = M2NMesh((1, 4), 32)
        engine._prepare_destination_plan(metadata_rank=32, num_workers=4)

    def test_sharded_destination_builds_before_complete_plan_agreement(
        self, monkeypatch
    ):
        destination = M2NDestination(
            "w",
            M2NDestinationMode.IN_PLACE,
            (REPLICATE, 0),
            (8, 8),
            torch.empty(8, 8),
        )

        self._init_32_to_4_plan(monkeypatch, destination)

    def test_update_preflights_all_names_before_reshard(self):
        vllm_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.model_config = Mock()
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )
        engine._handle = Mock()
        _set_pp1_worker_transport(engine, 1)
        engine._staging_pool = None
        engine._staged_targets = {}
        engine._next_parameter_index = 0
        engine._metas = [
            M2NParamMeta(
                "valid",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
        ]
        engine._reshard = Mock()
        group = engine.model_update_group
        handle = engine._handle

        with pytest.raises(ValueError, match=r"update order mismatch.*unknown"):
            engine.receive_weights(
                M2NWeightTransferUpdateInfo(names=["valid", "unknown"])
            )

        engine._reshard.assert_not_called()
        group.abort.assert_not_called()
        handle.destroy.assert_not_called()
        assert engine.model_update_group is group
        assert engine._handle is handle

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
        engine.device = torch.device("cpu")
        engine.model = model
        engine._staging_pool = None
        engine._staged_targets = {}
        engine._next_parameter_index = 0
        names = ["pair.weight", "direct", "pair.scale"]
        engine._metas = [
            M2NParamMeta(
                name,
                torch.float32,
                (1,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
            for name in names
        ]
        engine._parameter_destinations = [
            M2NDestination(
                "pair.weight",
                M2NDestinationMode.FULL_FALLBACK,
                REPLICATED,
                (1,),
                None,
            ),
            M2NDestination(
                "direct",
                M2NDestinationMode.IN_PLACE,
                REPLICATED,
                (1,),
                direct,
            ),
            M2NDestination(
                "pair.scale",
                M2NDestinationMode.FULL_FALLBACK,
                REPLICATED,
                (1,),
                None,
            ),
        ]
        stream = Mock()
        _set_pp1_worker_transport(engine, len(names), stream=stream)
        reshard_order = []

        def reshard(comm, stream, route_dst_mesh, meta, placements, buffer):
            reshard_order.append(meta.name)
            buffer.fill_(len(reshard_order))

        engine._reshard = reshard

        engine.receive_weights(M2NWeightTransferUpdateInfo(names=names))

        assert model.calls == 1
        assert [name for name, _ in model.loaded] == ["pair.weight", "pair.scale"]
        assert [tensor.item() for _, tensor in model.loaded] == [1, 3]
        assert direct.item() == 2
        assert reshard_order == ["pair.weight", "direct", "pair.scale"]
        assert stream.synchronize.call_count == 3

    def test_contiguous_chunks_follow_the_initialization_plan(self, monkeypatch):
        engine = object.__new__(M2NWeightTransferEngine)
        engine._handle = object()
        engine.device = torch.device("cpu")
        engine.model = Mock()
        engine._staging_pool = None
        engine._staged_targets = {}
        engine._next_parameter_index = 0
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        engine._metas = [
            M2NParamMeta("a", torch.float32, (1,), layout),
            M2NParamMeta("b", torch.float32, (1,), layout),
        ]
        engine._parameter_destinations = [
            M2NDestination(
                "a",
                M2NDestinationMode.IN_PLACE,
                REPLICATED,
                (1,),
                torch.zeros(1),
            ),
            M2NDestination(
                "b",
                M2NDestinationMode.IN_PLACE,
                REPLICATED,
                (1,),
                torch.zeros(1),
            ),
        ]
        _set_pp1_worker_transport(engine, 2)
        engine._reshard = Mock()

        engine.receive_weights(M2NWeightTransferUpdateInfo(names=["a"]))
        engine.receive_weights(M2NWeightTransferUpdateInfo(names=["b"]))

        assert engine._reshard.call_count == 2
        assert [call.args[3].name for call in engine._reshard.call_args_list] == [
            "a",
            "b",
        ]

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

    def test_trainer_destination_mesh_rejects_negative_dims_before_launch(self):
        with pytest.raises(ValueError, match="positive ints"):
            M2NTrainerInitInfo(
                master_address="127.0.0.1",
                master_port=1234,
                world_size=18,
                num_trainer_ranks=2,
                dst_mesh_dims=(-1, -16),
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


class TestWorkerStagingLifecycle:
    @staticmethod
    def _engine():
        vllm_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.model_config = Mock()
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )
        engine._m2n = Mock()
        engine._handle = Mock()
        _set_pp1_worker_transport(engine, 1)
        return engine

    def test_parse_failure_preserves_transport(self, monkeypatch):
        engine = self._engine()
        engine._uses_load_weights = True
        pool = Mock()
        engine._staging_pool = pool
        groups = {0: engine.model_update_group, 1: Mock()}
        engine._model_update_groups = groups
        engine._route_streams[1] = Mock()
        handle = engine._handle
        initialize = Mock()
        abort_reload = Mock()
        synchronize = Mock()
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
            initialize,
        )
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.abort_layerwise_reload",
            abort_reload,
        )
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

        engine.start_weight_update()
        with pytest.raises(ValueError, match="Invalid update_info"):
            engine.update_weights({})
        engine.start_weight_update()

        assert initialize.call_count == 2
        abort_reload.assert_called_once_with(engine.model)
        assert pool.start_update.call_count == 2
        pool.discard_update.assert_called_once_with()
        for group in groups.values():
            group.abort.assert_not_called()
        handle.destroy.assert_not_called()
        synchronize.assert_called_once_with()
        assert engine.model_update_group is groups[0]
        assert engine._model_update_groups == groups
        assert engine._handle is handle

    def test_start_failure_resets_update_but_preserves_transport(self, monkeypatch):
        engine = self._engine()
        engine._uses_load_weights = True
        pool = Mock()
        engine._staging_pool = pool
        group = engine.model_update_group
        handle = engine._handle
        initialize = Mock()
        abort_reload = Mock()
        synchronize = Mock()
        bind = Mock(side_effect=[RuntimeError("start failed"), None])
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
            initialize,
        )
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.abort_layerwise_reload",
            abort_reload,
        )
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
        monkeypatch.setattr(engine, "_bind_staged_targets", bind)

        with pytest.raises(RuntimeError, match="start failed"):
            engine.start_weight_update()
        engine.start_weight_update()

        assert initialize.call_count == 2
        abort_reload.assert_called_once_with(engine.model)
        synchronize.assert_called_once_with()
        assert pool.start_update.call_count == 2
        pool.discard_update.assert_called_once_with()
        group.abort.assert_not_called()
        handle.destroy.assert_not_called()
        assert engine.model_update_group is group
        assert engine._handle is handle

    def test_finish_failure_resets_update_for_full_retry(self):
        engine = self._engine()
        pool = Mock()
        pool.release_retained_after_finalize.side_effect = [
            RuntimeError("finish failed"),
            None,
        ]
        engine._staging_pool = pool
        group = engine.model_update_group
        handle = engine._handle

        with pytest.raises(RetryableWeightUpdateError, match="complete update"):
            engine.finish_weight_update()
        engine.start_weight_update()
        engine.finish_weight_update()

        assert pool.release_retained_after_finalize.call_count == 2
        pool.discard_update.assert_called_once_with()
        pool.start_update.assert_called_once_with()
        pool.finish_update.assert_called_once_with()
        group.abort.assert_not_called()
        handle.destroy.assert_not_called()
        assert engine.model_update_group is group
        assert engine._handle is handle

    def test_finalize_failure_resets_loader_without_aborting_transport(
        self, monkeypatch
    ):
        engine = self._engine()
        engine._uses_load_weights = True
        pool = Mock()
        engine._staging_pool = pool
        group = engine.model_update_group
        handle = engine._handle
        finalize = Mock(side_effect=RuntimeError("finalize failed"))
        abort_reload = Mock()
        synchronize = Mock()
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
            finalize,
        )
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.abort_layerwise_reload",
            abort_reload,
        )
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

        with pytest.raises(RetryableWeightUpdateError, match="complete update"):
            engine.finish_weight_update()

        finalize.assert_called_once_with(engine.model, engine.model_config)
        synchronize.assert_called_once_with()
        abort_reload.assert_called_once_with(engine.model)
        pool.discard_update.assert_called_once_with()
        group.abort.assert_not_called()
        handle.destroy.assert_not_called()
        assert engine.model_update_group is group
        assert engine._handle is handle

    def test_abort_precedes_release_of_worker_fallback_buffer(self, monkeypatch):
        events = []

        class FallbackBuffer:
            def __del__(self):
                events.append("released")

        engine = self._engine()
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        engine._metas = [M2NParamMeta("weight", torch.float32, (4,), layout)]
        engine._parameter_destinations = [
            M2NDestination(
                "weight",
                M2NDestinationMode.FULL_FALLBACK,
                REPLICATED,
                (4,),
                None,
            )
        ]
        engine.model = SimpleNamespace(load_weights=lambda weights: list(weights))
        group = engine.model_update_group
        group.abort.side_effect = lambda: events.append("abort")
        handle = engine._handle
        handle.destroy.side_effect = lambda: events.append("destroy")

        def fail_reshard(*_args):
            events.append("reshard")
            raise RuntimeError("reshard failed")

        engine._reshard = fail_reshard
        monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: FallbackBuffer())
        synchronize = Mock()
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

        with pytest.raises(RuntimeError, match="reshard failed"):
            engine.receive_weights(M2NWeightTransferUpdateInfo(names=["weight"]))
        gc.collect()

        assert events[0] == "reshard"
        assert events.index("abort") < events.index("released")
        synchronize.assert_not_called()
        group.abort.assert_called_once_with()
        handle.destroy.assert_called_once_with()

    def test_finish_allows_consecutive_updates(self):
        engine = self._engine()

        for _ in range(2):
            engine.start_weight_update()
            engine.finish_weight_update()

        assert engine._next_parameter_index == 0
        assert engine._staged_targets == {}

    def test_worker_execution_keys_transport_by_route_index(self):
        engine = self._engine()
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        engine._metas = [
            M2NParamMeta(name, torch.float32, (1,), layout)
            for name in ("first", "second")
        ]
        engine._parameter_destinations = [
            M2NDestination(
                meta.name,
                M2NDestinationMode.IN_PLACE,
                REPLICATED,
                (1,),
                torch.zeros(1, device=engine.device),
            )
            for meta in engine._metas
        ]
        # Route order deliberately differs from source-group order. Parameter
        # zero belongs to route one, while parameter one belongs to route zero.
        engine._routes = (
            M2NTransferRoute(
                source_group_index=1,
                members=(1, 2),
                param_indices=(1,),
                dst_mesh=M2NMesh((1, 1), 1),
            ),
            M2NTransferRoute(
                source_group_index=0,
                members=(0, 2),
                param_indices=(0,),
                dst_mesh=M2NMesh((1, 1), 1),
            ),
        )
        engine._param_routes = build_m2n_param_routes(engine._routes, 2)
        communicators = {0: object(), 1: object()}
        events = []
        streams = {
            index: Mock(
                synchronize=Mock(
                    side_effect=lambda index=index: events.append(f"sync:{index}")
                )
            )
            for index in range(2)
        }
        engine._model_update_groups = communicators
        engine.model_update_group = communicators[0]
        engine._route_streams = streams
        engine._reshard = Mock(
            side_effect=lambda _comm, _stream, _mesh, meta, *_: events.append(
                f"reshard:{meta.name}"
            )
        )

        engine.receive_weights(M2NWeightTransferUpdateInfo(names=["first", "second"]))

        assert engine._param_routes == (1, 0)
        calls = engine._reshard.call_args_list
        assert [reshard.args[3].name for reshard in calls] == ["second", "first"]
        assert calls[0].args[0] is communicators[0]
        assert calls[1].args[0] is communicators[1]
        streams[0].synchronize.assert_called_once_with()
        streams[1].synchronize.assert_called_once_with()
        assert events == ["reshard:second", "reshard:first", "sync:0", "sync:1"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_staged_pair_rebinds_and_orders_all_reuse_streams(self, monkeypatch):
        model = _AtomicSemanticModel()
        engine = self._engine()
        engine.model = model
        engine.model_config = SimpleNamespace(quantization=None, dtype=torch.float32)
        engine.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
        engine._dst_mesh = M2NMesh((1, 2), 1)
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        names = ["gate_up.weight", "down.weight"]
        shapes = [(4, 6, 4), (4, 4, 3)]
        engine._metas = [
            M2NParamMeta(name, torch.float32, shape, layout)
            for name, shape in zip(names, shapes)
        ]
        _set_pp1_worker_transport(engine, len(names), dst_mesh_dims=(1, 2))
        engine._prepare_destination_plan(metadata_rank=1, num_workers=2)
        engine._routes = (
            M2NTransferRoute(
                source_group_index=0,
                members=(0, 2, 3),
                param_indices=(0,),
                dst_mesh=M2NMesh((1, 2), 1),
            ),
            M2NTransferRoute(
                source_group_index=1,
                members=(1, 2, 3),
                param_indices=(1,),
                dst_mesh=M2NMesh((1, 2), 1),
            ),
        )
        engine._param_routes = build_m2n_param_routes(engine._routes, len(names))
        transfer_streams = {0: Mock(), 1: Mock()}
        engine._route_streams = transfer_streams
        engine._model_update_groups = {0: Mock(), 1: Mock()}
        engine.model_update_group = engine._model_update_groups[0]
        pool = engine._staging_pool
        assert pool is not None
        lifetime_events = []
        for route_index, transfer_stream in transfer_streams.items():
            transfer_stream.wait_stream.side_effect = (
                lambda _consumer, route_index=route_index: lifetime_events.append(
                    f"wait:{route_index}"
                )
            )
        real_release_group = pool.release_group

        def release_group_and_record(group):
            lifetime_events.append("release")
            real_release_group(group)

        release_group = Mock(side_effect=release_group_and_record)
        monkeypatch.setattr(pool, "release_group", release_group)

        events = []
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
            lambda _: events.append("initialize"),
        )
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
            lambda *_: events.append("finalize"),
        )
        monkeypatch.setattr(
            torch.accelerator,
            "synchronize",
            lambda: events.append("synchronize"),
        )
        consumer_stream = Mock()
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda *_args, **_kwargs: consumer_stream,
        )
        engine._reshard = (
            lambda _comm, _stream, _mesh, _meta, _placements, buffer: buffer.fill_(
                len(model.loaded) + 1
            )
        )

        update_bindings = []
        for _ in range(2):
            engine.start_weight_update()
            update_bindings.append(tuple(engine._staged_targets.values()))
            engine.receive_weights(M2NWeightTransferUpdateInfo(names=names))
            engine.finish_weight_update()
            assert engine._staged_targets == {}

        assert [name for name, _ in model.loaded] == names * 2
        assert release_group.call_args_list == [call(0), call(0)]
        for transfer_stream in transfer_streams.values():
            assert transfer_stream.wait_stream.call_args_list == [
                call(consumer_stream),
                call(consumer_stream),
            ]
        assert lifetime_events == [
            "wait:0",
            "wait:1",
            "release",
            "wait:0",
            "wait:1",
            "release",
        ]
        assert all(
            first is not second for first, second in zip(*update_bindings, strict=True)
        )
        assert model.resolve_calls == 6  # plan once, bind twice
        assert events == [
            "initialize",
            "finalize",
            "synchronize",
            "initialize",
            "finalize",
            "synchronize",
        ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_staged_consumer_failure_aborts_transport(self, monkeypatch):
        model = _AtomicSemanticModel()
        model.fail_consumers = True
        engine = self._engine()
        engine.model = model
        engine.model_config = SimpleNamespace(quantization=None, dtype=torch.float32)
        engine.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
        engine._dst_mesh = M2NMesh((1, 2), 1)
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        names = ["gate_up.weight", "down.weight"]
        engine._metas = [
            M2NParamMeta(name, torch.float32, shape, layout)
            for name, shape in zip(names, [(4, 6, 4), (4, 4, 3)])
        ]
        _set_pp1_worker_transport(engine, len(names), dst_mesh_dims=(1, 2))
        engine._prepare_destination_plan(metadata_rank=1, num_workers=2)
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
            lambda _: None,
        )
        engine._reshard = (
            lambda _comm, _stream, _mesh, _meta, _placements, buffer: buffer.fill_(1)
        )
        group = engine.model_update_group
        handle = engine._handle
        pool = engine._staging_pool
        assert pool is not None

        engine.start_weight_update()
        with pytest.raises(RuntimeError, match="semantic consumer failed"):
            engine.receive_weights(M2NWeightTransferUpdateInfo(names=names))
        with pytest.raises(RuntimeError, match="not initialized"):
            engine.start_weight_update()

        assert engine._staged_targets == {}
        assert not pool._updating
        group.abort.assert_called_once_with()
        handle.destroy.assert_called_once_with()
        assert engine.model_update_group is None
        assert engine._handle is None


class TestDestinationPlan:
    @staticmethod
    def _param(*, allow_full_fallback: bool = True) -> M2NWireParam:
        return M2NWireParam(
            "weight",
            "float32",
            (4, 3),
            (1, 1),
            REPLICATED,
            allow_full_fallback,
        )

    @staticmethod
    def _meta(param: M2NWireParam) -> M2NParamMeta:
        return M2NParamMeta(
            param.name,
            getattr(torch, param.dtype_name),
            param.shape,
            M2NLayout(M2NMesh(param.src_mesh_dims, 0), param.src_placements),
            param.allow_full_fallback,
        )

    @staticmethod
    def _successful_group(envelopes: list[dict], rank: int = 0) -> Mock:
        group = Mock(rank=rank, world_size=len(envelopes))
        group.all_gather_obj.side_effect = lambda envelope: (
            [envelope] * group.world_size
            if envelope.get("phase") == "destination_commit"
            else envelopes
        )
        return group

    @staticmethod
    def _sharded(rank: int) -> M2NWireDestination:
        return M2NWireDestination(
            name="weight",
            mode=M2NDestinationMode.IN_PLACE.value,
            dtype_name="float32",
            placements=(REPLICATE, 0),
            local_shape=(2, 3),
            semantic_id=None,
            shard_dim=0,
            shard_index=rank,
            num_shards=2,
        )

    @staticmethod
    def _envelopes(
        first: M2NWireDestination,
        second: M2NWireDestination,
        *,
        source_digest: str | None = None,
        dst_mesh_dims: tuple[int, int] = (1, 2),
        dst_mesh_start_rank: int = 1,
    ) -> list[dict]:
        digest = source_digest or source_plan_digest([TestDestinationPlan._param()])

        def envelope(plan: list[dict] | None) -> dict:
            return {
                "ok": True,
                "error": None,
                "plan": plan,
                "source_digest": digest,
                "dst_mesh_dims": list(dst_mesh_dims),
                "dst_mesh_start_rank": dst_mesh_start_rank,
            }

        return [
            envelope(None),
            envelope([first.to_dict()]),
            envelope([second.to_dict()]),
        ]

    def test_full_worker_plans_produce_commit_token(self):
        param = self._param()
        group = self._successful_group(
            self._envelopes(self._sharded(0), self._sharded(1))
        )

        reference, token = agree_destination_plan(
            group,
            source_digest=source_plan_digest([param]),
            params=[param],
            metas=[self._meta(param)],
            dst_mesh=M2NMesh((1, 2), 1),
            local_plan=None,
            local_error=None,
        )

        assert reference == [self._sharded(0)]
        assert token.startswith("m2n-plan:")
        assert group.all_gather_obj.call_count == 2

    def test_commit_token_hashes_source_topology_and_transfer_routes(self, monkeypatch):
        param = self._param()
        topology = M2NSourceTopology.single_group(1, 1)
        source_digest = source_plan_digest([param], topology)
        group = self._successful_group(
            self._envelopes(
                self._sharded(0), self._sharded(1), source_digest=source_digest
            )
        )
        hashed_payloads = []

        class CapturedDigest:
            @staticmethod
            def hexdigest():
                return "captured"

        def capture_digest(payload):
            hashed_payloads.append(json.loads(payload))
            return CapturedDigest()

        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_plan.hashlib.sha256",
            capture_digest,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_plan.source_plan_digest",
            lambda *_args, **_kwargs: source_digest,
        )

        _, token = agree_destination_plan(
            group,
            source_digest=source_digest,
            params=[param],
            source_topology=topology,
            metas=[self._meta(param)],
            dst_mesh=M2NMesh((1, 2), 1),
            local_plan=None,
            local_error=None,
        )

        assert token == "m2n-plan:captured"
        assert len(hashed_payloads) == 1
        payload = hashed_payloads[0]
        assert payload["source_topology"] == topology.to_dict()
        assert payload["transfer_routes"] == [
            {
                "source_group_index": 0,
                "members": [0, 1, 2],
                "param_indices": [0],
                "dst_mesh_dims": [1, 2],
                "dst_mesh_start_rank": 1,
            }
        ]

    def test_destination_commit_digest_must_match(self):
        param = self._param()
        envelopes = self._envelopes(self._sharded(0), self._sharded(1))
        group = Mock(rank=0, world_size=len(envelopes))

        def gather(envelope):
            if envelope.get("phase") != "destination_commit":
                return envelopes
            peer = dict(envelope)
            peer["digest"] = "different"
            return [envelope, peer, envelope]

        group.all_gather_obj.side_effect = gather

        with pytest.raises(RuntimeError, match="destination-commit digest disagrees"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
                metas=[self._meta(param)],
                dst_mesh=M2NMesh((1, 2), 1),
                local_plan=None,
                local_error=None,
            )

    def test_later_cross_sharded_fanin_failure_rejects_complete_plan(self):
        params = [
            M2NWireParam("valid", "float32", (32, 8), (8, 4), (REPLICATE, 0)),
            M2NWireParam("invalid", "float32", (32, 8), (1, 32), (REPLICATE, 0)),
        ]
        dst_mesh = M2NMesh((1, 4), 32)

        def worker_plan(shard_index: int) -> list[M2NWireDestination]:
            return [
                M2NWireDestination(
                    name="valid",
                    mode=M2NDestinationMode.IN_PLACE.value,
                    dtype_name="float32",
                    placements=(REPLICATE, 0),
                    local_shape=(8, 8),
                    semantic_id=None,
                    shard_dim=0,
                    shard_index=shard_index,
                    num_shards=4,
                ),
                M2NWireDestination(
                    name="invalid",
                    mode=M2NDestinationMode.IN_PLACE.value,
                    dtype_name="float32",
                    placements=(REPLICATE, 1),
                    local_shape=(32, 2),
                    semantic_id=None,
                    shard_dim=1,
                    shard_index=shard_index,
                    num_shards=4,
                ),
            ]

        digest = source_plan_digest(params)

        def envelope(plan: list[M2NWireDestination] | None) -> dict:
            return {
                "ok": True,
                "error": None,
                "plan": None if plan is None else [item.to_dict() for item in plan],
                "source_digest": digest,
                "dst_mesh_dims": list(dst_mesh.dims),
                "dst_mesh_start_rank": dst_mesh.start_rank,
            }

        envelopes = [envelope(None) for _ in range(32)] + [
            envelope(worker_plan(index)) for index in range(4)
        ]
        group = self._successful_group(envelopes)

        with pytest.raises(RuntimeError, match=r"parameter 'invalid'.*MAX_SOURCES=16"):
            agree_destination_plan(
                group,
                source_digest=digest,
                params=params,
                metas=[self._meta(param) for param in params],
                dst_mesh=dst_mesh,
                local_plan=None,
                local_error=None,
            )

        assert group.all_gather_obj.call_count == 2

    def test_worker_coordinate_disagreement_fails_before_nccl(self):
        param = self._param()
        group = self._successful_group(
            self._envelopes(self._sharded(0), self._sharded(0))
        )

        with pytest.raises(RuntimeError, match="invalid shard coordinates"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
                metas=[self._meta(param)],
                dst_mesh=M2NMesh((1, 2), 1),
                local_plan=None,
                local_error=None,
            )

    @pytest.mark.parametrize("caller_rank", [0, 1, 2])
    def test_destination_mesh_disagreement_fails_on_every_rank(self, caller_rank):
        param = self._param()
        group = Mock(rank=caller_rank, world_size=3)
        first = self._sharded(0)
        second = self._sharded(1)
        local_plans = [None, [first], [second]]
        envelopes = self._envelopes(self._sharded(0), self._sharded(1))
        envelopes[2]["dst_mesh_dims"] = [2, 1]
        group.all_gather_obj.side_effect = lambda envelope: (
            envelopes
            if envelope == envelopes[caller_rank]
            else pytest.fail("caller published the wrong destination envelope")
        )
        local_mesh = M2NMesh((2, 1), 1) if caller_rank == 2 else M2NMesh((1, 2), 1)

        with pytest.raises(RuntimeError, match="destination mesh disagrees"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
                metas=[self._meta(param)],
                dst_mesh=local_mesh,
                local_plan=local_plans[caller_rank],
                local_error=None,
            )

    @pytest.mark.parametrize("caller_rank", [0, 1, 2])
    def test_source_digest_disagreement_fails_on_every_rank(self, caller_rank):
        param = self._param()
        other_param = M2NWireParam("weight", "float32", (8, 3), (1, 1), REPLICATED)
        canonical_digest = source_plan_digest([param])
        other_digest = source_plan_digest([other_param])
        group = Mock(rank=caller_rank, world_size=3)
        first = self._sharded(0)
        second = self._sharded(1)
        local_plans = [None, [first], [second]]
        envelopes = self._envelopes(first, second)
        envelopes[2]["source_digest"] = other_digest
        group.all_gather_obj.side_effect = lambda envelope: (
            envelopes
            if envelope == envelopes[caller_rank]
            else pytest.fail("caller published the wrong destination envelope")
        )
        local_param = other_param if caller_rank == 2 else param
        local_digest = other_digest if caller_rank == 2 else canonical_digest

        with pytest.raises(RuntimeError, match="source_digest disagrees"):
            agree_destination_plan(
                group,
                source_digest=local_digest,
                params=[local_param],
                metas=[self._meta(local_param)],
                dst_mesh=M2NMesh((1, 2), 1),
                local_plan=local_plans[caller_rank],
                local_error=None,
            )

    def test_empty_source_manifest_fails_before_nccl(self):
        group = Mock(rank=0, world_size=2)
        group.all_gather_obj.side_effect = lambda envelope: [envelope, envelope]

        with pytest.raises(RuntimeError, match="manifest must not be empty"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([]),
                params=[],
                metas=[],
                dst_mesh=M2NMesh((1, 1), 1),
                local_plan=None,
                local_error=None,
            )

    def test_forbidden_full_fallback_is_rejected_from_wire_plan(self):
        param = self._param(allow_full_fallback=False)
        full = M2NWireDestination(
            name="weight",
            mode=M2NDestinationMode.FULL_FALLBACK.value,
            dtype_name="float32",
            placements=REPLICATED,
            local_shape=(4, 3),
            semantic_id=None,
            shard_dim=None,
            shard_index=0,
            num_shards=1,
        )
        group = self._successful_group(self._envelopes(full, full))

        with pytest.raises(RuntimeError, match="forbidden full fallback"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
                metas=[self._meta(param)],
                dst_mesh=M2NMesh((1, 2), 1),
                local_plan=None,
                local_error=None,
            )

    def test_incomplete_atomic_staging_group_is_rejected(self):
        param = self._param(allow_full_fallback=False)

        def staged(rank: int) -> M2NWireDestination:
            return M2NWireDestination(
                name="weight",
                mode=M2NDestinationMode.SHARDED_STAGING.value,
                dtype_name="float32",
                placements=(REPLICATE, 0),
                local_shape=(2, 3),
                semantic_id="test.weight.v1",
                shard_dim=0,
                shard_index=rank,
                num_shards=2,
                staging_group=0,
                staging_slot=0,
                staging_group_size=2,
            )

        group = self._successful_group(self._envelopes(staged(0), staged(1)))
        with pytest.raises(RuntimeError, match="incomplete staging group"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
                metas=[self._meta(param)],
                dst_mesh=M2NMesh((1, 2), 1),
                local_plan=None,
                local_error=None,
            )

    def test_local_serialization_error_is_gathered_before_raise(self):
        param = self._param()
        invalid_destination = Mock()
        invalid_destination.to_dict.side_effect = ValueError("cannot serialize")
        group = Mock(rank=1, world_size=2)
        group.all_gather_obj.side_effect = lambda envelope: [
            {
                "ok": True,
                "error": None,
                "plan": None,
                "source_digest": source_plan_digest([param]),
                "dst_mesh_dims": [1, 1],
                "dst_mesh_start_rank": 1,
            },
            envelope,
        ]

        with pytest.raises(RuntimeError, match=r"rank 1.*cannot serialize"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
                metas=[self._meta(param)],
                dst_mesh=M2NMesh((1, 1), 1),
                local_plan=[invalid_destination],
                local_error=None,
            )

        assert "cannot serialize" in group.all_gather_obj.call_args.args[0]["error"]


class TestStagingPool:
    @staticmethod
    def _requirements() -> list[M2NStagingRequirement]:
        return [
            M2NStagingRequirement(0, 0, torch.float32, (4,)),
            M2NStagingRequirement(0, 1, torch.float32, (2,)),
            M2NStagingRequirement(1, 0, torch.float32, (3,)),
            M2NStagingRequirement(1, 1, torch.float32, (1,)),
        ]

    def test_retained_groups_get_distinct_buffers_then_reuse_them(self):
        pool = M2NStagingPool(self._requirements(), torch.device("cpu"))
        pool.start_update()
        first = pool.acquire(group=0, slot=0, dtype=torch.float32, shape=(4,))
        pool.acquire(group=0, slot=1, dtype=torch.float32, shape=(2,))
        second = pool.acquire(group=1, slot=0, dtype=torch.float32, shape=(3,))
        assert first.data_ptr() != second.data_ptr()
        pool.release_group(0)
        pool.acquire(group=1, slot=1, dtype=torch.float32, shape=(1,))
        pool.release_group(1)
        pool.finish_update()
        allocated_buffers = sum(len(buffers) for buffers in pool._buffers.values())

        pool.start_update()
        reused = pool.acquire(group=0, slot=0, dtype=torch.float32, shape=(4,))
        assert reused.data_ptr() in {first.data_ptr(), second.data_ptr()}
        pool.release_group(0)
        pool.discard_update()
        assert sum(len(buffers) for buffers in pool._buffers.values()) == (
            allocated_buffers
        )

    def test_finalize_releases_only_groups_that_were_received(self):
        pool = M2NStagingPool(self._requirements(), torch.device("cpu"))
        pool.start_update()
        pool.acquire(group=0, slot=0, dtype=torch.float32, shape=(4,))
        pool.acquire(group=0, slot=1, dtype=torch.float32, shape=(2,))
        pool.release_retained_after_finalize()
        with pytest.raises(RuntimeError, match=r"not consumed: \[1\]"):
            pool.finish_update()


class _StaticM2NSource(M2NWeightSource):
    def __init__(self, metadata, values):
        self._metadata = metadata
        self._values = values

    def metadata(self):
        return list(self._metadata)

    def __iter__(self):
        return iter(self._values)


class _GroupedM2NSource(_StaticM2NSource):
    def __init__(self, metadata, values, topology, group_values):
        super().__init__(metadata, values)
        self._topology = topology
        self._group_values = group_values

    def source_topology(self):
        return self._topology

    def iter_group(self, group_index):
        return iter(self._group_values[group_index])


class TestTrainerSourceContract:
    @staticmethod
    def _meta(name="weight", mesh_dims=(1, 1)):
        return M2NParamMeta(
            name,
            torch.float32,
            (4,),
            M2NLayout(M2NMesh(mesh_dims, 0), REPLICATED),
        )

    @staticmethod
    def _engine(source):
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )
        engine._m2n = _fake_m2n()
        engine._handle = Mock()
        engine._metas = source.metadata()
        engine._dst_mesh = M2NMesh((1, 1), 1)
        engine._dst_placements = [REPLICATED] * len(engine._metas)
        engine._trainer_rank = 0
        if engine._metas:
            engine._source_topology = M2NSourceTopology.single_group(
                1, len(engine._metas)
            )
            engine._routes = build_m2n_transfer_routes(
                engine._source_topology,
                num_trainer_ranks=1,
                world_size=2,
                dst_mesh_dims=(1, 1),
            )
            engine._param_routes = build_m2n_param_routes(
                engine._routes, len(engine._metas)
            )
            engine._transfer_waves = build_m2n_transfer_waves(
                engine._routes,
                [None] * len(engine._metas),
            )
            communicator = Mock(comm=1)
            engine._route_communicators = {0: communicator}
            engine._route_streams = {0: Mock()}
            engine.group = communicator
        else:
            engine._source_topology = M2NSourceTopology((), ())
            engine._routes = ()
            engine._param_routes = ()
            engine._transfer_waves = ()
            engine._route_communicators = {}
            engine._route_streams = {}
            engine.group = None
        return engine

    def test_legacy_source_defaults_to_one_group(self):
        values = [("weight", _BufferLike())]
        source = _StaticM2NSource(
            [self._meta(mesh_dims=(1, 2))],
            values,
        )
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(), source=source, is_sender=False
        )
        engine._trainer_rank = 0

        assert source.source_topology() is None
        assert list(source.iter_group(0)) == values
        with pytest.raises(ValueError, match="only supports source group 0"):
            list(source.iter_group(1))
        engine._prepare_source_plan(source, num_trainer_ranks=2)
        assert engine._source_topology == M2NSourceTopology.single_group(2, 1)

    def test_source_mesh_is_validated_against_its_group_size(self):
        metas = [
            self._meta("first", (1, 2)),
            self._meta("second", (2, 1)),
        ]
        topology = M2NSourceTopology(
            groups=((0, 2), (1, 3)),
            param_groups=(0, 1),
        )
        source = _GroupedM2NSource(
            metas,
            [],
            topology,
            {0: [], 1: []},
        )
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(),
            source=source,
            is_sender=False,
        )

        engine._prepare_source_plan(source, num_trainer_ranks=4)

        assert engine._source_topology == topology

    def test_source_mesh_cannot_cover_more_ranks_than_its_group(self):
        topology = M2NSourceTopology(
            groups=((0, 2),),
            param_groups=(0,),
        )
        source = _GroupedM2NSource(
            [self._meta(mesh_dims=(1, 4))],
            [],
            topology,
            {0: []},
        )
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(),
            source=source,
            is_sender=False,
        )

        with pytest.raises(ValueError, match="source mesh covers 4 ranks"):
            engine._prepare_source_plan(source, num_trainer_ranks=4)

    def test_held_names_must_cover_groups_joined_by_this_rank(self):
        topology = M2NSourceTopology(
            groups=((0,), (1,)),
            param_groups=(0, 1),
        )
        source = _GroupedM2NSource(
            [self._meta("first"), self._meta("second")],
            [],
            topology,
            {0: [], 1: []},
        )
        source.held_names = Mock(return_value=("second",))
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(),
            source=source,
            is_sender=False,
        )
        engine._trainer_rank = 0

        with pytest.raises(ValueError, match="held_names.*omits.*'first'"):
            engine._prepare_source_plan(source, num_trainer_ranks=3)

    def test_trainer_sends_only_canonical_group_for_tied_copy(self, monkeypatch):
        first = _BufferLike()
        second = _BufferLike()
        metas = [
            self._meta("first", (1, 2)),
            self._meta("second", (1, 2)),
        ]
        topology = M2NSourceTopology(
            groups=((0, 2), (1, 3)),
            param_groups=(0, 1),
        )
        source = _GroupedM2NSource(
            metas,
            [],
            topology,
            {0: [("first", first)], 1: [("second", second)]},
        )
        source.held_names = Mock(return_value=("first", "second"))
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(),
            source=source,
            is_sender=False,
        )
        engine._m2n = _fake_m2n()
        engine._handle = object()
        engine._trainer_rank = 1
        engine._round_id = 1
        engine._prepare_source_plan(source, num_trainer_ranks=4)
        engine._routes = build_m2n_transfer_routes(
            topology,
            num_trainer_ranks=4,
            world_size=5,
            dst_mesh_dims=(1, 1),
        )
        engine._param_routes = build_m2n_param_routes(engine._routes, len(metas))
        engine._transfer_waves = build_m2n_transfer_waves(
            engine._routes,
            [None] * len(metas),
        )
        communicator = Mock(comm=1)
        stream = Mock()
        engine._route_communicators = {1: communicator}
        engine._route_streams = {1: stream}
        engine._dst_placements = [REPLICATED, REPLICATED]
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        engine._m2n.reshard.assert_called_once()
        reshard_call = engine._m2n.reshard.call_args
        assert reshard_call.args[0] is second
        assert reshard_call.args[2] is communicator
        stream.synchronize.assert_called_once_with()

    def test_partial_trainer_initialization_releases_transport_resources(
        self, monkeypatch
    ):
        metadata_group = Mock(rank=0, world_size=3)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer."
            "stateless_init_metadata_group",
            Mock(return_value=metadata_group),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.import_m2n",
            Mock(return_value=Mock()),
        )
        metas = [
            self._meta("first"),
            self._meta("second"),
        ]
        topology = M2NSourceTopology(
            groups=((0,), (1,)),
            param_groups=(0, 1),
        )
        source = _GroupedM2NSource(
            metas,
            [],
            topology,
            {0: [], 1: []},
        )
        digest = source_plan_digest(
            [
                M2NWireParam(
                    meta.name,
                    "float32",
                    meta.shape,
                    meta.source_layout.mesh.dims,
                    meta.source_layout.placements,
                )
                for meta in metas
            ],
            topology,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.check_source_plan_agreement",
            Mock(return_value=digest),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.agree_destination_plan",
            Mock(
                return_value=(
                    [
                        M2NDestination(
                            meta.name,
                            M2NDestinationMode.FULL_FALLBACK,
                            REPLICATED,
                            meta.shape,
                            None,
                        )
                        for meta in metas
                    ],
                    "m2n-plan:test",
                )
            ),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.check_data_plane_agreement",
            Mock(),
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer."
            "check_runtime_ready_agreement",
            Mock(),
        )

        def check_communicator_init(_group, local_errors):
            assert local_errors[0] is None
            assert "route 1 init failed" in local_errors[1]
            raise RuntimeError(local_errors[1])

        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer."
            "check_communicator_init_agreement",
            check_communicator_init,
        )
        handle = Mock()
        nccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.prepare_m2n_local_runtime",
            Mock(return_value=(nccl, 0, handle)),
        )
        first_communicator = Mock()
        make_communicator = Mock(
            side_effect=[
                first_communicator,
                RuntimeError("route 1 init failed"),
            ]
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_trainer.create_m2n_communicator",
            make_communicator,
        )
        monkeypatch.setattr(torch.cuda, "Stream", Mock(return_value=Mock()))
        client = Mock()
        engine = M2NTrainerWeightTransferEngine(
            client=client, source=source, is_sender=True
        )
        init_info = M2NTrainerInitInfo(
            rank=0,
            master_address="127.0.0.1",
            master_port=1234,
            world_size=3,
            num_trainer_ranks=2,
        )

        with pytest.raises(RuntimeError, match="route 1 init failed"):
            engine._initialize(init_info, source)

        assert make_communicator.call_args_list == [
            call(nccl, metadata_group, None, members=(0, 2)),
            call(nccl, metadata_group, None, members=(1, 2)),
        ]
        first_communicator.abort.assert_called_once_with()
        first_communicator.destroy.assert_not_called()
        handle.destroy.assert_called_once_with()
        assert engine.group is None
        assert engine._route_communicators == {}
        assert engine._route_streams == {}
        assert engine._routes == ()
        assert engine._transfer_waves == ()
        assert engine._handle is None
        assert engine._destination_plan == []
        assert engine._destination_commit_token == ""
        assert engine._dst_placements == []
        assert engine._dst_mesh is None

    @pytest.mark.parametrize(
        "rpc_failure",
        [None, RuntimeError("late worker init failed")],
        ids=["success", "failure"],
    )
    def test_trainer_init_outcome_fanout_reaches_non_sender(self, rpc_failure):
        source = _StaticM2NSource(
            [self._meta()],
            [("weight", _BufferLike())],
        )
        token = "m2n-plan:test"
        expected = {
            "phase": "trainer_init_outcome",
            "plan_token": token,
            "ok": rpc_failure is None,
            "error": (
                None if rpc_failure is None else "RuntimeError: late worker init failed"
            ),
        }

        sender = self._engine(source)
        sender.is_sender = True
        sender._num_trainer_ranks = 2
        sender._destination_commit_token = token
        sender_group = Mock()
        init_future = Mock()
        init_future.result.side_effect = rpc_failure
        if rpc_failure is None:
            sender._sync_trainer_init_outcome(sender_group, init_future)
        else:
            with pytest.raises(RuntimeError, match="late worker init failed"):
                sender._sync_trainer_init_outcome(sender_group, init_future)
        sender_group.send_obj.assert_called_once_with(expected, 1)

        receiver = self._engine(source)
        receiver._destination_commit_token = token
        receiver_group = Mock()
        receiver_group.recv_obj.return_value = expected
        if rpc_failure is None:
            receiver._sync_trainer_init_outcome(receiver_group, None)
        else:
            with pytest.raises(
                RuntimeError,
                match="worker initialization failed on trainer rank zero",
            ):
                receiver._sync_trainer_init_outcome(receiver_group, None)
        receiver_group.recv_obj.assert_called_once_with(0)

    def test_finish_failure_replays_the_full_round_on_every_trainer(self):
        controls: list[dict[str, object]] = []
        sender_group = Mock()
        sender_group.send_obj.side_effect = lambda message, _rank: controls.append(
            message
        )
        sender_group.recv_obj.side_effect = [
            {
                "phase": "round_dispatch_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
            {
                "phase": "round_outcome_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
            {
                "phase": "round_dispatch_consumed",
                "round_id": 1,
                "rank": 1,
                "error": None,
            },
            {
                "phase": "round_outcome_consumed",
                "round_id": 1,
                "rank": 1,
                "error": None,
            },
        ]

        source = _StaticM2NSource(
            [self._meta()],
            [("weight", _BufferLike())],
        )
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = sender_group
        sender._num_trainer_ranks = 2
        sender._send = Mock()
        sender.client = Mock()
        sender.client.finish_weight_update.side_effect = [
            RuntimeError("finish failed"),
            None,
        ]
        sender_communicator = sender.group
        sender_handle = sender._handle

        sender.send_weights()

        assert sender._send.call_count == 2
        assert sender.client.start_weight_update.call_count == 2
        assert sender.client.update_weights.call_count == 2
        assert sender.client.finish_weight_update.call_count == 2
        assert sender_group.recv_obj.call_args_list == [call(1)] * 4
        assert controls == [
            {
                "phase": "round_dispatch",
                "round_id": 0,
                "ok": True,
                "error": None,
                "requires_reinit": False,
                "retry_update": False,
            },
            {
                "phase": "round_outcome",
                "round_id": 0,
                "ok": False,
                "error": "RuntimeError: finish failed",
                "requires_reinit": False,
                "retry_update": True,
            },
            {
                "phase": "round_dispatch",
                "round_id": 1,
                "ok": True,
                "error": None,
                "requires_reinit": False,
                "retry_update": False,
            },
            {
                "phase": "round_outcome",
                "round_id": 1,
                "ok": True,
                "error": None,
                "requires_reinit": False,
                "retry_update": False,
            },
        ]
        sender_communicator.abort.assert_not_called()
        sender_handle.destroy.assert_not_called()
        receiver_group = Mock()
        receiver_group.recv_obj.side_effect = lambda _source_rank: controls.pop(0)
        receiver = self._engine(source)
        receiver._metadata_group = receiver_group
        receiver._num_trainer_ranks = 2
        receiver._trainer_rank = 1
        receiver._send = Mock()
        receiver_communicator = receiver.group
        receiver_handle = receiver._handle

        receiver.send_weights()

        assert receiver._send.call_count == 2
        assert receiver_group.recv_obj.call_args_list == [call(0)] * 4
        assert receiver_group.send_obj.call_args_list == [
            call(
                {
                    "phase": "round_dispatch_consumed",
                    "round_id": 0,
                    "rank": 1,
                    "error": None,
                },
                0,
            ),
            call(
                {
                    "phase": "round_outcome_consumed",
                    "round_id": 0,
                    "rank": 1,
                    "error": None,
                },
                0,
            ),
            call(
                {
                    "phase": "round_dispatch_consumed",
                    "round_id": 1,
                    "rank": 1,
                    "error": None,
                },
                0,
            ),
            call(
                {
                    "phase": "round_outcome_consumed",
                    "round_id": 1,
                    "rank": 1,
                    "error": None,
                },
                0,
            ),
        ]
        receiver_communicator.abort.assert_not_called()
        receiver_handle.destroy.assert_not_called()
        assert not controls

    def test_non_sender_observes_start_failure_without_reinitializing(self):
        controls: list[dict[str, object]] = []
        sender_group = Mock()
        sender_group.send_obj.side_effect = lambda message, _rank: controls.append(
            message
        )
        sender_group.recv_obj.side_effect = [
            {
                "phase": "round_dispatch_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
            {
                "phase": "round_outcome_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
        ]
        source = _StaticM2NSource(
            [self._meta()],
            [("weight", _BufferLike())],
        )
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = sender_group
        sender._num_trainer_ranks = 2
        sender.client.start_weight_update.side_effect = RuntimeError("start failed")
        sender_communicator = sender.group
        sender_handle = sender._handle

        with pytest.raises(RuntimeError, match="start failed"):
            sender.send_weights()

        assert controls == [
            {
                "phase": "round_dispatch",
                "round_id": 0,
                "ok": False,
                "error": "RuntimeError: start failed",
                "requires_reinit": False,
                "retry_update": False,
            },
            {
                "phase": "round_outcome",
                "round_id": 0,
                "ok": False,
                "error": "RuntimeError: start failed",
                "requires_reinit": False,
                "retry_update": False,
            },
        ]
        sender_communicator.abort.assert_not_called()
        sender_handle.destroy.assert_not_called()

        receiver_group = Mock()
        receiver_group.recv_obj.side_effect = lambda _source_rank: controls.pop(0)
        receiver = self._engine(source)
        receiver._metadata_group = receiver_group
        receiver._num_trainer_ranks = 2
        receiver._trainer_rank = 1
        receiver._send = Mock()
        receiver_communicator = receiver.group
        receiver_handle = receiver._handle

        with pytest.raises(
            RuntimeError,
            match="worker update was not dispatched: RuntimeError: start failed",
        ):
            receiver.send_weights()

        receiver._send.assert_not_called()
        receiver_communicator.abort.assert_not_called()
        receiver_handle.destroy.assert_not_called()
        assert not controls

    def test_sender_transfer_failure_requires_reinitialization(self):
        controls: list[dict[str, object]] = []
        source = _StaticM2NSource(
            [self._meta()],
            [("weight", _BufferLike())],
        )
        sender = self._engine(source)
        sender.is_sender = True
        sender._num_trainer_ranks = 2
        sender._metadata_group = Mock()
        sender._metadata_group.send_obj.side_effect = (
            lambda message, _rank: controls.append(message)
        )
        sender._metadata_group.recv_obj.side_effect = [
            {
                "phase": "round_dispatch_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
            {
                "phase": "round_outcome_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
        ]
        sender._send = Mock(side_effect=RuntimeError("transfer failed"))
        communicator = sender.group
        handle = sender._handle

        with pytest.raises(RuntimeError, match="transfer failed"):
            sender.send_weights()

        assert controls[-1] == {
            "phase": "round_outcome",
            "round_id": 0,
            "ok": False,
            "error": "RuntimeError: transfer failed",
            "requires_reinit": True,
            "retry_update": False,
        }
        communicator.abort.assert_called_once_with()
        handle.destroy.assert_called_once_with()

    def test_sender_rejects_invalid_round_outcome_acknowledgement(self):
        source = _StaticM2NSource(
            [self._meta()],
            [("weight", _BufferLike())],
        )
        sender = self._engine(source)
        sender.is_sender = True
        sender._metadata_group = Mock()
        sender._metadata_group.recv_obj.side_effect = [
            {
                "phase": "round_dispatch_consumed",
                "round_id": 0,
                "rank": 1,
                "error": None,
            },
            {"phase": "wrong"},
        ]
        sender._num_trainer_ranks = 2
        sender._send = Mock()
        sender.client = Mock()

        with pytest.raises(RuntimeError, match="invalid round_outcome acknowledgement"):
            sender.send_weights()

    def test_non_sender_reports_invalid_control_before_acknowledging_it(self):
        receiver = self._engine(
            _StaticM2NSource([self._meta()], [("weight", _BufferLike())])
        )
        receiver._trainer_rank = 1
        receiver._metadata_group = Mock()
        receiver._metadata_group.recv_obj.return_value = {
            "phase": "round_outcome",
            "round_id": 0,
            "ok": True,
            "error": None,
            "requires_reinit": False,
            "retry_update": False,
        }

        with pytest.raises(RuntimeError, match="out-of-order round_dispatch"):
            receiver._receive_round_dispatch()

        receiver._metadata_group.send_obj.assert_called_once_with(
            {
                "phase": "round_dispatch_consumed",
                "round_id": 0,
                "rank": 1,
                "error": ("trainer rank zero sent an out-of-order round_dispatch"),
            },
            0,
        )

    def test_non_sender_consumes_outcome_then_aborts_after_local_failure(self):
        source = _StaticM2NSource(
            [self._meta()],
            [("weight", _BufferLike())],
        )
        receiver = self._engine(source)
        receiver._send = Mock(side_effect=ValueError("local send failed"))
        receiver._metadata_group = Mock()
        receiver._trainer_rank = 1
        handle = Mock()
        receiver._handle = handle
        communicator = receiver.group
        receiver._metadata_group.recv_obj.side_effect = [
            {
                "phase": "round_dispatch",
                "round_id": 0,
                "ok": True,
                "error": None,
                "requires_reinit": False,
                "retry_update": False,
            },
            {
                "phase": "round_outcome",
                "round_id": 0,
                "ok": False,
                "error": "RuntimeError: first round failed",
                "requires_reinit": True,
                "retry_update": False,
            },
        ]

        with pytest.raises(ValueError, match="local send failed"):
            receiver.send_weights()
        with pytest.raises(RuntimeError, match="not initialized"):
            receiver.send_weights()

        receiver._send.assert_called_once_with()
        assert receiver._metadata_group.recv_obj.call_count == 2
        assert receiver._metadata_group.send_obj.call_args_list == [
            call(
                {
                    "phase": "round_dispatch_consumed",
                    "round_id": 0,
                    "rank": 1,
                    "error": None,
                },
                0,
            ),
            call(
                {
                    "phase": "round_outcome_consumed",
                    "round_id": 0,
                    "rank": 1,
                    "error": None,
                },
                0,
            ),
        ]
        communicator.abort.assert_called_once_with()
        handle.destroy.assert_called_once_with()

    def test_plan_summary_reports_static_transfer_plan(self):
        engine = self._engine(_StaticM2NSource([], []))
        engine._metas = [
            M2NParamMeta(
                "replicated",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            ),
            M2NParamMeta(
                "sharded",
                torch.float16,
                (6,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            ),
        ]
        engine._dst_mesh = M2NMesh((1, 2), 2)
        engine._source_topology = M2NSourceTopology(
            groups=((0,), (1,)), param_groups=(0, 1)
        )
        engine._routes = build_m2n_transfer_routes(
            engine._source_topology,
            num_trainer_ranks=2,
            world_size=4,
            dst_mesh_dims=(1, 2),
        )
        engine._transfer_waves = build_m2n_transfer_waves(engine._routes, (-1, None))
        engine._source_digest = "source-digest"
        engine._destination_commit_token = "m2n-plan:destination-digest"
        engine._destination_plan = [
            M2NWireDestination(
                name="replicated",
                mode=M2NDestinationMode.FULL_FALLBACK.value,
                dtype_name="float32",
                placements=REPLICATED,
                local_shape=(4,),
                semantic_id=None,
                shard_dim=None,
                shard_index=0,
                num_shards=1,
            ),
            M2NWireDestination(
                name="sharded",
                mode=M2NDestinationMode.IN_PLACE.value,
                dtype_name="float16",
                placements=(REPLICATE, 0),
                local_shape=(3,),
                semantic_id=None,
                shard_dim=0,
                shard_index=0,
                num_shards=2,
            ),
        ]

        summary = engine.plan_summary()

        assert summary["source_digest"] == "source-digest"
        assert summary["destination_commit_token"] == ("m2n-plan:destination-digest")
        assert summary["logical_global_bytes"] == 28
        assert summary["planned_receiver_payload_bytes"] == 44
        assert summary["destination_mode_counts"] == {
            "full_fallback": 1,
            "in_place": 1,
        }
        assert summary["transfer_communicators"] == [
            {
                "source_group_index": 0,
                "source_ranks": [0],
                "destination_ranks": [2, 3],
                "param_indices": [0],
            },
            {
                "source_group_index": 1,
                "source_ranks": [1],
                "destination_ranks": [2, 3],
                "param_indices": [1],
            },
        ]
        assert summary["transfer_waves"] == [
            {"issue_indices": [0, 1], "consume_index": 0},
        ]

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("missing", "group 0 ended before parameter 'weight'"),
            ("extra", "group 0 yielded extra parameter 'extra'"),
            ("reordered", "group 0 yielded 'other'.*expected 'weight'"),
        ],
    )
    def test_source_cardinality_and_order(self, monkeypatch, case, match):
        buffer = _BufferLike()
        values = {
            "missing": [],
            "extra": [("weight", buffer), ("extra", buffer)],
            "reordered": [("other", buffer)],
        }[case]
        source = _StaticM2NSource([self._meta()], values)
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match=match):
            engine._send()

    @pytest.mark.parametrize(
        ("case", "match"), [("dtype", "dtype"), ("shape", "local shape")]
    )
    def test_materialized_buffer_matches_declared_metadata(
        self, monkeypatch, case, match
    ):
        buffer = _BufferLike()
        if case == "dtype":
            buffer.dtype = torch.float16
        else:
            buffer.shape = (2,)

        source = _StaticM2NSource([self._meta()], [("weight", buffer)])
        engine = self._engine(source)
        m2n = engine._m2n
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(ValueError, match=match):
            engine._send()

        m2n.reshard.assert_not_called()

    def test_m2n_compatible_buffer_needs_no_torch_methods(self, monkeypatch):
        buffer = _BufferLike()
        source = _StaticM2NSource([self._meta()], [("weight", buffer)])
        engine = self._engine(source)
        stream = engine._route_streams[0]
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        assert engine._m2n.reshard.call_args.args[0] is buffer
        stream.synchronize.assert_called_once_with()

    def test_abort_precedes_release_of_non_torch_source_buffer(self, monkeypatch):
        events = []

        class SourceBuffer(_BufferLike):
            def __del__(self):
                events.append("released")

        class OneShot:
            done = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.done:
                    raise StopIteration
                self.done = True
                return "weight", SourceBuffer()

        class EphemeralSource(_StaticM2NSource):
            def __iter__(self):
                return OneShot()

        source = EphemeralSource([self._meta()], [])
        engine = self._engine(source)
        group = engine.group
        group.abort.side_effect = lambda: events.append("abort")
        handle = Mock()
        handle.destroy.side_effect = lambda: events.append("destroy")
        engine._handle = handle

        def fail_reshard(*_args, **_kwargs):
            events.append("reshard")
            raise RuntimeError("reshard failed")

        engine._m2n.reshard = fail_reshard
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match="reshard failed"):
            engine._send()
        gc.collect()

        assert events[0] == "reshard"
        assert events.index("abort") < events.index("released")
        group.abort.assert_called_once_with()
        handle.destroy.assert_called_once_with()

    def test_allocator_aware_buffer_records_send_stream(self, monkeypatch):
        buffer = _AllocatorAwareBuffer()
        source = _StaticM2NSource([self._meta()], [("weight", buffer)])
        engine = self._engine(source)
        stream = engine._route_streams[0]
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        buffer.record_stream.assert_called_once_with(stream)

    def test_raw_pointer_uses_declared_shape_and_dtype(self, monkeypatch):
        pointer = 1234
        source = _StaticM2NSource([self._meta()], [("weight", pointer)])
        engine = self._engine(source)
        monkeypatch.setattr(torch.cuda, "stream", _fake_cuda_stream_context)
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        engine._send()

        call = engine._m2n.reshard.call_args
        assert call.args[0] == pointer
        assert call.kwargs["src_local_shape"] == (4,)
        assert call.kwargs["src_dtype"] == torch.float32


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
        include_dashboard=False,
        num_cpus=4,
        num_gpus=torch.accelerator.device_count(),
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            }
        },
    )


@pytest.fixture
def _isolated_multi_route_ray(monkeypatch):
    """Start every multi-route participant with NCCL launch ordering enabled."""
    if ray.is_initialized():
        ray.shutdown()

    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    pytest.importorskip("nccl.m2n", reason="nccl_m2n backend needs the m2n runtime")
    env_vars = {
        "NCCL_LAUNCH_ORDER_IMPLICIT": "1",
        **{name: "1" for name in current_platform.ray_noset_device_env_vars},
    }
    ray.init(
        include_dashboard=False,
        num_cpus=4,
        num_gpus=torch.accelerator.device_count(),
        runtime_env={"env_vars": env_vars},
    )
    try:
        yield
    finally:
        if ray.is_initialized():
            ray.shutdown()


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
                # Opt into the same known-safe raw destination contract as a
                # real column-parallel vLLM weight. Shape alone is not enough
                # to prove that bypassing a model loader is safe.
                self.weight.output_dim = 0
                self.weight.weight_loader = MethodType(
                    ColumnParallelLinear.weight_loader,
                    MagicMock(tp_size=world_size - 1),
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
    """Send two tensors with different source-mesh factorizations."""
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


@ray.remote(num_gpus=1)
def _semantic_trainer_send(master_address: str, master_port: int) -> dict:
    """Send one tensor that forbids full replication on the receivers."""
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

    full = torch.arange(
        SHAPE[0] * SHAPE[1], dtype=torch.float32, device=device
    ).reshape(SHAPE)
    source = ManifestM2NWeightSource(
        [
            M2NManifestEntry(
                "logical.weight",
                torch.float32,
                tuple(SHAPE),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
                lambda: full,
                allow_full_fallback=False,
            )
        ]
    )
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=M2NTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=3,
            num_trainer_ranks=1,
            dst_mesh_dims=(1, 2),
            rank=0,
        ),
        client=NoopClient(),
        source=source,
    )
    summary = engine.plan_summary()
    engine.send_weights()
    engine.shutdown()
    return summary


@ray.remote(num_gpus=1)
def _semantic_worker_receive(
    master_address: str,
    master_port: int,
    worker_rank: int,
) -> dict:
    """Receive one generic semantic target through real M2N staging."""
    from unittest.mock import MagicMock, patch

    from vllm.config.parallel import ParallelConfig

    device = _assigned_device()

    class SemanticRecorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = None

        def resolve_sharded_weight_target(self, relative_name, request):
            if relative_name != "logical.weight":
                return None
            local_shape = (request.global_shape[0] // 2, request.global_shape[1])

            def consume(weight):
                self.loaded = weight.clone()
                return True

            return ShardedWeightTarget(
                spec=ShardedWeightSpec(
                    semantic_id="test.logical.weight.v1",
                    dtype=request.dtype,
                    shard_dim=0,
                    local_shape=local_shape,
                    shard_index=worker_rank,
                    num_shards=2,
                ),
                retention_key=self,
                consume=consume,
            )

        def load_weights(self, weights):
            raise AssertionError("semantic destination unexpectedly used full fallback")

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = worker_rank
    parallel_config.world_size = 2
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = SimpleNamespace(quantization=None)
    recorder = SemanticRecorder()
    engine = M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"),
        vllm_config,
        device,
        recorder,
    )
    param = M2NWireParam(
        "logical.weight",
        DTYPE,
        tuple(SHAPE),
        (1, 1),
        REPLICATED,
        allow_full_fallback=False,
    )
    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            schema_version=M2N_WIRE_SCHEMA_VERSION,
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=3,
            dst_mesh_dims=[1, 2],
            source_digest=source_plan_digest([param]),
            params=[param.to_dict()],
        )
    )
    mode = engine._parameter_destinations[0].mode.value
    with (
        patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload"),
        patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload"),
    ):
        engine.start_weight_update()
        engine.receive_weights(M2NWeightTransferUpdateInfo(names=["logical.weight"]))
        engine.finish_weight_update()
    torch.accelerator.synchronize()

    full = torch.arange(
        SHAPE[0] * SHAPE[1], dtype=torch.float32, device=device
    ).reshape(SHAPE)
    expected = full.chunk(2, dim=0)[worker_rank]
    got = recorder.loaded
    result = {
        "mode": mode,
        "shape": list(got.shape) if got is not None else None,
        "exact": bool(torch.equal(got, expected)) if got is not None else False,
    }
    engine.shutdown()
    return result


_MULTI_ROUTE_NAMES = ("param_0", "param_1", "param_2", "param_3")
_MULTI_ROUTE_SHAPE = (8, 4)
_MULTI_ROUTE_TOPOLOGY = M2NSourceTopology(
    groups=((0,), (1, 2)),
    param_groups=(1, 1, 0, 0),
)
_MULTI_ROUTE_ROUNDS = 2


def _multi_route_source_layout(param_index: int) -> M2NLayout:
    group_index = _MULTI_ROUTE_TOPOLOGY.param_groups[param_index]
    group_size = len(_MULTI_ROUTE_TOPOLOGY.groups[group_index])
    placements = REPLICATED if group_size == 1 else (REPLICATE, 0)
    return M2NLayout(M2NMesh((1, group_size), 0), placements)


def _multi_route_tensor(
    param_index: int,
    round_index: int,
    device: torch.device | str,
) -> torch.Tensor:
    return (
        torch.arange(
            _MULTI_ROUTE_SHAPE[0] * _MULTI_ROUTE_SHAPE[1],
            dtype=torch.float32,
            device=device,
        ).reshape(_MULTI_ROUTE_SHAPE)
        + param_index * 100
        + round_index * 1000
    )


@ray.remote(num_gpus=1, max_calls=1)
def _multi_route_trainer_send(
    master_address: str,
    master_port: int,
    trainer_rank: int,
) -> dict:
    """Run one rank of two disjoint source groups for two updates."""
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

    class MultiRouteSource(M2NWeightSource):
        def __init__(self):
            self._metadata = [
                M2NParamMeta(
                    name,
                    torch.float32,
                    _MULTI_ROUTE_SHAPE,
                    _multi_route_source_layout(index),
                    allow_full_fallback=index not in (1, 2),
                )
                for index, name in enumerate(_MULTI_ROUTE_NAMES)
            ]
            self.group_rounds = {0: 0, 1: 0}

        def metadata(self):
            return list(self._metadata)

        def source_topology(self):
            return _MULTI_ROUTE_TOPOLOGY

        def held_names(self):
            return tuple(
                name
                for index, name in enumerate(_MULTI_ROUTE_NAMES)
                if trainer_rank
                in _MULTI_ROUTE_TOPOLOGY.groups[
                    _MULTI_ROUTE_TOPOLOGY.param_groups[index]
                ]
            )

        def __iter__(self):
            raise AssertionError("multi-route sources must be iterated by group")

        def iter_group(self, group_index):
            members = _MULTI_ROUTE_TOPOLOGY.groups[group_index]
            if trainer_rank not in members:
                raise AssertionError(
                    f"trainer rank {trainer_rank} does not own source group "
                    f"{group_index}"
                )
            round_index = self.group_rounds[group_index]
            self.group_rounds[group_index] += 1
            local_rank = members.index(trainer_rank)
            for param_index, name in enumerate(_MULTI_ROUTE_NAMES):
                if _MULTI_ROUTE_TOPOLOGY.param_groups[param_index] != group_index:
                    continue
                full = _multi_route_tensor(param_index, round_index, device)
                yield name, full.chunk(len(members), dim=0)[local_rank].contiguous()

    source = MultiRouteSource()
    engine = None
    try:
        engine = WeightTransferTrainerFactory.trainer_init(
            init_info=M2NTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                world_size=4,
                num_trainer_ranks=3,
                dst_mesh_dims=(1, 1),
                rank=trainer_rank,
            ),
            client=NoopClient(),
            source=source,
        )
        diagnostics: dict = {
            "joined_routes": sorted(engine._route_communicators),
            "routes": [
                {
                    "members": list(route.members),
                    "param_indices": list(route.param_indices),
                }
                for route in engine._routes
            ],
            "waves": [
                {
                    "issue": list(wave.issue_indices),
                    "consume": wave.consume_index,
                }
                for wave in engine._transfer_waves
            ],
        }
        for _ in range(_MULTI_ROUTE_ROUNDS):
            engine.send_weights()
        torch.accelerator.synchronize()
        diagnostics["group_rounds"] = dict(source.group_rounds)
        return diagnostics
    finally:
        if engine is not None:
            engine.shutdown()


@ray.remote(num_gpus=1, max_calls=1)
def _multi_route_worker_receive(
    master_address: str,
    master_port: int,
) -> dict:
    """Receive two real routes through fallback and cross-route staging."""
    from unittest.mock import MagicMock, patch

    from vllm.config.parallel import ParallelConfig

    device = _assigned_device()

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = {
                name: torch.full(
                    _MULTI_ROUTE_SHAPE,
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                )
                for name in _MULTI_ROUTE_NAMES
            }
            self.events = []
            self._staging_owner = object()
            self._staged_count = 0

        def reset(self):
            for value in self.values.values():
                value.fill_(float("nan"))
            self.events = []
            self._staged_count = 0

        def resolve_sharded_weight_target(self, relative_name, request):
            if relative_name not in {"param_1", "param_2"}:
                return None

            def consume(weight, name=relative_name):
                self.values[name].copy_(weight)
                self.events.append((name, "sharded_staging"))
                self._staged_count += 1
                return self._staged_count == 2

            return ShardedWeightTarget(
                spec=ShardedWeightSpec(
                    semantic_id=f"test.{relative_name}.v1",
                    dtype=request.dtype,
                    shard_dim=0,
                    local_shape=request.global_shape,
                    shard_index=0,
                    num_shards=1,
                ),
                retention_key=self._staging_owner,
                retention_group_size=2,
                consume=consume,
            )

        def load_weights(self, weights):
            for name, weight in weights:
                if name not in {"param_0", "param_3"}:
                    raise AssertionError(f"unexpected fallback for {name!r}")
                self.values[name].copy_(weight)
                self.events.append((name, "full_fallback"))

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.tensor_parallel_size = 1
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
    params = []
    for index, name in enumerate(_MULTI_ROUTE_NAMES):
        source_layout = _multi_route_source_layout(index)
        params.append(
            M2NWireParam(
                name,
                DTYPE,
                _MULTI_ROUTE_SHAPE,
                source_layout.mesh.dims,
                source_layout.placements,
                allow_full_fallback=index not in (1, 2),
            )
        )
    try:
        engine.init_transfer_engine(
            M2NWeightTransferInitInfo(
                schema_version=M2N_WIRE_SCHEMA_VERSION,
                master_address=master_address,
                master_port=master_port,
                rank_offset=3,
                world_size=4,
                dst_mesh_dims=[1, 1],
                source_digest=source_plan_digest(params, _MULTI_ROUTE_TOPOLOGY),
                params=[param.to_dict() for param in params],
                source_topology=_MULTI_ROUTE_TOPOLOGY.to_dict(),
            )
        )
        diagnostics: dict = {
            "communicators": sorted(engine._model_update_groups),
            "streams": sorted(engine._route_streams),
            "routes": [
                {
                    "members": list(route.members),
                    "param_indices": list(route.param_indices),
                }
                for route in engine._routes
            ],
            "modes": [
                destination.mode.value for destination in engine._parameter_destinations
            ],
            "rounds": [],
        }
        with (
            patch(
                "vllm.model_executor.model_loader.reload.initialize_layerwise_reload"
            ),
            patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload"),
        ):
            for round_index in range(_MULTI_ROUTE_ROUNDS):
                recorder.reset()
                engine.start_weight_update()
                engine.receive_weights(
                    M2NWeightTransferUpdateInfo(names=list(_MULTI_ROUTE_NAMES))
                )
                engine.finish_weight_update()
                torch.accelerator.synchronize()
                diagnostics["rounds"].append(
                    {
                        "events": list(recorder.events),
                        "exact": {
                            name: bool(
                                torch.equal(
                                    recorder.values[name],
                                    _multi_route_tensor(
                                        param_index, round_index, device
                                    ),
                                )
                            )
                            for param_index, name in enumerate(_MULTI_ROUTE_NAMES)
                        },
                    }
                )
        return diagnostics
    finally:
        engine.shutdown()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs: one trainer rank and one inference worker.",
)
@pytest.mark.parametrize("data_plane", ["tcp", "uid"])
@pytest.mark.parametrize("worker_rank_offset", [None, 1], ids=["default", "explicit"])
def test_m2n_weight_transfer_between_processes(data_plane, worker_rank_offset):
    """A parameter survives a real reshard from a trainer process to a worker.

    This test builds both engines, joins one NCCL communicator across two
    processes, and reshards. The preceding unit tests only check how a transfer
    is *described*.
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
        0,
        worker_rank_offset,
        nccl_unique_id_b64,
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


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 4,
    reason="Need 4 GPUs: three trainer ranks and one inference worker.",
)
def test_m2n_transfer_with_disjoint_source_groups(_isolated_multi_route_ray):
    """Disjoint groups transfer their owned parameters for two updates."""
    master_address = "127.0.0.1"
    master_port = get_open_port()
    trainers = [
        _multi_route_trainer_send.remote(master_address, master_port, rank)
        for rank in range(3)
    ]
    worker = _multi_route_worker_receive.remote(master_address, master_port)
    *trainer_results, worker_result = ray.get(trainers + [worker], timeout=300)

    expected_routes = [
        {"members": [0, 3], "param_indices": [2, 3]},
        {"members": [1, 2, 3], "param_indices": [0, 1]},
    ]
    expected_waves = [
        {"issue": [2, 0], "consume": 0},
        {"issue": [1], "consume": 1},
        {"issue": [], "consume": 2},
        {"issue": [3], "consume": 3},
    ]
    assert [result["joined_routes"] for result in trainer_results] == [
        [0],
        [1],
        [1],
    ]
    assert [result["group_rounds"] for result in trainer_results] == [
        {0: 2, 1: 0},
        {0: 0, 1: 2},
        {0: 0, 1: 2},
    ]
    for result in trainer_results:
        assert result["routes"] == expected_routes
        assert result["waves"] == expected_waves

    assert worker_result["communicators"] == [0, 1]
    assert worker_result["streams"] == [0, 1]
    assert worker_result["routes"] == expected_routes
    assert worker_result["modes"] == [
        "full_fallback",
        "sharded_staging",
        "sharded_staging",
        "full_fallback",
    ]
    expected_events = [
        ("param_0", "full_fallback"),
        ("param_1", "sharded_staging"),
        ("param_2", "sharded_staging"),
        ("param_3", "full_fallback"),
    ]
    assert len(worker_result["rounds"]) == _MULTI_ROUTE_ROUNDS
    for round_result in worker_result["rounds"]:
        assert round_result["events"] == expected_events
        assert round_result["exact"] == dict.fromkeys(_MULTI_ROUTE_NAMES, True)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 3,
    reason="Need 3 GPUs: one trainer rank and two inference workers.",
)
def test_m2n_transfer_to_generic_rank_local_semantic_targets():
    """Real M2N sends only the shard owned by each generic model consumer."""
    pytest.importorskip("nccl.m2n", reason="nccl_m2n backend needs the m2n runtime")
    _init_ray()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    workers = [
        _semantic_worker_receive.remote(master_address, master_port, rank)
        for rank in range(2)
    ]
    trainer = _semantic_trainer_send.remote(master_address, master_port)
    summary, *results = ray.get([trainer, *workers], timeout=300)

    logical_bytes = SHAPE[0] * SHAPE[1] * 4
    assert summary["wire_schema_version"] == 4
    assert summary["destination_mode_counts"] == {"sharded_staging": 1}
    assert summary["logical_global_bytes"] == logical_bytes
    assert summary["planned_receiver_payload_bytes"] == logical_bytes
    assert summary["planned_receiver_payload_bytes"] < logical_bytes * 2
    for result in results:
        assert result == {
            "mode": "sharded_staging",
            "shape": [SHAPE[0] // 2, SHAPE[1]],
            "exact": True,
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
        self.other = torch.nn.Module()
        self.other.weight = torch.nn.Parameter(torch.zeros(16, 8))
        self.other.weight.input_dim = 1
        self.other.weight.weight_loader = MethodType(
            RowParallelLinear.weight_loader, Mock(tp_size=2)
        )


def _resolve(names, dtypes, shapes, **kwargs):
    defaults = dict(
        num_workers=2,
        shard_axis_size=2,
        allow_direct=True,
        destination_shard_index=0,
        allow_full_fallback=[True] * len(names),
    )
    defaults.update(kwargs)
    return resolve_parameter_destinations(_Model(), names, dtypes, shapes, **defaults)


class _AtomicSemanticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resolve_calls = 0
        self.loaded = []
        self.fail_consumers = False

    def resolve_sharded_weight_target(self, relative_name, request):
        self.resolve_calls += 1
        if relative_name not in {"gate_up.weight", "down.weight"}:
            return None
        local_shape = (request.global_shape[0] // 2, *request.global_shape[1:])

        def consume(weight):
            if self.fail_consumers:
                raise RuntimeError("semantic consumer failed")
            self.loaded.append((relative_name, weight.clone()))
            return relative_name == "down.weight"

        return ShardedWeightTarget(
            spec=ShardedWeightSpec(
                semantic_id=f"test.{relative_name}.v1",
                dtype=request.dtype,
                shard_dim=0,
                local_shape=local_shape,
                shard_index=0,
                num_shards=2,
            ),
            retention_key=self,
            consume=consume,
            retention_group_size=2,
        )


class _DecliningSemanticModel(torch.nn.Module):
    def resolve_sharded_weight_target(self, relative_name, request):
        return None


class TestDestinationResolution:
    def test_declined_semantic_target_honors_full_fallback_policy(self):
        kwargs = dict(
            model=_DecliningSemanticModel(),
            names=["logical.weight"],
            dtypes=[torch.float32],
            shapes=[(4, 3)],
            num_workers=2,
            shard_axis_size=2,
            destination_shard_index=0,
            allow_direct=True,
        )

        [destination] = resolve_parameter_destinations(
            **kwargs,
            allow_full_fallback=[True],
        )
        assert destination.mode is M2NDestinationMode.FULL_FALLBACK
        assert destination.placements is REPLICATED
        assert destination.local_shape == (4, 3)

        with pytest.raises(ValueError, match="requires a sharded destination"):
            resolve_parameter_destinations(
                **kwargs,
                allow_full_fallback=[False],
            )

    @pytest.mark.parametrize(
        ("quantization", "pipeline_parallel_size"),
        [("test-quantization", 1), (None, 2)],
    )
    def test_quantization_or_pp_disables_semantic_targets(
        self,
        quantization,
        pipeline_parallel_size,
    ):
        engine = object.__new__(M2NWeightTransferEngine)
        engine._dst_mesh = M2NMesh((1, 2), 1)
        engine._metas = [
            M2NParamMeta(
                "gate_up.weight",
                torch.float32,
                (4, 6, 4),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
        ]
        engine.model = _AtomicSemanticModel()
        engine.model_config = SimpleNamespace(quantization=quantization)
        engine.parallel_config = SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size
        )
        engine.device = torch.device("cuda:0")

        engine._prepare_destination_plan(metadata_rank=1, num_workers=2)

        [destination] = engine._parameter_destinations
        assert destination.mode is M2NDestinationMode.FULL_FALLBACK
        assert engine.model.resolve_calls == 1

    def test_semantic_retention_group_must_be_complete(self):
        model = _AtomicSemanticModel()
        with pytest.raises(ValueError, match="provider requires 2"):
            resolve_parameter_destinations(
                model,
                ["gate_up.weight"],
                [torch.float32],
                [(4, 6, 4)],
                num_workers=2,
                shard_axis_size=2,
                destination_shard_index=0,
                allow_direct=True,
                allow_full_fallback=[False],
            )

    def test_semantic_pair_gets_one_atomic_staging_group(self):
        destinations = resolve_parameter_destinations(
            _AtomicSemanticModel(),
            ["gate_up.weight", "down.weight"],
            [torch.float32, torch.float32],
            [(4, 6, 4), (4, 4, 3)],
            num_workers=2,
            shard_axis_size=2,
            destination_shard_index=0,
            allow_direct=True,
            allow_full_fallback=[False, False],
        )
        assert [destination.staging_group for destination in destinations] == [
            0,
            0,
        ]
        assert [destination.staging_slot for destination in destinations] == [
            0,
            1,
        ]
        assert all(destination.staging_group_size == 2 for destination in destinations)

    def test_one_plan_can_mix_staged_in_place_and_full_fallback(self):
        model = _AtomicSemanticModel()
        model.direct = torch.nn.Parameter(torch.zeros(4, 3))
        model.direct.weight_loader = default_weight_loader

        destinations = resolve_parameter_destinations(
            model,
            [
                "gate_up.weight",
                "down.weight",
                "direct",
                "missing.weight",
            ],
            [torch.float32] * 4,
            [(4, 6, 4), (4, 4, 3), (4, 3), (4, 3)],
            num_workers=2,
            shard_axis_size=2,
            destination_shard_index=0,
            allow_direct=True,
            allow_full_fallback=[False, False, True, True],
        )

        assert [destination.mode for destination in destinations] == [
            M2NDestinationMode.SHARDED_STAGING,
            M2NDestinationMode.SHARDED_STAGING,
            M2NDestinationMode.IN_PLACE,
            M2NDestinationMode.FULL_FALLBACK,
        ]

    @pytest.mark.parametrize(
        "name,expected_dim",
        [("column", 0), ("row", 1)],
    )
    def test_sharded_parameter_resolves_to_its_tp_dim(self, name, expected_dim):
        [destination] = _resolve([name], [torch.float32], [(16, 16)])
        assert destination.mode is M2NDestinationMode.IN_PLACE
        assert destination.placements == (REPLICATE, expected_dim)

    def test_replicated_parameter_needs_no_placement(self):
        """A parameter every rank holds in full is REPLICATED, so it works
        whatever way the inference mesh happens to be factored."""
        [destination] = _resolve(["norm"], [torch.float32], [(16,)])
        assert destination.mode is M2NDestinationMode.IN_PLACE
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
            destination_shard_index=0,
            allow_full_fallback=[True],
        )
        assert destination.name == "column"
        assert destination.tensor is not None
        assert destination.tensor.data_ptr() == model.column.data_ptr()

    def test_unknown_name_falls_back(self):
        """Fused parameters reach the worker under checkpoint names that do not
        exist in the model; those must take the full-tensor path."""
        [destination] = _resolve(["mlp.gate_proj.weight"], [torch.float32], [(16, 16)])
        assert destination.mode is not M2NDestinationMode.IN_PLACE
        assert destination.placements is REPLICATED

    def test_shape_the_tp_factor_cannot_explain_falls_back(self):
        [destination] = _resolve(["fused"], [torch.float32], [(16, 16)])
        assert destination.mode is not M2NDestinationMode.IN_PLACE

    def test_dtype_mismatch_falls_back(self):
        """A parameter stored in a different dtype than the wire dtype means a
        quantized or otherwise transformed layout — never write into it."""
        [destination] = _resolve(["column"], [torch.bfloat16], [(16, 16)])
        assert destination.mode is not M2NDestinationMode.IN_PLACE

    def test_allow_direct_false_forces_every_parameter_to_fall_back(self):
        destinations = _resolve(
            ["column", "norm"],
            [torch.float32] * 2,
            [(16, 16), (16,)],
            allow_direct=False,
        )
        assert not any(
            destination.mode is M2NDestinationMode.IN_PLACE
            for destination in destinations
        )

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
            destination_shard_index=0,
            allow_full_fallback=[True],
        )
        assert destination.mode is M2NDestinationMode.FULL_FALLBACK

    def test_fallback_parameter_demotes_only_direct_siblings_in_same_module(self):
        engine = object.__new__(M2NWeightTransferEngine)
        engine._dst_mesh = M2NMesh((1, 2), 1)
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        names = ["proj.weight", "proj.bias", "other.weight"]
        shapes = [(16, 16), (16,), (16, 16)]
        engine._metas = [
            M2NParamMeta(name, torch.float32, shape, layout)
            for name, shape in zip(names, shapes)
        ]
        engine.model = _MixedModel()
        engine.model_config = SimpleNamespace(quantization=None)
        engine.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
        engine.device = torch.device("cuda:0")

        engine._prepare_destination_plan(metadata_rank=1, num_workers=2)

        assert [destination.mode for destination in engine._parameter_destinations] == [
            M2NDestinationMode.FULL_FALLBACK,
            M2NDestinationMode.FULL_FALLBACK,
            M2NDestinationMode.IN_PLACE,
        ]

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
            destination_shard_index=0,
            allow_full_fallback=[True],
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
            destination_shard_index=0,
            allow_full_fallback=[True],
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
            destination_shard_index=0,
            allow_full_fallback=[True],
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
            destination_shard_index=0,
            allow_full_fallback=[True],
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
            destination_shard_index=0,
            allow_full_fallback=[True],
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
            destination_shard_index=0,
            allow_full_fallback=[True],
        )
        assert not destination.direct

    def test_forbidden_full_fallback_fails_closed(self):
        with pytest.raises(ValueError, match="requires a sharded destination"):
            _resolve(
                ["mlp.gate_up_proj.weight"],
                [torch.float32],
                [(16, 16)],
                allow_full_fallback=[False],
            )

    def test_plan_reports_byte_coverage(self, caplog_vllm):
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
            "1/2 parameters resharded directly into the model, 0 via sharded "
            "staging, 1 via full-tensor fallback; direct byte coverage: "
            "64/2097216 bytes (0.0%)" in caplog_vllm.text
        )
