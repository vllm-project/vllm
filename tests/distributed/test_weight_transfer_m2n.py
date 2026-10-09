# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NCCL M2N weight transfer backend.

These cover how a transfer is *described* — layout encoding and the wire-type
validation that keeps a bad plan from reaching a collective, since a mismatch
there is a hang rather than an error. The transfer itself needs the m2n runtime
and multiple GPUs, so it is exercised separately.
"""

import logging
import os
import socket
import sys
from contextlib import contextmanager
from functools import wraps
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, call

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
    M2NDestinationMode,
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
    prepare_m2n_local_runtime,
    prepare_m2n_nccl_runtime,
    resolve_layout,
    source_plan_digest,
    validate_layout,
    validate_local_tensor,
    validate_m2n_nccl_communicator,
    validate_m2n_nccl_library_binding,
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
    pynccl_from_metadata_group,
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

    def test_local_tensor_dtype_must_match_manifest(self):
        meta = M2NParamMeta(
            "w",
            torch.bfloat16,
            (4,),
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
        )
        with pytest.raises(ValueError, match=r"parameter 'w'.*dtype"):
            validate_local_tensor(meta, torch.zeros(4, dtype=torch.float32))

    def test_local_tensor_must_be_on_expected_device(self):
        meta = M2NParamMeta(
            "w",
            torch.float32,
            (4,),
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
        )
        with pytest.raises(ValueError, match=r"parameter 'w'.*expected cuda:0"):
            validate_local_tensor(
                meta, torch.zeros(4), expected_device=torch.device("cuda:0")
            )

    def test_local_tensor_must_be_contiguous(self):
        meta = M2NParamMeta(
            "w",
            torch.float32,
            (3, 2),
            M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
        )
        with pytest.raises(ValueError, match=r"parameter 'w'.*non-contiguous"):
            validate_local_tensor(meta, torch.zeros(2, 3).T)


class TestTransferable:
    def test_unsupported_dtype_names_the_parameter(self):
        with pytest.raises(ValueError, match="'w'"):
            check_transferable("w", torch.complex64, (4,))

    def test_rank_four_rejected(self):
        with pytest.raises(ValueError, match="rank 4"):
            check_transferable("w", torch.bfloat16, (2, 2, 2, 2))


class TestNCCLRuntimeIdentity:
    def test_resolved_nccl_is_promoted_globally(self, monkeypatch, request):
        path = "/opt/nccl-m2n/lib/libnccl.so.2"
        loaded = SimpleNamespace(abs_path=path, _handle_uint=0x1234)
        load_library = Mock(return_value=loaded)
        pathfinder = ModuleType("cuda.pathfinder")
        monkeypatch.setattr(
            pathfinder, "load_nvidia_dynamic_lib", load_library, raising=False
        )
        cuda = ModuleType("cuda")
        monkeypatch.setattr(cuda, "pathfinder", pathfinder, raising=False)
        monkeypatch.setitem(sys.modules, "cuda", cuda)
        monkeypatch.setitem(sys.modules, "cuda.pathfinder", pathfinder)
        promoted = SimpleNamespace(_handle=loaded._handle_uint)
        cdll = Mock(return_value=promoted)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_common.ctypes.CDLL", cdll
        )
        monkeypatch.delenv("VLLM_NCCL_SO_PATH", raising=False)
        prepare_m2n_nccl_runtime.cache_clear()
        request.addfinalizer(prepare_m2n_nccl_runtime.cache_clear)

        runtime = prepare_m2n_nccl_runtime()

        assert runtime == M2NNcclRuntime(path, loaded._handle_uint, promoted)
        load_library.assert_called_once_with("nccl")
        cdll.assert_called_once_with(
            path,
            mode=os.RTLD_NOW | os.RTLD_GLOBAL,
        )

    def test_local_runtime_prepares_wrapper_device_and_handle(self, monkeypatch):
        from vllm.distributed.device_communicators import pynccl_wrapper
        from vllm.distributed.weight_transfer import m2n_common

        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 0x1234, object())
        wrapper = SimpleNamespace(
            lib=SimpleNamespace(_handle=runtime.library_handle),
            ncclGetRawVersion=Mock(return_value=23005),
        )
        constructor = Mock(return_value=wrapper)
        monkeypatch.setattr(
            m2n_common, "prepare_m2n_nccl_runtime", Mock(return_value=runtime)
        )
        monkeypatch.setattr(pynccl_wrapper, "NCCLLibrary", constructor)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
        config = object()
        handle = object()
        m2n = SimpleNamespace(
            Config=Mock(return_value=config),
            Handle=SimpleNamespace(create=Mock(return_value=handle)),
        )

        assert prepare_m2n_local_runtime(m2n, 8) == (runtime, 2, handle)
        constructor.assert_called_once_with(runtime.library_path)
        wrapper.ncclGetRawVersion.assert_called_once_with()
        m2n.Config.assert_called_once_with(max_cta=8)
        m2n.Handle.create.assert_called_once_with(config)

    def test_different_pynccl_handle_is_rejected(self):
        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 0x1234, object())
        comm = SimpleNamespace(
            nccl=SimpleNamespace(
                lib=SimpleNamespace(
                    _handle=0x5678,
                    _name="/other/libnccl.so.2",
                )
            )
        )

        with pytest.raises(RuntimeError, match="different NCCL runtimes"):
            validate_m2n_nccl_communicator(comm, runtime)

    def test_pynccl_receives_canonical_library_path(self, monkeypatch):
        from vllm.distributed.device_communicators import pynccl
        from vllm.distributed.weight_transfer import nccl_common

        events = []

        @contextmanager
        def unpinned_nccl_env():
            events.append("enter")
            yield
            events.append("exit")

        def construct(*args, **kwargs):
            events.append("construct")
            return object()

        constructor = Mock(side_effect=construct)
        monkeypatch.setattr(pynccl, "PyNcclCommunicator", constructor)
        monkeypatch.setattr(nccl_common, "unpinned_nccl_env", unpinned_nccl_env)
        group = Mock()

        pynccl_from_metadata_group(
            group,
            device=3,
            library_path="/canonical/libnccl.so.2",
        )

        constructor.assert_called_once_with(
            group,
            device=3,
            library_path="/canonical/libnccl.so.2",
        )
        assert events == ["enter", "construct", "exit"]

    def test_m2n_dependency_resolves_to_pynccl_runtime(self, monkeypatch):
        from vllm.distributed.weight_transfer import m2n_common

        runtime_library = object()
        m2n_library = object()
        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 0x1234, runtime_library)
        monkeypatch.setattr(
            m2n_common, "_load_m2n_library", Mock(return_value=m2n_library)
        )
        monkeypatch.setattr(m2n_common.os.path, "realpath", lambda path: path)
        monkeypatch.setattr(
            m2n_common,
            "_symbol_address",
            lambda library, symbol: {
                runtime_library: 0xCAFE,
                m2n_library: 0xCAFE,
            }[library],
        )

        validate_m2n_nccl_library_binding(runtime, "/canonical/libnccl_m2n.so")

    def test_different_m2n_dependency_is_rejected(self, monkeypatch):
        from vllm.distributed.weight_transfer import m2n_common

        runtime_library = object()
        m2n_library = object()
        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 0x1234, runtime_library)
        monkeypatch.setattr(
            m2n_common, "_load_m2n_library", Mock(return_value=m2n_library)
        )
        monkeypatch.setattr(m2n_common.os.path, "realpath", lambda path: path)
        monkeypatch.setattr(
            m2n_common,
            "_symbol_address",
            lambda library, symbol: {
                runtime_library: 0xCAFE,
                m2n_library: 0xDEAD,
            }[library],
        )

        with pytest.raises(RuntimeError, match="different runtimes"):
            validate_m2n_nccl_library_binding(runtime, "/canonical/libnccl_m2n.so")


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
            wire_params = [M2NWireParam.from_dict(value) for value in fields["params"]]
            fields["source_digest"] = source_plan_digest(wire_params)
        return M2NWeightTransferInitInfo(**fields)

    def test_accepts_a_consistent_plan(self):
        assert [param.name for param in self._init_info().parse_wire_params()] == ["w"]

    def test_destination_policy_uses_wire_schema_v3(self):
        param = M2NWireParam(
            "w",
            "bfloat16",
            (16, 16),
            (1, 1),
            REPLICATED,
            allow_full_fallback=False,
        )

        assert M2N_WIRE_SCHEMA_VERSION == 3
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

    def test_metadata_rendezvous_is_required_with_uid(self):
        with pytest.raises(ValueError, match="master_address"):
            self._init_info(
                master_address="",
                nccl_unique_id_b64=VALID_UID_B64,
            )

    def test_uid_uses_tcp_metadata_before_uid_data_plane(self, monkeypatch):
        group = Mock(rank=1, world_size=3)
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
        runtime = M2NNcclRuntime("/canonical/libnccl.so.2", 0x1234, object())
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(return_value=(runtime, 0, Mock())),
        )
        comm = Mock()
        uid_init = Mock(return_value=comm)
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
            "vllm.distributed.weight_transfer.m2n_engine.validate_m2n_nccl_communicator",
            Mock(),
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
        assert init_metadata.call_args.args[0].rank_offset == info.rank_offset
        uid_init.assert_called_once_with(
            b"\x00" * 128,
            rank=1,
            world_size=3,
            device=0,
            library_path="/canonical/libnccl.so.2",
        )
        tcp_init.assert_not_called()

    def test_wire_carries_a_mesh_per_parameter_in_caller_order(self):
        dense = self._param(
            "dense", src_mesh_dims=(4, 1), src_placements=(REPLICATE, 0)
        )
        expert = self._param(
            "expert", src_mesh_dims=(1, 4), src_placements=(REPLICATE, 0)
        )
        info = self._init_info(
            rank_offset=4,
            world_size=6,
            params=[dense.to_dict(), expert.to_dict()],
        )

        params = info.parse_wire_params()
        assert [param.name for param in params] == ["dense", "expert"]
        assert [param.src_mesh_dims for param in params] == [
            (4, 1),
            (1, 4),
        ]

    def test_unknown_schema_rejected(self):
        with pytest.raises(ValueError, match="schema version"):
            self._init_info(
                schema_version=M2N_WIRE_SCHEMA_VERSION + 1
            ).parse_wire_params()

    def test_source_digest_must_match_params(self):
        with pytest.raises(ValueError, match="source digest"):
            self._init_info(source_digest="0" * 64).parse_wire_params()

    def test_wire_param_rejects_missing_fields(self):
        param = self._param().to_dict()
        del param["shape"]
        with pytest.raises(ValueError, match=r"missing.*shape"):
            self._init_info(params=[param], source_digest="0" * 64).parse_wire_params()

    def test_source_digest_includes_caller_order(self):
        first = self._param("first")
        second = self._param("second")
        assert source_plan_digest([first, second]) != source_plan_digest(
            [second, first]
        )

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("duplicate", "duplicate parameter 'first'"),
            ("missing", "source digest"),
            ("reordered", "source digest"),
            ("inconsistent", "source digest"),
        ],
    )
    def test_invalid_source_records_fail_preflight(self, case, match):
        first = self._param("first")
        second = self._param("second")
        canonical = [first, second]
        if case == "duplicate":
            actual = [first, first]
        elif case == "missing":
            actual = [first]
        elif case == "reordered":
            actual = [second, first]
        else:
            actual = [first, self._param("second", shape=(8, 16))]

        info = self._init_info(
            params=[param.to_dict() for param in actual],
            source_digest=source_plan_digest(canonical),
        )
        with pytest.raises(ValueError, match=match):
            info.parse_wire_params()

    @pytest.mark.parametrize("caller_rank", [0, 1])
    def test_preflight_reports_a_rank_with_a_different_source_plan(self, caller_rank):
        param = self._param()
        digest = source_plan_digest([param])
        other_param = self._param(shape=(8, 16))
        other_digest = source_plan_digest([other_param])
        group = Mock(rank=caller_rank, world_size=2)
        envelopes = [
            {
                "phase": "source",
                "digest": digest,
                "declared_digest": digest,
                "error": None,
            },
            {
                "phase": "source",
                "digest": other_digest,
                "declared_digest": other_digest,
                "error": None,
            },
        ]
        group.all_gather_obj.side_effect = lambda envelope: (
            envelopes
            if envelope == envelopes[caller_rank]
            else pytest.fail("caller published the wrong source envelope")
        )
        local_param = param if caller_rank == 0 else other_param
        local_digest = digest if caller_rank == 0 else other_digest

        with pytest.raises(RuntimeError, match=r"rank 1.*source digest disagrees"):
            check_source_plan_agreement(group, [local_param], local_digest)

    def test_preflight_reports_a_rank_with_a_bad_declared_digest(self):
        param = self._param()
        digest = source_plan_digest([param])
        group = Mock(rank=0, world_size=2)
        group.all_gather_obj.return_value = [
            {
                "phase": "source",
                "digest": digest,
                "declared_digest": digest,
                "error": None,
            },
            {
                "phase": "source",
                "digest": digest,
                "declared_digest": "declared-wrong",
                "error": None,
            },
        ]

        with pytest.raises(RuntimeError, match=r"rank 1.*declared-wrong"):
            check_source_plan_agreement(group, [param], digest)

    @pytest.mark.parametrize("caller_rank", [0, 1])
    def test_runtime_readiness_reports_rank_failure_on_every_rank(self, caller_rank):
        group = Mock(rank=caller_rank, world_size=2)
        envelopes = [
            {"phase": "runtime_ready", "error": None},
            {
                "phase": "runtime_ready",
                "error": "RuntimeError: handle allocation failed",
            },
        ]
        group.all_gather_obj.side_effect = lambda envelope: (
            envelopes
            if envelope == envelopes[caller_rank]
            else pytest.fail("caller published the wrong runtime envelope")
        )
        local_error = envelopes[caller_rank]["error"]

        with pytest.raises(RuntimeError, match=r"rank 1.*handle allocation failed"):
            check_runtime_ready_agreement(group, local_error)

    @pytest.mark.parametrize(
        ("peer", "match"),
        [
            (
                {
                    "phase": "data_plane",
                    "mode": "uid",
                    "uid_digest": "different",
                    "max_cta": 8,
                    "error": None,
                },
                "identity disagrees",
            ),
            (
                {
                    "phase": "data_plane",
                    "mode": "tcp",
                    "uid_digest": None,
                    "max_cta": 16,
                    "error": None,
                },
                "max_cta disagrees",
            ),
        ],
    )
    def test_data_plane_preflight_rejects_mismatched_config(self, peer, match):
        group = Mock(rank=0, world_size=2)
        group.all_gather_obj.side_effect = lambda envelope: [
            envelope,
            {**envelope, **peer},
        ]

        with pytest.raises(RuntimeError, match=match):
            check_data_plane_agreement(group, None, 8)

    @pytest.mark.parametrize(
        ("name", "local_value", "peer_value"),
        [
            ("NCCL_RESHARD_NUM_CTAS", "4", "8"),
            ("NCCL_RESHARD_COPY_ALGORITHM", "PACK", "DIRECT"),
        ],
    )
    def test_data_plane_preflight_rejects_mismatched_reshard_environment(
        self, monkeypatch, name, local_value, peer_value
    ):
        monkeypatch.setenv(name, local_value)
        group = Mock(rank=0, world_size=2)

        def gather(envelope):
            peer_env = {**envelope["reshard_env"], name: peer_value}
            return [envelope, {**envelope, "reshard_env": peer_env}]

        group.all_gather_obj.side_effect = gather

        with pytest.raises(RuntimeError, match=rf"rank 1.*{name}"):
            check_data_plane_agreement(group, None, 8)

    @pytest.mark.parametrize(
        "name",
        [
            "NCCL_RESHARD_LOG_LEVEL",
            "NCCL_RESHARD_SPLIT_KERNEL_TRACE",
        ],
    )
    def test_data_plane_preflight_ignores_diagnostic_reshard_environment(
        self, monkeypatch, name
    ):
        monkeypatch.setenv(name, "1")
        group = Mock(rank=0, world_size=1)
        group.all_gather_obj.side_effect = lambda envelope: [envelope]

        check_data_plane_agreement(group, None, 8)

        envelope = group.all_gather_obj.call_args.args[0]
        assert name not in envelope["reshard_env"]

    def test_data_plane_preflight_exchanges_decode_error(self):
        group = Mock(rank=0, world_size=1)
        group.all_gather_obj.side_effect = lambda envelope: [envelope]

        with pytest.raises(RuntimeError, match="bad uid"):
            check_data_plane_agreement(
                group,
                None,
                None,
                "ValueError: bad uid",
            )

        assert group.all_gather_obj.call_args.args[0]["error"] == "ValueError: bad uid"

    def test_source_preflight_exchanges_error_before_raising(self):
        param = self._param()
        group = Mock(rank=0, world_size=1)
        group.all_gather_obj.return_value = [
            {
                "phase": "source",
                "digest": None,
                "declared_digest": None,
                "error": "ValueError: bad local layout",
            }
        ]

        with pytest.raises(RuntimeError, match="bad local layout"):
            check_source_plan_agreement(
                group, [param], local_error="ValueError: bad local layout"
            )

        envelope = group.all_gather_obj.call_args.args[0]
        assert envelope["error"] == "ValueError: bad local layout"

    @pytest.mark.parametrize(
        "error",
        [
            "schema",
            "VLLM_DISABLE_PYNCCL",
            "M2N symbol-binding mismatch",
        ],
    )
    def test_source_error_is_exchanged_before_pynccl(self, monkeypatch, error):
        group = Mock(rank=2, world_size=3)
        make_pynccl = Mock()
        init_metadata_group = Mock(return_value=group)
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            init_metadata_group,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.pynccl_from_metadata_group",
            make_pynccl,
        )
        if error == "schema":
            monkeypatch.setattr(
                "vllm.distributed.weight_transfer.m2n_engine.import_m2n",
                Mock(return_value=object()),
            )
        elif error == "VLLM_DISABLE_PYNCCL":
            monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "1")
        else:
            from vllm.distributed.weight_transfer import m2n_common

            nccl = ModuleType("nccl")
            nccl.__path__ = []
            m2n = ModuleType("nccl.m2n")
            monkeypatch.setitem(sys.modules, "nccl", nccl)
            monkeypatch.setitem(sys.modules, "nccl.m2n", m2n)
            monkeypatch.setattr(
                m2n_common,
                "prepare_m2n_nccl_runtime",
                Mock(
                    return_value=M2NNcclRuntime(
                        "/canonical/libnccl.so.2", 0x1234, object()
                    )
                ),
            )
            monkeypatch.setattr(
                m2n_common,
                "validate_m2n_nccl_library_binding",
                Mock(side_effect=RuntimeError(error)),
            )
            monkeypatch.setenv("NCCL_M2N_LIBRARY", "/canonical/libnccl_m2n.so")
        vllm_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.model_config = Mock()
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cuda:0"),
            Mock(),
        )
        info = self._init_info(
            schema_version=M2N_WIRE_SCHEMA_VERSION + (error == "schema"),
            worker_rank_offset=2,
        )
        peer = {
            "phase": "source",
            "digest": info.source_digest,
            "declared_digest": info.source_digest,
            "error": None,
        }
        group.all_gather_obj.side_effect = lambda envelope: [peer, peer, envelope]

        with pytest.raises(RuntimeError, match=rf"source preflight.*{error}"):
            engine.init_transfer_engine(info)

        envelope = group.all_gather_obj.call_args.args[0]
        assert error in envelope["error"]
        assert init_metadata_group.call_args.args[0].rank_offset == 2
        assert info.rank_offset == 1
        make_pynccl.assert_not_called()

    def test_local_runtime_failure_is_agreed_before_pynccl(self, monkeypatch):
        group = Mock(rank=1, world_size=3)

        def gather_runtime(envelope):
            if envelope["phase"] == "data_plane":
                return [envelope] * 3
            assert envelope["phase"] == "runtime_ready"
            return [
                {"phase": "runtime_ready", "error": None},
                envelope,
                {"phase": "runtime_ready", "error": None},
            ]

        group.all_gather_obj.side_effect = gather_runtime
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.worker_init_metadata_group",
            Mock(return_value=group),
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
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.prepare_m2n_local_runtime",
            Mock(side_effect=RuntimeError("handle allocation failed")),
        )
        make_pynccl = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.pynccl_from_metadata_group",
            make_pynccl,
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

        with pytest.raises(
            RuntimeError,
            match=r"runtime preflight failed.*handle allocation failed",
        ):
            engine.init_transfer_engine(self._init_info())

        make_pynccl.assert_not_called()
        assert engine.model_update_group is None

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

    def test_source_mesh_must_cover_the_trainers_on_worker(self):
        engine = object.__new__(M2NWeightTransferEngine)
        param = self._param(src_mesh_dims=(2, 1))

        with pytest.raises(ValueError, match="covers 2 ranks, but there are 1"):
            engine._prepare_source_plan((param,), num_trainer_ranks=1)

    def test_destination_planning_failure_precedes_pynccl(self, monkeypatch):
        group = Mock(rank=1)
        make_pynccl = Mock()
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
            "vllm.distributed.weight_transfer.m2n_engine.pynccl_from_metadata_group",
            make_pynccl,
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

        make_pynccl.assert_not_called()

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
            engine._prepare_source_plan((param,), num_trainer_ranks=1)

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
        engine._state = type(engine._state).UPDATING
        engine._handle = Mock()
        engine.model_update_group = Mock()
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

        with pytest.raises(ValueError, match=r"update order mismatch.*unknown"):
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


class TestWorkerLifecycle:
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
        engine._state = type(engine._state).READY
        engine._m2n = Mock()
        engine._handle = Mock()
        engine.model_update_group = Mock()
        return engine

    def test_update_parse_failure_poisons_engine(self):
        engine = self._engine()

        with pytest.raises(ValueError, match="Invalid update_info"):
            engine.update_weights({})

        with pytest.raises(RuntimeError, match="cannot be reused"):
            engine.start_weight_update()

    def test_receive_failure_poisons_engine(self):
        engine = self._engine()
        engine.start_weight_update()

        with pytest.raises(ValueError, match="update order mismatch"):
            engine.update_weights({"names": ["unknown"]})

        with pytest.raises(RuntimeError, match="cannot be reused"):
            engine.start_weight_update()

    def test_final_synchronization_failure_poisons_engine(self, monkeypatch):
        engine = self._engine()
        engine.start_weight_update()
        engine.receive_weights = Mock()
        monkeypatch.setattr(
            torch.accelerator,
            "synchronize",
            Mock(side_effect=RuntimeError("synchronize failed")),
        )

        with pytest.raises(RuntimeError, match="synchronize failed"):
            engine.update_weights({"names": []})

        with pytest.raises(RuntimeError, match="cannot be reused"):
            engine.start_weight_update()

    def test_finish_restores_ready_state_for_consecutive_updates(self):
        engine = self._engine()

        for _ in range(2):
            engine.start_weight_update()
            engine.finish_weight_update()

        assert engine._next_parameter_index == 0
        assert engine._staged_targets == {}

    def test_fallback_tensors_share_one_load_weights_invocation(self, monkeypatch):
        class CoupledModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.loaded = []

            def load_weights(self, weights):
                self.calls += 1
                received = list(weights)
                if [name for name, _ in received] == [
                    "pair.weight",
                    "pair.scale",
                ]:
                    self.loaded = [(name, tensor.clone()) for name, tensor in received]

        model = CoupledModel()
        vllm_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.model_config = Mock()
        engine = M2NWeightTransferEngine(
            WeightTransferConfig(backend="nccl_m2n"),
            vllm_config,
            torch.device("cpu"),
            model,
        )
        engine._state = type(engine._state).UPDATING
        engine._handle = Mock()
        engine.model_update_group = Mock(device=torch.device("cpu"))
        engine._dst_mesh = M2NMesh((3, 1), 1)
        layout = M2NLayout(M2NMesh((1, 1), 0), REPLICATED)
        names = ["pair.weight", "direct", "pair.scale"]
        engine._metas = [
            M2NParamMeta(name, torch.float32, (1,), layout) for name in names
        ]
        direct = torch.zeros(1)
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
        reshard_order = []

        def reshard(comm, stream, meta, placements, buffer):
            reshard_order.append(meta.name)
            buffer.fill_(len(reshard_order))

        engine._reshard = reshard
        stream = Mock()
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.comm_ptr", lambda _: 0
        )
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

        engine.receive_weights(M2NWeightTransferUpdateInfo(names=names))

        assert model.calls == 1
        assert [name for name, _ in model.loaded] == ["pair.weight", "pair.scale"]
        assert [tensor.item() for _, tensor in model.loaded] == [1, 3]
        assert direct.item() == 2
        assert reshard_order == names
        assert stream.synchronize.call_count == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_staged_pair_rebinds_for_each_update(self, monkeypatch):
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
        engine._prepare_destination_plan(metadata_rank=1, num_workers=2)
        pool = engine._staging_pool
        assert pool is not None
        release_group = Mock(wraps=pool.release_group)
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
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.comm_ptr", lambda _: 0
        )
        engine._reshard = lambda _comm, _stream, _meta, _placements, buffer: (
            buffer.fill_(len(model.loaded) + 1)
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
        assert engine._state.value == "ready"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_staged_consumer_failure_poisons_update(self, monkeypatch):
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
        engine._prepare_destination_plan(metadata_rank=1, num_workers=2)
        monkeypatch.setattr(
            "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
            lambda _: None,
        )
        monkeypatch.setattr(
            "vllm.distributed.weight_transfer.m2n_engine.comm_ptr", lambda _: 0
        )
        engine._reshard = lambda _comm, _stream, _meta, _placements, buffer: (
            buffer.fill_(1)
        )

        engine.start_weight_update()
        with pytest.raises(RuntimeError, match="semantic consumer failed"):
            engine.receive_weights(M2NWeightTransferUpdateInfo(names=names))

        assert engine._state.value == "poisoned"
        assert engine._staged_targets == {}
        assert engine.model_update_group is None
        with pytest.raises(RuntimeError, match="cannot be reused"):
            engine.start_weight_update()


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
        group = Mock(rank=0, world_size=3)
        group.all_gather_obj.return_value = self._envelopes(
            self._sharded(0), self._sharded(1)
        )

        reference, token = agree_destination_plan(
            group,
            source_digest=source_plan_digest([param]),
            params=[param],
            dst_mesh=M2NMesh((1, 2), 1),
            local_plan=None,
            local_error=None,
        )

        assert reference == [self._sharded(0)]
        assert token.startswith("m2n-plan:")

    def test_worker_coordinate_disagreement_fails_before_nccl(self):
        param = self._param()
        group = Mock(rank=0, world_size=3)
        group.all_gather_obj.return_value = self._envelopes(
            self._sharded(0), self._sharded(0)
        )

        with pytest.raises(ValueError, match="invalid shard coordinates"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
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
        group = Mock(rank=0, world_size=3)
        group.all_gather_obj.return_value = self._envelopes(full, full)

        with pytest.raises(RuntimeError, match="forbidden full fallback"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
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

        group = Mock(rank=0, world_size=3)
        group.all_gather_obj.return_value = self._envelopes(staged(0), staged(1))
        with pytest.raises(ValueError, match="incomplete staging group"):
            agree_destination_plan(
                group,
                source_digest=source_plan_digest([param]),
                params=[param],
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


class TestTrainerSourceContract:
    @staticmethod
    def _engine(source):
        engine = M2NTrainerWeightTransferEngine(
            client=Mock(),
            source=source,
            is_sender=False,
        )
        engine._m2n = Mock()
        engine._handle = object()
        engine._dst_mesh = M2NMesh((1, 1), 1)
        engine._dst_placements = []
        engine.group = Mock(comm=1)
        engine._metadata_group = Mock()
        engine._metadata_group.recv_obj.return_value = {"ok": True, "error": None}
        engine._num_trainer_ranks = 2
        return engine

    @pytest.mark.parametrize(
        ("mesh", "num_trainer_ranks", "match"),
        [
            (M2NMesh((2, 1), 1), 2, "must start at rank 0"),
            (M2NMesh((1, 1), 0), 2, "covers 1 ranks, but there are 2"),
        ],
    )
    def test_source_mesh_must_cover_the_trainer_ranks(
        self, mesh, num_trainer_ranks, match
    ):
        source = ManifestM2NWeightSource(
            [
                M2NManifestEntry(
                    "weight",
                    torch.float32,
                    (4,),
                    M2NLayout(mesh, REPLICATED),
                    lambda: torch.zeros(4),
                )
            ]
        )
        engine = self._engine(source)

        with pytest.raises(ValueError, match=match):
            engine._prepare_source_plan(source, num_trainer_ranks)

    def test_source_must_yield_every_declared_parameter(self, monkeypatch):
        engine = self._engine([])
        engine._metas = [
            M2NParamMeta(
                "missing",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
        ]
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match="first missing: 'missing'"):
            engine._send()

        assert engine._failure is not None
        assert engine.group is None

    def test_source_must_not_yield_undeclared_parameters(self, monkeypatch):
        engine = self._engine([("extra", torch.zeros(1))])
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match="first extra parameter: 'extra'"):
            engine._send()

        assert engine._failure is not None
        assert engine.group is None

    def test_source_iteration_order_must_match_metadata(self, monkeypatch):
        engine = self._engine([("second", torch.zeros(4))])
        m2n = engine._m2n
        engine._metas = [
            M2NParamMeta(
                "first",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
        ]
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(RuntimeError, match="yielded 'second'.*declared 'first'"):
            engine._send()

        m2n.reshard.assert_not_called()

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("dtype", "dtype"),
            ("shape", "local shape"),
            ("device", "expected cuda:0"),
            ("contiguity", "non-contiguous"),
        ],
    )
    def test_send_validates_materialized_tensor_before_reshard(
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

        engine = self._engine([("weight", tensor)])
        m2n = engine._m2n
        engine._metas = [
            M2NParamMeta(
                "weight",
                torch.float32,
                (4,),
                M2NLayout(M2NMesh((1, 1), 0), REPLICATED),
            )
        ]
        engine._dst_placements = [REPLICATED]
        monkeypatch.setattr(torch.cuda, "current_stream", Mock())
        monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

        with pytest.raises(ValueError, match=rf"parameter 'weight'.*{match}"):
            engine._send()

        m2n.reshard.assert_not_called()

    def test_failed_engine_is_not_reused(self):
        engine = self._engine([])
        failure = ValueError("materialization failed")
        engine._send = Mock(side_effect=failure)

        with pytest.raises(ValueError, match="materialization failed"):
            engine.send_weights()
        with pytest.raises(RuntimeError, match="cannot be reused") as error:
            engine.send_weights()

        assert error.value.__cause__ is failure
        engine._send.assert_called_once_with()

    def test_non_sender_waits_for_controller_round_outcome(self):
        engine = self._engine([])
        engine._send = Mock()

        engine.send_weights()

        engine._metadata_group.recv_obj.assert_called_once_with(0)

    def test_non_sender_observes_worker_finish_failure(self):
        engine = self._engine([])
        engine._send = Mock()
        engine._metadata_group.recv_obj.return_value = {
            "ok": False,
            "error": "RuntimeError: finish failed",
        }

        with pytest.raises(RuntimeError, match="finish failed"):
            engine.send_weights()

        assert engine._failure is not None

    def test_sender_publishes_worker_finish_failure(self):
        engine = self._engine([])
        engine.is_sender = True
        engine._send = Mock()
        engine._executor = Mock()
        future = Mock()
        engine._executor.submit.return_value = future
        engine.client.finish_weight_update.side_effect = RuntimeError("finish failed")
        metadata_group = engine._metadata_group

        with pytest.raises(RuntimeError, match="finish failed"):
            engine.send_weights()

        future.result.assert_called_once_with()
        envelope, rank = metadata_group.send_obj.call_args.args
        assert rank == 1
        assert envelope == {
            "ok": False,
            "error": "RuntimeError: finish failed",
        }

    def test_sender_publishes_success_after_worker_finish(self):
        engine = self._engine([])
        engine.is_sender = True
        engine._send = Mock()
        engine._executor = Mock()
        future = Mock()
        engine._executor.submit.return_value = future
        events = []
        engine.client.finish_weight_update.side_effect = lambda: events.append("finish")
        engine._metadata_group.send_obj.side_effect = lambda *_args: events.append(
            "publish"
        )

        engine.send_weights()

        assert events == ["finish", "publish"]
        engine._metadata_group.send_obj.assert_called_once_with(
            {"ok": True, "error": None}, 1
        )

    def test_abort_preserves_transfer_error_and_attempts_all_cleanup(self):
        engine = self._engine([])
        failure = ValueError("materialization failed")
        engine._send = Mock(side_effect=failure)
        group = engine.group
        handle = Mock()
        engine._handle = handle
        executor = Mock()
        engine._executor = executor

        def fail_group_cleanup():
            assert engine.group is None
            assert engine._handle is None
            assert engine._executor is None
            raise RuntimeError("communicator cleanup failed")

        group.destroy.side_effect = fail_group_cleanup
        handle.destroy.side_effect = RuntimeError("handle cleanup failed")
        executor.shutdown.side_effect = RuntimeError("executor cleanup failed")

        with pytest.raises(ValueError, match="materialization failed") as error:
            engine.send_weights()

        assert error.value is failure
        assert engine._failure is failure
        group.destroy.assert_called_once_with()
        handle.destroy.assert_called_once_with()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        assert engine.group is None
        assert engine._handle is None
        assert engine._executor is None

    def test_successful_shutdown_releases_handle_group_and_executor(self, monkeypatch):
        engine = self._engine([])
        handle = Mock()
        group = engine.group
        executor = Mock()
        engine._handle = handle
        engine._executor = executor
        synchronize = Mock()
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

        engine.shutdown()
        engine.shutdown()

        synchronize.assert_called_once_with()
        handle.destroy.assert_called_once_with()
        group.destroy.assert_called_once_with()
        executor.shutdown.assert_called_once_with()
        assert engine._closed

    def test_shutdown_attempts_every_cleanup_and_raises_first_error(self, monkeypatch):
        engine = self._engine([])
        handle = Mock()
        group = engine.group
        executor = Mock()
        engine._handle = handle
        engine._executor = executor
        synchronize = Mock(side_effect=ValueError("synchronize failed"))
        handle.destroy.side_effect = RuntimeError("handle failed")
        group.destroy.side_effect = RuntimeError("group failed")
        executor.shutdown.side_effect = RuntimeError("executor failed")
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

        with pytest.raises(ValueError, match="synchronize failed"):
            engine.shutdown()

        handle.destroy.assert_called_once_with()
        group.destroy.assert_called_once_with()
        executor.shutdown.assert_called_once_with()

    def test_abort_cancels_requests_and_does_not_wait_for_executor(self):
        engine = self._engine([])
        init_future = Mock()
        update_future = Mock()
        executor = Mock()
        engine._init_future = init_future
        engine._update_future = update_future
        engine._executor = executor
        engine._handle = Mock()

        engine._abort(RuntimeError("failed request"))

        init_future.cancel.assert_called_once_with()
        update_future.cancel.assert_called_once_with()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_shutdown_rejects_concurrent_send_without_destroying_resources(self):
        engine = self._engine([])
        group = engine.group
        handle = Mock()
        engine._handle = handle
        engine._lifecycle_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="during a weight update"):
                engine.shutdown()
        finally:
            engine._lifecycle_lock.release()

        group.destroy.assert_not_called()
        handle.destroy.assert_not_called()

    def test_plan_summary_reports_logical_and_receiver_bytes_by_mode(self):
        engine = self._engine([])
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
        engine._dst_mesh = M2NMesh((1, 2), 1)
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
        assert summary["parameter_count"] == 2
        assert summary["logical_global_bytes"] == 28
        assert summary["planned_receiver_payload_bytes"] == 44
        assert summary["destination_mode_counts"] == {
            "full_fallback": 1,
            "in_place": 1,
        }
        assert summary["logical_global_bytes_by_destination_mode"] == {
            "full_fallback": 16,
            "in_place": 12,
        }
        assert summary["planned_receiver_payload_bytes_by_destination_mode"] == {
            "full_fallback": 32,
            "in_place": 12,
        }
        assert (
            sum(summary["logical_global_bytes_by_destination_mode"].values())
            == (summary["logical_global_bytes"])
        )
        assert (
            sum(summary["planned_receiver_payload_bytes_by_destination_mode"].values())
            == summary["planned_receiver_payload_bytes"]
        )

    def test_reserved_socket_is_not_sent_to_workers(self):
        engine = self._engine([])
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
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

        nccl_unique_id_b64 = base64.b64encode(
            bytes(NCCLLibrary().ncclGetUniqueId().internal)
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
    assert summary["wire_schema_version"] == 3
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
