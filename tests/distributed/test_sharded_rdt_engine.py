# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receive-pool and staging tests for the sharded-RDT consumer."""

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_engine as engine_module
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    HostStagingAllocationError,
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _CallPlan,
    _Chunk,
)


def _chunk(nbytes: int = 32, *, stream: int = 0) -> _Chunk:
    return _Chunk(
        scatters=[],
        keys=[("w", ())],
        pack_layout=[(0, torch.uint8, nbytes, (nbytes,))],
        pack_bytes=nbytes,
        materialize=[],
        quant=[],
        free=[],
        owner=0,
        stream=stream,
    )


def _plan(*, streams: int = 1, chunk_bytes: tuple[int, ...] = (32,)) -> _CallPlan:
    chunks = [
        _chunk(nbytes, stream=index % max(1, streams))
        for index, nbytes in enumerate(chunk_bytes)
    ]
    return _CallPlan(chunks=chunks, pre_free=[], num_streams=streams)


def _engine(
    *,
    staging_device: str = "gpu",
    stages_in_flight: int = 1,
    ring_depth: int = 2,
    streams: int = 1,
    chunk_bytes: tuple[int, ...] = (32,),
) -> ShardedRDTWeightTransferEngine:
    engine = object.__new__(ShardedRDTWeightTransferEngine)
    engine.device = torch.device("cpu")
    engine._proc_thread = None
    engine._slot_read_done = []
    engine._slot_queued = []
    engine._slot_done = []
    engine._slot_cv = None
    engine._pool_geometry = None
    engine._staging_device = "gpu"
    engine._stages_in_flight = 1
    engine._ring_depth = 1
    engine._lanes = 1
    engine._dest_buffers = [{}]
    engine._lane_cursors = [0]
    engine._targets_cache = {}
    engine._buffer_presize = 0
    engine._cached_plan = _plan(streams=streams, chunk_bytes=chunk_bytes)
    engine._router = SimpleNamespace(
        num_producers=1,
        reserve_serve_buffers=lambda _sizes, _digests: [],
        pull=lambda _owner, _keys, _seq: object(),
    )
    config = engine._configure_ring(
        ShardedRDTWeightTransferInitInfo(
            staging_device=staging_device,
            stages_in_flight=stages_in_flight,
            num_rdt_buffers=ring_depth,
        )
    )
    engine._size_pool(config, engine._cached_plan)
    return engine


@pytest.fixture
def registered(monkeypatch):
    buffers: list[torch.Tensor] = []
    monkeypatch.setattr(engine_module, "register_nixl_memory", buffers.append)
    monkeypatch.setattr(
        engine_module,
        "buffer_alloc_bytes",
        lambda nbytes, presize=0: max(nbytes, presize),
    )
    return buffers


@pytest.fixture
def allocations(monkeypatch):
    real_empty = torch.empty
    state = SimpleNamespace(calls=[], failure=None)

    def empty(*args, **kwargs):
        state.calls.append(dict(kwargs))
        if kwargs.get("pin_memory"):
            if state.failure is not None:
                raise state.failure
            kwargs = {
                key: value for key, value in kwargs.items() if key != "pin_memory"
            }
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    return state


class TestPoolGeometry:
    def test_configuration_records_settings_without_allocating(self, monkeypatch):
        engine = _engine()
        old_pool = engine._dest_buffers
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *_args, **_kwargs: pytest.fail("configuration allocated"),
        )

        config = engine._configure_ring(
            ShardedRDTWeightTransferInitInfo(
                staging_device="host",
                stages_in_flight=3,
                num_rdt_buffers=4,
                buffer_presize_gb=0.5,
            )
        )

        assert (
            config.staging_device,
            config.stages_in_flight,
            config.ring_depth,
            config.buffer_presize,
        ) == ("host", 3, 4, 1 << 29)
        assert engine._dest_buffers is old_pool

    def test_pool_has_one_k_slot_block_per_effective_lane(self):
        engine = _engine(stages_in_flight=2, ring_depth=3, streams=4)
        assert engine._lanes == 2
        assert engine._num_slots == 6
        assert len(engine._dest_buffers) == 6
        assert list(engine._lane_slots(0)) == [0, 1, 2]
        assert list(engine._lane_slots(1)) == [3, 4, 5]

    def test_requested_stages_are_clamped_to_plan_streams(self):
        engine = _engine(stages_in_flight=5, ring_depth=2, streams=3)
        assert (engine._lanes, engine._num_slots) == (3, 6)

    def test_empty_plan_keeps_one_lane(self):
        engine = _engine(
            stages_in_flight=5,
            ring_depth=2,
            streams=0,
            chunk_bytes=(),
        )
        assert (engine._lanes, engine._num_slots) == (1, 2)


class TestPoolReinitialization:
    @staticmethod
    def _activate(engine):
        engine._proc_thread = object()
        engine._slot_read_done = [object() for _ in range(engine._num_slots)]
        engine._slot_queued = [0] * engine._num_slots
        engine._slot_done = [0] * engine._num_slots

    @staticmethod
    def _references(engine):
        return (
            engine._dest_buffers,
            engine._slot_read_done,
            engine._slot_queued,
            engine._slot_done,
            engine._lane_cursors,
            engine._targets_cache,
        )

    @pytest.mark.parametrize(
        ("info", "plan"),
        [
            pytest.param(
                ShardedRDTWeightTransferInitInfo(
                    staging_device="host",
                    stages_in_flight=2,
                    num_rdt_buffers=2,
                ),
                _plan(streams=2),
                id="staging",
            ),
            pytest.param(
                ShardedRDTWeightTransferInitInfo(
                    stages_in_flight=1,
                    num_rdt_buffers=2,
                ),
                _plan(streams=1),
                id="lanes",
            ),
            pytest.param(
                ShardedRDTWeightTransferInitInfo(
                    stages_in_flight=1,
                    num_rdt_buffers=4,
                ),
                _plan(streams=1),
                id="equal-product-different-k",
            ),
        ],
    )
    def test_geometry_change_is_rejected_atomically(self, info, plan):
        engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
        self._activate(engine)
        references = self._references(engine)
        values = (
            engine._pool_geometry,
            engine._staging_device,
            engine._lanes,
            engine._ring_depth,
            engine._num_slots,
        )

        config = engine._configure_ring(info)
        with pytest.raises(RuntimeError, match=r"old=.*L=2, K=2.*new="):
            engine._size_pool(config, plan)

        assert all(
            actual is expected
            for actual, expected in zip(self._references(engine), references)
        )
        assert (
            engine._pool_geometry,
            engine._staging_device,
            engine._lanes,
            engine._ring_depth,
            engine._num_slots,
        ) == values

    def test_exact_geometry_reuses_every_pool_reference(self):
        engine = _engine(stages_in_flight=2, ring_depth=2, streams=2)
        self._activate(engine)
        references = self._references(engine)
        config = engine._configure_ring(
            ShardedRDTWeightTransferInitInfo(
                stages_in_flight=5,
                num_rdt_buffers=2,
            )
        )

        engine._size_pool(config, _plan(streams=2))

        assert engine._pool_geometry == ("gpu", 2, 2)
        assert all(
            actual is expected
            for actual, expected in zip(self._references(engine), references)
        )


class TestReceiveAllocation:
    def test_gpu_slots_use_engine_device_without_pinning(
        self,
        registered,
        allocations,
    ):
        engine = _engine()

        engine._preregister_at_init()

        slot_calls = [
            call for call in allocations.calls if call.get("dtype") is torch.uint8
        ]
        assert len(slot_calls) == 2
        assert all(call["device"] == engine.device for call in slot_calls)
        assert all("pin_memory" not in call for call in slot_calls)
        assert len(registered) == 2

    def test_preregisters_every_l_by_k_slot_once(self, registered):
        engine = _engine(
            stages_in_flight=2,
            ring_depth=2,
            streams=2,
            chunk_bytes=(48, 16),
        )

        engine._preregister_at_init()

        buffers = [slot[torch.uint8] for slot in engine._dest_buffers]
        assert registered == buffers
        assert len({buffer.data_ptr() for buffer in buffers}) == 4
        assert all(buffer.numel() == 48 for buffer in buffers)

        engine._preregister_at_init()
        assert registered == buffers

    def test_defensive_grow_uses_the_same_registered_seam(
        self,
        registered,
        monkeypatch,
    ):
        import ray.experimental

        monkeypatch.setattr(
            ray.experimental,
            "set_target_for_ref",
            lambda _ref, _targets: None,
        )
        engine = _engine(ring_depth=1)
        engine._preregister_at_init()
        engine._dest_buffers[0].clear()

        engine._issue_pull(engine._cached_plan.chunks[0], 0)

        assert registered[-1] is engine._dest_buffers[0][torch.uint8]
        assert len(registered) == 2

    def test_host_slots_are_pinned_cpu_memory(
        self,
        registered,
        allocations,
    ):
        engine = _engine(
            staging_device="host",
            stages_in_flight=2,
            ring_depth=2,
            streams=2,
        )

        engine._preregister_at_init()

        slot_calls = [
            call for call in allocations.calls if call.get("dtype") is torch.uint8
        ]
        assert len(slot_calls) == 4
        assert all(
            call["device"] == "cpu" and call["pin_memory"] is True
            for call in slot_calls
        )
        assert len(registered) == 4

    def test_host_allocation_failure_is_named_and_preserves_cause(
        self,
        registered,
        allocations,
    ):
        del registered
        engine = _engine(
            staging_device="host",
            stages_in_flight=2,
            ring_depth=2,
            streams=2,
        )
        allocations.failure = RuntimeError("pinned allocator exhausted")

        with pytest.raises(HostStagingAllocationError) as raised:
            engine._preregister_at_init()

        assert "4 slots (2 lane(s) x K=2)" in str(raised.value)
        assert "128 bytes per worker" in str(raised.value)
        assert raised.value.__cause__ is allocations.failure

    def test_host_registration_failure_preserves_cause(
        self,
        monkeypatch,
        allocations,
        registered,
    ):
        del registered
        engine = _engine(staging_device="host", ring_depth=1)
        cause = RuntimeError("NIXL registration failed")
        monkeypatch.setattr(
            engine_module,
            "register_nixl_memory",
            lambda _buffer: (_ for _ in ()).throw(cause),
        )

        with pytest.raises(
            HostStagingAllocationError,
            match="NIXL registration failed",
        ) as raised:
            engine._preregister_at_init()

        assert raised.value.__cause__ is cause


def test_current_scheduler_remains_a_single_lane_adapter(monkeypatch):
    engine = _engine(
        stages_in_flight=2,
        ring_depth=2,
        streams=2,
        chunk_bytes=(8, 8, 8),
    )
    engine._completed_syncs = 1
    issued: list[tuple[int, int]] = []

    def issue(chunk, slot):
        issued.append((chunk.stream, slot))
        return SimpleNamespace(slot=slot)

    monkeypatch.setattr(engine, "_issue_pull", issue)
    monkeypatch.setattr(engine, "_complete_pull", lambda _pending: {})
    monkeypatch.setattr(engine, "_dispatch_item", lambda _item: None)

    engine._run_chunk_pipeline(engine._cached_plan)

    assert issued == [(0, 0), (1, 1), (0, 0)]
