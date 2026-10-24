# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    MemoryRuntime,
    _CancellableWorkerHandle,
    _DelayedLoadWorkerHandle,
    _DelayedStoreWorkerHandle,
    _EmbeddedRuntime,
    _kv_cache_config,
    _SchedulerHandle,
    _StoreRecordingWorkerHandle,
    _vllm_config,
    _WaitRecordingDelayedLoadHandle,
    _WorkerHandle,
    handshake,
    install_memory_runtime,
    make_kv_caches,
    make_request,
    make_scheduler_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockStoreBatch,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVLayoutPlanner,
    KVRange,
    KVRegion,
    LoadSpec,
    RankTopology,
    RequestTracker,
    StoreEventResult,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

mori_available = importlib.util.find_spec("mori") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_available),
    reason="UMBP is only available on ROCm with the mori package installed",
)

_LAYER1_SPEC = FullAttentionSpec(
    block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
)
_LAYER2_SPEC = replace(_LAYER1_SPEC, num_kv_heads=2)


def _uniform_type_kv_cache_config(
    layer2: FullAttentionSpec = _LAYER2_SPEC,
    *,
    block_stride: int | None = None,
    num_blocks: int = 8,
) -> KVCacheConfig:
    """One UniformTypeKVCacheSpecs group whose two layers have unequal pages.

    Without ``block_stride`` each layer is laid out layer-major with its own
    page as the block stride; with it both pages share one block-major stride.
    """
    specs = {"layer1": _LAYER1_SPEC, "layer2": layer2}
    tensors = []
    offset = 0
    for name, spec in specs.items():
        stride = block_stride or spec.page_size_bytes
        tensors.append(
            KVCacheTensor(
                size=stride * num_blocks,
                layers=[name],
                layer_stride=stride * num_blocks,
                block_stride=stride,
                offset=offset if block_stride else 0,
            )
        )
        offset += spec.page_size_bytes
    group_spec = UniformTypeKVCacheSpecs.from_specs(specs)
    assert group_spec is not None
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[KVCacheGroupSpec(list(specs), group_spec)],
    )


def test_block_identity_codec_does_not_use_physical_block_id():
    codec = BlockIdentityCodec(UMBPNamespace("namespace"), 2, 3)

    assert codec.key(b"\x01\x02", group_id=4) == (
        "umbp:vllm:v1:namespace:tp2:pcp0:dcp0:pp3:g4:0102"
    )
    assert codec.key(b"\x01\x02", group_id=4) == codec.key(b"\x01\x02", group_id=4)


def test_layout_planner_orders_all_transfer_layers():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())

    assert [region.layer_name for region in planner.regions] == ["layer1", "layer2"]
    assert planner.regions[0].object_offset == 0
    assert planner.regions[1].object_offset == planner.regions[0].block_bytes
    assert planner.object_size == 2 * planner.regions[0].block_bytes


def test_rank_topology_is_part_of_layout_identity():
    topology = RankTopology(
        tp_rank=1,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        pcp_rank=1,
        pcp_size=2,
        dcp_rank=0,
        dcp_size=1,
    )

    assert topology.local_namespace == (1, 1, 0, 0)
    assert topology.rank_count == 4
    assert len(topology.all_namespaces()) == 4


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 0)), (2, (0, 1, 0, 0)), (5, (1, 0, 0, 1))],
)
def test_rank_topology_derives_ranks_from_worker_rank(rank, expected):
    # Workers are laid out as [PP, PCP, TP], TP innermost.
    topology = RankTopology.from_vllm_config(
        _vllm_config(
            {},
            rank=rank,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            pipeline_parallel_size=2,
            world_size=8,
        )
    )

    assert topology.local_namespace == expected


def test_rank_topology_maps_dcp_onto_tp_workers():
    topology = RankTopology(tp_rank=1, tp_size=4, dcp_rank=1, dcp_size=2)

    assert topology.rank_count == 4
    assert topology.all_namespaces() == (
        (0, 0, 0, 0),
        (1, 0, 1, 0),
        (2, 0, 0, 0),
        (3, 0, 1, 0),
    )


def test_layout_descriptor_carries_topology_and_format():
    topology = RankTopology(tp_size=2)
    descriptor = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config()).describe(
        topology, "lbh_nc"
    )

    assert isinstance(descriptor, KVLayoutDescriptor)
    assert descriptor.layout_format == "lbh_nc"
    assert descriptor.topology.tp_size == 2


def test_namespace_distinguishes_draft_models():
    config = _kv_cache_config()

    def namespace(speculative_config):
        vllm_config = _vllm_config({"mode": "embedded"})
        if speculative_config is not None:
            vllm_config.speculative_config = speculative_config
        return UMBPNamespace.from_vllm_config(vllm_config, config).value

    def draft(model, revision="main"):
        return SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(model=model, revision=revision),
        )

    plain = namespace(None)
    assert namespace(SimpleNamespace()) != plain
    assert namespace(draft("drafter-a")) != plain
    assert namespace(draft("drafter-a")) == namespace(draft("drafter-a"))
    assert namespace(draft("drafter-a")) != namespace(draft("drafter-b"))
    assert namespace(draft("drafter-a")) != namespace(draft("drafter-a", "v2"))


def test_layout_planner_sizes_uniform_type_layers_independently():
    """A wrapper's page size is the sum of its layers, not any one layer's."""
    planner = KVLayoutPlanner.from_kv_cache_config(_uniform_type_kv_cache_config())

    assert [region.block_bytes for region in planner.regions] == [64, 128]
    assert [region.object_offset for region in planner.regions] == [0, 64]
    assert planner.object_size == 192


def test_uniform_type_block_major_round_trip_stays_within_layer_pages():
    num_blocks = 8
    block_bytes = 192
    planner = KVLayoutPlanner.from_kv_cache_config(
        _uniform_type_kv_cache_config(block_stride=block_bytes, num_blocks=num_blocks)
    )
    backing = torch.zeros(num_blocks * block_bytes, dtype=torch.uint8)
    caches = {
        "layer1": backing.as_strided((num_blocks, 64), (block_bytes, 1)),
        "layer2": backing.as_strided((num_blocks, 128), (block_bytes, 1), 64),
    }
    planner.register_kv_caches(caches)

    source = planner.plan_registered_block("key", block_id=1)
    assert [
        (item.base_address - backing.data_ptr(), item.length, item.object_offset)
        for item in source.ranges
    ] == [(192, 64, 0), (256, 128, 64)]

    def block(index):
        return backing[index * block_bytes : (index + 1) * block_bytes]

    block(1).copy_(torch.arange(block_bytes, dtype=torch.uint8))
    block(4).fill_(0xAB)
    handle = MemoryRuntime().create_worker_handle(
        "namespace", RankTopology(), planner.describe(RankTopology())
    )
    handle.register_buffers(caches)
    handle.publish(handle.store([source]))
    loaded = handle.load([planner.plan_registered_block("key", block_id=3)])

    assert loaded.status is TransferJobStatus.COMPLETED
    assert torch.equal(block(3), block(1))
    assert torch.all(block(4) == 0xAB)


@pytest.mark.parametrize(
    ("baseline", "variant"),
    [
        pytest.param(
            _LAYER2_SPEC, replace(_LAYER2_SPEC, dtype=torch.bfloat16), id="dtype"
        ),
        pytest.param(
            _LAYER2_SPEC,
            replace(_LAYER2_SPEC, num_kv_heads=1, head_size=2),
            id="geometry",
        ),
        pytest.param(
            replace(
                _LAYER2_SPEC,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            ),
            replace(
                _LAYER2_SPEC,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            ),
            id="quantization",
        ),
    ],
)
def test_namespace_fingerprints_uniform_type_layer_specs(baseline, variant):
    vllm_config = _vllm_config({"mode": "embedded"})

    def namespace(layer2):
        return UMBPNamespace.from_vllm_config(
            vllm_config, _uniform_type_kv_cache_config(layer2)
        ).value

    assert namespace(baseline) == namespace(baseline)
    assert namespace(baseline) != namespace(variant)


def test_scheduler_uses_worker_namespace_for_uniform_type_groups(monkeypatch):
    """The scheduler sees a flattened group spec; keys must still match."""
    install_memory_runtime(monkeypatch)
    worker_config = _uniform_type_kv_cache_config(block_stride=192)
    vllm_config = _vllm_config({"mode": "embedded"})
    scheduler = UMBPStoreConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        generate_scheduler_kv_cache_config([worker_config]),
    )
    worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, worker_config)
    try:
        handshake(scheduler, worker)

        assert scheduler.connector_scheduler.codec.key(b"a") == (
            worker.connector_worker.codec.key(b"a")
        )
    finally:
        scheduler.shutdown()
        worker.shutdown()


def test_scheduler_defers_lookup_until_worker_handshake(monkeypatch):
    runtime = _EmbeddedRuntime()
    monkeypatch.setitem(UMBPRuntimeFactory._builders, "embedded", lambda _: runtime)
    scheduler = UMBPStoreConnector(
        _vllm_config({"mode": "embedded"}),
        KVConnectorRole.SCHEDULER,
        _kv_cache_config(),
    )
    request = SimpleNamespace(
        request_id="early", num_tokens=33, block_hashes=[b"a", b"b"]
    )

    assert runtime.scheduler_args is None
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


@pytest.mark.parametrize("no_reports", [False, True])
def test_scheduler_rejects_inconsistent_worker_namespaces(monkeypatch, no_reports):
    install_memory_runtime(monkeypatch)
    parallel = {"tensor_parallel_size": 2, "world_size": 2}
    config = _kv_cache_config()
    other_model = _vllm_config({"mode": "embedded"}, rank=1, **parallel)
    other_model.model_config.model = "other-model"
    workers = [
        UMBPStoreConnector(
            _vllm_config({"mode": "embedded"}, rank=0, **parallel),
            KVConnectorRole.WORKER,
            config,
        ),
        UMBPStoreConnector(other_model, KVConnectorRole.WORKER, config),
    ]
    scheduler = UMBPStoreConnector(
        _vllm_config({"mode": "embedded"}, **parallel),
        KVConnectorRole.SCHEDULER,
        config,
    )
    reports = {
        rank: worker.get_handshake_metadata() for rank, worker in enumerate(workers)
    }
    if no_reports:
        reports = {0: None}
    try:
        with pytest.raises(ValueError, match="namespace"):
            scheduler.set_xfer_handshake_metadata(reports)
    finally:
        scheduler.shutdown()
        for worker in workers:
            worker.shutdown()


def test_scheduler_binds_when_ranks_share_a_report_key(monkeypatch):
    """Without DCP, the engine keys every PCP rank's report by its TP rank."""
    install_memory_runtime(monkeypatch)
    vllm_config = _vllm_config(
        {"mode": "embedded"}, prefill_context_parallel_size=2, world_size=2
    )
    config = _kv_cache_config()
    scheduler = UMBPStoreConnector(vllm_config, KVConnectorRole.SCHEDULER, config)
    worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
    try:
        scheduler.set_xfer_handshake_metadata({0: worker.get_handshake_metadata()})

        assert scheduler.connector_scheduler.codec == worker.connector_worker.codec
    finally:
        scheduler.shutdown()
        worker.shutdown()


def test_layout_planner_builds_scatter_gather_ranges():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    plan = planner.plan_for_block(
        "key", block_id=3, base_addresses={"layer1": 1000, "layer2": 2000}
    )

    assert [item.base_address for item in plan.ranges] == [4072, 5072]
    assert [item.object_offset for item in plan.ranges] == [
        0,
        planner.regions[0].block_bytes,
    ]


def test_layout_planner_registers_real_tensor_addresses():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    caches = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in ("layer1", "layer2")
    }

    planner.register_kv_caches(caches)
    plan = planner.plan_registered_block("key", block_id=2)

    assert plan.ranges[0].base_address == caches["layer1"].data_ptr() + 2048
    assert plan.ranges[1].base_address == caches["layer2"].data_ptr() + 2048


def test_layout_planner_expands_kernel_subblocks():
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    caches = {
        name: torch.empty((8, 2, 16, 8), dtype=torch.float16)
        for name in ("layer1", "layer2")
    }

    planner.register_kv_caches(caches)
    plan = planner.plan_registered_block("key", block_id=2)

    assert len(plan.ranges) == 4
    assert [item.base_address for item in plan.ranges[:2]] == [
        caches["layer1"].data_ptr() + 2048,
        caches["layer1"].data_ptr() + 2560,
    ]


def test_layout_planner_builds_compact_group_objects():
    planner = KVLayoutPlanner(
        (
            KVRegion("attention.0", 0, 256, 256, 0, 16),
            KVRegion("mamba.0", 1, 512, 512, 256, 32),
            KVRegion("attention.1", 0, 256, 256, 768, 16),
        )
    )
    addresses = {"attention.0": 1000, "mamba.0": 2000, "attention.1": 3000}

    attention = planner.plan_for_block(
        "attention",
        2,
        addresses,
        group_id=0,
    )
    mamba = planner.plan_for_block(
        "mamba",
        2,
        addresses,
        group_id=1,
    )

    assert [item.layer_name for item in attention.ranges] == [
        "attention.0",
        "attention.1",
    ]
    assert [item.object_offset for item in attention.ranges] == [0, 256]
    assert max(item.object_offset + item.length for item in attention.ranges) == 512
    assert [item.layer_name for item in mamba.ranges] == ["mamba.0"]
    assert [item.object_offset for item in mamba.ranges] == [0]
    assert max(item.object_offset + item.length for item in mamba.ranges) == 512


def test_layout_planner_rejects_unknown_group():
    planner = KVLayoutPlanner((KVRegion("attention", 0, 256, 256, 0, 16),))

    with pytest.raises(ValueError, match="does not contain cache group 1"):
        planner.plan_for_block("missing", 0, {"attention": 1000}, group_id=1)


def test_worker_materializes_only_the_requested_group():
    planner = KVLayoutPlanner(
        (
            KVRegion("attention", 0, 32, 32, 0, 16),
            KVRegion("mamba", 1, 32, 32, 32, 16),
        )
    )
    planner.register_kv_caches(
        {
            "attention": torch.empty((8, 16), dtype=torch.float16),
            "mamba": torch.empty((8, 16), dtype=torch.float16),
        }
    )
    worker = UMBPStoreConnectorWorker(_WorkerHandle(), planner)

    [materialized] = worker._materialize_plans(
        [BlockTransferPlan("mamba-key", 2, group_id=1)]
    )

    assert [item.layer_name for item in materialized.ranges] == ["mamba"]
    assert [item.object_offset for item in materialized.ranges] == [0]


def test_materialization_preserves_plan_metadata_after_reregistration():
    """Compact ranges must use current buffers without losing logical identity."""
    planner = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    worker = UMBPStoreConnectorWorker(_WorkerHandle(), planner)
    plan = BlockTransferPlan(
        "key",
        2,
        request_id="req",
        group_id=0,
        block_hash=b"hash",
        parent_block_hash=b"parent",
        token_ids=(1, 2),
        block_size=16,
    )
    previous = None
    for _ in range(2):
        caches = {
            name: torch.empty((8, 2, 16, 8), dtype=torch.float16)
            for name in ("layer1", "layer2")
        }
        planner.register_kv_caches(caches)
        [actual] = worker._materialize_plans([plan])
        expected = planner.plan_registered_block(
            plan.key,
            plan.block_id,
            group_id=0,
        )
        assert actual == replace(plan, ranges=expected.ranges)
        assert actual.ranges[0].base_address == (caches["layer1"].data_ptr() + 2048)
        if previous is not None:
            assert actual.ranges != previous.ranges
        previous = actual


def test_transfer_job_isolates_failed_keys():
    plans = (
        BlockTransferPlan("key-a", 7),
        BlockTransferPlan("key-b", 8),
    )
    job = TransferJobState(plans)
    job.start()
    job.complete(["key-a"])
    job.fail(["key-b"], "timeout")

    assert job.status is TransferJobStatus.FAILED
    assert job.failed_block_ids == {8}
    assert job.completed_keys == {"key-a"}


def test_request_tracker_advances_only_complete_block_watermarks():
    tracker = RequestTracker()

    assert tracker.mark_saved(31, 16) == 16
    assert tracker.mark_saved(15, 16) == 16
    tracker.reset()

    assert tracker.saved_tokens == 0


def test_load_spec_reports_only_external_tokens():
    spec = LoadSpec(local_tokens=32, external_tokens=48)

    assert spec.num_tokens_to_load == 16


def test_worker_metadata_aggregates_store_events():
    metadata = UMBPConnectorWorkerMetadata(
        store_events={7: StoreEventResult(1, {(0, b"a")})}
    )
    other = UMBPConnectorWorkerMetadata(
        store_events={7: StoreEventResult(1, {(1, b"b")})}
    )
    metadata.aggregate(other)

    assert metadata.store_events == {7: StoreEventResult(2, {(0, b"a"), (1, b"b")})}
    assert other.store_events == {7: StoreEventResult(1, {(1, b"b")})}

    restored = MsgpackDecoder(UMBPConnectorWorkerMetadata).decode(
        MsgpackEncoder().encode(metadata)
    )
    assert restored == metadata


@pytest.mark.parametrize("mode", ["embedded"])
def test_runtime_config_accepts_local_modes(mode):
    config = UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": mode}))

    assert config.mode == mode


@pytest.mark.parametrize("mode", ["embedded"])
def test_runtime_adapter_owns_its_configuration(monkeypatch, mode):
    runtime = SimpleNamespace()

    def build(config):
        if not config.options.get("adapter_key"):
            raise ValueError("adapter_key required")
        assert config.rank_count == 4
        return runtime

    monkeypatch.setitem(UMBPRuntimeFactory._builders, mode, build)
    config = UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": mode}))
    with pytest.raises(ValueError, match="adapter_key required"):
        UMBPRuntimeFactory.build(config)
    config = UMBPRuntimeConfig.from_vllm(
        _vllm_config({"mode": mode, "adapter_key": "pool"})
    ).resolve_for_rank_count(4)
    assert UMBPRuntimeFactory.build(config) is runtime


def test_scheduler_tracks_consecutive_load_plan():
    config = _kv_cache_config()
    handle = _SchedulerHandle([True, False])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        config,
        handle,
        BlockIdentityCodec(UMBPNamespace("test")),
    )
    request = make_request("req", num_tokens=33)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, False)
    assert handle.queries[0] == [
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g0:61",
        "umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g0:62",
    ]


@pytest.mark.parametrize("failure", [TimeoutError("lookup timeout"), [True, False]])
def test_scheduler_lookup_failure_recomputes_and_allows_retry(failure):
    class _Handle(_SchedulerHandle):
        def lookup(self, keys):
            if isinstance(self.hits, Exception):
                raise self.hits
            return self.hits

    handle = _Handle(failure)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"load_async": False}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lookup-retry")),
    )
    request = make_request("retry")
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        handle.hits = [True]
        assert scheduler.get_num_new_matched_tokens(request, 0) == (16, False)
    finally:
        scheduler.close()


@pytest.mark.parametrize(("use_eagle", "expected"), [(False, 32), (True, 16)])
def test_scheduler_single_group_lookup_drops_last_block_for_eagle(use_eagle, expected):
    vllm_config = _vllm_config({"mode": "embedded", "load_async": False})
    vllm_config.speculative_config = SimpleNamespace(
        use_eagle_block_drop=lambda: use_eagle
    )
    scheduler = UMBPStoreConnectorScheduler(
        vllm_config,
        _kv_cache_config(),
        _SchedulerHandle([True, True]),
        BlockIdentityCodec(UMBPNamespace("eagle")),
    )
    request = make_request("req", num_tokens=33)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (expected, False)


def test_scheduler_emits_async_load_without_scheduled_model_tokens():
    config = _kv_cache_config()
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": True}),
        config,
        _SchedulerHandle([True]),
        BlockIdentityCodec(UMBPNamespace("async-load")),
    )
    request = make_request("async-load")

    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, True)
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7],)),
        16,
    )
    metadata = scheduler.build_connector_meta(make_scheduler_output())

    assert [plan.block_id for plan in metadata.load_requests["async-load"]] == [7]
    assert scheduler._pending_loads == {}


def test_load_batch_metadata_roundtrip_preserves_destinations_and_identity():
    batch = BlockLoadBatch(["g1", "g0"], [7, 3], [1, 0], "r")
    meta = UMBPConnectorMetadata(load_requests={"r": batch})
    restored = (
        MsgpackDecoder(UMBPConnectorMetadata)
        .decode(MsgpackEncoder().encode(meta))
        .load_requests["r"]
    )
    assert list(restored) == [
        BlockTransferPlan("g1", 7, group_id=1, request_id="r"),
        BlockTransferPlan("g0", 3, group_id=0, request_id="r"),
    ]
    assert restored[:1] == [restored[0]]
    with pytest.raises(ValueError, match="equal lengths"):
        BlockLoadBatch(["missing-destination"], [], [0], "r")


def test_scheduler_cached_decode_uses_save_watermark_and_new_blocks():
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "load_async": False,
            "save_decode_cache": True,
        }
    )
    vllm_config.kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    scheduler = UMBPStoreConnectorScheduler(
        vllm_config,
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("decode")),
    )
    request = make_request(
        "decode",
        num_tokens=48,
        num_prompt_tokens=32,
        all_token_ids=list(range(48)),
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([1, 2, 3],),
        num_computed_tokens=32,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )

    def output(num_computed_tokens, new_block_ids):
        return make_scheduler_output(
            cached_reqs=SimpleNamespace(
                req_ids=["decode"],
                new_block_ids=[new_block_ids],
                num_computed_tokens=[num_computed_tokens],
            ),
            scheduled_tokens={"decode": 1},
        )

    first = scheduler.build_connector_meta(output(32, ([3],)))
    assert [plan.block_id for plan in first.iter_store_plans()] == [1, 2]

    second = scheduler.build_connector_meta(output(48, ()))
    second_plans = list(second.iter_store_plans())
    assert [plan.block_id for plan in second_plans] == [3]
    assert second_plans[0].token_ids == tuple(range(32, 48))


def test_scheduler_decode_store_skips_unverified_speculative_tokens():
    """Scheduled draft tokens reach past the last hashed block; stop there."""
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "load_async": False,
                "save_decode_cache": True,
            }
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("spec")),
    )
    request = make_request(
        "spec",
        num_tokens=47,
        num_prompt_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([1, 2, 3],),
        num_computed_tokens=46,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2, 3],)),
        0,
    )
    # One verified token plus three draft tokens: 46 + 4 = 50 > 47 tokens.
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            cached_reqs=SimpleNamespace(
                req_ids=["spec"],
                new_block_ids=[()],
                num_computed_tokens=[46],
            ),
            scheduled_tokens={"spec": 4},
        )
    )
    assert [plan.block_id for plan in metadata.iter_store_plans()] == [1, 2]


def test_scheduler_skips_decode_blocks_without_save_decode_cache():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("no-decode")),
    )
    request = make_request(
        "r",
        num_tokens=48,
        num_prompt_tokens=32,
        block_hashes=[b"a", b"b", b"c"],
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2, 3],)), 0
    )
    scheduler._request_trackers["r"].saved_tokens = 32
    # A decode step under synchronous scheduling: num_tokens - 1 are computed.
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            cached_reqs=SimpleNamespace(
                req_ids=["r"], new_block_ids=[None], num_computed_tokens=[47]
            ),
            scheduled_tokens={"r": 1},
        )
    )

    assert metadata.store_plans == []


@pytest.mark.parametrize("kv_role", ["kv_both", "kv_consumer"])
def test_kv_consumer_never_stores(kv_role):
    vllm_config = _vllm_config({"mode": "embedded", "load_async": False})
    vllm_config.kv_transfer_config.kv_role = kv_role
    scheduler = UMBPStoreConnectorScheduler(
        vllm_config,
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("role")),
    )
    request = make_request(
        "r",
        block_ids=([3, 4],),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4],)),
        0,
    )
    scheduled = SimpleNamespace(req_id="r", num_computed_tokens=0, block_ids=([3, 4],))
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(new_reqs=[scheduled], scheduled_tokens={"r": 32})
    )
    stored = [plan.block_id for plan in metadata.iter_store_plans()]
    assert stored == ([] if kv_role == "kv_consumer" else [3, 4])
    assert bool(metadata.store_requests) == (kv_role != "kv_consumer")


@pytest.mark.parametrize("resident", [False, True])
def test_restored_prefix_skips_resident_blocks(resident):
    """A restored block that is still in the pool is not stored again."""
    codec = BlockIdentityCodec(UMBPNamespace("restored-residency"))
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}), _kv_cache_config(), handle, codec
    )
    request = make_request(
        "req",
        num_tokens=49,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([1, 2, 3],),
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids), 0
    )
    scheduler._request_trackers["req"].load_spec = LoadSpec(0, 32)
    handle.hits = {codec.key(h, 0): resident for h in (b"a", b"b")}
    meta = scheduler.build_connector_meta(
        make_scheduler_output(new_reqs=[request], scheduled_tokens={"req": 48})
    )

    stored = [b"c"] if resident else [b"a", b"b", b"c"]
    assert [plan.key for plan in meta.iter_store_plans()] == [
        codec.key(h, 0) for h in stored
    ]
    scheduler.close()


def test_store_event_owns_refs_until_all_ranks_finish():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded"},
            tensor_parallel_size=2,
            world_size=2,
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("store-refs")),
        RankTopology(tp_size=2),
    )
    block = SimpleNamespace(block_id=4)
    freed = []
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=[None, None, None, None, block],
            touch=lambda blocks: None,
            free_blocks=lambda blocks: freed.extend(blocks),
        )
    )
    request = make_request(
        "store-refs",
        num_tokens=16,
        block_hashes=[b"a"],
        block_ids=([4],),
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([4],)), 0
    )
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(new_reqs=[request], scheduled_tokens={request.req_id: 16})
    )
    one_rank = SimpleNamespace(
        kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
            store_events={metadata.store_event: StoreEventResult(1)}
        )
    )

    scheduler.update_connector_output(one_rank)
    assert freed == []
    assert scheduler.has_pending_push_work()
    assert not scheduler.reset_store()

    scheduler.update_connector_output(one_rank)
    assert freed == [block]
    assert not scheduler.has_pending_push_work()
    assert scheduler.reset_store()
    # A late report for a finished event releases nothing twice.
    scheduler.update_connector_output(one_rank)
    assert freed == [block]


def test_store_events_deduplicate_in_flight_shared_prefix_objects():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("event-refs")),
    )
    pinned, freed = [], []
    scheduler.bind_gpu_block_pool(
        SimpleNamespace(
            blocks=list(range(8)),
            touch=lambda blocks: pinned.extend(blocks),
            free_blocks=lambda blocks: freed.extend(blocks),
        )
    )
    events = []
    for request_id in ("first", "second"):
        request = make_request(
            request_id,
            num_tokens=16,
            block_hashes=[b"a"],
            block_ids=([4],),
        )
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda group_ids: ([4],)),
            0,
        )
        metadata = scheduler.build_connector_meta(
            make_scheduler_output(new_reqs=[request], scheduled_tokens={request_id: 16})
        )
        events.append(metadata.store_event)
    assert pinned == [4]
    assert events == [0, -1]

    for event_id in (-1, 0):
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={event_id: StoreEventResult(1)},
                ),
            )
        )
        assert len(freed) == (1 if event_id == 0 else 0)
        assert scheduler.has_pending_push_work() == (event_id != 0)


def test_scheduler_resumed_request_replaces_stale_block_table():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "save_decode_cache": True,
            }
        ),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("resumed")),
    )
    request = make_request(
        "resumed",
        block_ids=([7, 8],),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            cached_reqs=SimpleNamespace(
                req_ids=["resumed"],
                new_block_ids=[([7, 8],)],
                num_computed_tokens=[31],
                resumed_req_ids={"resumed"},
            ),
            scheduled_tokens={"resumed": 1},
        )
    )

    tracker = scheduler._request_trackers["resumed"]
    assert tracker.block_ids == ([7, 8],)
    assert [plan.block_id for plan in metadata.iter_store_plans()] == [7, 8]


def test_scheduler_stores_from_cores_current_block_table():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("current-table")),
    )
    request = make_request(
        "r",
        num_tokens=64,
        num_prompt_tokens=64,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4],)), 0
    )
    # Core has since freed block 3, e.g. a sliding-window block out of the window.
    current = {"r": ([NULL_BLOCK_ID, 4],)}
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            cached_reqs=SimpleNamespace(
                req_ids=["r"],
                new_block_ids=[None],
                num_computed_tokens=[0],
                resumed_req_ids=set(),
            ),
            scheduled_tokens={"r": 32},
            block_state=SimpleNamespace(
                get_block_ids=current.get, boundary_state_offloads={}
            ),
        )
    )

    assert [plan.block_id for plan in metadata.iter_store_plans()] == [4]


def test_eager_store_deduplicates_same_step_shared_prefix_keys():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("same-step-dedup")),
    )
    first = make_request("first", block_ids=([1, 2],))
    second = make_request("second", block_ids=([3, 4],))
    reserved: set[str] = set()

    first_batch = scheduler._store_plans(
        first, RequestTracker(), 32, skip_keys=reserved
    )
    reserved.update(first_batch.keys)
    second_batch = scheduler._store_plans(
        second, RequestTracker(), 32, skip_keys=reserved
    )

    assert [plan.block_id for plan in first_batch] == [1, 2]
    assert len(second_batch) == 0


def test_scheduler_readmission_after_failed_load_uses_new_blocks():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("readmitted")),
    )
    request = make_request(
        "readmitted",
        num_tokens=64,
        num_prompt_tokens=64,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)), 0
    )
    # The load failed: core freed blocks 1 and 2 and recomputes from token 0.
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([5, 6],)), 0
    )
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            cached_reqs=SimpleNamespace(
                req_ids=["readmitted"],
                new_block_ids=[None],
                num_computed_tokens=[0],
                resumed_req_ids=set(),
            ),
            scheduled_tokens={"readmitted": 32},
        )
    )

    assert scheduler._request_trackers["readmitted"].block_ids == ([5, 6],)
    assert [plan.block_id for plan in metadata.iter_store_plans()] == [5, 6]


@pytest.mark.parametrize("fail_load", [False, True])
def test_worker_polls_async_load_without_forward_execution(fail_load):
    handle = _DelayedLoadWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    plan = BlockTransferPlan("async-load", 1, request_id="req")
    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(
            async_load=True,
            load_requests={"req": [plan]},
        ),
    )

    assert worker.get_finished(set()) == (None, None)

    if fail_load:
        handle.load_job.fail([plan.key], "load failed")
    else:
        handle.load_job.complete()

    assert worker.get_finished(set()) == (None, {"req"})
    assert worker.get_block_ids_with_load_errors() == ({1} if fail_load else set())


def test_forward_does_not_wait_for_async_loads():
    """Async loads fill blocks of requests the running batch does not contain."""
    handle = _WaitRecordingDelayedLoadHandle()
    worker = UMBPStoreConnectorWorker(handle)
    plan = BlockTransferPlan(
        "async-load",
        1,
        request_id="req",
        ranges=(KVRange("layer1", 0, 1, 1000, 16, 16, 0),),
    )
    worker.start_load_kv(
        None, UMBPConnectorMetadata(async_load=True, load_requests={"req": [plan]})
    )

    worker.wait_for_layer_load("layer1")
    worker.wait_for_layer_load("")

    assert handle.waited == []
    assert worker.get_finished(set()) == (None, None)


@pytest.mark.parametrize("fail_first", [False, True])
def test_async_stores_keep_independent_events_across_steps(fail_first):
    """Later stores of one request must not unpin its earlier in-flight KV."""
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    for event_id in (1, 2):
        plan = BlockTransferPlan(str(event_id), event_id, request_id="req")
        worker.enqueue_stores(
            UMBPConnectorMetadata(store_event=event_id, store_plans=[plan])
        )
        worker.wait_for_save()
        assert worker.get_finished({"req"}) == (None, None)
        assert worker.build_connector_worker_meta().store_events == {}
    assert not handle.waited
    assert not handle.publications

    handle.jobs[1].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {2: StoreEventResult(1)}
    assert handle.publications == [handle.jobs[1]]
    if fail_first:
        handle.jobs[0].fail(["1"], "copy failed")
    else:
        handle.jobs[0].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {1: StoreEventResult(1)}
    assert len(handle.publications) == (1 if fail_first else 2)
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {}
    assert not handle.waited


def test_preemption_cancels_pending_stores_from_previous_steps():
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    for event_id, request_id in enumerate(("first", "other", "first")):
        worker.enqueue_stores(
            UMBPConnectorMetadata(
                store_event=event_id,
                store_plans=[
                    BlockTransferPlan(str(event_id), event_id, request_id=request_id)
                ],
            )
        )
        worker.wait_for_save()
    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"first"}))
    worker.get_finished(set())
    assert handle.cancelled == [handle.jobs[0], handle.jobs[2]]
    assert worker.build_connector_worker_meta().store_events == {
        0: StoreEventResult(1),
        2: StoreEventResult(1),
    }
    assert not handle.publications
    assert not handle.waited
    handle.jobs[1].complete()
    worker.get_finished(set())
    assert worker.build_connector_worker_meta().store_events == {1: StoreEventResult(1)}


def test_async_store_drains_before_buffers_can_be_reused():
    handle = _DelayedStoreWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    worker.enqueue_stores(
        UMBPConnectorMetadata(store_event=4, store_plans=[BlockTransferPlan("kv", 1)])
    )
    worker.wait_for_save()
    assert not handle.waited
    worker.close()
    assert handle.waited == handle.jobs
    assert worker.build_connector_worker_meta().store_events == {4: StoreEventResult(1)}


def test_worker_preemption_cancels_only_matching_request():
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    first = BlockTransferPlan("first", 1, request_id="first")
    second = BlockTransferPlan("second", 2, request_id="second")
    worker.enqueue_stores(
        UMBPConnectorMetadata(
            store_event=21,
            store_plans=[first, second],
            store_requests={"first": [first], "second": [second]},
        )
    )

    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"first"}))

    assert len(handle.cancelled) == 1
    assert handle.cancelled[0].plans == (first,)
    assert worker.build_connector_worker_meta().store_events == {}
    worker.wait_for_save()
    assert handle.published.plans == (second,)
    assert worker.build_connector_worker_meta().store_events == {
        21: StoreEventResult(1)
    }


def test_async_load_cancellation_preserves_other_requests():
    handle = _CancellableWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    plans = {
        name: [
            BlockTransferPlan(
                name,
                block_id,
                request_id=name,
                ranges=(KVRange("layer1", 0, block_id, 1000, 16, 16, 0),),
            )
        ]
        for name, block_id in (("first", 1), ("second", 2))
    }
    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(
            async_load=True,
            load_requests=plans,
        ),
    )
    worker.handle_preemptions(UMBPConnectorMetadata(preempted_request_ids={"first"}))
    worker.wait_for_layer_load("")
    # Asynchronous loads settle through get_finished, not the forward pass.
    finished = worker.get_finished(set())

    assert len(handle.cancelled) == 1
    assert handle.cancelled[0].plans[0].request_id == "first"
    assert worker.get_kv_connector_stats().reduce()["load_completed"] == 1
    assert finished == (None, {"second"})
    worker.wait_for_layer_load("")
    assert worker.get_kv_connector_stats() is None


def test_worker_reports_load_completion_and_store_event_without_deferring_request():
    worker = UMBPStoreConnectorWorker(_WorkerHandle())
    metadata = UMBPConnectorMetadata(
        async_load=True,
        store_event=7,
        store_plans=[BlockTransferPlan("store", 4)],
        load_requests={"req": [BlockTransferPlan("load", 3)]},
        store_requests={"req": [BlockTransferPlan("store", 4)]},
    )

    worker.start_load_kv(None, metadata)
    worker.wait_for_layer_load("layer0")
    worker.enqueue_stores(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()

    assert result.store_events == {7: StoreEventResult(1)}
    assert worker.get_finished({"req"}) == (None, {"req"})


def test_worker_partial_store_failure_reports_unpublished_objects():
    class _PartialFailureHandle(_WorkerHandle):
        def store(self, plans):
            job = TransferJobState(tuple(plans))
            job.start()
            job.complete(["ok"])
            job.fail(["bad"], "store failed")
            return job

    worker = UMBPStoreConnectorWorker(_PartialFailureHandle())
    plans = [
        BlockTransferPlan("ok", 11),
        BlockTransferPlan("bad", 12),
    ]
    worker.enqueue_stores(UMBPConnectorMetadata(store_plans=plans, store_event=9))
    worker.wait_for_save()

    assert worker.get_finished(set()) == (None, None)
    assert worker.get_block_ids_with_load_errors() == set()
    metadata = worker.build_connector_worker_meta()
    assert metadata.store_events == {9: StoreEventResult(1)}
    assert not hasattr(worker.runtime, "published")
    assert worker.get_kv_connector_stats().reduce()["store_failed"] == 1


@pytest.mark.parametrize("forward", [False, True])
def test_connector_finalizes_each_store_once_when_forward_hook_is_skipped(
    monkeypatch, forward
):
    """No-forward steps must not strand scheduler events or resubmit stores."""
    runtime = _EmbeddedRuntime()
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders, "embedded", lambda config: runtime
    )
    connector = UMBPStoreConnector(
        _vllm_config({"mode": "embedded"}), KVConnectorRole.WORKER, _kv_cache_config()
    )
    for event_id in (7, 8):
        plan = BlockTransferPlan(
            key=f"store-{event_id}",
            block_id=1,
            ranges=(KVRange("layer1", 0, 1, 1000, 16, 16, 0),),
        )
        connector.bind_connector_metadata(
            UMBPConnectorMetadata(store_plans=[plan], store_event=event_id)
        )
        if forward:
            connector.wait_for_save()
            connector.wait_for_save()
        # Both model runners collect completions before clearing metadata.
        before_clear = connector.build_connector_worker_meta().store_events
        connector.clear_connector_metadata()
        connector.clear_connector_metadata()
        assert not connector.has_connector_metadata()

        # A no-forward store completion must survive until the next step.
        connector.bind_connector_metadata(UMBPConnectorMetadata())
        connector.wait_for_save()
        next_step = connector.build_connector_worker_meta().store_events
        connector.clear_connector_metadata()
        expected = {event_id: StoreEventResult(1)}
        assert before_clear == (expected if forward else {})
        assert next_step == ({} if forward else expected)
        assert plan.key in runtime.store

    stats = connector.get_kv_connector_stats().reduce()
    assert stats["store_submitted"] == 2
    assert stats["store_completed"] == 2


def test_worker_preserves_scheduler_supplied_ranges():
    handle = _WorkerHandle()
    worker = UMBPStoreConnectorWorker(
        handle, KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    )
    worker.register_kv_caches(make_kv_caches())
    supplied = BlockTransferPlan(
        key="custom",
        block_id=1,
        ranges=(
            KVRange(
                layer_name="custom",
                group_id=3,
                block_id=1,
                base_address=1234,
                stride=256,
                length=128,
                object_offset=0,
            ),
        ),
    )
    metadata = UMBPConnectorMetadata(
        load_requests={"req": [supplied]},
    )

    worker.start_load_kv(None, metadata)

    assert handle.loaded_plans == [supplied]


def test_embedded_runtime_register_store_and_load(monkeypatch):
    """Scheduler-built batches restore stored bytes into different blocks."""
    install_memory_runtime(monkeypatch)
    config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "load_async": False,
            "key_namespace": "builtin-embedded-test",
        }
    )
    source_caches = make_kv_caches(fill=True)
    destination_caches = make_kv_caches()
    scheduler_connector = UMBPStoreConnector(
        vllm_config, KVConnectorRole.SCHEDULER, config
    )
    store_worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
    load_worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
    try:
        store_worker.register_kv_caches(source_caches)
        load_worker.register_kv_caches(destination_caches)
        handshake(scheduler_connector, store_worker)

        producer = make_request(
            "builtin-producer",
            block_hashes=[b"builtin-a", b"builtin-b"],
            block_ids=([1, 2],),
        )
        scheduler_connector.update_state_after_alloc(
            producer,
            SimpleNamespace(get_block_ids=lambda group_ids: producer.block_ids),
            0,
        )
        store_metadata = scheduler_connector.build_connector_meta(
            make_scheduler_output(
                new_reqs=[producer], scheduled_tokens={producer.req_id: 32}
            )
        )
        store_worker.bind_connector_metadata(store_metadata)
        store_worker.wait_for_save()
        store_worker.clear_connector_metadata()

        consumer = make_request(
            "builtin-consumer",
            num_tokens=33,
            block_hashes=producer.block_hashes,
            block_ids=([5, 6],),
        )
        assert scheduler_connector.get_num_new_matched_tokens(consumer, 0) == (
            32,
            False,
        )
        scheduler_connector.update_state_after_alloc(
            consumer,
            SimpleNamespace(get_block_ids=lambda group_ids: consumer.block_ids),
            32,
        )
        load_metadata = scheduler_connector.build_connector_meta(
            make_scheduler_output(
                new_reqs=[consumer], scheduled_tokens={consumer.req_id: 1}
            )
        )
        assert set(load_metadata.load_requests) == {consumer.req_id}

        load_worker.bind_connector_metadata(load_metadata)
        load_worker.start_load_kv(None)
        load_worker.wait_for_layer_load("layer1")
        load_worker.wait_for_layer_load("layer2")
        assert load_worker.get_finished({consumer.req_id}) == (None, None)
        assert load_worker.get_block_ids_with_load_errors() == set()
        load_worker.clear_connector_metadata()

        for name, source in source_caches.items():
            destination = destination_caches[name]
            assert torch.equal(source[1], destination[5]), f"{name} block 1"
            assert torch.equal(source[2], destination[6]), f"{name} block 2"
            untouched = [block for block in range(8) if block not in (5, 6)]
            assert not destination[untouched].any(), f"{name} stray write"
    finally:
        scheduler_connector.shutdown()
        store_worker.shutdown()
        load_worker.shutdown()


@pytest.mark.parametrize(("tp_size", "dcp_size"), [(2, 1), (2, 2), (4, 2), (8, 8)])
def test_embedded_tp_dcp_rank_store_completeness(monkeypatch, tp_size, dcp_size):
    """A prefix is reusable only after every actual TP worker publishes."""
    install_memory_runtime(monkeypatch)
    config = _kv_cache_config()
    extra = {
        "mode": "embedded",
        "load_async": False,
        "key_namespace": f"builtin-tp{tp_size}-dcp{dcp_size}-test",
    }
    parallel = {
        "tensor_parallel_size": tp_size,
        "decode_context_parallel_size": dcp_size,
        "world_size": tp_size,
    }
    token_count = 32 * dcp_size
    request = make_request(
        "ranked-producer",
        num_tokens=token_count,
        block_hashes=[b"ranked-a", b"ranked-b"],
        block_ids=([1, 2],),
    )
    consumer = make_request(
        "ranked-consumer",
        num_tokens=token_count + 1,
        block_hashes=request.block_hashes,
    )
    scheduler = UMBPStoreConnector(
        _vllm_config(extra, **parallel), KVConnectorRole.SCHEDULER, config
    )
    scheduler.reset_cache()
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids),
        0,
    )
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            new_reqs=[request], scheduled_tokens={request.req_id: token_count}
        )
    )
    caches = make_kv_caches()
    try:
        for tp_rank in range(tp_size):
            assert scheduler.get_num_new_matched_tokens(consumer, 0) == (0, False)
            worker = UMBPStoreConnector(
                _vllm_config(extra, rank=tp_rank, **parallel),
                KVConnectorRole.WORKER,
                config,
            )
            try:
                worker.register_kv_caches(caches)
                worker.bind_connector_metadata(metadata)
                worker.wait_for_save()
            finally:
                worker.shutdown()
        assert scheduler.get_num_new_matched_tokens(consumer, 0) == (
            token_count,
            False,
        )
    finally:
        scheduler.shutdown()


def test_embedded_logical_hit_requires_all_tp_dcp_objects(monkeypatch):
    runtime = _EmbeddedRuntime()
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders,
        "embedded",
        lambda config: runtime,
    )
    config = _kv_cache_config()
    vllm_config = _vllm_config(
        {"mode": "embedded", "load_async": False},
        rank=0,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        world_size=2,
    )
    connector = UMBPStoreConnector(vllm_config, KVConnectorRole.SCHEDULER, config)
    handshake(
        connector, UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, config)
    )
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    request = make_request(
        "logical",
        num_tokens=65,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    for block_hash in request.block_hashes:
        runtime.store.update(
            scheduler.codec.keys_for_topology(block_hash, scheduler.topology, (0,))
        )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (64, False)
    runtime.store.remove(scheduler.codec.key_for_namespace(b"b", 0, (1, 0, 1, 0)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)


@pytest.mark.parametrize("fail_store", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 3])
def test_worker_localizes_scheduler_plan_key_to_its_tp_rank(
    fail_store, tp_rank, monkeypatch
):
    handle = _StoreRecordingWorkerHandle()
    if fail_store:
        store = handle.store

        def fail(plans):
            job = store(plans)
            job.completed_keys.clear()
            job.fail([plan.key for plan in plans], "store failed")
            return job

        monkeypatch.setattr(handle, "store", fail)
    codec = BlockIdentityCodec(UMBPNamespace("rank-local"), tp_rank=tp_rank)
    layout = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    layout.register_kv_caches(make_kv_caches())
    worker = UMBPStoreConnectorWorker(handle, layout, codec=codec)
    plan = BlockTransferPlan(
        key=BlockIdentityCodec(UMBPNamespace("rank-local")).key(b"hash", 0),
        block_id=1,
        group_id=0,
        block_hash=b"hash",
    )
    metadata = UMBPConnectorMetadata(store_plans=[plan], store_event=7)

    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert handle.store_calls[0][0].key == codec.key(b"hash", 0)
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta.store_events == {
        7: StoreEventResult(
            1,
            {(0, b"hash")} if fail_store else set(),
        )
    }
    assert hasattr(handle, "published") != fail_store


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("fast_path", [False, True])
def test_batched_load_localizes_rank_without_changing_request_or_destination(
    tp_rank, fast_path
):
    """Fast and fallback loads target the same rank and GPU blocks."""
    codec = BlockIdentityCodec(UMBPNamespace("rank-local"), tp_rank=tp_rank)
    scheduler_codec = replace(codec, tp_rank=0)
    calls = []

    def load(plans):
        calls.append(plans)
        job = TransferJobState(plans if fast_path else tuple(plans))
        job.start()
        job.complete([codec.key(b"a", 0)])
        job.fail([codec.key(b"b", 1)], "injected load failure")
        return job

    handle = SimpleNamespace(load=load)
    if fast_path:
        handle.load_blocks = load
    worker = UMBPStoreConnectorWorker(handle, codec=codec)
    batch = BlockLoadBatch(
        [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 1)],
        [7, 9],
        [0, 1],
        "request",
    )
    worker.start_load_kv(
        None,
        UMBPConnectorMetadata(async_load=True, load_requests={"request": batch}),
    )

    assert len(calls) == 1
    assert [(p.key, p.block_id, p.group_id, p.request_id) for p in calls[0]] == [
        (codec.key(b"a", 0), 7, 0, "request"),
        (codec.key(b"b", 1), 9, 1, "request"),
    ]
    job = worker._load_jobs["request"][None]
    assert job.failed_block_ids == {9}
    assert batch.keys == [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 1)]


@pytest.mark.parametrize("fail_store", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("fast_path", [False, True])
def test_batched_store_localizes_rank_without_changing_scheduler_batch(
    fail_store, tp_rank, fast_path
):
    """Fast and fallback stores target the same rank and publish alike."""
    codec = BlockIdentityCodec(UMBPNamespace("rank-local"), tp_rank=tp_rank)
    scheduler_codec = replace(codec, tp_rank=0)
    handle = _StoreRecordingWorkerHandle()
    calls = []

    def store(plans):
        calls.append(plans)
        job = TransferJobState(plans if fast_path else tuple(plans))
        job.start()
        if fail_store:
            # Only the second object fails; the whole job stays unpublished.
            job.complete([codec.key(b"a", 0)])
            job.fail([codec.key(b"b", 0)], "injected store failure")
        else:
            job.complete()
        return job

    handle.store = store
    if fast_path:
        handle.store_blocks = store
    layout = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config())
    layout.register_kv_caches(make_kv_caches())
    worker = UMBPStoreConnectorWorker(handle, layout, codec=codec)
    scheduler_keys = [scheduler_codec.key(b"a", 0), scheduler_codec.key(b"b", 0)]
    batch = BlockStoreBatch(
        keys=list(scheduler_keys),
        block_ids=[1, 2],
        group_ids=[0, 0],
        block_hashes=[b"a", b"b"],
        parent_block_hashes=[None, b"a"],
        token_ids=[tuple(range(16)), tuple(range(16, 32))],
        block_sizes=[16, 16],
        request_id="request",
    )
    metadata = UMBPConnectorMetadata(store_event=11, store_requests={"request": batch})

    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert len(calls) == 1
    if fast_path:
        assert isinstance(calls[0], BlockStoreBatch)
    else:
        assert all(plan.ranges for plan in calls[0])
    assert [
        (
            plan.key,
            plan.block_id,
            plan.group_id,
            plan.request_id,
            plan.block_hash,
            plan.parent_block_hash,
            plan.token_ids,
            plan.block_size,
        )
        for plan in calls[0]
    ] == [
        (codec.key(b"a", 0), 1, 0, "request", b"a", None, tuple(range(16)), 16),
        (
            codec.key(b"b", 0),
            2,
            0,
            "request",
            b"b",
            b"a",
            tuple(range(16, 32)),
            16,
        ),
    ]
    # The scheduler's batch is shared metadata and must not be rewritten.
    assert batch.keys == scheduler_keys
    assert metadata.store_requests == {"request": batch}
    assert worker.build_connector_worker_meta().store_events == {
        11: StoreEventResult(1, {(0, b"a"), (0, b"b")} if fail_store else set())
    }
    if fail_store:
        assert not hasattr(handle, "published")
    else:
        assert list(handle.published.keys) == [
            codec.key(b"a", 0),
            codec.key(b"b", 0),
        ]


@pytest.mark.parametrize("empty_group", [False, True])
def test_worker_merges_grouped_and_flat_store_plans_once(empty_group):
    """Request-grouped stores must not suppress independent flat stores."""
    handle = _StoreRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    plans = [] if empty_group else [BlockTransferPlan("grouped", 1)]
    flat = BlockTransferPlan("flat", 2)
    metadata = UMBPConnectorMetadata(
        store_event=20,
        store_plans=[*plans, flat, flat],
        store_requests={"req": plans},
    )
    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert handle.store_calls == ([plans] if plans else []) + [[flat]]
    assert metadata.store_requests == {"req": plans}
    assert worker.build_connector_worker_meta().store_events == {
        20: StoreEventResult(1)
    }


@pytest.mark.parametrize("grouped", [False, True])
def test_worker_submits_stores_once(grouped):
    """Grouped and flat fallback plans are each stored once."""
    handle = _StoreRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle)
    plan = BlockTransferPlan(
        key="layered-store",
        block_id=1,
        request_id="req",
        ranges=(
            KVRange("layer1", 0, 1, 1000, 16, 16, 0),
            KVRange("layer2", 0, 1, 2000, 16, 16, 16),
        ),
    )
    metadata = UMBPConnectorMetadata(
        store_event=19,
        store_plans=[plan],
        store_requests={"req": [plan]} if grouped else {},
    )

    worker.enqueue_stores(metadata)
    worker.wait_for_save()

    assert len(handle.store_calls) == 1
    assert [
        item.layer_name for call in handle.store_calls for item in call[0].ranges
    ] == ["layer1", "layer2"]
    assert metadata.store_plans == [plan]
    assert metadata.store_requests == ({"req": [plan]} if grouped else {})
    assert worker.get_finished({"req"}) == (None, None)
    completed = worker.build_connector_worker_meta()
    assert completed.store_events == {19: StoreEventResult(1)}
    assert worker.get_kv_connector_stats().reduce()["store_num_bytes"] == 32
