# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_events import AllBlocksCleared, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector as mooncake_store_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    protocol,
    scheduler,
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlockKey,
    KeyMetadata,
    LBHNCStoreLayout,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    PoolKey,
    RankLocalStoreLayout,
    StoreResidency,
    TailKeyBoundary,
    store_block_key,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from .utils import create_vllm_config


def _make_vllm_config():
    return create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
    )


@pytest.mark.parametrize(
    "extra_config",
    [
        {"store_tp_size": 4},
        {"enable_store_tp_lcm": True, "prefill_tp_sizes": [6, 4]},
        {"prefill_tp_sizes": [4, 2]},
        {"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 0]},
        None,
    ],
)
def test_store_tp_does_not_override_backend_kv_cache_layout(extra_config):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    assert (
        mooncake_store_connector.MooncakeStoreConnector.get_required_kvcache_layout(
            vllm_config
        )
        is None
    )


def _make_kv_cache_config() -> KVCacheConfig:
    """Single-group full-attention KVCacheConfig — enough for the connector
    constructor's validate() pass."""
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8192,
                layers=["layer0"],
                layer_stride=8192,
                block_stride=2048,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )


def _make_qsa_hybrid_kv_cache_config(mamba_mode: str):
    full_spec = FullAttentionSpec(
        block_size=800, num_kv_heads=8, head_size=64, dtype=None
    )
    qsa_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        head_size_v=0,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=800,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mamba_mode,
    )
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full_spec),
            KVCacheGroupSpec(["qsa"], qsa_spec),
            KVCacheGroupSpec(["mamba"], mamba_spec),
        ],
    )


def test_validation_accepts_aligned_mamba_after_block_size_rewrite():
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.block_size = 8

    mooncake_store_connector.MooncakeStoreConnector._validate_kv_cache_config(
        vllm_config, _make_qsa_hybrid_kv_cache_config("align")
    )


def test_validation_rejects_mamba_mode_directly():
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.block_size = 800

    with pytest.raises(ValueError, match="mamba_cache_mode=\x27none\x27"):
        mooncake_store_connector.MooncakeStoreConnector._validate_kv_cache_config(
            vllm_config, _make_qsa_hybrid_kv_cache_config("none")
        )


def test_scheduler_requires_align_mode_for_mamba():
    vllm_config = _make_vllm_config()
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], mamba_spec)],
    )

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ),
        pytest.raises(AssertionError, match="requires mamba_cache_mode='align'"),
    ):
        scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)


def _make_block_stored(
    block_hash: bytes = b"hash", group_idx: int | None = None
) -> BlockStored:
    return BlockStored(
        block_hashes=[block_hash],
        parent_block_hash=None,
        token_ids=[1, 2, 3],
        block_size=16,
        lora_id=None,
        medium="cpu",
        lora_name=None,
        group_idx=group_idx,
    )


def test_scheduler_role_initializes_store_scheduler_only():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler.assert_called_once_with(vllm_config, kv_cache_config)
    mock_worker.assert_not_called()
    assert connector.connector_scheduler is mock_scheduler.return_value
    assert connector.connector_worker is None
    block_pool = MagicMock()

    connector.bind_gpu_block_pool(block_pool)
    mock_scheduler.return_value.bind_gpu_block_pool.assert_called_once_with(block_pool)


def test_scheduler_reports_mooncake_cache_source():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    # Mooncake Store is an external cache service. Its internal memory and
    # local-storage policy is intentionally hidden behind that stable label.
    assert connector.get_external_cache_hit_sources(
        None,  # type: ignore[arg-type]
        32,
    ) == {CacheHitSource.EXTERNAL_UNSPECIFIED: 32}


def test_worker_methods_delegate_to_store_worker():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    kv_caches = {"layer0": MagicMock()}
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    finished_req_ids = {"req-1"}

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    worker = mock_worker_cls.return_value
    worker.get_finished.return_value = ({"req-1"}, {"req-2"})
    worker.get_block_ids_with_load_errors.return_value = {3, 4}
    connector.bind_connector_metadata(metadata)

    connector.register_kv_caches(kv_caches)
    connector.start_load_kv(MagicMock())
    connector.wait_for_save()
    result = connector.get_finished(finished_req_ids)
    invalid_block_ids = connector.get_block_ids_with_load_errors()

    worker.register_kv_caches.assert_called_once_with(kv_caches)
    worker.start_load_kv.assert_called_once_with(metadata)
    worker.wait_for_save.assert_called_once_with(metadata)
    worker.get_finished.assert_called_once_with(finished_req_ids, metadata)
    assert result == ({"req-1"}, {"req-2"})
    worker.get_block_ids_with_load_errors.assert_called_once_with()
    assert invalid_block_ids == {3, 4}


def test_get_transfer_results_delegates_to_store_worker():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    finished_req_ids = {"req-1"}

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    worker = mock_worker_cls.return_value
    transfer_results = KVConnectorTransferResults(
        finished_sending={"req-1"},
        finished_recving={"req-2"},
        failed_recving={"req-3"},
    )
    worker.get_transfer_results.return_value = transfer_results
    connector.bind_connector_metadata(metadata)

    result = connector.get_transfer_results(finished_req_ids)

    worker.get_transfer_results.assert_called_once_with(finished_req_ids, metadata)
    assert result is transfer_results


def test_get_kv_connector_kv_cache_events_returns_none_when_disabled():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.enable_kv_events = False
    assert connector.get_kv_connector_kv_cache_events() is None


def test_get_kv_connector_stats_delegates_to_worker():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    expected_stats = MooncakeStoreConnectorStats()
    expected_stats.record_operation("save_put", 0.01, 2, num_bytes=1024)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.get_kv_connector_stats.return_value = expected_stats
    stats = connector.get_kv_connector_stats()

    assert stats is expected_stats
    mock_worker_cls.return_value.get_kv_connector_stats.assert_called_once_with()


def test_build_kv_connector_stats_reconstructs_mooncake_stats():
    stats = mooncake_store_connector.MooncakeStoreConnector.build_kv_connector_stats(
        {
            "save_put": [
                {
                    "duration_seconds": 0.02,
                    "num_keys": 4,
                    "num_bytes": 2048,
                    "status": "ok",
                    "num_failed_keys": 0,
                }
            ]
        }
    )

    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data["save_put"][0]["num_bytes"] == 2048


def _make_scheduler_connector() -> mooncake_store_connector.MooncakeStoreConnector:
    vllm_config = _make_vllm_config()
    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        return mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, _make_kv_cache_config()
        )


def _namespace(
    group_idx: int = 0,
    *,
    tp_rank: int = 0,
    pcp_rank: int = 0,
    dcp_rank: int = 0,
    pp_rank: int = 0,
) -> str:
    """A Store key namespace, spelled the way the lookup table spells it."""
    return PoolKey.build_prefix(
        KeyMetadata(
            "test-model", tp_rank, pcp_rank, dcp_rank, pp_rank, group_id=group_idx
        )
    )


def _rank_local_required(
    group_idx: int, *rank_namespaces: tuple[int, int, int, int]
) -> tuple[str, ...]:
    """The namespaces a lookup probes for a rank-local group."""
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_idx), 16, 16
    )
    return layout.lookup_key_prefixes(rank_namespaces)


def _residency(
    events: list[BlockStored],
    covered: dict[BlockKey, list[str] | tuple[str, ...] | frozenset[str]],
) -> StoreResidency:
    """The blocks one rank reports for one poll."""
    return StoreResidency(
        events=list(events),
        covered={key: frozenset(names) for key, names in covered.items()},
    )


def _worker_container(
    residency: StoreResidency,
    required: dict[int, list[str] | tuple[str, ...] | frozenset[str]],
) -> mooncake_store_connector.MooncakeStoreKVEvents:
    """What a worker's connector polls for one engine step."""
    container = mooncake_store_connector.MooncakeStoreKVEvents(num_workers=1)
    container.add_residency(
        residency, {group: frozenset(names) for group, names in required.items()}
    )
    return container


def _clear_container() -> mooncake_store_connector.MooncakeStoreKVEvents:
    container = mooncake_store_connector.MooncakeStoreKVEvents(num_workers=1)
    container.add_events([AllBlocksCleared()])
    return container


def _step_output(
    container: mooncake_store_connector.MooncakeStoreKVEvents | None,
) -> ModelRunnerOutput:
    """One worker's step output; None is a worker that reported no output."""
    if container is None:
        return ModelRunnerOutput(
            req_ids=[], req_id_to_index={}, kv_connector_output=KVConnectorOutput()
        )
    return ModelRunnerOutput.with_kv_conn_output_only(
        KVConnectorOutput(kv_cache_events=container)
    )


def _run_engine_step(
    connector: mooncake_store_connector.MooncakeStoreConnector,
    *containers: mooncake_store_connector.MooncakeStoreKVEvents | None,
) -> list[KVCacheEvent]:
    """Push one engine step through the real aggregation hops.

    Every worker polls a container of its own, ``KVOutputAggregator`` merges
    them into the step's accumulator, and the scheduler connector folds that
    into the accumulator it keeps for the connector's lifetime before draining
    whatever is ready.
    """
    outputs = [_step_output(container) for container in containers]
    aggregator = KVOutputAggregator(expected_finished_count=len(outputs))
    aggregated = aggregator.aggregate(outputs)
    assert aggregated is not None
    assert aggregated.kv_connector_output is not None
    connector.update_connector_output(aggregated.kv_connector_output)
    return list(connector.take_events())


def _contribute(
    connector: mooncake_store_connector.MooncakeStoreConnector,
    event: BlockStored,
    namespaces: list[str] | tuple[str, ...],
    required: dict[int, list[str] | tuple[str, ...]],
) -> list[KVCacheEvent]:
    """Run one step in which a single rank covers ``namespaces`` of ``event``."""
    return _run_engine_step(
        connector,
        _worker_container(
            _residency([event], {store_block_key(event): namespaces}),
            required,
        ),
    )


def test_get_kv_connector_kv_cache_events_wraps_worker_events():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    event = _make_block_stored(group_idx=0)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.enable_kv_events = True
    mock_worker_cls.return_value.kv_send_thread = MagicMock()
    mock_worker_cls.return_value.lookup_key_prefixes = ((_namespace(),),)
    mock_worker_cls.return_value.drain_residency.return_value = _residency(
        [event],
        {store_block_key(event): [_namespace()]},
    )
    kv_events = connector.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, mooncake_store_connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == [event]


def test_get_kv_connector_kv_cache_events_checks_coverage_against_the_lookup_table():
    """A rank's reported coverage is checked against the namespaces a lookup
    probes for its group, which the worker's own table supplies."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    event = _make_block_stored(group_idx=0)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.enable_kv_events = True
    mock_worker_cls.return_value.kv_send_thread = MagicMock()
    mock_worker_cls.return_value.lookup_key_prefixes = ((_namespace(),),)
    mock_worker_cls.return_value.drain_residency.return_value = _residency(
        [event],
        {store_block_key(event): [_namespace(tp_rank=1)]},
    )

    with pytest.raises(ValueError, match="never probes"):
        connector.get_kv_connector_kv_cache_events()


def test_get_kv_connector_kv_cache_events_keeps_empty_worker_contribution():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    worker = mock_worker_cls.return_value
    worker.enable_kv_events = True
    worker.kv_send_thread = MagicMock()
    worker.lookup_key_prefixes = ((_namespace(),),)
    worker.drain_residency.return_value = _residency([], {})

    kv_events = connector.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, mooncake_store_connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == []


def test_block_waits_for_every_lookup_namespace_across_engine_steps():
    """Two ranks completing the same block in different steps announce it once,
    when the last of the group's lookup namespaces reports."""
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    first, second = _namespace(0, tp_rank=0), _namespace(0, tp_rank=1)
    required = {0: [first, second]}

    assert _contribute(connector, event, [first], required) == []
    assert _contribute(connector, event, [second], required) == [event]
    # A further report of the same block adds nothing once it is announced.
    assert _contribute(connector, event, [first], required) == []
    assert _contribute(connector, event, list(required[0]), required) == []


def test_repeated_namespace_report_does_not_cover_another_namespace():
    """One namespace reporting twice is still one namespace."""
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    first, second = _namespace(0, tp_rank=0), _namespace(0, tp_rank=1)
    required = {0: [first, second]}

    for _ in range(3):
        assert _contribute(connector, event, [first], required) == []

    assert _contribute(connector, event, [second], required) == [event]


def test_empty_poll_neither_completes_nor_resets_pending_coverage():
    """A worker with nothing to report, or no connector output at all, leaves
    the coverage another rank already reported untouched."""
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    first, second = _namespace(0, tp_rank=0), _namespace(0, tp_rank=1)
    required = {0: [first, second]}

    assert _contribute(connector, event, [first], required) == []

    assert _run_engine_step(connector, None) == []
    assert (
        _run_engine_step(connector, _worker_container(_residency([], {}), required))
        == []
    )
    assert _run_engine_step(connector, _worker_container(_residency([], {}), {})) == []

    assert _contribute(connector, event, [second], required) == [event]


@pytest.mark.parametrize(
    "rank_namespaces",
    [
        # Replicated GQA: four ranks in two byte-identical replicas each, so a
        # lookup probes one namespace per shard rank.
        pytest.param(tuple((shard, 0, 0, 0) for shard in range(2)), id="replicated"),
        # Rank-specific bytes: a lookup needs every rank's namespace.
        pytest.param(tuple((rank, 0, 0, 0) for rank in range(4)), id="rank-specific"),
        # PCP splits a sequence across ranks, each with its own namespace.
        pytest.param(tuple((0, pcp, 0, 0) for pcp in range(2)), id="pcp"),
        # PP: a lookup needs every pipeline stage's objects.
        pytest.param(tuple((0, 0, 0, pp) for pp in range(3)), id="pp"),
    ],
)
def test_block_needs_every_namespace_its_group_lookup_probes(rank_namespaces):
    """A block is announced only once every namespace a lookup probes for its
    group has reported, and draining between those reports releases nothing."""
    required = {0: list(_rank_local_required(0, *rank_namespaces))}
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)

    for namespace in required[0][:-1]:
        assert _contribute(connector, event, [namespace], required) == []

    assert _contribute(connector, event, required[0][-1:], required) == [event]
    # Draining again without a contribution yields nothing.
    assert list(connector.take_events()) == []


def test_groups_require_their_own_namespaces():
    """Per-group coverage follows each group's lookup prefixes: a replicated
    attention group completes on fewer reports than a rank-specific one."""
    connector = _make_scheduler_connector()
    replicated = _make_block_stored(b"replicated", group_idx=0)
    rank_specific = _make_block_stored(b"rank-specific", group_idx=1)
    required = {
        0: list(_rank_local_required(0, *((shard, 0, 0, 0) for shard in range(2)))),
        1: list(_rank_local_required(1, *((rank, 0, 0, 0) for rank in range(4)))),
    }

    # Rank 0 and rank 1 cover shard 0 and shard 1: enough for group 0 only.
    events = _run_engine_step(
        connector,
        _worker_container(
            _residency(
                [replicated, rank_specific],
                {
                    store_block_key(replicated): [required[0][0]],
                    store_block_key(rank_specific): [required[1][0]],
                },
            ),
            required,
        ),
        _worker_container(
            _residency(
                [replicated, rank_specific],
                {
                    store_block_key(replicated): [required[0][1]],
                    store_block_key(rank_specific): [required[1][1]],
                },
            ),
            required,
        ),
    )

    assert events == [replicated]

    # The two remaining ranks complete the rank-specific group.
    events = _run_engine_step(
        connector,
        _worker_container(
            _residency(
                [rank_specific],
                {store_block_key(rank_specific): required[1][2:]},
            ),
            required,
        ),
    )

    assert events == [rank_specific]


def test_store_tp_shards_need_every_store_namespace():
    """With a shared Store-TP layout each rank covers the shards it holds, and
    the block is announced once every store shard namespace has reported."""
    metadata = KeyMetadata("test-model", 0, 0, 0, 0, store_namespace="@store_tp:4")
    layouts = {
        tp_rank: LBHNCStoreLayout(
            metadata,
            16,
            16,
            local_tp_size=2,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        for tp_rank in range(2)
    }
    rank_namespaces = [(0, 0, 0, 0), (1, 0, 0, 0)]
    required = {0: list(layouts[0].lookup_key_prefixes(rank_namespaces))}
    covered = {
        tp_rank: frozenset(
            PoolKey.build_prefix(metadata, tp_rank=shard_id)
            for shard_id in layout.local_shard_ids
        )
        for tp_rank, layout in layouts.items()
    }

    assert len(required[0]) == 4
    assert covered[0] | covered[1] == set(required[0])

    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)

    assert _contribute(connector, event, list(covered[0]), required) == []
    assert _contribute(connector, event, list(covered[1]), required) == [event]


def test_residency_outside_the_lookup_namespaces_is_rejected():
    container = mooncake_store_connector.MooncakeStoreKVEvents(num_workers=1)
    event = _make_block_stored(b"block", group_idx=0)

    with pytest.raises(ValueError, match="never probes"):
        container.add_residency(
            _residency([event], {store_block_key(event): [_namespace(0, tp_rank=7)]}),
            {0: frozenset({_namespace(0, tp_rank=0)})},
        )


def test_ranks_must_agree_on_a_groups_lookup_namespaces():
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    namespaces = [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]

    _contribute(connector, event, namespaces[:1], {0: namespaces})

    with pytest.raises(ValueError, match="lookup namespaces differ between ranks"):
        _contribute(connector, event, namespaces[:1], {0: namespaces[:1]})


def test_one_block_is_announced_once_with_the_first_payload():
    """A rank's retry covers the token range of an earlier failed attempt, so
    the two reports of a block can carry different payloads; the block is still
    one block, announced once, with the payload seen first.
    """
    connector = _make_scheduler_connector()
    first = _make_block_stored(b"block", group_idx=0)
    retried = BlockStored(
        block_hashes=[b"block"],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4, 5, 6],
        block_size=16,
        lora_id=None,
        medium="cpu",
        lora_name=None,
        group_idx=0,
    )
    required = {0: [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]}

    assert _contribute(connector, first, required[0][:1], required) == []
    assert _contribute(connector, retried, required[0][1:], required) == [first]


def test_completed_block_leaves_no_pending_state_behind():
    """A step container that completes a block's coverage arrives with the event
    already released; folding it in must also drop the partial coverage an
    earlier step left, so no stale payload stays behind.
    """
    accumulator = mooncake_store_connector.MooncakeStoreKVEvents()
    event = _make_block_stored(b"block", group_idx=0)
    required = [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]

    accumulator.add_residency(
        _residency([event], {store_block_key(event): required[:1]}),
        {0: frozenset(required)},
    )
    assert accumulator.pop_ready_events() == []

    completed = _worker_container(
        _residency([event], {store_block_key(event): required}), {0: required}
    )
    assert completed.get_all_events() == [event]

    accumulator.merge(completed)

    assert accumulator.pop_ready_events() == [event]
    # Neither the partial payload nor a released copy is held any more, and a
    # further report of the announced block is not counted again.
    assert accumulator.get_all_events() == []
    accumulator.merge(
        _worker_container(
            _residency([event], {store_block_key(event): required}), {0: required}
        )
    )
    assert accumulator.pop_ready_events() == []


def test_worker_container_survives_the_engine_transport_pickle():
    """Worker containers are pickled into the engine's step output, so their
    state must survive the trip and stay mergeable on the engine side.
    """
    event = _make_block_stored(b"block", group_idx=0)
    required = [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]

    container = _worker_container(
        _residency([event], {store_block_key(event): required[:1]}), {0: required}
    )
    restored = pickle.loads(pickle.dumps(container))

    assert restored.get_all_events() == [event]
    accumulator = mooncake_store_connector.MooncakeStoreKVEvents()
    accumulator.merge(restored)
    accumulator.merge(
        _worker_container(
            _residency([event], {store_block_key(event): required[1:]}), {0: required}
        )
    )
    assert accumulator.pop_ready_events() == [event]


def test_clear_event_retires_pending_coverage():
    """Coverage reported before the Store was wiped cannot complete a block."""
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    required = {0: [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]}

    assert _contribute(connector, event, required[0][:1], required) == []
    assert _run_engine_step(connector, _clear_container()) == [AllBlocksCleared()]

    # The first namespace was reported before the clear, so it is gone: the
    # second one alone does not complete the block.
    assert _contribute(connector, event, required[0][1:], required) == []
    # Both namespaces reporting after the clear do announce it.
    assert _contribute(connector, event, required[0][:1], required) == [event]


def test_clear_event_lets_an_announced_block_be_announced_again():
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    required = {0: [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]}

    assert _contribute(connector, event, required[0], required) == [event]
    assert _contribute(connector, event, required[0], required) == []

    assert _run_engine_step(connector, _clear_container()) == [AllBlocksCleared()]

    # A block stored again after the reset is announced again.
    assert _contribute(connector, event, required[0], required) == [event]


def test_block_released_before_a_clear_is_published_before_it():
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    required = {0: [_namespace(0, tp_rank=0), _namespace(0, tp_rank=1)]}

    assert _run_engine_step(
        connector,
        _worker_container(
            _residency([event], {store_block_key(event): required[0]}), required
        ),
        _clear_container(),
    ) == [event, AllBlocksCleared()]


def test_update_connector_output_and_take_events():
    connector = _make_scheduler_connector()
    event = _make_block_stored(b"block", group_idx=0)
    required = {0: [_namespace(0, tp_rank=0)]}

    container = _worker_container(
        _residency([event], {store_block_key(event): required[0]}), required
    )
    output = KVConnectorOutput(kv_cache_events=container)
    connector.update_connector_output(output)

    connector.connector_scheduler.update_connector_output.assert_called_once_with(
        output
    )
    assert list(connector.take_events()) == [event]
    # The accumulator is kept for the connector's lifetime so a block whose
    # coverage completes in a later step can still be announced.
    assert isinstance(
        connector._kv_cache_events, mooncake_store_connector.MooncakeStoreKVEvents
    )
    assert connector._kv_cache_events.get_all_events() == []


# ============================================================
# reset_cache() — RL hard-reset path via typed LookupKey protocol
# ============================================================


def test_reset_cache_scheduler_role_delegates_to_reset_store():
    """SCHEDULER role reset_cache() routes to scheduler.reset_store()."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler_cls.return_value.reset_store.return_value = True
    assert conn.reset_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()


def test_reset_cache_scheduler_role_propagates_failure():
    """SCHEDULER role surfaces False when scheduler.reset_store() fails."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert conn.reset_cache() is False


def test_reset_cache_worker_role_returns_none():
    """WORKER role reset_cache() is a no-op; reset is driven via ZMQ admin."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ),
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    assert conn.reset_cache() is None


def test_scheduler_reset_store_returns_client_reset_result():
    """MooncakeStoreScheduler.reset_store() returns LookupKeyClient.reset()."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)

    mock_client_cls.return_value.reset.return_value = True
    assert sched.reset_store() is True
    mock_client_cls.return_value.reset.assert_called_once_with()


def test_scheduler_reset_store_handles_rpc_exception():
    """Exceptions from the ZMQ reset RPC convert to False, not raise."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)

    mock_client_cls.return_value.reset.side_effect = RuntimeError("rpc timed out")
    assert sched.reset_store() is False


def test_lookup_key_client_lookup_prepends_typed_tag():
    """LookupKeyClient.lookup() puts LOOKUP_MSG tag at frame 0."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    fake_socket.recv.return_value = (5).to_bytes(4, "big")

    # Blocking lookup (non_block defaults to False) runs on the executor and
    # returns the resolved hit length.
    result = client.lookup("req0", num_tokens=128, block_hashes=[])
    assert result is not None
    assert result.hit_length == 5

    sent_frames = fake_socket.send_multipart.call_args[0][0]
    assert sent_frames[0] == protocol.LOOKUP_MSG
    assert int.from_bytes(sent_frames[1], "big") == 128


def test_lookup_key_client_reset_uses_typed_protocol():
    """LookupKeyClient.reset() sends RESET_MSG and parses RESP_OK / RESP_ERR."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value

    # ACK path: server returns RESP_OK -> client returns True.
    fake_socket.recv.return_value = protocol.RESP_OK
    assert client.reset() is True
    assert fake_socket.send.call_args[0][0] == protocol.RESET_MSG

    # NACK path: server returns RESP_ERR -> client returns False.
    fake_socket.recv.return_value = protocol.RESP_ERR
    assert client.reset() is False


def _poll_lookup(client, req_id, num_tokens=128, block_hashes=(), timeout=5.0):
    """Drive non-blocking lookup until the executor completes it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.lookup(req_id, num_tokens, list(block_hashes), non_block=True)
        if result is not None:
            return result.hit_length
        time.sleep(0.005)
    return None


def _gated_recv(gate: threading.Event, value: int):
    """Mock recv side-effect that blocks until ``gate`` is set, so the
    executor's lookup can be held pending deterministically."""

    def recv():
        gate.wait()
        return value.to_bytes(4, "big")

    return recv


def test_lookup_key_client_non_block_lookup_async():
    """Non-blocking lookup defers to the executor: None first, hit once the
    Future resolves."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    # Hold the executor's lookup pending until we release the gate.
    gate = threading.Event()
    fake_socket.recv.side_effect = _gated_recv(gate, 7)

    # First query submits the lookup and returns None while it is in flight.
    assert client.lookup("req1", 128, [], non_block=True) is None
    # Release the executor; a later poll returns the hit length.
    gate.set()
    assert _poll_lookup(client, "req1") == 7
    # Future is consumed (popped) on read.
    assert "req1" not in client.futures


def test_lookup_key_client_discard_clears_state():
    """discard() drops a completed lookup Future so it is not served stale."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    gate = threading.Event()
    fake_socket.recv.side_effect = _gated_recv(gate, 9)

    # Submit while gated so the call returns None and the Future stays in
    # `futures` (unconsumed) once it resolves.
    assert client.lookup("req2", 128, [], non_block=True) is None
    gate.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.futures["req2"].done():
            break
        time.sleep(0.005)
    # discard() drops the completed result before any lookup consumes it.
    client.discard("req2")
    assert "req2" not in client.futures
    # A fresh query re-submits rather than returning a stale value: hold the
    # gate so the resubmitted lookup stays in flight.
    gate.clear()
    assert client.lookup("req2", 128, [], non_block=True) is None
    gate.set()  # release the executor so the worker thread can drain


def test_get_num_new_matched_tokens_async_defers_then_reports():
    """Async lookup returns (None, False) until ready, then the hit count."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lookup_async": True},
    )
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)

    assert sched.lookup_async is True
    mock_client = mock_client_cls.return_value

    block_size = sched._block_size
    request = MagicMock()
    request.request_id = "r1"
    request.num_tokens = 4 * block_size
    request.block_hashes = []

    # Lookup not ready -> defer.
    mock_client.lookup.return_value = None
    assert sched.get_num_new_matched_tokens(request, 0) == (None, False)
    assert "r1" not in sched.load_specs

    # Lookup ready with a hit -> report need_to_allocate + async-load flag.
    hit = 3 * block_size
    boundary = TailKeyBoundary(group_id=0, num_tokens=4 * block_size)
    mock_client.lookup.return_value = MooncakeLookupResult(hit, (boundary,))
    need, load_async = sched.get_num_new_matched_tokens(request, 0)
    assert need == hit
    assert load_async == sched.load_async
    assert sched.load_specs["r1"].kvpool_cached_tokens == hit
    assert sched.load_specs["r1"].tail_key_boundaries == (boundary,)


def test_protocol_tags_are_distinct_and_non_empty():
    """Protocol tags must be unique and non-empty to avoid collision."""
    tags = {protocol.LOOKUP_MSG, protocol.RESET_MSG}
    assert len(tags) == 2
    for tag in tags:
        assert isinstance(tag, bytes)
        assert len(tag) > 0
    assert protocol.RESP_OK != protocol.RESP_ERR


def test_lookup_response_round_trip_preserves_tail_keys():
    result = MooncakeLookupResult(
        hit_length=20,
        tail_key_boundaries=(
            TailKeyBoundary(group_id=0, num_tokens=24),
            TailKeyBoundary(group_id=1, num_tokens=20),
        ),
    )

    assert protocol.decode_lookup_response(protocol.encode_lookup_response(result)) == (
        result
    )


def test_scheduler_reset_connector_cache_invokes_connector_reset():
    """Cascade test: Scheduler.reset_prefix_cache(reset_connector=True)
    cascades into MooncakeStoreConnector.reset_cache without dragging in
    the heavy KVCacheManager fixtures.
    """
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler_cls.return_value.reset_store.return_value = True

    class _StubScheduler:
        def __init__(self, c):
            self.connector = c

        def reset_connector_cache(self):
            return self.connector.reset_cache() is not False

    sched = _StubScheduler(conn)
    assert sched.reset_connector_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()

    mock_scheduler_cls.return_value.reset_store.reset_mock()
    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert sched.reset_connector_cache() is False


def test_reset_cache_scheduler_role_clears_local_state():
    """SCHEDULER reset_cache() must clear scheduler-side state that points
    at master keys we're about to wipe -- pending load_specs and the
    accumulated Store residency both reference keys whose blobs are
    about to be remove_all'd, so reading them after reset would surface
    stale references to wiped keys.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
        LoadSpec,
    )

    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    # Seed both sentinel pieces of stale-reference state.
    sched_inst = mock_scheduler_cls.return_value
    sched_inst.load_specs = {
        "req-A": LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=True)
    }
    conn._kv_cache_events = mooncake_store_connector.MooncakeStoreKVEvents(
        num_workers=1
    )
    sched_inst.reset_store.return_value = True

    assert conn.reset_cache() is True

    # Both stale references must be cleared (load_specs flushed dict, residency
    # accumulator dropped) so nothing learned before the wipe survives it.
    assert sched_inst.load_specs == {}
    assert conn._kv_cache_events is None


def test_successful_store_reset_starts_a_new_residency_epoch():
    """A block stored again after a successful reset is announced again."""
    conn = _make_scheduler_connector()
    conn.connector_scheduler.reset_store.return_value = True
    event = _make_block_stored(b"block", group_idx=0)
    required = {0: [_namespace(0, tp_rank=0)]}

    assert _contribute(conn, event, required[0], required) == [event]
    assert _contribute(conn, event, required[0], required) == []

    assert conn.reset_cache() is True

    assert _contribute(conn, event, required[0], required) == [event]


def test_failed_store_reset_keeps_what_it_announced():
    """A reset that failed leaves the Store holding what it announced, so the
    residency ledger must survive it and no block may be announced twice."""
    conn = _make_scheduler_connector()
    conn.connector_scheduler.reset_store.return_value = False
    event = _make_block_stored(b"block", group_idx=0)
    required = {0: [_namespace(0, tp_rank=0)]}

    assert _contribute(conn, event, required[0], required) == [event]

    assert conn.reset_cache() is False

    assert _contribute(conn, event, required[0], required) == []


def _make_lookup_key_server_for_reset(
    send_thread: MagicMock | None,
) -> tuple[worker.LookupKeyServer, list[str]]:
    """A LookupKeyServer with mocked store objects, ready for _reset_store().

    Bypasses __init__ so the test drives the handler the RESET message reaches
    instead of binding a real ZMQ REP socket.
    """
    call_order: list[str] = []

    store = MagicMock()
    store.remove_all.side_effect = lambda force: call_order.append(
        f"remove_all(force={force})"
    )

    if send_thread is not None:
        send_thread.request_queue.join.side_effect = lambda: call_order.append("join")

    store_worker = MagicMock()
    store_worker.kv_send_thread = send_thread
    store_worker.store = store
    store_worker.retire_residency.side_effect = lambda: call_order.append(
        "retire_residency"
    )

    server = object.__new__(worker.LookupKeyServer)
    server.store_worker = store_worker
    return server, call_order


def test_reset_drains_puts_then_wipes_then_retires_reports():
    """RESET must drain in-flight puts before remove_all, and drop the reports
    buffered for the objects it wipes before this rank's next poll reports
    them as residency of the fresh Store.
    """
    server, call_order = _make_lookup_key_server_for_reset(MagicMock())

    assert server._reset_store() == protocol.RESP_OK
    assert call_order == ["join", "remove_all(force=True)", "retire_residency"]


def test_reset_skips_drain_when_no_send_thread():
    """A worker with no send thread (e.g. consumer-only) still wipes the store."""
    server, call_order = _make_lookup_key_server_for_reset(None)

    assert server._reset_store() == protocol.RESP_OK
    assert call_order == ["remove_all(force=True)", "retire_residency"]


def test_failed_reset_keeps_buffered_reports():
    """A wipe that failed leaves the Store holding what it held, so the reports
    buffered for those objects stay valid and must not be dropped.
    """
    server, call_order = _make_lookup_key_server_for_reset(MagicMock())
    server.store_worker.store.remove_all.side_effect = RuntimeError("master down")

    assert server._reset_store() == protocol.RESP_ERR
    assert call_order == ["join"]


def test_shutdown_closes_worker_store():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    worker = mock_worker_cls.return_value
    connector.shutdown()

    worker.close.assert_called_once_with()


def test_del_invokes_shutdown_and_closes_store():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    worker = mock_worker_cls.return_value
    # __del__ is the GC backstop; it must route through shutdown() -> close().
    connector.__del__()

    worker.close.assert_called_once_with()


def test_shutdown_scheduler_role_is_noop():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    # Scheduler role holds no store handle, so shutdown must be a safe no-op.
    assert connector.connector_worker is None
    connector.shutdown()
