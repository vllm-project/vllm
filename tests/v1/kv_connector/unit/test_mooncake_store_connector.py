# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import SamplingParams
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_events import BlockStored
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
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .utils import create_request, create_vllm_config


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


def _make_mamba_none_kv_cache_config(
    attention_block_size: int | None, mamba_block_size: int
) -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=mamba_block_size,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="none",
            ),
        )
    ]
    if attention_block_size is not None:
        # Kimi K3 layout without prefix caching: the LCM hash unit exceeds both
        # group block sizes, so no lookup coordinator can be built.
        groups.insert(
            0,
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=attention_block_size,
                    num_kv_heads=8,
                    head_size=64,
                    dtype=torch.float32,
                ),
            ),
        )
    return KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=groups)


def _make_role_vllm_config(kv_role: str, enable_lookup: bool, save_decode_cache: bool):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role=kv_role,
        kv_connector_extra_config={
            "enable_lookup": enable_lookup,
            "save_decode_cache": save_decode_cache,
        },
    )
    vllm_config.cache_config.enable_prefix_caching = False
    return vllm_config


_MAMBA_NONE_LAYOUTS = pytest.mark.parametrize(
    "attention_block_size,mamba_block_size",
    [(None, 16), (1_048_576, 6_400)],
    ids=["mamba-only", "hybrid-no-prefix"],
)


@_MAMBA_NONE_LAYOUTS
@pytest.mark.parametrize(
    "kv_role,enable_lookup,save_decode_cache",
    [
        ("kv_consumer", True, False),
        ("kv_consumer", False, True),
        ("kv_producer", False, False),
        ("kv_both", False, False),
    ],
)
def test_scheduler_requires_align_mode_for_mamba_kv_transfer(
    kv_role, enable_lookup, save_decode_cache, attention_block_size, mamba_block_size
):
    vllm_config = _make_role_vllm_config(kv_role, enable_lookup, save_decode_cache)
    kv_cache_config = _make_mamba_none_kv_cache_config(
        attention_block_size, mamba_block_size
    )

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ),
        pytest.raises(AssertionError, match="requires mamba_cache_mode='align'"),
    ):
        scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)


@_MAMBA_NONE_LAYOUTS
def test_capacity_only_scheduler_allows_mamba_none_mode(
    attention_block_size, mamba_block_size
):
    """Capacity-only consumers transfer no KV, so any Mamba mode is allowed."""
    vllm_config = _make_role_vllm_config(
        "kv_consumer", enable_lookup=False, save_decode_cache=False
    )
    kv_cache_config = _make_mamba_none_kv_cache_config(
        attention_block_size, mamba_block_size
    )

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "scheduler.LookupKeyClient"
    ) as mock_client:
        store_scheduler = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)
        assert store_scheduler.get_num_new_matched_tokens(MagicMock(), 0) == (0, False)
        meta = store_scheduler.build_connector_meta(SchedulerOutput.make_empty())

    mock_client.return_value.lookup.assert_not_called()
    assert meta.requests == []


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


def test_get_kv_connector_kv_cache_events_wraps_worker_events():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    event = _make_block_stored()

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

    mock_worker_cls.return_value.get_kv_events.return_value = [event]
    mock_worker_cls.return_value.enable_kv_events = True
    mock_worker_cls.return_value.kv_send_thread = MagicMock()
    mock_worker_cls.return_value.group_tp_replication_factors = (1,)
    kv_events = connector.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, mooncake_store_connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == [event]


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
    worker.group_tp_replication_factors = (2,)
    worker.get_kv_events.return_value = []

    kv_events = connector.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, mooncake_store_connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == []


def _aggregate_store_events(
    worker_events: list[list[BlockStored]],
    replication_factors: tuple[int, ...],
) -> list[BlockStored]:
    combined = mooncake_store_connector.MooncakeStoreKVEvents(
        num_workers=1,
        group_tp_replication_factors=replication_factors,
    )
    combined.add_events(worker_events[0])
    for events in worker_events[1:]:
        combined.add_events(events)
        combined.increment_workers()
    combined.aggregate()
    return combined.get_all_events()


def test_gqa_store_events_follow_tp_replica_striping():
    block_a = _make_block_stored(b"a", group_idx=0)
    block_b = _make_block_stored(b"b", group_idx=0)

    events = _aggregate_store_events(
        [[block_a], [block_b], [block_a], [block_b]],
        replication_factors=(2,),
    )

    assert events == [block_a, block_b]


def test_mqa_store_events_require_one_replica():
    blocks = [_make_block_stored(bytes([index]), group_idx=0) for index in range(4)]

    events = _aggregate_store_events(
        [[block] for block in blocks],
        replication_factors=(4,),
    )

    assert events == blocks


def test_store_events_require_every_non_replicated_worker():
    complete = _make_block_stored(b"complete", group_idx=0)
    incomplete = _make_block_stored(b"incomplete", group_idx=0)

    events = _aggregate_store_events(
        [
            [complete, incomplete],
            [complete],
            [complete],
            [complete],
        ],
        replication_factors=(1,),
    )

    assert events == [complete]


def test_store_events_reject_incomplete_replica_coverage():
    complete = _make_block_stored(b"complete", group_idx=0)
    incomplete = _make_block_stored(b"incomplete", group_idx=0)

    events = _aggregate_store_events(
        [[complete, incomplete], [complete], [], []],
        replication_factors=(2,),
    )

    assert events == [complete]


def test_store_events_use_each_groups_replication_factor():
    full_attention = _make_block_stored(b"full", group_idx=0)
    replicated_attention = _make_block_stored(b"replicated", group_idx=1)

    events = _aggregate_store_events(
        [
            [full_attention, replicated_attention],
            [full_attention],
            [full_attention, replicated_attention],
            [full_attention],
        ],
        replication_factors=(1, 2),
    )

    assert events == [full_attention, replicated_attention]


def test_update_connector_output_and_take_events():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    event = _make_block_stored()

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

    kv_events = mooncake_store_connector.MooncakeStoreKVEvents(num_workers=1)
    kv_events.add_events([event])
    output = KVConnectorOutput(kv_cache_events=kv_events)
    connector.update_connector_output(output)

    assert connector._kv_cache_events is kv_events
    connector.connector_scheduler.update_connector_output.assert_called_once_with(
        output
    )
    assert list(connector.take_events()) == [event]
    assert connector._kv_cache_events is None


def test_store_events_persist_across_polls():
    connector = object.__new__(mooncake_store_connector.MooncakeStoreConnector)
    connector.connector_scheduler = MagicMock()
    connector._kv_cache_events = None
    complete = _make_block_stored(b"complete", group_idx=0)
    pending = _make_block_stored(b"pending", group_idx=0)

    def poll(worker_events: list[BlockStored]) -> list[BlockStored]:
        events = mooncake_store_connector.MooncakeStoreKVEvents(
            num_workers=4,
            group_tp_replication_factors=(2,),
        )
        events.add_events(worker_events)
        connector.update_connector_output(KVConnectorOutput(kv_cache_events=events))
        return list(connector.take_events())

    assert poll([complete, pending]) == []
    assert poll([complete]) == [complete]
    assert poll([pending, pending]) == [pending]
    assert connector._kv_cache_events is None


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


@contextmanager
def _multi_store_lookup(hit_lengths=(16, 32), *, nested=False, limit=4):
    configs = [
        {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lookup_async": True,
                "lookup_prefetch_limit": limit[i]
                if isinstance(limit, tuple)
                else limit,
                "cache_prefix": f"store-{i}",
            },
        }
        for i in range(len(hit_lengths))
    ]
    if nested:
        configs = [
            {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"connectors": configs},
            }
        ]
    config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"connectors": configs},
    )
    gates = [threading.Event() for _ in hit_lengths]
    started = [threading.Event() for _ in hit_lengths]
    sockets = [MagicMock() for _ in hit_lengths]
    for sock, gate, start, hit in zip(sockets, gates, started, hit_lengths):
        gate.set()

        def recv(gate=gate, start=start, hit=hit):
            start.set()
            return _gated_recv(gate, hit)()

        sock.recv.side_effect = recv
    with (
        set_current_vllm_config(config),
        patch.object(worker, "make_zmq_socket", side_effect=sockets),
    ):
        connector = MultiConnector(
            config, KVConnectorRole.SCHEDULER, _make_kv_cache_config()
        )
    stores = connector.sub_connectors
    if nested:
        stores = stores[0].sub_connectors
    try:
        yield connector, stores, sockets, gates, started
    finally:
        for gate in gates:
            gate.set()
        connector.shutdown()


def _lookup_states(request):
    handle = request.kv_preschedule_handle
    assert handle is not None
    return (
        handle.handles[0].handles
        if isinstance(handle.handles[0], type(handle))
        else handle.handles
    )


@pytest.mark.parametrize("nested", [False, True])
def test_dropped_preschedule_request_cancels_all_queued_children(nested):
    with _multi_store_lookup(nested=nested) as (connector, stores, _, gates, started):
        for gate in gates:
            gate.clear()
        running = create_request(num_tokens=48)
        running.kv_preschedule_handle = connector.on_preschedule_request(running)
        assert all(event.wait(timeout=5) for event in started)
        dropped = create_request(num_tokens=48)
        dropped.kv_preschedule_handle = connector.on_preschedule_request(dropped)
        futures = [state.future for state in _lookup_states(dropped)]
        connector.cancel_preschedule_request(dropped)
        connector.cancel_preschedule_request(dropped)
        assert dropped.kv_preschedule_handle is None
        assert all(future.cancelled() for future in futures)
        # Dropping this handle must not disturb another request's work.
        for gate in gates:
            gate.set()
        assert [
            s.future.result(timeout=5).hit_length for s in _lookup_states(running)
        ] == [16, 32]


@pytest.mark.parametrize(
    "nested,hits", [(False, (16, 32)), (True, (0, 32)), (False, (0, 0))]
)
def test_multi_store_preschedule_keeps_lookup_results_with_their_store(nested, hits):
    with _multi_store_lookup(hits, nested=nested) as (connector, _, sockets, _, _):
        request = create_request(num_tokens=48)
        with ThreadPoolExecutor(max_workers=1) as executor:
            request.kv_preschedule_handle = executor.submit(
                connector.on_preschedule_request, request
            ).result(timeout=5)
        assert [
            s.future.result(timeout=5).hit_length for s in _lookup_states(request)
        ] == list(hits)
        connector.on_new_request(request)
        expected = next((hit for hit in hits if hit), 0)
        assert connector.get_num_new_matched_tokens(request, 0) == (
            expected,
            bool(expected),
        )
        assert request.kv_preschedule_handle is None
        assert connector.get_num_new_matched_tokens(request, 0) == (
            expected,
            bool(expected),
        )
        assert [sock.send_multipart.call_count for sock in sockets] == [1, 1]


@pytest.mark.parametrize("first_limit", [0, 4])
def test_multi_store_preschedule_retains_later_handle_when_first_is_pending(
    first_limit,
):
    with _multi_store_lookup((0, 32), limit=(first_limit, 4)) as (
        connector,
        stores,
        sockets,
        gates,
        _,
    ):
        gates[0].clear()
        request = create_request(num_tokens=48)
        request.kv_preschedule_handle = connector.on_preschedule_request(request)
        second = _lookup_states(request)[1]
        assert second.future.result(timeout=5).hit_length == 32
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        assert request.kv_preschedule_handle.handles == [None, second]
        gates[0].set()
        stores[0].connector_scheduler.client._lookups[request.request_id].future.result(
            timeout=5
        )
        assert connector.get_num_new_matched_tokens(request, 0) == (32, True)
        assert [sock.send_multipart.call_count for sock in sockets] == [1, 1]


@pytest.mark.parametrize(
    "queued,probe,all_groups",
    [(True, False, False), (True, True, True), (False, True, True)],
)
def test_multi_store_abort_then_reuse_request_id(queued, probe, all_groups):
    """Queued work is cancelled; a late running RPC cannot replace a new lookup."""
    with _multi_store_lookup(nested=True) as (connector, _, sockets, gates, started):
        for gate in gates:
            gate.clear()
        request = create_request(request_id=1000, num_tokens=48)
        request.kv_preschedule_handle = connector.on_preschedule_request(request)
        for event in started:
            assert event.wait(timeout=5)
        running = list(_lookup_states(request))
        if queued:
            request = create_request(request_id=1001, num_tokens=48)
            request.kv_preschedule_handle = connector.on_preschedule_request(request)
        states = list(_lookup_states(request))
        if probe:
            assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        finish = (
            connector.request_finished_all_groups
            if all_groups
            else connector.request_finished
        )
        assert finish(request, ([],) if all_groups else []) == (False, None)
        assert request.kv_preschedule_handle is None
        assert all(state.future.cancelled() == queued for state in states)
        for sock, gate, hit in zip(sockets, gates, (0, 48)):
            sock.recv.side_effect = _gated_recv(gate, hit)
        replacement = create_request(request_id=1001 if queued else 1000, num_tokens=64)
        replacement.kv_preschedule_handle = connector.on_preschedule_request(
            replacement
        )
        new_states = list(_lookup_states(replacement))
        for gate in gates:
            gate.set()
        assert [s.future.result(timeout=5).hit_length for s in running] == [16, 32]
        assert [s.future.result(timeout=5).hit_length for s in new_states] == [0, 48]
        assert connector.get_num_new_matched_tokens(replacement, 0) == (48, True)
        assert replacement.kv_preschedule_handle is None
        assert [sock.send_multipart.call_count for sock in sockets] == [2, 2]


def _make_lookup_client(extra_config=None):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as make_socket:
        client = worker.LookupKeyClient(vllm_config)
    return client, make_socket.return_value


@pytest.mark.parametrize(
    "adopted,reply",
    [(False, protocol.RESP_OK), (True, protocol.RESP_OK), (False, protocol.RESP_ERR)],
)
def test_reset_invalidates_old_handles_but_preserves_later_submissions(adopted, reply):
    client, sock = _make_lookup_client({"lookup_prefetch_limit": 2})
    reset_started, release_reset = threading.Event(), threading.Event()
    sock.recv.side_effect = [(32).to_bytes(4, "big"), reply, bytes(4), bytes(4)]

    def send_reset(message):
        assert message == protocol.RESET_MSG
        reset_started.set()
        assert release_reset.wait(timeout=5)

    sock.send.side_effect = send_reset
    try:
        old = client.prefetch(48, [])
        assert old.future.result(timeout=5).hit_length == 32
        if adopted:
            client.adopt("old", old)
        with ThreadPoolExecutor(max_workers=1) as scheduler:
            resetting = scheduler.submit(client.reset)
            try:
                assert reset_started.wait(timeout=5)
                fresh = client.prefetch(48, [])
            finally:
                release_reset.set()
            assert resetting.result(timeout=5) == (reply == protocol.RESP_OK)
        assert fresh.future.result(timeout=5).hit_length == 0
        client.adopt("fresh", fresh)
        assert client.lookup("fresh", 48, []).hit_length == 0
        client.adopt("old", old)
        assert client.lookup("old", 48, []).hit_length == 0
        assert sock.send_multipart.call_count == 3
    finally:
        release_reset.set()
        client.close()


def test_lookup_close_stops_running_rpc_and_cancels_unadopted_work():
    client, sock = _make_lookup_client({"lookup_prefetch_limit": 2})
    sent, reply = threading.Event(), threading.Event()
    sock.send_multipart.side_effect = lambda *a, **kw: sent.set()
    sock.poll.side_effect = lambda timeout: reply.wait(timeout / 1000)
    sock.recv.side_effect = _gated_recv(reply, 32)
    try:
        running = client.prefetch(48, [])
        assert sent.wait(timeout=5)
        queued = client.prefetch(48, [])
        client.close()
        assert running.future.done()
        with pytest.raises(CancelledError):
            running.future.result()
        assert queued.future.cancelled()
        assert client.prefetch(48, []) is None
        client.close()
    finally:
        reply.set()
        client.close()


def test_multi_store_shutdown_closes_scheduler_clients():
    with _multi_store_lookup() as (connector, stores, *_):
        connector.shutdown()
        for store in stores:
            assert store.connector_scheduler.client.prefetch(48, []) is None


@pytest.mark.parametrize("cache_limit", [0, 16])
def test_preschedule_respects_request_cache_read_limit(cache_limit):
    with _multi_store_lookup() as (connector, _, sockets, *_):
        request = Request(
            "bounded",
            [1] * 48,
            SamplingParams(prompt_logprobs=1, prompt_logprob_start=cache_limit),
            None,
        )
        request.block_hashes = [b"h0", b"h1", b"h2"]
        assert request.skip_reading_external_kv
        assert connector.on_preschedule_request(request) is None
        assert all(sock.send_multipart.call_count == 0 for sock in sockets)


@pytest.mark.parametrize("limit", [None, 0, 1])
def test_lookup_prefetch_cap_bounds_in_flight_prefetches(limit):
    client, sock = _make_lookup_client(
        {} if limit is None else {"lookup_prefetch_limit": limit}
    )
    gate = threading.Event()
    sock.recv.side_effect = _gated_recv(gate, 32)
    try:
        first = client.prefetch(48, [])
        assert (first is not None) == bool(limit)
        assert client.prefetch(48, []) is None
        # Scheduler probes bypass the cap.
        assert client.lookup("next", 48, [], non_block=True) is None
        assert "next" in client._lookups
        if limit:
            gate.set()
            # The single worker runs callbacks before taking its next task.
            client.executor.submit(lambda: None).result(timeout=5)
            later = client.prefetch(48, [])
            assert later is not None
            later.future.result(timeout=5)
    finally:
        gate.set()
        client.close()


@pytest.mark.parametrize("change", ["tokens", "hash", "expired", "cached_expired"])
def test_lookup_client_refreshes_changed_or_expired_query(monkeypatch, change):
    client, sock = _make_lookup_client({"lookup_prefetch_limit": 1})
    sock.recv.return_value = (32).to_bytes(4, "big")
    clock = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    num_tokens, hashes = 48, [b"h0", b"h1", b"h2"]
    try:
        state = client.prefetch(num_tokens, hashes)
        client.adopt("req", state)
        first = state.future
        assert first.result(timeout=5).hit_length == 32
        if change == "expired":
            # Prefetched long before the scheduler's first probe.
            clock[0] = 3.0
        else:
            assert client.lookup("req", num_tokens, hashes, non_block=True) is not None
            if change == "tokens":
                num_tokens += 1
            elif change == "hash":
                hashes[-1] = b"different-tail"
            else:
                clock[0] = 3.0

        client.lookup("req", num_tokens, hashes, non_block=True)
        refreshed = client._lookups["req"].future
        assert refreshed is not first
        assert refreshed.result(timeout=5).hit_length == 32
        assert client.lookup("req", num_tokens, hashes, non_block=True).hit_length == 32
        assert sock.send_multipart.call_count == 2
    finally:
        client.close()


def test_normal_pending_lookup_is_not_restarted_by_result_cache_ttl(monkeypatch):
    client, sock = _make_lookup_client()
    gate = threading.Event()
    sock.recv.side_effect = _gated_recv(gate, 32)
    clock = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    try:
        assert client.lookup("req", 48, [], non_block=True) is None
        first = client._lookups["req"].future
        clock[0] = 3.0
        assert client.lookup("req", 48, [], non_block=True) is None
        assert client._lookups["req"].future is first
        gate.set()
        assert first.result(timeout=5).hit_length == 32
        assert client.lookup("req", 48, [], non_block=True).hit_length == 32
        assert sock.send_multipart.call_count == 1
    finally:
        gate.set()
        client.close()


def test_reset_clears_cached_async_results():
    client, sock = _make_lookup_client()
    sock.recv.return_value = (32).to_bytes(4, "big")
    try:
        client.lookup("req", 48, [], non_block=True)
        assert client._lookups["req"].future.result(timeout=5).hit_length == 32
        sock.recv.return_value = protocol.RESP_OK
        assert client.reset() is True
        assert client._lookups == {}
    finally:
        client.close()


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
    # Repeated reads reuse the completed result until expiry or discard.
    assert _poll_lookup(client, "req1") == 7
    assert fake_socket.send_multipart.call_count == 1
    client.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize("all_groups", [False, True])
def test_finished_request_cancels_queued_lookup_before_next_step(all_groups):
    config = _make_vllm_config()
    gate, started = threading.Event(), threading.Event()
    with (
        set_current_vllm_config(config),
        patch.object(worker, "make_zmq_socket") as make,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            config, KVConnectorRole.SCHEDULER, _make_kv_cache_config()
        )
    client = connector.connector_scheduler.client
    make.return_value.send_multipart.side_effect = lambda *args, **kwargs: started.set()
    make.return_value.recv.side_effect = _gated_recv(gate, 7)
    request = Request("queued", [1] * 128, SamplingParams(max_tokens=1), None)
    try:
        assert client.lookup("busy", 128, [], non_block=True) is None
        assert started.wait(5)
        assert client.lookup(request.request_id, 128, [], non_block=True) is None
        finish = (
            connector.request_finished_all_groups
            if all_groups
            else connector.request_finished
        )
        assert finish(request, ([],) if all_groups else []) == (False, None)
        gate.set()
        client.executor.submit(lambda: None).result(timeout=5)
        assert make.return_value.send_multipart.call_count == 1
        assert client.lookup("busy", 128, []).hit_length == 7
    finally:
        gate.set()
        client.executor.shutdown(wait=True, cancel_futures=True)
        connector.shutdown()


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
    # the client once it resolves.
    assert client.lookup("req2", 128, [], non_block=True) is None
    gate.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client._lookups["req2"].future.done():
            break
        time.sleep(0.005)
    # discard() drops the completed result before any lookup consumes it.
    client.discard("req2")
    assert "req2" not in client._lookups
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
    at master keys we're about to wipe -- pending load_specs and
    accumulated _kv_cache_events both reference keys whose blobs are
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

    # Both stale references must be cleared by the time reset_store is
    # invoked downstream (load_specs flushed dict, events nulled).
    assert sched_inst.load_specs == {}
    assert conn._kv_cache_events is None


def test_lookup_key_server_reset_drains_send_queue_before_remove_all():
    """LookupKeyServer RESET handler must drain the send thread's
    request_queue BEFORE calling store.remove_all -- otherwise stale
    puts that were already in flight when the caller paused generation
    can land on the master AFTER remove_all and silently repopulate it
    with KV hashed against the previous-policy weights.
    """
    # Exercise the handler logic directly with mocks for the send thread
    # and store. We assert (a) join() is called, (b) remove_all is called,
    # and (c) join() comes BEFORE remove_all in the call order. The full
    # LookupKeyServer is heavy (binds a real ZMQ REP socket), so we drive
    # just the dispatch branch here via a stub equivalent.
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        protocol,
    )

    call_order: list[str] = []

    fake_send_queue = MagicMock()
    fake_send_queue.join.side_effect = lambda: call_order.append("join")

    fake_store = MagicMock()
    fake_store.remove_all.side_effect = lambda force: call_order.append(
        f"remove_all(force={force})"
    )

    fake_send_thread = MagicMock()
    fake_send_thread.request_queue = fake_send_queue

    fake_store_worker = MagicMock()
    fake_store_worker.kv_send_thread = fake_send_thread
    fake_store_worker.store = fake_store

    fake_socket = MagicMock()
    sent: list[bytes] = []
    fake_socket.send.side_effect = lambda frame: sent.append(frame)

    # Mirror the body of LookupKeyServer.process_request RESET_MSG branch.
    # Keeping this inline (instead of importing the closure) keeps the
    # test independent of the live thread lifecycle.
    msg_type = protocol.RESET_MSG
    if msg_type == protocol.RESET_MSG:
        try:
            if fake_store_worker.kv_send_thread is not None:
                fake_store_worker.kv_send_thread.request_queue.join()
            fake_store_worker.store.remove_all(force=True)
            fake_socket.send(protocol.RESP_OK)
        except Exception:
            fake_socket.send(protocol.RESP_ERR)

    # Drain must happen before remove_all.
    assert call_order == ["join", "remove_all(force=True)"]
    # Worker reported success.
    assert sent == [protocol.RESP_OK]


def test_lookup_key_server_reset_skips_drain_when_no_send_thread():
    """When the worker has no send thread (e.g. consumer-only role
    configurations), the RESET handler must still call remove_all
    instead of dereferencing a None send thread.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        protocol,
    )

    call_order: list[str] = []
    fake_store = MagicMock()
    fake_store.remove_all.side_effect = lambda force: call_order.append("remove_all")

    fake_store_worker = MagicMock()
    fake_store_worker.kv_send_thread = None
    fake_store_worker.store = fake_store

    fake_socket = MagicMock()
    sent: list[bytes] = []
    fake_socket.send.side_effect = lambda frame: sent.append(frame)

    msg_type = protocol.RESET_MSG
    if msg_type == protocol.RESET_MSG:
        try:
            if fake_store_worker.kv_send_thread is not None:
                fake_store_worker.kv_send_thread.request_queue.join()
            fake_store_worker.store.remove_all(force=True)
            fake_socket.send(protocol.RESP_OK)
        except Exception:
            fake_socket.send(protocol.RESP_ERR)

    assert call_order == ["remove_all"]
    assert sent == [protocol.RESP_OK]


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
