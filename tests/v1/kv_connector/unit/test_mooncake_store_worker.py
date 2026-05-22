# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    rdma_utils,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)


def _default_send_coord() -> mooncake_store_worker.MooncakeStoreCoordinator:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    return mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )


def _make_store_sending_thread(
    store: MagicMock,
    *,
    coord: mooncake_store_worker.MooncakeStoreCoordinator | None = None,
    token_databases: list[ChunkedTokenDatabase] | None = None,
    block_size: int = 16,
    replicate_config: object | None = None,
) -> mooncake_store_worker.KVCacheStoreSendingThread:
    if coord is None:
        coord = _default_send_coord()
    if token_databases is None:
        db = ChunkedTokenDatabase(KeyMetadata("test-model", 0, 0, 0, 0), block_size=16)
        db.set_kv_caches_base_addr([0x1000])
        db.set_block_len([256])
        token_databases = [db]
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        token_databases=token_databases,
        block_size=block_size,
        coord=coord,
        tp_rank=0,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
        replicate_config=replicate_config,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_store_recving_thread(
    store: MagicMock,
    *,
    tp_rank: int = 0,
    disk_offload_buffer_budget_bytes: int | None = None,
) -> mooncake_store_worker.KVCacheStoreRecvingThread:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_databases=[token_database],
        block_size=16,
        tp_rank=tp_rank,
        ready_event=threading.Event(),
        coord=coord,
        disk_offload_buffer_budget_bytes=disk_offload_buffer_budget_bytes,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_load_req(
    req_id: str,
    block_hashes: list[bytes],
    *,
    token_len: int,
    vllm_cached_tokens: int = 0,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=(list(range(len(block_hashes))),),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
        ),
    )


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=([0, 1],),
        block_hashes=block_hashes,
        can_save=True,
        original_block_size=16,
    )


_DISK_OFFLOAD_SINGLE_KEY_BYTES = worker._estimate_disk_offload_staging_bytes([256])
_DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS = 4 * _DISK_OFFLOAD_SINGLE_KEY_BYTES
_DISK_OFFLOAD_BUDGET_FOR_SPLIT = math.ceil(
    2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES / _DISK_OFFLOAD_USABLE_BUDGET_RATIO
)  # Allows two 256-byte chunks but not the third.
_DISK_OFFLOAD_BUDGET_TOO_SMALL = (
    _DISK_OFFLOAD_SINGLE_KEY_BYTES - 1
)  # Smaller than a single 256-byte chunk.


class _FakeKVTransferConfig:
    def __init__(
        self,
        *,
        kv_role: str = "kv_both",
        extra_config: dict[str, object] | None = None,
    ) -> None:
        self.kv_role = kv_role
        self.kv_connector_extra_config = extra_config or {}

    def get_from_extra_config(self, key: str, default: object) -> object:
        return self.kv_connector_extra_config.get(key, default)


class _FakeModelConfig:
    model = "test-model"
    use_mla = False

    def get_num_layers(self, parallel_config) -> int:
        return 1

    def get_total_num_kv_heads(self) -> int:
        return 1


def _make_vllm_config(
    *, extra_config: dict[str, object] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=_FakeModelConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            rank=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=_FakeKVTransferConfig(extra_config=extra_config),
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=10),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
        speculative_config=None,
    )


def _make_kv_cache_config(*, block_size: int = 16) -> object:
    """Minimal single-group KVCacheConfig for topology tests."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )


def _write_mooncake_config(tmp_path, config: dict[str, object]) -> str:
    config_path = tmp_path / "mooncake_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def _install_fake_mooncake(monkeypatch, store_instance: MagicMock):
    class FakeReplicateConfig:
        def __init__(self) -> None:
            self.preferred_segment = ""

    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = lambda: store_instance  # type: ignore[attr-defined]
    fake_store_module.ReplicateConfig = FakeReplicateConfig  # type: ignore[attr-defined]
    fake_mooncake_module = types.ModuleType("mooncake")
    fake_mooncake_module.store = fake_store_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)
    return FakeReplicateConfig


def _patch_worker_runtime(monkeypatch, *, local_ip: str = "10.0.0.7") -> None:
    single_rank_group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(worker, "get_mooncake_dp_engine_index", lambda _: 0)
    monkeypatch.setattr(worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(worker, "get_pcp_group", lambda: single_rank_group)
    monkeypatch.setattr(worker, "get_dcp_group", lambda: single_rank_group)
    monkeypatch.setattr(worker, "get_ip", lambda: local_ip)


def test_default_local_buffer_size_matches_pr40900():
    """PR-40900 shipped a 4 GiB default for local_buffer_size; the dual-mode
    patch preserves it (and the JSON key) so unchanged PR-40900 configs work."""
    assert worker.DEFAULT_LOCAL_BUFFER_SIZE == 4 * 1024**3


def test_get_requester_local_hostname_prefers_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "worker-a:50053")

    assert rdma_utils.get_requester_local_hostname("10.0.0.7") == "worker-a:50053"


def test_get_configured_preferred_segment_returns_explicit_override():
    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "10.0.0.8:50053")

    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_returns_env_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "10.0.0.8:50053")

    assert rdma_utils.get_configured_preferred_segment({}) == "10.0.0.8:50053"


def test_get_configured_preferred_segment_rejects_empty_override():
    with pytest.raises(ValueError, match="preferred_segment"):
        rdma_utils.get_configured_preferred_segment({"preferred_segment": "  "})


def test_get_configured_worker_rnic_prefers_explicit_device_name(monkeypatch):
    store_config = worker.MooncakeStoreConfig(
        metadata_server="",
        local_buffer_size=1,
        protocol="rdma",
        device_name="rocep139s0",
        master_server_address="",
    )

    assert (
        rdma_utils.get_configured_worker_rnic(
            protocol=store_config.protocol,
            configured_device=store_config.device_name,
        )
        == "rocep139s0"
    )


def test_get_configured_worker_rnic_selects_device_from_explicit_csv(monkeypatch):
    monkeypatch.setattr(
        rdma_utils,
        "get_current_physical_gpu_index",
        lambda: 1,
    )
    store_config = worker.MooncakeStoreConfig(
        metadata_server="",
        local_buffer_size=1,
        protocol="rdma",
        device_name="rocep139s0,rocep140s0",
        master_server_address="",
    )

    assert (
        rdma_utils.get_configured_worker_rnic(
            protocol=store_config.protocol,
            configured_device=store_config.device_name,
        )
        == "rocep140s0"
    )


def test_get_configured_worker_rnic_warns_and_returns_empty_for_rdma_with_no_device(
    caplog, monkeypatch
):
    """No device configured + protocol=rdma → emit a clear warning and return ""
    so the C++ side handles auto-selection. There is no Python-side fallback."""
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.WARNING):
        result = rdma_utils.get_configured_worker_rnic(
            protocol="rdma",
            configured_device="",
        )
    assert result == ""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("No RDMA devices specified" in r.message for r in warnings), (
        f"expected fallback warning, got {[r.message for r in warnings]}"
    )


def test_get_configured_worker_rnic_silent_for_tcp_with_no_device(caplog, monkeypatch):
    """protocol=tcp + no device → return "" silently (no RDMA, no warning)."""
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.WARNING):
        result = rdma_utils.get_configured_worker_rnic(
            protocol="tcp",
            configured_device="",
        )
    assert result == ""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("RDMA" in r.message for r in warnings), (
        "did not expect RDMA warning for tcp protocol, got "
        f"{[r.message for r in warnings]}"
    )


def test_get_configured_worker_rnic_rejects_short_explicit_csv(monkeypatch):
    monkeypatch.setattr(
        rdma_utils,
        "get_current_physical_gpu_index",
        lambda: 2,
    )
    with pytest.raises(ValueError, match="does not cover local GPU 2"):
        rdma_utils.get_configured_worker_rnic(
            protocol="rdma",
            configured_device="rocep139s0,rocep140s0",
        )


class _ReplicaDesc:
    def __init__(self, tier: str):
        self.tier = tier

    def is_memory_replica(self) -> bool:
        return self.tier == "memory"

    def is_disk_replica(self) -> bool:
        return self.tier == "disk"

    def is_local_disk_replica(self) -> bool:
        return self.tier == "disk"


def test_store_sending_thread_skips_request_during_cpu_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-200, -200],
        [256, 256],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is True
    assert "req-a" in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-b")
    thread._handle_request(_make_store_req("req-b", [b"b0", b"b1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 2

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a4", b"a5"]))

    assert store.batch_put_from_multi_buffers.call_count == 3


def test_store_sending_thread_only_skips_on_no_available_handle():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-500, -500],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 2


def test_store_recving_thread_reports_failed_block_ids():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -5, -7]
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_finished_requests() == {"req-a"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {1, 2}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


def test_store_recving_thread_reports_failed_block_ids_after_rotation():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -5, 256]
    thread = _make_store_recving_thread(store, tp_rank=1)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_block_ids_with_load_errors() == {2}


def test_store_recving_thread_reports_all_attempted_blocks_on_exception():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = RuntimeError("boom")
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_finished_requests() == {"req-a"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1, 2}


def test_store_worker_get_block_ids_with_load_errors_delegates_to_recv_thread():
    recv_thread = MagicMock()
    recv_thread.get_and_clear_block_ids_with_load_errors.return_value = {3, 4}
    w = _make_bare_worker()
    w.kv_recv_thread = recv_thread

    assert w.get_block_ids_with_load_errors() == {3, 4}
    recv_thread.get_and_clear_block_ids_with_load_errors.assert_called_once_with()


def test_store_sending_thread_passes_replicate_config_when_preferred_segment_set():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(preferred_segment="10.0.0.7:50053")
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_store_sending_thread_passes_default_replicate_config_when_no_preferred_segment():  # noqa: E501
    """Without a preferred_segment the SendingThread still forwards a
    (default-constructed) ReplicateConfig so the C++ side always sees a
    well-defined config object."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace()
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_estimate_disk_offload_staging_bytes_sums_multi_segment_sizes():
    assert worker._estimate_disk_offload_staging_bytes([256, 512]) == 12288


def test_recv_thread_uses_single_batch_when_no_disk_offload_budget(monkeypatch):
    monkeypatch.delenv("VLLM_MOONCAKE_STORE_TIER_LOG", raising=False)
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert sizes == [[256], [256], [256]]
    store.batch_get_replica_desc.assert_not_called()


def test_recv_thread_logs_tier_summary_when_enabled(monkeypatch, caplog_vllm):
    monkeypatch.setenv("VLLM_MOONCAKE_STORE_TIER_LOG", "1")
    caplog_vllm.set_level(logging.INFO, logger=worker.logger.name)

    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )
    expected_keys = [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    store.batch_get_replica_desc.return_value = {
        expected_keys[0]: [_ReplicaDesc("memory")],
        expected_keys[1]: [_ReplicaDesc("disk")],
        expected_keys[2]: [],
    }

    thread._handle_request(req)

    assert store.batch_get_replica_desc.call_args.args == (expected_keys,)
    assert store.method_calls[0][0] == "batch_get_replica_desc"
    assert store.method_calls[1][0] == "batch_get_into_multi_buffers"

    messages = [record.getMessage() for record in caplog_vllm.records]
    assert any(
        "Mooncake load tier summary" in message
        and "req_id=req-a" in message
        and "batch_keys=3" in message
        and "memory_keys=1" in message
        and "disk_keys=1" in message
        and "unknown_keys=1" in message
        and "success_keys=2" in message
        and "failed_keys=1" in message
        and "bytes_by_tier={'memory': 256, 'disk': 256, 'unknown': 0}" in message
        for message in messages
    )


def test_recv_thread_uses_ratio_scaled_budget_for_first_pass_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]


def test_recv_thread_splits_disk_offload_loads_by_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2

    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    first_addrs = store.batch_get_into_multi_buffers.call_args_list[0].args[1]
    second_addrs = store.batch_get_into_multi_buffers.call_args_list[1].args[1]
    first_sizes = store.batch_get_into_multi_buffers.call_args_list[0].args[2]
    second_sizes = store.batch_get_into_multi_buffers.call_args_list[1].args[2]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    base_addr = thread.token_databases[0].kv_caches_base_addr[0]
    block_len = thread.token_databases[0].block_len[0]
    assert first_addrs == [[base_addr], [base_addr + block_len]]
    assert second_addrs == [[base_addr + 2 * block_len]]
    expected_size = block_len
    assert first_sizes == [[expected_size], [expected_size]]
    assert second_sizes == [[expected_size]]


def test_recv_thread_stops_after_first_failing_disk_offload_sub_batch():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-10, -10]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}


def test_recv_thread_skips_split_when_budget_holds_all_keys():
    """PR-36 removed the count-based split trigger; with budget for 3 keys,
    all three should be requested in a single call."""
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    assert store.batch_get_into_multi_buffers.call_args_list[0].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]


def test_recv_thread_reports_unsplittable_key_larger_than_budget():
    # Non-zero tp_rank exercises the rotation: the oversized key may not sit
    # at the original request's first block. Every block must still be marked
    # invalid, since none are loaded.
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        tp_rank=2,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 0
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1, 2}


def test_requester_worker_init_uses_positional_setup(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4gb",
                "local_buffer_size": "64mb",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": True,
            },
        ),
    )
    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert not hasattr(w, "_isolate_offload_resources")
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        4 * 1024 * 1024 * 1024,  # global_segment_size: "4gb" honored
        64 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )


def test_requester_worker_init_prefers_local_hostname_override(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "worker-a:50053")
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "local_buffer_size": "64mb",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert store.setup.call_args.args[0] == "worker-a:50053"


def test_requester_worker_init_skips_disk_budget_when_offload_disabled(
    tmp_path,
    monkeypatch,
):
    """enable_offload=False zeroes out the disk budget so we don't generate
    redundant owner GET-RPCs."""
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": False,
            },
        ),
    )
    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert w.disk_offload_buffer_budget_bytes is None


def test_requester_worker_init_builds_replicate_config_for_preferred_segment(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "preferred_segment": "10.0.0.7:50053",
            }
        ),
        _make_kv_cache_config(),
    )

    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == "10.0.0.7:50053"


# ---------------------------------------------------------------------------
# Helpers for register_kv_caches tests
# ---------------------------------------------------------------------------


def test_store_sending_thread_clamps_token_len_to_lcm():
    """Partial chunks past the last lcm boundary aren't stored — cache hits
    are always lcm-aligned (mirrors HybridKVCacheCoordinator)."""
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    # Default coord: single full-attn block_size=16, lcm=16.
    # token_len_chunk=33 clamps to 32 → 2 chunks (not 3 with a partial 1-token chunk).
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("r0")
    thread._handle_request(
        ReqMeta(
            req_id="r0",
            token_len_chunk=33,
            block_ids=([0, 1, 2],),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
            original_block_size=16,
        )
    )

    keys = store.batch_put_from_multi_buffers.call_args.args[0]
    assert len(keys) == 2


def test_store_sending_thread_skips_when_token_len_below_lcm():
    """Requests shorter than lcm_block_size cannot produce any aligned chunk,
    so neither the existence check nor the put should be issued."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    # lcm=64 via single full-attn block_size=64.
    spec = FullAttentionSpec(block_size=64, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L"], spec)],
        scheduler_block_size=64,
        hash_block_size=64,
    )
    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=64,
        hash_block_size=64,
    )
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([1024])
    thread = _make_store_sending_thread(
        store, coord=coord, token_databases=[db], block_size=64
    )

    thread.add_stored_request("r0")
    thread._handle_request(
        ReqMeta(
            req_id="r0",
            token_len_chunk=32,
            block_ids=([0, 1],),
            block_hashes=[b"a0", b"a1"],
            can_save=True,
            original_block_size=64,
        )
    )

    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()
    assert thread.stored_requests["r0"] == 0


def test_store_sending_thread_only_stores_swa_blocks_in_window():
    """For SWA groups, only blocks within ``sliding_window`` of an
    lcm-aligned boundary are stored — a block outside every such window
    can never participate in any future hit (mirrors recv-side masking).

    Setup: full-attn (block_size=32) + SWA (block_size=8, sliding_window=8)
    → lcm=32. With token_len=64 there are two LCM boundaries (32, 64);
    the SWA group should store exactly one block per boundary: the block
    ending at 32 and the block ending at 64.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, addrs, sizes: (
        [256] * len(keys)
    )

    full_spec = FullAttentionSpec(
        block_size=32, num_kv_heads=8, head_size=64, dtype=None
    )
    swa_spec = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=8,
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], swa_spec)],
        scheduler_block_size=32,
        hash_block_size=8,
    )

    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=32,
        hash_block_size=8,
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([512])
    db_swa = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=8,
        hash_block_size=8,
    )
    db_swa.set_kv_caches_base_addr([0x2000])
    db_swa.set_block_len([128])

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_swa],
        block_size=32,
    )

    hs = [bytes([i + 1]) * 4 for i in range(8)]
    thread.add_stored_request("r0")
    thread._handle_request(
        ReqMeta(
            req_id="r0",
            token_len_chunk=64,
            block_ids=([0, 1], list(range(8))),
            block_hashes=hs,
            can_save=True,
            original_block_size=32,
        )
    )

    keys = store.batch_put_from_multi_buffers.call_args.args[0]
    full_keys = [k for k in keys if "@group:0" in k]
    swa_keys = [k for k in keys if "@group:1" in k]
    # Full-attn: 2 blocks (chunks ending at 32 and 64).
    assert len(full_keys) == 2
    # SWA: only the two blocks ending at lcm boundaries (32 and 64), i.e.
    # blocks covering tokens [24,32) and [56,64) — hashes hs[3] and hs[7].
    assert len(swa_keys) == 2
    swa_hashes = {k.rsplit("@", 1)[-1] for k in swa_keys}
    assert swa_hashes == {hs[3].hex(), hs[7].hex()}


def _auto_set_ready_event(*args, **kwargs):
    """Side effect for mocked thread constructors that auto-sets ready_event."""
    for arg in args:
        if isinstance(arg, threading.Event):
            arg.set()
    for val in kwargs.values():
        if isinstance(val, threading.Event):
            val.set()
    return MagicMock()


def _register_with_mocked_threads(
    worker: mooncake_store_worker.MooncakeStoreWorker,
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """Call register_kv_caches with the I/O transfer threads mocked out."""
    prefix = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker."
    with (
        patch(prefix + "KVCacheStoreSendingThread", side_effect=_auto_set_ready_event),
        patch(prefix + "KVCacheStoreRecvingThread", side_effect=_auto_set_ready_event),
    ):
        worker.register_kv_caches(kv_caches)


def _make_bare_worker(
    *,
    num_gpu_blocks: int = 10,
    block_size: int = 16,
    kv_role: str = "kv_both",
) -> mooncake_store_worker.MooncakeStoreWorker:
    """Construct a MooncakeStoreWorker via __new__, bypassing __init__.

    Sets only the attributes that register_kv_caches() reads so we can
    test the stride-based layout detection without a real
    MooncakeDistributedStore.
    """
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker.cache_config = MagicMock()
    worker.cache_config.num_gpu_blocks = num_gpu_blocks
    worker.store = MagicMock()
    worker.store.register_buffer.return_value = 0
    worker.use_mla = False
    worker.kv_role = kv_role
    worker.block_size = block_size
    worker.tp_rank = 0
    worker.put_step = 1
    worker.enable_kv_events = False
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.tp_size = 1
    worker.num_kv_head = 1
    worker.pp_size = 1
    # Minimal single-full-attention-group config so the coordinator-based
    # lookup path works (the connector no longer carries a legacy single-group
    # path; everything flows through the coordinator).
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
    )

    worker.disk_offload_buffer_budget_bytes = None
    worker.store_replicate_config = SimpleNamespace()

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    group = KVCacheGroupSpec(["layer0", "__cross_layer__"], spec)
    worker._kv_cache_groups = [group]
    worker.pcp_size = 1
    worker.dcp_size = 1
    worker.hash_block_size = block_size
    worker.metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    # Pre-build a single-group token_dbs so lookup-only tests don't have to
    # call register_kv_caches.
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
            block_size=block_size,
            hash_block_size=block_size,
        )
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    return worker


def test_lookup_partial_prefix_returns_first_hit_length():
    worker = _make_bare_worker()
    worker.store.batch_is_exist.return_value = [1, 1, 0]
    assert worker.lookup(48, [b"a0", b"a1", b"a2"]) == 32


def test_lookup_swa_single_group_returns_full_when_tail_window_present():
    """Single-SWA, sliding_window=32 (= 2 blocks): producer stored only the
    tail. Coordinator-driven lookup returns full prefix even though the
    pre-window blocks are absent."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec

    worker = _make_bare_worker(block_size=16)
    swa = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None, sliding_window=32
    )
    worker._kv_cache_groups = [KVCacheGroupSpec(["layer0"], swa)]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=worker.hash_block_size,
        hash_block_size=worker.hash_block_size,
    )
    worker.store.batch_is_exist.return_value = [0, 0, 1, 1]
    assert worker.lookup(64, [b"h0", b"h1", b"h2", b"h3"]) == 64


# ---------------------------------------------------------------------------
# register_kv_caches tests
# ---------------------------------------------------------------------------


def test_register_kv_caches_blocks_first_single_segment():
    """Blocks-first layout (FlashInfer/MLA): one segment per layer."""
    num_blocks = 10
    page_size_elements = 64
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (num_blocks, page_size_elements) — blocks outermost, no outer_dims
    tensor = torch.zeros(num_blocks, page_size_elements, dtype=torch.float16)
    _register_with_mocked_threads(worker, {"layer0": tensor})

    db = worker.token_dbs[0]
    assert db.kv_caches_base_addr == [tensor.untyped_storage().data_ptr()]
    assert db.block_len == [tensor.untyped_storage().nbytes() // num_blocks]
    worker.store.register_buffer.assert_called_once_with(
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


def test_register_kv_caches_kv_first_two_segments():
    """K/V-first layout (FlashAttn): two segments (K, V) per layer."""
    num_blocks = 10
    block_size_tokens = 16
    num_kv_heads = 4
    head_size = 8
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (2, num_blocks, block_size, num_kv_heads, head_size) — K/V outermost
    tensor = torch.zeros(
        2,
        num_blocks,
        block_size_tokens,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
    )
    _register_with_mocked_threads(worker, {"layer0": tensor})

    db = worker.token_dbs[0]
    seg_stride = tensor.stride(0) * tensor.element_size()
    base = tensor.untyped_storage().data_ptr()
    assert db.kv_caches_base_addr == [base, base + seg_stride]
    assert db.block_len == [seg_stride // num_blocks] * 2


def test_register_kv_caches_cross_layer_single_segment():
    """Cross-layer tensor: single segment with block_len = page_size * num_layers."""
    num_blocks = 10
    num_layers = 4
    per_layer_page_elements = 64  # elements per layer per block

    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Cross-layer blocks-first tensor: all layers packed into a single
    # contiguous block.  Shape (num_blocks, num_layers * per_layer_page)
    # mimics the physical layout after stride reordering.
    total_page_elements = num_layers * per_layer_page_elements
    tensor = torch.zeros(num_blocks, total_page_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        # Use the cross-layer wrapper key, same as register_cross_layers_kv_caches
        worker.register_kv_caches({"__cross_layer__": tensor})

    db = worker.token_dbs[0]
    assert len(db.kv_caches_base_addr) == 1
    assert db.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    # block_len should be per_layer_page_size * num_layers
    assert (
        expected_block_len
        == num_layers * per_layer_page_elements * tensor.element_size()
    )
    assert len(db.block_len) == 1
    assert db.block_len[0] == expected_block_len

    # Also verify via register_cross_layers_kv_caches wrapper
    worker2 = _make_bare_worker(num_gpu_blocks=num_blocks)
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker2.register_cross_layers_kv_caches(tensor)

    db2 = worker2.token_dbs[0]
    assert db2.kv_caches_base_addr == db.kv_caches_base_addr
    assert db2.block_len == db.block_len


# ---------------------------------------------------------------------------
# Dual-mode (embedded / standalone-store) config validation tests
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Build a MooncakeStoreConfig with sensible defaults for validation tests.

    Required dataclass fields are populated; callers override only the field
    under test.
    """
    base = dict(
        metadata_server="http://metadata/endpoint",
        master_server_address="10.0.0.7:50051",
        protocol="rdma",
        device_name="mlx5_0",
    )
    base.update(overrides)
    return worker.MooncakeStoreConfig(**base)


def test_config_defaults_to_embedded():
    """A JSON without explicit mode parses as embedded with 4 GiB segment."""
    cfg = _make_config()
    assert cfg.mode == "embedded"
    assert cfg.global_segment_size == worker.DEFAULT_GLOBAL_SEGMENT_SIZE
    assert cfg.local_buffer_size == worker.DEFAULT_LOCAL_BUFFER_SIZE
    assert cfg.enable_offload is False


def test_config_pr40900_unchanged(tmp_path):
    """A literal PR-40900 config (no mode, no enable_offload, no preferred_segment)
    parses without raising and resolves to embedded mode."""
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "global_segment_size": "4GB",
            "local_buffer_size": "4GB",
            "protocol": "rdma",
            "device_name": "mlx5_0",
            "master_server_address": "10.0.0.7:50051",
        },
    )
    cfg = worker.MooncakeStoreConfig.from_file(config_path)
    assert cfg.mode == "embedded"
    assert cfg.global_segment_size == 4 * 1024**3
    assert cfg.local_buffer_size == 4 * 1024**3
    assert cfg.enable_offload is False


def test_config_embedded_rejects_zero_segment():
    with pytest.raises(
        ValueError, match=r"embedded mode requires global_segment_size > 0"
    ):
        _make_config(mode="embedded", global_segment_size=0)


def test_config_standalone_store_rejects_nonzero_segment():
    with pytest.raises(
        ValueError,
        match=r"standalone-store mode requires global_segment_size == 0",
    ):
        _make_config(mode="standalone-store", global_segment_size=4 * 1024**3)


def test_config_standalone_store_accepts_zero_segment():
    cfg = _make_config(mode="standalone-store", global_segment_size=0)
    assert cfg.mode == "standalone-store"
    assert cfg.global_segment_size == 0


def test_config_unknown_mode():
    with pytest.raises(ValueError, match=r"unknown Mooncake mode"):
        _make_config(mode="something-else")


def test_config_zero_local_buffer():
    with pytest.raises(ValueError, match=r"local_buffer_size must be > 0"):
        _make_config(local_buffer_size=0)


# ---------------------------------------------------------------------------
# End-to-end topology tests
# Covers the two supported recipes:
#   (A) standalone-store mode + disk offload (mode="standalone-store",
#       segment=0, enable_offload=true, preferred_segment set)
#   (B) embedded mode + CPU only      (mode default, segment>0,
#       enable_offload=false, no preferred_segment)
# ---------------------------------------------------------------------------


def test_topology_standalone_store_with_disk_offload(tmp_path, monkeypatch):
    """standalone-store + disk: global_segment_size=0, enable_offload=True,
    preferred_segment set. Assert setup() positional args, ReplicateConfig
    wiring, and that the disk-offload buffer budget is allocated."""
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "mode": "standalone-store",
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": 0,
                "local_buffer_size": "1GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": True,
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(extra_config={"preferred_segment": "10.0.0.7:50053"}),
        _make_kv_cache_config(),
    )

    # setup() receives global_segment_size=0 and the configured local buffer.
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        0,
        1024 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    # ReplicateConfig is built and carries the preferred_segment.
    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == "10.0.0.7:50053"
    # Disk-offload staging budget is allocated (enable_offload=True).
    assert w.disk_offload_buffer_budget_bytes is not None
    assert w.disk_offload_buffer_budget_bytes > 0


def test_topology_embedded_cpu_only(tmp_path, monkeypatch):
    """embedded + CPU-only: no mode key (defaults to embedded),
    global_segment_size>0, enable_offload absent, no preferred_segment.
    This is the PR-40900 baseline recipe."""
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    # setup() receives global_segment_size=4 GiB (rank contributes a segment).
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        4 * 1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    # No preferred_segment — ReplicateConfig is default-constructed (so the
    # preferred_segment field keeps its default value).
    assert w.preferred_segment is None
    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == ""
    # No disk budget — enable_offload was absent (defaults to False).
    assert w.disk_offload_buffer_budget_bytes is None
