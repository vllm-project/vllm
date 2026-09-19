# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU method-level tests. All native/GPU resources below are simulated."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from peer_retirement import (
    PEER_MAPS,
    DropResult,
    PeerRetirement,
    all_workers_acknowledged,
)
from source_harness import (
    SimulatedAllocation,
    add_peer,
    load_worker_methods,
    make_worker,
    recv_meta,
)

PEER = "p-old-generation"


def test_baseline_owner_exit_does_not_call_consumer_cleanup():
    worker, allocation = make_worker()
    allocation.owner_exit()
    assert allocation.resident
    assert PEER in worker._remote_agents
    assert worker.nixl_wrapper.events == []


def test_baseline_stale_but_unpolled_peer_keeps_imports():
    worker, allocation = make_worker()
    allocation.owner_exit()
    worker._engine_last_active[PEER] = time.perf_counter() - 7200
    assert allocation.resident
    assert worker.nixl_wrapper.events == []
    worker._evict_stale_engines()
    assert not allocation.resident


def test_baseline_ttl_preserves_a_known_inflight_transfer():
    worker, _ = make_worker()
    worker._engine_last_active[PEER] = time.perf_counter() - 7200
    worker._recving_metadata["request"] = recv_meta()
    worker._recving_transfers["request"] = ["native-read"]
    worker._evict_stale_engines()
    assert worker.nixl_wrapper.events == []


def test_drained_drop_uses_real_cleanup_and_preserves_healthy_peer():
    worker, allocation = make_worker()
    healthy = SimulatedAllocation()
    add_peer(worker, "p-healthy-generation", healthy)
    before = {
        name: copy.deepcopy(getattr(worker, name)["p-healthy-generation"])
        for name in PEER_MAPS
    }
    allocation.owner_exit()
    result = PeerRetirement(worker).drop_peer(PEER)
    assert result.state == "cleanup_returned"
    assert not result.native_release_verified
    assert not allocation.resident
    assert healthy.imports
    assert worker.nixl_wrapper.events == [
        ("release_dlist", f"{PEER}-dlist0"),
        ("release_dlist", f"{PEER}-dlist1"),
        ("remove_agent", f"{PEER}-agent0"),
        ("remove_agent", f"{PEER}-agent1"),
    ]
    for name in PEER_MAPS:
        assert PEER not in getattr(worker, name)
        assert getattr(worker, name)["p-healthy-generation"] == before[name]
    assert worker._topology_peers == {"p-healthy-generation"}
    assert worker.src_xfer_handles_by_block_size == {16: "local-source-do-not-release"}


@pytest.mark.parametrize(
    "reason",
    [
        "handshake_callbacks_pending",
        "ready_callbacks_pending",
        "failed_callbacks_pending",
        "send_side_not_drained",
        "target_receive_pending",
        "unattributed_receive_metadata",
        "unattributed_transfer",
        "unattributed_notification",
    ],
)
def test_drop_refuses_uncertain_or_busy_native_state(reason):
    worker, _ = make_worker()
    if reason == "handshake_callbacks_pending":
        worker._handshake_futures["another-peer"] = object()
    elif reason == "ready_callbacks_pending":
        worker._ready_requests.put(("request", recv_meta()))
    elif reason == "failed_callbacks_pending":
        worker._failed_recv_reqs.put("request")
    elif reason == "send_side_not_drained":
        worker._reqs_to_send["request"] = time.perf_counter()
    elif reason == "target_receive_pending":
        worker._recving_metadata["request"] = recv_meta()
    elif reason == "unattributed_receive_metadata":
        worker._recving_metadata["request"] = SimpleNamespace(remote=None)
    elif reason == "unattributed_transfer":
        worker._recving_transfers["orphan"] = ["live-native-handle"]
    else:
        worker._pending_recv_notifs["orphan"] = [("agent", b"notification")]
    result = PeerRetirement(worker).drop_peer(PEER)
    assert result.state == "busy"
    assert result.reason == reason
    assert worker.nixl_wrapper.events == []
    assert PEER in worker._remote_agents


def test_busy_drop_blocks_new_old_generation_requests_then_retries():
    worker, _ = make_worker()
    worker._recving_metadata["request"] = recv_meta()
    worker._recving_transfers["request"] = ["native-read"]
    controller = PeerRetirement(worker)
    assert controller.drop_peer(PEER).state == "busy"
    with pytest.raises(RuntimeError, match="retired"):
        controller.check_admission(PEER)
    controller.check_admission("p-new-generation-same-address")
    # Simulate normal completion processing, NOT a forced native cancellation.
    worker._recving_transfers.clear()
    worker._recving_metadata.clear()
    assert controller.drop_peer(PEER).state == "cleanup_returned"


def test_target_pending_notification_does_not_get_freed():
    worker, _ = make_worker()
    worker._recving_metadata["request"] = recv_meta()
    worker._pending_recv_notifs["request"] = [("agent", b"done")]
    assert PeerRetirement(worker).drop_peer(PEER).state == "busy"
    assert worker.nixl_wrapper.events == []


def test_finished_handshake_before_ready_callback_still_blocks_cleanup():
    worker, _ = make_worker()
    worker._recving_metadata["request"] = recv_meta()
    # The future was removed, but the per-request callback has not queued yet.
    assert not worker._handshake_futures
    assert worker._ready_requests.empty()
    result = PeerRetirement(worker).drop_peer(PEER)
    assert result.state == "busy"
    assert worker.nixl_wrapper.events == []


def test_pending_send_processing_without_expiry_still_blocks_cleanup():
    worker, _ = make_worker()
    worker._reqs_to_process.add("not-yet-expiring")
    result = PeerRetirement(worker).drop_peer(PEER)
    assert result.state == "busy"
    assert result.reason == "send_side_not_drained"


def test_one_busy_rank_prevents_pool_acknowledgement():
    workers = {name: make_worker(name)[0] for name in ["rank0", "rank1"]}
    workers["rank1"]._recving_metadata["pending"] = recv_meta()
    results = {
        name: PeerRetirement(worker).drop_peer(PEER) for name, worker in workers.items()
    }
    assert results["rank0"].state == "cleanup_returned"
    assert results["rank1"].state == "busy"
    assert not all_workers_acknowledged(results, set(workers))


def test_known_other_peer_receive_is_not_aborted():
    worker, _ = make_worker()
    worker._recving_metadata["healthy-request"] = recv_meta("healthy")
    worker._recving_transfers["healthy-request"] = ["healthy-read"]
    assert PeerRetirement(worker).drop_peer(PEER).state == "cleanup_returned"
    assert worker._recving_transfers["healthy-request"] == ["healthy-read"]


def test_idempotence_never_double_releases_native_handles():
    worker, _ = make_worker()
    controller = PeerRetirement(worker)
    result = controller.drop_peer(PEER)
    events = list(worker.nixl_wrapper.events)
    assert controller.drop_peer(PEER) == result
    assert worker.nixl_wrapper.events == events
    assert controller.drop_peer("unknown-generation").state == "absent"
    with pytest.raises(RuntimeError, match="retired"):
        controller.check_admission("unknown-generation")


@pytest.mark.parametrize("failure", ["release_dlist", "remove_agent"])
def test_partial_native_failure_is_not_misreported_as_absent_on_retry(failure):
    worker, allocation = make_worker()
    allocation.owner_exit()
    suffix = "dlist1" if failure == "release_dlist" else "agent1"
    worker.nixl_wrapper.fail_on = (failure, f"{PEER}-{suffix}")
    controller = PeerRetirement(worker)
    result = controller.drop_peer(PEER)
    assert result.state == "cleanup_failed"
    events = list(worker.nixl_wrapper.events)
    assert controller.drop_peer(PEER) == result
    assert worker.nixl_wrapper.events == events
    assert allocation.resident
    assert not all_workers_acknowledged({"rank0": result}, {"rank0"})


def test_partially_missing_python_state_fails_closed():
    worker, _ = make_worker()
    del worker._remote_agents[PEER]
    result = PeerRetirement(worker).drop_peer(PEER)
    assert result.state == "cleanup_failed"
    assert result.reason == "inconsistent_peer_state"
    assert worker.nixl_wrapper.events == []


def test_return_from_native_wrapper_does_not_prove_mapping_closed():
    worker, allocation = make_worker(delayed_close=True)
    allocation.owner_exit()
    result = PeerRetirement(worker).drop_peer(PEER)
    assert result.state == "cleanup_returned"
    assert not result.native_release_verified
    assert allocation.resident
    assert worker.nixl_wrapper.pending_closes
    worker.nixl_wrapper.progress()
    assert not allocation.resident


def test_all_decode_instances_and_ranks_must_release_their_imports():
    allocation = SimulatedAllocation()
    expected = {"d0-rank0", "d0-rank1", "d1-rank0", "d1-rank1"}
    workers = {
        name: make_worker(name, allocation=allocation)[0] for name in sorted(expected)
    }
    allocation.owner_exit()
    results = {}
    for name, worker in workers.items():
        results[name] = PeerRetirement(worker).drop_peer(PEER)
        assert all_workers_acknowledged(results, expected) == (len(results) == 4)
        assert allocation.resident == (len(results) < 4)


def test_acknowledgement_does_not_accept_mixed_generations_or_missing_ranks():
    results = {
        "rank0": DropResult(PEER, "absent", "test"),
        "rank1": DropResult("p-new-generation", "absent", "test"),
    }
    assert not all_workers_acknowledged(results, {"rank0", "rank1"})
    assert not all_workers_acknowledged({}, set())
    assert not all_workers_acknowledged(results, {"rank0", "rank1", "rank2"})


def test_off_worker_thread_call_cannot_race_with_connector():
    worker, _ = make_worker()
    controller = PeerRetirement(worker)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controller.drop_peer, PEER)
        with pytest.raises(RuntimeError, match="serialized"):
            future.result()
    assert worker.nixl_wrapper.events == []


@pytest.mark.parametrize("invalid_id", [None, "", " ", 123])
def test_invalid_id_cannot_release_resources(invalid_id):
    worker, _ = make_worker()
    with pytest.raises(ValueError, match="engine generation"):
        PeerRetirement(worker).drop_peer(invalid_id)
    assert worker.nixl_wrapper.events == []


def test_harness_refuses_unreviewed_source(tmp_path):
    source = tmp_path / "changed_worker.py"
    source.write_text("class NixlBaseConnectorWorker: pass\n")
    with pytest.raises(ValueError, match="Source changed"):
        load_worker_methods(source)
