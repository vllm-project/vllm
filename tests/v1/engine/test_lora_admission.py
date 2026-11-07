# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import msgspec
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.v1.engine.lora_admission import GroupLoRAAdmission, gather_metadata
from vllm.v1.request import Request, RequestStatus


def request(adapter: int, arrival: float = 0, name: str | None = None) -> Request:
    return Request(
        request_id=name or f"req-{adapter}",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=arrival,
        lora_request=LoRARequest(f"adapter-{adapter}", adapter, f"/adapters/{adapter}"),
    )


def publish(admission, action, workers=4):
    assert action[0] == "prepare"
    fingerprints = {
        (
            adapter := msgspec.msgpack.decode(item, type=LoRARequest).lora_int_id
        ): f"hash-{adapter}"
        for item in action[1]
    }
    admission.publish(
        action,
        [
            {"error": None, "fingerprints": fingerprints, "result": True}
            for _ in range(workers)
        ],
    )


def test_global_union_is_bounded_and_other_adapters_make_progress():
    engines = [GroupLoRAAdmission(2) for _ in range(2)]
    requests = [request(i, i) for i in range(1, 5)]
    for engine, pending in zip(engines, (requests[:2], requests[2:])):
        for req in pending:
            engine.track(req)
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        action = engine.plan(frames)
        publish(engine, action)
        assert engine.admitted == {1, 2}
    requests[0].status = RequestStatus.RUNNING
    frames = [engines[0].snapshot([requests[0]]), engines[1].snapshot([])]
    for engine in engines:
        assert engine.plan(frames) is None
        assert engine.admitted == {2}
    for req in requests[:2]:
        req.status = RequestStatus.FINISHED_STOPPED
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        publish(engine, engine.plan(frames))
        assert engine.admitted == {3, 4}


@pytest.mark.parametrize("failure", ["missing", "weights", "load"])
def test_partial_worker_readiness_cannot_publish(failure):
    admission = GroupLoRAAdmission(2)
    req = request(1)
    admission.track(req)
    action = admission.plan([admission.snapshot([])])
    reports: list[dict] = [
        {"error": None, "fingerprints": {1: "same"}, "result": True},
        {"error": None, "fingerprints": {1: "same"}, "result": True},
    ]
    if failure == "missing":
        reports[1]["fingerprints"] = {}
    elif failure == "weights":
        reports[1]["fingerprints"][1] = "different"
    else:
        reports[1]["error"] = "injected load failure"
    with pytest.raises(RuntimeError):
        admission.publish(action, reports)
    assert not admission.ready and not admission.admitted


def test_mutation_waits_for_all_engines_and_running_requests():
    engines = [GroupLoRAAdmission(2) for _ in range(2)]
    req = request(1)
    for engine in engines:
        engine.track(req)
        publish(engine, engine.plan([engine.snapshot([])]))
    futures = [
        engine.enqueue("load-2", "add_lora", request(2).lora_request)
        for engine in engines
    ]
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        assert engine.plan(frames) is None
    req.status = RequestStatus.RUNNING
    frames = [engines[0].snapshot([req]), engines[1].snapshot([])]
    for engine in engines:
        assert engine.plan(frames) is None
    assert not any(future.done() for future in futures)
    req.status = RequestStatus.FINISHED_STOPPED
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        action = engine.plan(frames)
        assert action[0] == "add_lora"
        engine.publish(
            action, [{"error": None, "fingerprints": {2: "weight"}, "result": True}]
        )
    assert all(future.result() for future in futures)


def test_pinned_slot_counts_against_the_admitted_union():
    admission = GroupLoRAAdmission(2)
    pinned = request(10).lora_request
    admission.identities[10] = msgspec.msgpack.encode(pinned)
    admission.pinned.add(10)
    pending = [request(1, 1), request(2, 2)]
    for req in pending:
        admission.track(req)
    publish(admission, admission.plan([admission.snapshot([])]))
    assert admission.admitted == {10, 1}


def test_stale_adapter_heads_are_removed_after_completion():
    admission = GroupLoRAAdmission(1)
    late, early = request(1, 3, "late"), request(1, 1, "early")
    other = request(2, 2)
    for req in (late, early, other):
        admission.track(req)
    early.status = RequestStatus.FINISHED_STOPPED
    publish(admission, admission.plan([admission.snapshot([])]))
    assert admission.admitted == {2}
    late.status = other.status = RequestStatus.FINISHED_STOPPED
    assert admission.snapshot([])["candidates"] == []


def _metadata_rank(rank, rendezvous):
    store = dist.FileStore(rendezvous, 2)
    group = dist.ProcessGroupGloo(store, rank, 2, timeout=timedelta(seconds=30))
    data = {"rank": rank, "adapter": bytes([rank]) * (rank + 1)}
    assert gather_metadata(group, data) == [
        {"rank": 0, "adapter": b"\x00"},
        {"rank": 1, "adapter": b"\x01\x01"},
    ]


def test_metadata_gather_uses_the_actual_stateless_group(tmp_path):
    mp.spawn(_metadata_rank, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_blocked_adapter_does_not_starve_ready_requests():
    admission = GroupLoRAAdmission(1)
    blocked, ready = request(1, 1), request(2, 2)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    admission.track(blocked)
    admission.track(ready)
    publish(admission, admission.plan([admission.snapshot([])]))
    assert admission.admitted == {2}


def test_blocked_request_does_not_hide_a_ready_request_for_the_same_adapter():
    admission = GroupLoRAAdmission(1)
    blocked, ready = request(1, 1, "blocked"), request(1, 2, "ready")
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    admission.track(blocked)
    admission.track(ready)
    publish(admission, admission.plan([admission.snapshot([])]))
    assert admission.admitted == {1}


def test_a_free_slot_can_be_reused_without_draining_an_unrelated_request():
    admission = GroupLoRAAdmission(2)
    active, finished, waiting = [request(i, i) for i in range(1, 4)]
    for req in (active, finished):
        admission.track(req)
    publish(admission, admission.plan([admission.snapshot([])]))
    active.status = RequestStatus.RUNNING
    finished.status = RequestStatus.FINISHED_STOPPED
    admission.track(waiting)
    publish(admission, admission.plan([admission.snapshot([active])]))
    assert admission.ready == {1, 3}
    assert active.status == RequestStatus.RUNNING


def _register(engines, operation, lora):
    futures = [engine.enqueue(operation, "add_lora", lora) for engine in engines]
    frames = [engine.snapshot([]) for engine in engines]
    assert all(engine.plan(frames) is None for engine in engines)
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        action = engine.plan(frames)
        canonical = msgspec.msgpack.decode(action[1], type=LoRARequest)
        engine.publish(
            action,
            [
                {
                    "error": None,
                    "result": True,
                    "fingerprints": {canonical.lora_int_id: "same"},
                }
            ],
        )
    return [future.result() for future in futures]


def test_frontend_local_id_collisions_receive_group_wide_ids():
    engines = [GroupLoRAAdmission(2) for _ in range(2)]
    assert _register(engines, "first", LoRARequest("one", 1, "/one")) == [1, 1]
    assert _register(engines, "second", LoRARequest("two", 1, "/two")) == [2, 2]
    assert _register(engines, "duplicate", LoRARequest("one", 99, "/one")) == [1, 1]


def test_rejected_load_restores_admission_and_completes_every_call():
    engines = [GroupLoRAAdmission(2) for _ in range(2)]
    req = request(1)
    for engine in engines:
        engine.track(req)
        publish(engine, engine.plan([engine.snapshot([])]))
    futures = [
        engine.enqueue("bad", "add_lora", request(2).lora_request) for engine in engines
    ]
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        assert engine.plan(frames) is None
    frames = [engine.snapshot([]) for engine in engines]
    for engine in engines:
        assert not engine.reject(engine.plan(frames), "checkpoint missing on rank 1")
        assert engine.ready == engine.admitted == {1}
        assert not engine.pending and engine.target is None
    for future in futures:
        with pytest.raises(ValueError, match="previous epoch restored"):
            future.result()


def test_failed_promotion_does_not_retry_or_block_other_adapters():
    engine = GroupLoRAAdmission(2)
    active, bad, other = [request(i, i) for i in range(1, 4)]
    engine.track(active)
    publish(engine, engine.plan([engine.snapshot([])]))
    active.status = RequestStatus.RUNNING
    engine.track(bad)
    action = engine.plan([engine.snapshot([active])])
    assert engine.reject(action, "bad weights") == {2}
    assert engine.admitted == {1}
    engine.track(other)
    publish(engine, engine.plan([engine.snapshot([active])]))
    assert engine.ready == {1, 3}


def test_an_adapter_cannot_silently_change_weights_between_epochs():
    engine = GroupLoRAAdmission(1)
    first = request(1)
    engine.track(first)
    publish(engine, engine.plan([engine.snapshot([])]))
    encoded = msgspec.msgpack.encode(first.lora_request)
    with pytest.raises(RuntimeError, match="immutable weights"):
        engine.validate(
            ("prepare", [encoded]), [{"error": None, "fingerprints": {1: "changed"}}]
        )
