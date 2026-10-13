# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the simulated NIXL connector worker (no engine)."""

import time
from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.v1.worker.simulated.kv_transfer import (
    TransferRate,
)
from vllm.v1.worker.simulated.timing import (
    RealClock,
    StepLogWriter,
    VirtualClock,
)


def _nixl_worker(
    engine_id: str,
    clock,
    role="kv_consumer",
    port=None,
    mla=False,
    kv_heads=8,
    push=False,
    pp_rank=0,
    bytes_per_s=10.0,
    tp_rank=0,
    tp_size=1,
):
    from vllm.v1.worker.simulated.kv_nixl import (
        SimulatedNixlPushWorker,
        SimulatedNixlWorker,
    )

    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(page_size_bytes=10), layer_names=["layer"]
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[group], transfer_group_ids=(0,), transfer_groups=(group,)
    )
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(kv_connector="NixlConnector", kv_role=role),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            use_mla=mla, get_total_num_kv_heads=lambda: kv_heads
        ),
    )
    worker_cls = SimulatedNixlPushWorker if push else SimulatedNixlWorker
    nixl = worker_cls(config, engine_id, kv_cache_config)
    nixl.pp_rank, nixl.tp_rank, nixl.tp_size = pp_rank, tp_rank, tp_size
    if port is not None:
        nixl.side_channel_port = port
    nixl.configure(clock, TransferRate(latency_s=0.0, bytes_per_s=bytes_per_s))
    return nixl


def test_simulated_nixl_reads_notify_the_prefill_instance():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )

    clock = VirtualClock()
    prefill = _nixl_worker("sim-p", clock)
    decode = _nixl_worker("sim-d", clock)
    sent = NixlConnectorMetadata()
    sent.reqs_in_batch = {"p1", "p2"}
    prefill.start_load_kv(sent)

    def recv(blocks, remote_req, awaiting=True):
        return ReqMeta(
            local_block_ids=(blocks,),
            local_physical_block_ids=(blocks,),
            tp_size=1,
            awaiting_kvs=awaiting,
            remote=RemoteMeta(blocks, "", 0, "sim-p", remote_req),
        )

    meta = NixlConnectorMetadata()
    # 20 and 10 bytes at 10 B/s, serial on the link; r3 only notifies, about
    # a request the prefill instance no longer tracks.
    meta.reqs_to_recv = {
        "r1": recv([0, 1], "p1"),
        "r2": recv([2], "p2"),
        "r3": recv([], "p3", awaiting=False),
    }
    decode.start_load_kv(meta)
    assert decode.next_completion() == pytest.approx(2.0)

    assert decode.get_finished() == (set(), set())
    assert prefill.get_finished() == (set(), set())
    clock.sleep_until(2.5)
    assert decode.get_finished() == (set(), {"r1"})
    assert prefill.get_finished() == ({"p1"}, set())
    clock.sleep_until(3.0)
    assert decode.get_finished() == (set(), {"r2"})
    assert prefill.get_finished() == ({"p2"}, set())
    assert decode.next_completion() is None and prefill.next_completion() is None


def test_nixl_read_posting_cost(tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )
    from vllm.v1.worker.simulated.kv_transfer import (
        PostCost,
        post_cost_from_log,
        read_runs,
    )
    from vllm.v1.worker.simulated.timing import load_step_log

    # Runs of blocks consecutive on both sides; on a local prefix hit only the
    # last remote blocks are read.
    assert read_runs(([4, 5, 6],), ([7, 8, 9],)) == 1
    assert read_runs(([4, 5, 9],), ([7, 8, 9],)) == 2
    assert read_runs(([5, 6],), ([1, 2, 8],)) == 2

    path = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(path))
    cost = PostCost(per_read_s=200e-6, per_run_s=100e-6)
    for reads, runs in [(1, 1), (2, 2), (1, 128), (4, 512)]:
        writer.write_kv_post(reads, runs, cost.seconds(reads, runs), t=0.0)
    # A first posting to new memory, far slower than the rest: left out.
    writer.write_kv_post(1, 100, 0.5, t=0.0)
    writer.close()
    fitted = post_cost_from_log(str(path))
    assert fitted.per_read_s == pytest.approx(cost.per_read_s)
    assert fitted.per_run_s == pytest.approx(cost.per_run_s)
    assert load_step_log(str(path)) == []

    # Posting holds the decode rank's engine thread, and each request's reads
    # start once posted: 0.75 s for r1 (one run), then r2's.
    clock = VirtualClock()
    prefill = _nixl_worker("sim-post-p", clock)  # noqa: F841 (held weakly)
    decode = _nixl_worker("sim-post-d", clock)
    decode.configure(
        clock,
        TransferRate(latency_s=0.0, bytes_per_s=10.0),
        PostCost(per_read_s=0.5, per_run_s=0.25),
    )
    meta = NixlConnectorMetadata()
    meta.reqs_to_recv = {
        req_id: ReqMeta(
            local_block_ids=(local,),
            local_physical_block_ids=(local,),
            tp_size=1,
            awaiting_kvs=True,
            remote=RemoteMeta((remote,), "", 0, "sim-post-p", f"p-{req_id}"),
        )
        for req_id, local, remote in [("r1", [0, 1], [0, 1]), ("r2", [2], [5])]
    }
    decode.start_load_kv(meta)
    assert clock.now() == pytest.approx(1.5)
    # r1's 20 bytes from 0.75 s; r2's 10 bytes queue behind them on the link.
    assert decode.next_completion() == pytest.approx(2.75)
    clock.sleep_until(3.75)
    assert decode.get_finished() == (set(), {"r1", "r2"})


def test_simulated_nixl_books_reads_across_processes(monkeypatch):
    """A served decode rank books its read with the prefill rank over ZMQ,
    and fails the request when the prefill rank does not answer."""
    import zmq

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.simulated import kv_nixl

    base = get_open_port() - 10_000
    prefill = _nixl_worker("sim-served-p", RealClock(), "kv_producer", base)
    decode = _nixl_worker("sim-served-d", RealClock(), "kv_consumer")
    # Served instances run in separate processes.
    monkeypatch.delitem(kv_nixl._workers, "sim-served-p")

    def recv(req_id, remote_req):
        meta = NixlConnectorMetadata()
        meta.reqs_to_recv = {
            req_id: ReqMeta(
                local_block_ids=([0, 1],),
                local_physical_block_ids=([0, 1],),
                tp_size=1,
                awaiting_kvs=True,
                remote=RemoteMeta(
                    [0, 1], "localhost", base, "sim-served-p", remote_req
                ),
            )
        }
        return meta

    try:
        t0 = time.monotonic()
        decode.start_load_kv(recv("r1", "p1"))
        # 20 bytes at 10 B/s; the bookings' round trips are local.
        assert decode.next_completion() == pytest.approx(t0 + 2.0, abs=0.5)
        assert prefill.next_completion() == pytest.approx(t0 + 2.0, abs=0.5)

        # A malformed booking is rejected; the prefill rank keeps serving.
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.connect(f"tcp://localhost:{base + 10_000}")
        sock.send_json({"req": "x"})
        assert sock.recv_json() is None
        sock.close(linger=0)
        decode.start_load_kv(recv("r2", "p2"))
        assert decode.next_completion() == pytest.approx(t0 + 2.0, abs=0.5)

        # A prefill rank that does not answer fails the read.
        monkeypatch.setattr(kv_nixl, "_BOOKING_TIMEOUT_MS", 100)
        prefill.shutdown()
        decode.start_load_kv(recv("r3", "p3"))
        assert decode.get_transfer_results().failed_recving == {"r3"}
    finally:
        prefill.shutdown()
        decode.shutdown()


def test_simulated_nixl_frees_blocks_after_all_readers():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    clock = VirtualClock()
    prefill = _nixl_worker("sim-two-readers", clock)
    meta = NixlConnectorMetadata()
    meta.reqs_in_batch = {"p1"}
    prefill.start_load_kv(meta)
    # Two decode ranks each read half of this rank's heads, concurrently;
    # the blocks are freed once both have.
    assert prefill.book_read("p1", 10, 0.0, num_readers=2) == pytest.approx(1.0)
    assert prefill.next_completion() == pytest.approx(1.0)
    assert prefill.get_finished() == (set(), set())
    assert prefill.book_read("p1", 20, 0.0, num_readers=2) == pytest.approx(2.0)
    clock.sleep_until(1.5)
    assert prefill.get_finished() == (set(), set())
    clock.sleep_until(2.0)
    assert prefill.get_finished() == ({"p1"}, set())
    # Another request's read waits for the link.
    assert prefill.book_read("p2", 10, 0.0) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("prefill_tp", "decode_tp", "kv_heads"),
    [(2, 2, 8), (2, 4, 8), (4, 2, 8), (2, 4, 2), (4, 2, 2)],
)
def test_simulated_nixl_notifies_every_prefill_rank(prefill_tp, decode_tp, kv_heads):
    """Whether the layout is accepted or not, every prefill rank is notified
    by as many decode ranks as it waits for, so it frees the blocks."""
    worker = _nixl_worker(
        f"sim-notify-{prefill_tp}-{decode_tp}-{kv_heads}",
        VirtualClock(),
        kv_heads=kv_heads,
    )
    notified = [0] * prefill_tp
    readers = 1
    for rank in range(decode_tp):
        worker.tp_rank, worker.tp_size = rank, decode_tp
        for r in worker._paired_ranks(prefill_tp):
            notified[r] += 1
        if (mapping := worker._mapping(prefill_tp)) is not None:
            readers = mapping.local_consumers
    assert notified == [readers] * prefill_tp


def test_simulated_nixl_push_writes_from_every_prefill_stage():
    """In push mode a decode rank registers with each prefill stage, which
    writes its layers once the request is finished; the decode rank is done
    after both stages' writes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )
    from vllm.utils.network_utils import get_open_port

    p_base = get_open_port() - 10_000
    d_base = p_base + 100

    def worker(name, role, base, pp_rank=0):
        return _nixl_worker(
            name, RealClock(), role, base, push=True, pp_rank=pp_rank, bytes_per_s=100.0
        )

    # The two stages of one prefill instance, and a decode instance.
    stages = [worker(f"push-p{i}", "kv_producer", p_base, i) for i in (0, 1)]
    decode = worker("push-d", "kv_consumer", d_base)

    meta = NixlConnectorMetadata()
    meta.reqs_to_recv = {
        "d1-aaaaaaaa": ReqMeta(
            local_block_ids=([0, 1],),
            local_physical_block_ids=([0, 1],),
            tp_size=1,
            pp_size=2,
            remote=RemoteMeta([], "localhost", p_base, "push-p", "base"),
        )
    }
    meta.push_registrations = {
        "d1-aaaaaaaa": {
            "decode_host": "localhost",
            "decode_port": d_base,
            "decode_tp_size": 1,
            "local_block_ids": ([0, 1],),
            "remote_host": "localhost",
            "remote_port": p_base,
            "remote_tp_size": 1,
            "remote_pp_size": 2,
        }
    }
    decode.start_load_kv(meta)
    # The stages finish the request 0.1 s apart; each writes the 2 blocks the
    # decode instance has (20 bytes at 100 B/s) as it does.
    started = []
    for stage in stages:
        finished = NixlConnectorMetadata()
        finished.push_finished_blocks = {"d1-bbbbbbbb": ([5, 6, 7],)}
        started.append(time.monotonic())
        stage.start_load_kv(finished)
        time.sleep(0.1)
    first_end, last_end = started[0] + 0.2, started[1] + 0.2
    assert stages[0].next_completion() == pytest.approx(first_end, abs=0.02)
    # Notifications carry durations, so a new connection's setup delays them.
    assert decode.next_completion() == pytest.approx(last_end, abs=0.08)
    time.sleep(max(0.0, first_end + 0.02 - time.monotonic()))
    assert stages[0].get_finished() == ({"d1-bbbbbbbb"}, set())
    assert decode.get_finished() == (set(), set())
    time.sleep(max(0.0, last_end + 0.1 - time.monotonic()))
    assert decode.get_finished() == (set(), {"d1-aaaaaaaa"})
    for w in (decode, *stages):
        w.shutdown()


@pytest.mark.parametrize("finish_first", [True, False])
def test_simulated_nixl_push_with_more_decode_ranks(finish_first):
    """With more decode ranks than prefill ranks, the prefill rank writes to
    all of them once the request is finished and one has registered: a decode
    rank that schedules the request only afterwards still completes, and
    registrations sent before the request finished leave nothing behind."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )
    from vllm.utils.network_utils import get_open_port

    p_base = get_open_port() - 10_000
    d_base = p_base + 100

    def worker(name, role, base, tp_rank=0, tp_size=1):
        return _nixl_worker(
            name,
            RealClock(),
            role,
            base,
            push=True,
            bytes_per_s=1000.0,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    prefill = worker("push-hp", "kv_producer", p_base)
    # Decode TP=2: each rank listens at its own port.
    decode = [worker(f"push-hd{r}", "kv_consumer", d_base, r, 2) for r in (0, 1)]
    try:

        def schedule(d):
            meta = NixlConnectorMetadata()
            meta.reqs_to_recv = {
                "d-aaaaaaaa": ReqMeta(
                    local_block_ids=([0, 1],),
                    local_physical_block_ids=([0, 1],),
                    tp_size=1,
                    remote=RemoteMeta([], "localhost", p_base, "push-hp", "x"),
                )
            }
            meta.push_registrations = {
                "d-aaaaaaaa": {
                    "decode_host": "localhost",
                    "decode_port": d_base,
                    "decode_tp_size": 2,
                    "local_block_ids": ([0, 1],),
                    "remote_host": "localhost",
                    "remote_port": p_base,
                    "remote_tp_size": 1,
                    "remote_pp_size": 1,
                }
            }
            d.start_load_kv(meta)

        finished = NixlConnectorMetadata()
        finished.push_finished_blocks = {"d-bbbbbbbb": ([5, 6],)}
        if finish_first:
            prefill.start_load_kv(finished)
            schedule(decode[0])  # matches, and writes to both decode ranks
            time.sleep(0.05)
            schedule(decode[1])  # its write is already in
        else:
            schedule(decode[0])
            schedule(decode[1])
            prefill.start_load_kv(finished)  # matches both registrations
        time.sleep(0.05)
        for d in decode:
            assert d.get_finished() == (set(), {"d-aaaaaaaa"})
        assert prefill._registrations == {}
        assert prefill._registration_copies == {}
        assert prefill._late_registrations == {}
    finally:
        for w in (*decode, prefill):
            w.shutdown()


def test_simulated_nixl_push_fails_rejected_layouts():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )

    decode = _nixl_worker("push-reject", RealClock(), "kv_consumer", push=True)
    try:
        decode.tp_size = 2
        meta = NixlConnectorMetadata()
        meta.reqs_to_recv = {
            "r1": ReqMeta(
                local_block_ids=([0],),
                local_physical_block_ids=([0],),
                tp_size=3,
                remote=RemoteMeta([], "localhost", 1, "x", "r1"),
            )
        }
        decode.start_load_kv(meta)
        assert decode.get_transfer_results().failed_recving == {"r1"}
    finally:
        decode.shutdown()
