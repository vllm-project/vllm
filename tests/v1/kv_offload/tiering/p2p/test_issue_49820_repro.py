# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #49820: a later lookup round's pinned supply
was destroyed when an earlier fetch round for the same kv_request_id
finalized. Rounds are now isolated by the wire ``round_seq``.

Core sequence (all one kv_request_id):
    Fetch A (round 0) -> transfer A in flight -> Lookup B (round 1)
    resolves and pins -> transfer A completes and finalizes -> Fetch B
    (round 1) must be served from round 1's supply.
"""

from tests.v1.kv_offload.tiering.p2p.test_sessions import (
    FakeParent,
    _activate,
    _make_session,
    _serve,
    _srv_outbound,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortFetchMsg,
    FetchMsg,
    LookupMsg,
    LookupRespMsg,
    TransferDoneMsg,
)

KV = "req-1"


def _send_lookup(conn, keys, round_seq):
    conn.enqueue(
        {
            TYPE_KEY: LookupMsg.TYPE,
            LookupMsg.KV_REQUEST_ID: KV,
            LookupMsg.KEYS: keys,
            LookupMsg.ROUND_SEQ: round_seq,
        }
    )


def _fetch(conn, keys, idxs, round_seq):
    conn.enqueue(
        {
            TYPE_KEY: FetchMsg.TYPE,
            FetchMsg.KV_REQUEST_ID: KV,
            FetchMsg.KEYS: keys,
            FetchMsg.BLOCK_INDEXES: idxs,
            FetchMsg.ROUND_SEQ: round_seq,
        }
    )


def _drive_round_0_then_lookup_1(cb, session, conn, transport):
    """Round 0 fetch with transfer in flight, then round 1 lookup resolves."""
    _send_lookup(conn, [b"hA"], 0)
    session.poll()
    _serve(session, cb)  # pins hA (job 1000) under round 0
    _fetch(conn, [b"hA"], [20], 0)
    session.poll()
    assert len(transport._transfers) == 1  # transfer A (tid 0) in flight

    cb.stored[b"hB"] = 8
    _send_lookup(conn, [b"hB"], 1)
    session.poll()
    _serve(session, cb)  # pins hB (job 1001) under round 1
    resp = [m for m in conn._sent if m[TYPE_KEY] == LookupRespMsg.TYPE][-1]
    assert resp[LookupRespMsg.HITS] == [True]  # promised hB to the consumer


def test_completing_round_0_must_not_destroy_round_1_supply():
    """Round 0's completion must neither settle round 1's pin job nor drop
    round 1's available supply before round 1's FetchMsg arrives."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_0_then_lookup_1(cb, session, conn, transport)

    transport._poll_done.append(0)  # transfer A completes
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)  # deferred finalize results

    assert any(s.job_id == 1000 and s.success for s in stores)  # round 0's job
    assert all(s.job_id != 1001 for s in stores), (
        f"round 1 pin job settled by round 0 finalize: {stores}"
    )
    out = _srv_outbound(session, KV)
    assert out is not None and b"hB" in out.available, (
        "round 1 supply destroyed by round 0 finalize"
    )


def test_round_1_fetch_served_after_round_0_finalizes():
    """Full two-round resolution: round 1's fetch transfers its own pinned
    block, and each round gets its own TransferDone carrying its round."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_0_then_lookup_1(cb, session, conn, transport)

    transport._poll_done.append(0)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    assert [s.job_id for s in stores] == [1000]

    _fetch(conn, [b"hB"], [21], 1)
    session.poll()
    assert len(transport._transfers) == 2
    _, (_, local, remote) = sorted(transport._transfers.items())[-1]
    assert local == [8] and remote == [21]  # round 1's pinned block

    transport._poll_done.append(1)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    assert [(s.job_id, s.success) for s in stores] == [(1001, True)]
    dones = [m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE]
    assert [
        (d[TransferDoneMsg.SUCCESS], d.get(TransferDoneMsg.ROUND_SEQ)) for d in dones
    ] == [(True, 0), (True, 1)]


def test_terminal_empty_fetch_leaves_no_zombie_round():
    """An all-miss round's terminal empty FetchMsg must not park a
    demand-received round: the next round's real fetch is not a
    duplicate, and the empty fetch itself sends no TransferDone."""
    cb = FakeParent(stored={})
    session, conn, transport = _make_session()
    _activate(session, conn)

    _send_lookup(conn, [b"hA"], 0)
    session.poll()
    _serve(session, cb)  # all-miss round; client closes with empty fetch
    sent_before = len(conn._sent)
    _fetch(conn, [], [], 0)
    session.poll()
    assert not [
        m for m in conn._sent[sent_before:] if m[TYPE_KEY] == TransferDoneMsg.TYPE
    ]
    assert session._server._requests.get(KV) is None  # fully pruned

    # Next round: producer now has the block; fetch must be served, not
    # rejected as a duplicate.
    cb.stored[b"hA"] = 7
    _send_lookup(conn, [b"hA"], 1)
    session.poll()
    _serve(session, cb)
    _fetch(conn, [b"hA"], [20], 1)
    session.poll()
    assert len(transport._transfers) == 1


def test_unmatched_symmetric_fetch_fails_fast():
    """A round-tagged fetch demanding keys its lookup round never pinned is
    unservable: the server must fail it immediately (TransferDone
    success=False) and settle the round's pins, not park the demand."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)

    _send_lookup(conn, [b"hA"], 0)
    session.poll()
    _serve(session, cb)  # pins hA (job 1000)
    sent_before = len(conn._sent)
    _fetch(conn, [b"hA", b"hX"], [20, 21], 0)  # hX was never pinned
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)

    dones = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == TransferDoneMsg.TYPE]
    assert [d[TransferDoneMsg.SUCCESS] for d in dones] == [False]
    assert len(transport._transfers) == 0  # no partial transfer
    assert [(s.job_id, s.success) for s in stores] == [(1000, False)]
    assert session._server._requests.get(KV) is None


def test_pd_fetch_keeps_fetch_before_store_flow():
    """A round without lookup-pinned supply (PD) parks unmatched demand:
    a later store fulfills it instead of the fetch failing fast."""
    session, conn, transport = _make_session()
    _activate(session, conn)
    _fetch(conn, [b"k1"], [5], 0)
    session.poll()
    assert len(transport._transfers) == 0  # parked, not failed
    session.add_stored_blocks(KV, [b"k1"], [3], job_id=1)
    assert len(transport._transfers) == 1


def test_abort_of_round_0_preserves_round_1_supply():
    """AbortFetchMsg for one round settles that round's jobs promptly and
    leaves the next round's parked supply intact."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_0_then_lookup_1(cb, session, conn, transport)

    conn.enqueue(
        {
            TYPE_KEY: AbortFetchMsg.TYPE,
            AbortFetchMsg.KV_REQUEST_ID: KV,
            AbortFetchMsg.ROUND_SEQ: 0,
        }
    )
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    # Round 0's job fails promptly (no 30s store-timeout leak)...
    assert [(s.job_id, s.success) for s in stores] == [(1000, False)]
    # ...and round 1's supply survives the abort.
    out = _srv_outbound(session, KV)
    assert out is not None and b"hB" in out.available

    _fetch(conn, [b"hB"], [21], 1)
    session.poll()
    # Round 1 served with its own pinned block (round 0's cancelled
    # transfer was removed from the fake transport by the abort drain).
    assert [local for _, local, _ in transport._transfers.values()] == [[8]]


def test_concurrent_loads_complete_by_round():
    """Several loads can be in flight per id (the scheduler submits loads
    incrementally); fetches carry their round and each TransferDone
    completes its own job."""
    session, conn, _ = _make_session()
    _activate(session, conn)
    client = session._client

    # Round 0: probe hA, resolve, fetch.
    client.register_lookup(KV, b"hA")
    client.flush_pending_lookups()
    client.on_lookup_resp(KV, [b"hA"], [True])
    session.request_blocks(1, KV, [b"hA"], [10])
    # Round 1 starts while round 0's load is still in flight.
    client.register_lookup(KV, b"hB")
    client.flush_pending_lookups()
    client.on_lookup_resp(KV, [b"hB"], [True])
    session.request_blocks(2, KV, [b"hB"], [11])

    fetches = [m for m in conn._sent if m[TYPE_KEY] == FetchMsg.TYPE]
    assert [f[FetchMsg.ROUND_SEQ] for f in fetches] == [0, 1]
    lookups = [m for m in conn._sent if m[TYPE_KEY] == LookupMsg.TYPE]
    assert [lu[LookupMsg.ROUND_SEQ] for lu in lookups] == [0, 1]

    # Completions arrive out of order and match by round.
    conn.enqueue(
        {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: KV,
            TransferDoneMsg.SUCCESS: True,
            TransferDoneMsg.ROUND_SEQ: 1,
        }
    )
    result = session.poll()
    assert [(r.job_id, r.success) for r in result.loads] == [(2, True)]
    conn.enqueue(
        {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: KV,
            TransferDoneMsg.SUCCESS: True,
            TransferDoneMsg.ROUND_SEQ: 0,
        }
    )
    result = session.poll()
    assert [(r.job_id, r.success) for r in result.loads] == [(1, True)]


def test_unfetched_keys_repinned_next_round():
    """Keys probed but not fetched in a round are re-probed under the next
    round (probes clear at fetch) and served from a fresh pin."""
    cb = FakeParent(stored={b"hA": 7, b"hB": 8})
    session, conn, transport = _make_session()
    _activate(session, conn)

    _send_lookup(conn, [b"hA", b"hB"], 0)
    session.poll()
    _serve(session, cb)  # one pin job (1000) supplies both keys
    _fetch(conn, [b"hA"], [20], 0)  # round 0 fetches only hA
    session.poll()
    transport._poll_done.append(0)
    session.poll()
    session.poll()

    # hB re-probed under round 1 pins fresh (job 1001) and is served.
    _send_lookup(conn, [b"hB"], 1)
    session.poll()
    _serve(session, cb)
    _fetch(conn, [b"hB"], [21], 1)
    session.poll()
    assert len(transport._transfers) == 2
    transport._poll_done.append(1)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    assert (1001, True) in [(s.job_id, s.success) for s in stores]


def test_finish_with_inflight_load_still_closes_lookup_state():
    """finish() with a load in flight must still send the terminal empty
    FetchMsg when a later round's LookupMsg re-opened the peer's lookup
    state — otherwise its pinned supply leaks until the store timeout."""
    session, conn, _ = _make_session()
    _activate(session, conn)
    client = session._client

    client.register_lookup(KV, b"hA")
    client.flush_pending_lookups()
    client.on_lookup_resp(KV, [b"hA"], [True])
    session.request_blocks(1, KV, [b"hA"], [10])  # round 0 load in flight
    client.register_lookup(KV, b"hB")
    client.flush_pending_lookups()  # round 1 lookup re-opens peer state

    session.finish_request(KV)
    aborts = [m for m in conn._sent if m[TYPE_KEY] == AbortFetchMsg.TYPE]
    assert [a[AbortFetchMsg.ROUND_SEQ] for a in aborts] == [0]
    terminal = [
        m for m in conn._sent if m[TYPE_KEY] == FetchMsg.TYPE and not m[FetchMsg.KEYS]
    ]
    assert len(terminal) == 1 and terminal[0][FetchMsg.ROUND_SEQ] == 1
