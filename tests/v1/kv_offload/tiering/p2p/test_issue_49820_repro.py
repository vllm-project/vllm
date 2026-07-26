# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repro for issue #49820: a later lookup round's pinned supply is destroyed
when an earlier fetch round for the same kv_request_id finalizes.

Sequence (wire-ordered, all one kv_request_id):
    Fetch A -> transfer A in flight -> Lookup B resolves (pins, merged into
    A's outbound) -> transfer A completes (finalize wipes outbound, settles
    B's pin job) -> Fetch B (matches nothing, no transfer, no terminal).
"""

from tests.v1.kv_offload.tiering.p2p.test_sessions import (
    FakeParent,
    _activate,
    _make_session,
    _send_lookup,
    _serve,
    _srv_outbound,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortFetchMsg,
    FetchMsg,
    LookupRespMsg,
    TransferDoneMsg,
)

KV = "req-1"


def _fetch(conn, keys, idxs):
    conn.enqueue(
        {
            TYPE_KEY: FetchMsg.TYPE,
            FetchMsg.KV_REQUEST_ID: KV,
            FetchMsg.KEYS: keys,
            FetchMsg.BLOCK_INDEXES: idxs,
        }
    )


def _drive_round_a_then_lookup_b(cb, session, conn, transport):
    """Round A fetch with transfer in flight, then round B lookup resolves."""
    _send_lookup(conn, KV, [b"hA"])
    session.poll()
    _serve(session, cb)  # pins hA (job 1000)
    _fetch(conn, [b"hA"], [20])
    session.poll()
    assert len(transport._transfers) == 1  # transfer A (tid 0) in flight

    cb.stored[b"hB"] = 8
    _send_lookup(conn, KV, [b"hB"])
    session.poll()
    _serve(session, cb)  # pins hB (job 1001), merged into A's outbound
    resp = [m for m in conn._sent if m[TYPE_KEY] == LookupRespMsg.TYPE][-1]
    assert resp[LookupRespMsg.HITS] == [True]  # promised hB to the consumer


def test_completing_round_a_must_not_destroy_round_b_supply():
    """Round A's completion must neither settle round B's pin job nor drop
    round B's available supply before round B's FetchMsg arrives."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_a_then_lookup_b(cb, session, conn, transport)

    transport._poll_done.append(0)  # transfer A completes
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)  # deferred finalize results

    assert any(s.job_id == 1000 and s.success for s in stores)  # A's own job
    # Round B's pin job (1001) must not be settled by round A's completion.
    assert all(s.job_id != 1001 for s in stores), (
        f"round B pin job settled by round A finalize: {stores}"
    )
    # Round B's supply must still await round B's fetch.
    out = _srv_outbound(session, KV)
    assert out is not None and b"hB" in out.available, (
        "round B supply discarded by round A finalize"
    )


def test_round_b_fetch_is_served_or_promptly_terminal():
    """After round A finalizes, round B's fetch for its confirmed hit must
    either submit a transfer or reach a prompt terminal failure. Today it
    does neither and the consumer waits out _LOAD_TIMEOUT_S (30s)."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_a_then_lookup_b(cb, session, conn, transport)

    transport._poll_done.append(0)
    session.poll()
    session.poll()
    sent_before = len(conn._sent)

    _fetch(conn, [b"hB"], [21])  # round B fetches its confirmed hit
    session.poll()

    transferred = len(transport._transfers) == 2
    terminal = [
        m for m in conn._sent[sent_before:] if m[TYPE_KEY] == TransferDoneMsg.TYPE
    ]
    assert transferred or terminal, (
        "round B fetch neither served nor terminally failed: "
        "consumer stalls for _LOAD_TIMEOUT_S"
    )


def test_cross_round_fetch_b_served_after_a_finalizes():
    """Full two-round resolution: round B's fetch transfers its own pinned
    blocks after round A completed, and each round gets its own success
    TransferDone and StoreResult."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_a_then_lookup_b(cb, session, conn, transport)

    transport._poll_done.append(0)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    assert [s.job_id for s in stores] == [1000]

    _fetch(conn, [b"hB"], [21])
    session.poll()
    assert len(transport._transfers) == 2
    _, (_, local, remote) = sorted(transport._transfers.items())[-1]
    assert local == [8] and remote == [21]  # round B's pinned block

    transport._poll_done.append(1)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    assert [(s.job_id, s.success) for s in stores] == [(1001, True)]
    dones = [m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE]
    assert [d[TransferDoneMsg.SUCCESS] for d in dones] == [True, True]


def test_terminal_empty_fetch_leaves_no_zombie_round():
    """An all-miss round's terminal empty FetchMsg must not park a
    demand-received round: the next round's real fetch is not a
    duplicate, and the empty fetch itself sends no TransferDone."""
    cb = FakeParent(stored={})
    session, conn, transport = _make_session()
    _activate(session, conn)

    _send_lookup(conn, KV, [b"hA"])
    session.poll()
    _serve(session, cb)  # all-miss round; client will close with empty fetch
    sent_before = len(conn._sent)
    _fetch(conn, [], [])
    session.poll()
    assert not [
        m for m in conn._sent[sent_before:] if m[TYPE_KEY] == TransferDoneMsg.TYPE
    ]
    assert session._server._requests.get(KV) is None  # fully pruned

    # Next round: producer now has the block; fetch must be served, not
    # rejected as a duplicate.
    cb.stored[b"hA"] = 7
    _send_lookup(conn, KV, [b"hA"])
    session.poll()
    _serve(session, cb)
    _fetch(conn, [b"hA"], [20])
    session.poll()
    assert len(transport._transfers) == 1


def test_unmatched_symmetric_fetch_fails_fast():
    """A fetch demanding keys its closed lookup phase never pinned is
    unservable: the server must fail it immediately (TransferDone
    success=False) and settle the round's pins, not park the demand."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)

    _send_lookup(conn, KV, [b"hA"])
    session.poll()
    _serve(session, cb)  # pins hA (job 1000)
    sent_before = len(conn._sent)
    _fetch(conn, [b"hA", b"hX"], [20, 21])  # hX was never pinned
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)

    dones = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == TransferDoneMsg.TYPE]
    assert [d[TransferDoneMsg.SUCCESS] for d in dones] == [False]
    assert len(transport._transfers) == 0  # no partial transfer
    assert [(s.job_id, s.success) for s in stores] == [(1000, False)]
    assert session._server._requests.get(KV) is None


def test_abort_of_round_a_preserves_round_b_supply():
    """AbortFetchMsg for the in-flight round settles that round's jobs
    promptly and leaves the next round's parked supply intact."""
    cb = FakeParent(stored={b"hA": 7})
    session, conn, transport = _make_session()
    _activate(session, conn)
    _drive_round_a_then_lookup_b(cb, session, conn, transport)

    conn.enqueue({TYPE_KEY: AbortFetchMsg.TYPE, AbortFetchMsg.KV_REQUEST_ID: KV})
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    # Round A's job fails promptly (no 30s store-timeout leak)...
    assert [(s.job_id, s.success) for s in stores] == [(1000, False)]
    # ...and round B's supply survives the abort.
    st = session._server._requests.get(KV)
    assert st is not None and st.supply is not None
    assert b"hB" in st.supply.available

    _fetch(conn, [b"hB"], [21])
    session.poll()
    # Round B served with its own pinned block (round A's cancelled
    # transfer was removed from the fake transport by the abort drain).
    assert [local for _, local, _ in transport._transfers.values()] == [[8]]


def test_overlapping_loads_serialize_fetches():
    """A second load submitted while one is in flight must not put a
    second FetchMsg on the wire; it goes out when the first load
    completes, and each TransferDone completes its own job."""
    session, conn, _ = _make_session()
    _activate(session, conn)

    session.request_blocks(1, KV, [b"k1"], [10])
    session.request_blocks(2, KV, [b"k2"], [11])
    fetches = [m for m in conn._sent if m[TYPE_KEY] == FetchMsg.TYPE]
    assert len(fetches) == 1 and fetches[0][FetchMsg.KEYS] == [b"k1"]

    conn.enqueue(
        {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: KV,
            TransferDoneMsg.SUCCESS: True,
        }
    )
    result = session.poll()
    assert [(r.job_id, r.success) for r in result.loads] == [(1, True)]
    fetches = [m for m in conn._sent if m[TYPE_KEY] == FetchMsg.TYPE]
    assert len(fetches) == 2 and fetches[1][FetchMsg.KEYS] == [b"k2"]

    conn.enqueue(
        {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: KV,
            TransferDoneMsg.SUCCESS: True,
        }
    )
    result = session.poll()
    assert [(r.job_id, r.success) for r in result.loads] == [(2, True)]


def test_lookahead_supply_survives_intermediate_fetch():
    """One lookup round can pin supply for several upcoming fetches. A
    fetch consuming a subset must not settle or drop the remainder: the
    next fetch is served from the moved-back supply."""
    cb = FakeParent(stored={b"hA": 7, b"hB": 8})
    session, conn, transport = _make_session()
    _activate(session, conn)

    _send_lookup(conn, KV, [b"hA", b"hB"])
    session.poll()
    _serve(session, cb)  # one pin job (1000) supplies both keys

    _fetch(conn, [b"hA"], [20])  # first fetch consumes only hA
    session.poll()
    transport._poll_done.append(0)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    # Job 1000 still has hB parked, so it must not settle yet.
    assert stores == []

    _fetch(conn, [b"hB"], [21])
    session.poll()
    assert len(transport._transfers) == 2
    transport._poll_done.append(1)
    stores = list(session.poll().stores)
    stores += list(session.poll().stores)
    assert [(s.job_id, s.success) for s in stores] == [(1000, True)]
