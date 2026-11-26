# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy GrantTable: transitions, refusals and invariants I1-I5, I9."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vllm.v1.core.grant_table import (
    NO_OWNER,
    NULL_BLOCK_ID,
    NULL_GEN,
    BlockState,
    GrantError,
    GrantTable,
    row_digest,
    row_digest_term,
    row_digest_terms,
)

BS = 4


def make_table(num_blocks: int = 8, caching: bool = True) -> GrantTable:
    return GrantTable(num_blocks, [BS], enable_caching=caching)


def test_initial_state():
    t = make_table()
    assert t.state(NULL_BLOCK_ID) == BlockState.NULL
    assert t.gen(NULL_BLOCK_ID) == NULL_GEN
    assert all(t.state(b) == BlockState.FREE for b in range(1, 8))
    assert all(t.gen(b) == 1 for b in range(1, 8))
    assert 28 <= t.host_bytes_per_block <= 36


def test_t1_grant():
    t = make_table()
    gens = t.grant([1, 2], owner=7, group=0, domain=3)
    assert gens.tolist() == [1, 1]
    for b in (1, 2):
        assert t.state(b) == BlockState.EXCLUSIVE
        assert (t.owner(b), t.readers(b), t.valid(b), t.domain(b)) == (7, 1, 0, 3)
    with pytest.raises(GrantError):
        t.grant([2, 3], owner=8, group=0, domain=0)  # 2 is not FREE
    assert t.state(3) == BlockState.FREE  # all-or-nothing


def test_check_writes():
    t = make_table()
    (g,) = t.grant([1], owner=1, group=0, domain=0)
    assert t.check_writes(1, [1], [g])
    assert not t.check_writes(2, [1], [g])  # wrong owner (I2)
    assert not t.check_writes(1, [1], [g - 1])  # stale gen (I6)
    assert not t.check_writes(1, [2], [1])  # not granted
    assert not t.check_writes(1, [NULL_BLOCK_ID], [NULL_GEN])  # null block
    assert t.find_refused_writes(1, [1, 2], [g, 1]).tolist() == [2]
    assert t.check_writes(1, [], [])


def test_t2_publish_needs_full_valid():
    t = make_table()
    t.grant([1], owner=1, group=0, domain=0)
    assert not t.publish([1])  # I3: valid 0 < block_size
    t.confirm_valid([1], BS - 1)
    assert not t.publish([1])
    t.confirm_valid([1], 1)  # never lowers
    assert t.valid(1) == BS - 1
    t.confirm_valid([1], BS + 9)  # capped
    assert t.publish([1])
    assert t.state(1) == BlockState.SHARED and t.owner(1) == NO_OWNER
    assert not t.check_writes(1, [1], [1])  # frozen
    assert not t.publish([NULL_BLOCK_ID])


def test_publish_refused_without_caching():
    t = make_table(caching=False)
    t.grant([1], owner=1, group=0, domain=0)
    t.confirm_valid([1], BS)
    assert not t.publish([1])


def test_publish_uses_group_block_size():
    t = GrantTable(8, [4, 16])
    t.grant([1], owner=1, group=1, domain=0)
    t.confirm_valid([1], 4)
    assert not t.publish([1])
    t.confirm_valid([1], 16)
    assert t.publish([1])
    with pytest.raises(GrantError):
        t.grant([2], owner=1, group=2, domain=0)


def published(t: GrantTable, block: int, domain: int = 0) -> None:
    t.grant([block], owner=1, group=0, domain=domain)
    t.confirm_valid([block], BS)
    assert t.publish([block])


def test_t3_t4_t5_share_release_revive():
    t = make_table()
    published(t, 1, domain=5)
    assert t.share([1], group=0, domain=5)  # T3
    assert t.readers(1) == 2
    t.release([1, 1], step_seq=3)  # T4, no gen bump
    assert t.state(1) == BlockState.CACHED_IDLE and t.gen(1) == 1
    assert t.share([1], group=0, domain=5)  # T5
    assert t.state(1) == BlockState.SHARED and t.readers(1) == 1


def test_share_refusals_i9():
    t = make_table()
    published(t, 1, domain=5)
    t.release([1], 0)
    assert not t.share([1], group=0, domain=6)
    assert not t.share([1], group=1, domain=5)
    assert t.state(1) == BlockState.CACHED_IDLE  # no mutation
    assert not t.share([2])  # FREE
    assert not t.share([1, 2])  # all-or-nothing
    assert t.readers(1) == 0
    assert t.share([NULL_BLOCK_ID])  # skipped


def test_share_pins_exclusive_block():
    t = make_table()
    t.grant([1], owner=4, group=0, domain=0)
    assert t.share([1])
    assert (t.state(1), t.readers(1), t.owner(1)) == (BlockState.EXCLUSIVE, 2, 4)
    t.release([1], 1)
    assert t.state(1) == BlockState.EXCLUSIVE
    t.release([1], 1)
    assert t.state(1) == BlockState.DRAINING


def test_t6_release_exclusive_revokes_then_t9_drains():
    t = make_table()
    t.grant([1], owner=1, group=0, domain=0)
    t.release([1], step_seq=5)  # T6
    assert t.state(1) == BlockState.DRAINING and t.gen(1) == 2
    assert t.owner(1) == NO_OWNER
    assert t.drain(4).size == 0  # I5: step 5 not processed
    assert t.drain(5).tolist() == [1]  # T9
    assert t.state(1) == BlockState.FREE
    ids, gens = t.take_gen_updates()
    assert ids.tolist() == [1] and gens.tolist() == [2]
    assert t.take_gen_updates()[0].size == 0


def test_t7_forced_revoke_waits_for_readers():
    t = make_table()
    published(t, 1)
    t.share([1])
    t.revoke([1], step_seq=2)  # T7
    assert t.state(1) == BlockState.DRAINING and t.gen(1) == 2
    assert t.drain(10).size == 0  # readers still hold it
    t.release([1, 1], step_seq=3)
    assert t.drain(10).tolist() == [1]


def test_t8_evict_cached_idle():
    t = make_table()
    published(t, 1)
    t.release([1], 4)
    t.revoke([1], step_seq=0)
    assert t.state(1) == BlockState.DRAINING and t.last_use(1) == 4
    assert t.drain(3).size == 0
    assert t.drain(4).tolist() == [1]


def test_revoke_refusals():
    t = make_table()
    with pytest.raises(GrantError):
        t.revoke([1], 0)  # FREE
    with pytest.raises(GrantError):
        t.revoke([NULL_BLOCK_ID], 0)
    t.grant([1], owner=1, group=0, domain=0)
    t.revoke([1], 0)
    with pytest.raises(GrantError):
        t.revoke([1], 0)  # already DRAINING


def test_release_refusals():
    t = make_table()
    with pytest.raises(GrantError):
        t.release([1], 0)
    t.grant([1], owner=1, group=0, domain=0)
    with pytest.raises(GrantError):
        t.release([1, 1], 0)
    assert t.readers(1) == 1  # all-or-nothing
    t.release([NULL_BLOCK_ID], 0)  # skipped


def test_write_lease_blocks_drain_then_quarantine_and_reclaim():
    t = make_table()
    t.grant([1, 2], owner=1, group=0, domain=0)
    t.lease([1], "write")
    t.lease([2], "read")
    t.release([1, 2], 0)
    assert t.drain(0).size == 0
    t.expire_read_leases([2])  # read lease timed out: reader re-checks gen
    assert t.drain(0).tolist() == [2]
    t.quarantine([1])  # T10
    assert t.state(1) == BlockState.QUARANTINED and t.num_draining == 0
    with pytest.raises(GrantError):
        t.grant([1], owner=2, group=0, domain=0)
    assert t.reclaim_quarantined([1]).tolist() == [1]  # T11
    assert t.state(1) == BlockState.FREE


def test_quarantined_block_is_not_reclaimed_while_held():
    t = make_table()
    t.grant([1], owner=1, group=0, domain=0)
    t.revoke([1], 0)  # forced revoke keeps the holder
    t.quarantine([1])
    with pytest.raises(GrantError):
        t.reclaim_quarantined([1])
    t.release([1], 0)
    assert t.reclaim_quarantined([1]).tolist() == [1]
    assert t.readers(1) == 0


def test_ack_releases_lease():
    t = make_table()
    t.grant([1], owner=1, group=0, domain=0)
    t.lease([1], "write")
    t.release([1], 0)
    assert t.drain(0).size == 0
    t.ack([1], "write")
    assert t.drain(0).tolist() == [1]
    with pytest.raises(GrantError):
        t.ack([1], "write")
    with pytest.raises(GrantError):
        t.quarantine([2])
    with pytest.raises(GrantError):
        t.lease([NULL_BLOCK_ID], "read")


def test_gen_wrap_skips_zero_and_null_gen():
    t = make_table()
    t._gen[1] = NULL_GEN - 1  # test-only setup of a near-wrap generation
    t.grant([1], owner=1, group=0, domain=0)
    t.release([1], 0)
    assert t.gen(1) == 1


def test_owner_ids_never_recycled():
    t = make_table()
    a = t.owner_for("a")
    assert t.owner_for("a") == a
    t.forget_owner("a")
    assert t.owner_for("a") != a
    assert t.owner_for("b") not in (a,)


def test_seq_monotonic_and_on_step_done():
    t = make_table()
    t.grant([1], owner=1, group=0, domain=0)
    t.begin_step(2)
    assert t.latest_sched_seq == 2
    with pytest.raises(ValueError):
        t.begin_step(1)
    t.release([1], 2)
    assert t.on_step_done(1).size == 0
    assert t.on_step_done(2).tolist() == [1]
    with pytest.raises(ValueError):
        t.drain(1)


def test_digest_terms():
    cols, ids, gens = [0, 1, 2], [5, 9, NULL_BLOCK_ID], [1, 3, NULL_GEN]
    terms = row_digest_terms(cols, ids, gens)
    assert terms.dtype == np.uint64
    assert terms.tolist() == [row_digest_term(*x) for x in zip(cols, ids, gens)]
    assert row_digest(cols, ids, gens) == sum(terms.tolist()) % (1 << 64)
    # int32 bit pattern of NULL_GEN (a GPU mirror) digests identically.
    assert row_digest_terms([2], [0], np.array([-1], np.int32))[0] == terms[2]
    assert row_digest_term(0, 5, 1) != row_digest_term(0, 5, 2)
    assert row_digest_term(0, 5, 1) != row_digest_term(1, 5, 1)


# --------------------------------------------------------------------------
# Property test: random API sequences against a reference model.
# --------------------------------------------------------------------------

NUM_BLOCKS = 6
OPS = (
    "grant",
    "share",
    "publish",
    "confirm",
    "release",
    "revoke",
    "lease",
    "ack",
    "drain",
    "quarantine",
    "reclaim",
    "begin",
)

op_strategy = st.tuples(
    st.sampled_from(OPS),
    # Mostly single real blocks so multi-block all-or-nothing refusals do not
    # starve deep states (e.g. DRAINING with readers); null and lists stay in.
    st.one_of(
        st.lists(st.integers(1, NUM_BLOCKS - 1), min_size=1, max_size=1),
        st.lists(st.integers(0, NUM_BLOCKS - 1), min_size=1, max_size=3),
    ),
    st.integers(0, 2),  # owner / group / domain selector
    st.integers(0, BS + 1),
    st.sampled_from(("read", "write")),
)


class Ref:
    """Independent Python model of the spec's state machine."""

    def __init__(self) -> None:
        self.s = {b: "FREE" for b in range(1, NUM_BLOCKS)}
        self.gen = {b: 1 for b in range(1, NUM_BLOCKS)}
        self.readers = dict.fromkeys(self.s, 0)
        self.valid = dict.fromkeys(self.s, 0)
        self.owner = dict.fromkeys(self.s, -1)
        self.group = dict.fromkeys(self.s, 0)
        self.domain = dict.fromkeys(self.s, 0)
        self.last = dict.fromkeys(self.s, 0)
        self.wacks = dict.fromkeys(self.s, 0)
        self.racks = dict.fromkeys(self.s, 0)
        self.sched = 0
        self.processed = 0

    def bump(self, b: int, seq: int) -> None:
        self.gen[b] += 1
        self.s[b] = "DRAINING"
        self.owner[b] = -1
        self.valid[b] = 0
        self.last[b] = max(self.last[b], seq)


def check_invariants(t: GrantTable, ref: Ref) -> None:
    for b in range(1, NUM_BLOCKS):
        state = t.state(b).name
        assert state == ref.s[b], b
        assert t.gen(b) == ref.gen[b]
        assert t.readers(b) == ref.readers[b]
        assert t.owner(b) == ref.owner[b]
        # I2: only EXCLUSIVE blocks have an owner.
        assert (t.owner(b) != NO_OWNER) == (state == "EXCLUSIVE")
        # I3: published blocks are full.
        if state in ("SHARED", "CACHED_IDLE"):
            assert t.valid(b) == BS
        if state == "EXCLUSIVE":
            assert t.readers(b) >= 1
        if state == "FREE":
            assert t.readers(b) == 0 and t.pending_acks(b) == (0, 0)
    assert t.state(NULL_BLOCK_ID) == BlockState.NULL
    assert t.gen(NULL_BLOCK_ID) == NULL_GEN
    assert set(t.draining_ids()) == {b for b, s in ref.s.items() if s == "DRAINING"}


def apply(t: GrantTable, ref: Ref, op, blocks, sel, num, kind) -> None:
    real = [b for b in blocks if b != NULL_BLOCK_ID]
    seq = ref.sched
    if op == "grant":
        ok = (
            len(set(blocks)) == len(blocks)
            and NULL_BLOCK_ID not in blocks
            and all(ref.s[b] == "FREE" for b in blocks)
        )
        if not ok:
            with pytest.raises(GrantError):
                t.grant(blocks, owner=sel, group=0, domain=sel)
            return
        t.grant(blocks, owner=sel, group=0, domain=sel)
        for b in blocks:
            ref.s[b], ref.owner[b], ref.readers[b] = "EXCLUSIVE", sel, 1
            ref.valid[b], ref.domain[b], ref.group[b] = 0, sel, 0
    elif op == "share":
        ok = all(
            ref.s[b] in ("SHARED", "CACHED_IDLE", "EXCLUSIVE")
            and ref.domain[b] == sel
            and ref.group[b] == 0
            for b in real
        )
        assert t.share(blocks, group=0, domain=sel) == ok  # I9
        if ok:
            for b in real:
                if ref.s[b] == "CACHED_IDLE":
                    ref.s[b] = "SHARED"
                ref.readers[b] += 1
    elif op == "publish":
        ok = (
            NULL_BLOCK_ID not in blocks
            and all(ref.s[b] in ("EXCLUSIVE", "SHARED") for b in blocks)
            and all(ref.valid[b] == BS for b in blocks)
        )
        assert t.publish(blocks) == ok
        if ok:
            for b in blocks:
                ref.s[b], ref.owner[b] = "SHARED", -1
    elif op == "confirm":
        t.confirm_valid(blocks, num)
        for b in real:
            if ref.s[b] in ("EXCLUSIVE", "SHARED"):
                ref.valid[b] = max(ref.valid[b], min(num, BS))
    elif op == "release":
        counts = {b: real.count(b) for b in set(real)}
        ok = all(
            ref.readers[b] >= c
            and ref.s[b] in ("EXCLUSIVE", "SHARED", "DRAINING", "QUARANTINED")
            for b, c in counts.items()
        )
        if not ok:
            with pytest.raises(GrantError):
                t.release(blocks, seq)
            return
        t.release(blocks, seq)
        for b, c in counts.items():
            ref.readers[b] -= c
            ref.last[b] = max(ref.last[b], seq)
            if ref.readers[b] == 0:
                if ref.s[b] == "SHARED":
                    ref.s[b] = "CACHED_IDLE"
                elif ref.s[b] == "EXCLUSIVE":
                    ref.bump(b, seq)  # I4
    elif op == "revoke":
        ok = NULL_BLOCK_ID not in blocks and all(
            ref.s[b] in ("EXCLUSIVE", "SHARED", "CACHED_IDLE") for b in blocks
        )
        if not ok:
            with pytest.raises(GrantError):
                t.revoke(blocks, seq)
            return
        t.revoke(blocks, seq)
        for b in set(blocks):
            ref.bump(b, seq)
    elif op == "lease":
        ok = NULL_BLOCK_ID not in blocks and all(
            ref.s[b] not in ("FREE", "QUARANTINED") for b in blocks
        )
        if not ok:
            with pytest.raises(GrantError):
                t.lease(blocks, kind)
            return
        t.lease(blocks, kind)
        acks = ref.racks if kind == "read" else ref.wacks
        for b in blocks:
            acks[b] += 1
    elif op == "ack":
        acks = ref.racks if kind == "read" else ref.wacks
        counts = {b: real.count(b) for b in set(real)}
        if not all(acks[b] >= c for b, c in counts.items()):
            with pytest.raises(GrantError):
                t.ack(blocks, kind)
            return
        t.ack(blocks, kind)
        for b, c in counts.items():
            acks[b] -= c
    elif op == "drain":
        ref.processed = max(ref.processed, min(ref.sched, ref.processed + num))
        out = set(t.drain(ref.processed).tolist())
        expect = {
            b
            for b in ref.s
            if ref.s[b] == "DRAINING"
            and ref.readers[b] == 0
            and ref.last[b] <= ref.processed
            and ref.wacks[b] == 0
            and ref.racks[b] == 0
        }
        assert out == expect  # I5
        for b in out:
            ref.s[b], ref.owner[b], ref.valid[b] = "FREE", -1, 0
            ref.domain[b] = ref.group[b] = 0
    elif op == "quarantine":
        if NULL_BLOCK_ID in blocks or not all(ref.s[b] == "DRAINING" for b in blocks):
            with pytest.raises(GrantError):
                t.quarantine(blocks)
            return
        t.quarantine(blocks)
        for b in blocks:
            ref.s[b] = "QUARANTINED"
    elif op == "reclaim":
        if not all(
            b != NULL_BLOCK_ID and ref.s[b] == "QUARANTINED" and ref.readers[b] == 0
            for b in blocks
        ):
            with pytest.raises(GrantError):
                t.reclaim_quarantined(blocks)
            return
        t.reclaim_quarantined(blocks)
        for b in blocks:
            ref.s[b], ref.valid[b] = "FREE", 0
            ref.wacks[b] = ref.racks[b] = 0
            ref.domain[b] = ref.group[b] = 0
    elif op == "begin":
        ref.sched += num
        t.begin_step(ref.sched)


@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(op_strategy, min_size=30, max_size=120))
def test_random_api_sequences_preserve_invariants(ops):
    t = GrantTable(NUM_BLOCKS, [BS])
    ref = Ref()
    gens_seen = {b: {1} for b in range(1, NUM_BLOCKS)}
    for op in ops:
        before = {b: (ref.s[b], ref.owner[b]) for b in ref.s}
        apply(t, ref, *op)
        check_invariants(t, ref)
        for b in ref.s:
            # I4: an owner only ever leaves through DRAINING with a new gen.
            was_owned = before[b][0] == "EXCLUSIVE"
            if was_owned and ref.s[b] not in ("EXCLUSIVE", "SHARED"):
                assert ref.s[b] == "DRAINING"
                assert t.gen(b) not in gens_seen[b]
            # I1-adjacent: FREE is only entered from DRAINING or QUARANTINED.
            if ref.s[b] == "FREE" and before[b][0] != "FREE":
                assert before[b][0] in ("DRAINING", "QUARANTINED")
            gens_seen[b].add(t.gen(b))
