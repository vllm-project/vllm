# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The cross-rank exchange of the PLE row gather (`specialized_decode/ple/
gather.py`), checked over every interleaving of small models.

Exchange. The logits gather's generation protocol (`test_logits_protocol.py`)
with one writer a pack: a pack's owner is the rank whose vocabulary range holds
its row's id, which the call's ids decide, so it changes from call to call. A
warp (standing for its threads, warp 0 for thread 0) owning a pack stores it
into the pack's slot of every peer's mailbox, one step per peer; a warp of any
other rank polls its own mailbox's slot of the pack and resets it. Thread 0 of
each CTA reads the generation for the CTA (the others take it after a barrier)
and arrives; the last CTA to arrive advances the generation at its end. A run
fails when a store lands in a slot that still holds data, a poll reads data of
another call or of a rank that does not own the pack, or the exchange hangs.
Owners either move among all ranks from call to call, or among all ranks but
one, which owns nothing; that rank does not hold the others back, so those
runs model the collective the callers run after each call (a rank starts a
call only once every rank completed the one before). Defects it must catch:
one generation, a missing reset, a pack no rank owns (an id outside every
range), a pack two ranks own (overlapping ranges) and a rank that owns
nothing without the collective.

Residency. A GPU holds only so many CTAs at once and starts a grid's CTAs in no
promised order. The residency model gives every rank `slots` CTA slots: a
pending CTA starts, in any order, when a slot is free, and holds it until it
exits. CTA `c` of `grid` holds packs `c, c + grid, ...`; it publishes the
packs its rank owns, then awaits the others in increasing order. The kernel's
grid is at most one CTA a slot. A run fails when no rank can move before all are
done. Defect it must catch: a grid of one CTA per pack beyond the slots.
"""

import dataclasses

import pytest

from .test_logits_protocol import Violation, _explore


@dataclasses.dataclass(frozen=True)
class Ple:
    """The PLE gather's protocol as the kernel runs it, or with a defect."""

    generations: int = 2
    reset: bool = True
    # How a pack's owners follow from the call: "one" (the kernel's ranges
    # partition the ids), "idle" (one owner, but the last rank owns no pack of
    # any call), "none" (one pack of call 1 has no owner) or "two" (one pack
    # of call 0 has two owners).
    owners: str = "one"
    # A collective that every rank must enter before any completes it after
    # each call: a rank starts a call only once every rank completed the one
    # before (the executor's all-reduces between two calls).
    separated: bool = False


EMPTY = 0
START, BARRIER, STORE, POLL, RESET, END = range(6)


def owners_of(protocol: Ple, ranks: int, call: int, pack: int) -> tuple[int, ...]:
    """The ranks that own `pack` in `call`: one, moving with the call, so
    a slot's writer changes between calls (with "idle", among all ranks but
    the last)."""
    if protocol.owners == "idle":
        return ((call + 2 * pack) % (ranks - 1),)
    owner = (call + 2 * pack) % ranks
    if protocol.owners == "none" and call == 1 and pack == 0:
        return ()
    if protocol.owners == "two" and call == 0 and pack == 0:
        return (owner, (owner + 1) % ranks)
    return (owner,)


def explore_ple(protocol: Ple, ranks: int, warps: tuple[int, ...], calls: int) -> int:
    """The PLE gather's exchange (module docstring): `warps` per CTA of a
    rank, one pack each."""
    ctas = len(warps)
    local = [(cta, w == 0) for cta, count in enumerate(warps) for w in range(count)]
    packs = len(local)
    generations = protocol.generations

    def program(rank: int, call: int, pack: int) -> list[tuple[int, int]]:
        mine = rank in owners_of(protocol, ranks, call, pack)
        middle = (
            [(STORE, p) for p in range(ranks - 1)] if mine else [(POLL, 0), (RESET, 0)]
        )
        return [(START, 0), (BARRIER, 0)] + middle + [(END, 0)]

    def peers(rank: int) -> list[int]:
        return [p for p in range(ranks) if p != rank]

    def slot(owner: int, generation: int, pack: int) -> int:
        return (owner * generations + generation) * packs + pack

    def data(call: int, source: int) -> int:
        return 1 + call * ranks + source

    # State: per warp (call, step, generation, arrived last); per rank
    # (generation word, arrivals); per CTA the generation in shared memory;
    # the mailboxes.
    initial = (
        ((0, 0, 0, False),) * (ranks * packs),
        ((0, 0),) * ranks,
        (0,) * (ranks * ctas),
        (EMPTY,) * (ranks * generations * packs),
    )

    def successors(state):
        threads, words, shared, mailbox = state
        for index in range(ranks * packs):
            call, position, g, last = threads[index]
            if call == calls:
                continue
            rank, pack = divmod(index, packs)
            cta, first = local[pack]
            steps = program(rank, call, pack)
            step, which = steps[position]
            cta_warps = [rank * packs + f for f in range(packs) if local[f][0] == cta]
            words_next, shared_next, mailbox_next = words, shared, mailbox
            if step == START:
                # The previous call completed on this rank (with the
                # collective, on every rank).
                waited = range(ranks) if protocol.separated else (rank,)
                if any(
                    threads[r * packs + f][0] < call
                    for r in waited
                    for f in range(packs)
                ):
                    continue
                if first:
                    g = words[rank][0]
                    shared_next = list(shared)
                    shared_next[rank * ctas + cta] = g
                    shared_next = tuple(shared_next)
                    arrivals = words[rank][1]
                    last = arrivals == ctas - 1
                    words_next = list(words)
                    words_next[rank] = (words[rank][0], arrivals + 1)
                    words_next = tuple(words_next)
            elif step in (BARRIER, END):
                if step == BARRIER:
                    if any(
                        (threads[w][0], threads[w][1]) < (call, 1) for w in cta_warps
                    ):
                        continue
                    g = shared[rank * ctas + cta]
                elif first and last:
                    words_next = list(words)
                    words_next[rank] = ((g + 1) % generations, 0)
                    words_next = tuple(words_next)
            elif step == STORE:
                target = slot(peers(rank)[which], g, pack)
                if mailbox[target] != EMPTY:
                    raise Violation(f"rank {rank} call {call} stored into a full slot")
                mailbox_next = list(mailbox)
                mailbox_next[target] = data(call, rank)
                mailbox_next = tuple(mailbox_next)
            elif step == POLL:
                held = mailbox[slot(rank, g, pack)]
                if held == EMPTY:
                    continue
                expected = [
                    data(call, o) for o in owners_of(protocol, ranks, call, pack)
                ]
                if held not in expected:
                    raise Violation(
                        f"rank {rank} call {call} read data of another call or rank"
                    )
            elif step == RESET and protocol.reset:
                mailbox_next = list(mailbox)
                mailbox_next[slot(rank, g, pack)] = EMPTY
                mailbox_next = tuple(mailbox_next)
            if position + 1 == len(steps):
                thread = (call + 1, 0, g, last)
            else:
                thread = (call, position + 1, g, last)
            threads_next = threads[:index] + (thread,) + threads[index + 1 :]
            yield threads_next, words_next, shared_next, mailbox_next

    return _explore(
        initial,
        successors,
        lambda state: all(thread[0] == calls for thread in state[0]),
    )


# Two CTAs (two warps and one) over three calls at two ranks: a CTA's thread 0
# can arrive while its other warp has not read the generation yet, and every
# pack changes owner between calls. One CTA of two warps over four calls at
# three ranks: every generation serves two calls, each time another rank
# writing the slot.
CTAS_AND_WARPS = dict(ranks=2, warps=(2, 1), calls=3)
GENERATION_REUSE = dict(ranks=3, warps=(2,), calls=4)
# A rank that owns no pack: the other two alternate as the owner, so without
# the collective they complete calls 1 and 2 while it is still in call 0.
IDLE_RANK = dict(ranks=3, warps=(1, 1), calls=4)


@pytest.mark.parametrize(
    "protocol, sizes",
    [
        (Ple(), CTAS_AND_WARPS),
        (Ple(), GENERATION_REUSE),
        (Ple(owners="idle", separated=True), IDLE_RANK),
        (Ple(owners="idle", separated=True), GENERATION_REUSE),
    ],
    ids=["ctas-and-warps", "reuse", "idle-rank-separated", "idle-reuse-separated"],
)
def test_the_ple_gather_holds_in_every_interleaving(protocol: Ple, sizes: dict):
    assert explore_ple(protocol, **sizes) > 0


@pytest.mark.parametrize(
    "defect",
    [
        Ple(generations=1),
        Ple(reset=False),
        Ple(owners="none"),
        Ple(owners="two"),
        Ple(owners="idle"),
    ],
    ids=["one-generation", "no-reset", "unowned-pack", "two-owners", "idle-rank"],
)
def test_a_broken_ple_gather_fails(defect: Ple):
    with pytest.raises(Violation):
        explore_ple(defect, **GENERATION_REUSE)


def test_an_idle_rank_without_the_collective_fails_with_two_ctas():
    with pytest.raises(Violation):
        explore_ple(Ple(owners="idle"), **IDLE_RANK)


PENDING = -1


def explore_ple_residency(ranks: int, packs: int, grid: int, slots: int) -> int:
    """The gather's CTAs on a GPU of `slots` CTA slots a rank (module
    docstring, Residency), one call: pack `p`'s owner is rank `p % ranks`."""
    held = [range(cta, packs, grid) for cta in range(grid)]

    def program(rank: int, cta: int) -> list[tuple[int, int]]:
        mine = [p for p in held[cta] if p % ranks == rank]
        others = [p for p in held[cta] if p % ranks != rank]
        return [(STORE, p) for p in mine] + [(POLL, p) for p in others]

    programs = [[program(rank, cta) for cta in range(grid)] for rank in range(ranks)]

    # State: per rank, every CTA's position in its program (PENDING before it
    # starts, the program's length once it exited); per pack whether its owner
    # published it.
    initial = (((PENDING,) * grid,) * ranks, (False,) * packs)

    def successors(state):
        positions, published = state
        for rank in range(ranks):
            mine = positions[rank]
            mine_programs = programs[rank]
            running = sum(
                0 <= at < len(mine_programs[cta]) for cta, at in enumerate(mine)
            )
            for cta, at in enumerate(mine):
                published_next = published
                if at == PENDING:
                    if running == slots:
                        continue  # no free slot
                elif at == len(mine_programs[cta]):
                    continue
                else:
                    step, pack = mine_programs[cta][at]
                    if step == STORE:
                        published_next = list(published)
                        published_next[pack] = True
                        published_next = tuple(published_next)
                    elif not published[pack]:
                        continue  # the poll waits
                mine_next = mine[:cta] + (at + 1,) + mine[cta + 1 :]
                positions_next = positions[:rank] + (mine_next,) + positions[rank + 1 :]
                yield positions_next, published_next

    return _explore(
        initial,
        successors,
        lambda state: all(
            at == len(programs[rank][cta])
            for rank, mine in enumerate(state[0])
            for cta, at in enumerate(mine)
        ),
    )


@pytest.mark.parametrize(
    "sizes",
    [
        dict(ranks=2, packs=6, grid=2, slots=2),
        dict(ranks=3, packs=6, grid=2, slots=3),
    ],
    ids=["at-bound", "three-ranks"],
)
def test_the_bounded_ple_grid_finishes_in_every_start_order(sizes: dict):
    assert explore_ple_residency(**sizes) > 0


def test_a_ple_grid_beyond_the_slots_hangs():
    # One CTA per pack: rank 0 starts the CTAs of packs 1 and 3 (rank 1's),
    # rank 1 those of packs 0 and 2 (rank 0's); all four poll for packs whose
    # CTAs have no slot to start in.
    with pytest.raises(Violation):
        explore_ple_residency(ranks=2, packs=4, grid=4, slots=2)
