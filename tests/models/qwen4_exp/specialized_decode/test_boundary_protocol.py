# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Lamport exchange of the specialized decode's sublayer boundary kernel
(`specialized_decode/boundary/all_reduce_hc.py`), checked over every
interleaving of small models.

The kernel's ranks exchange their partial block outputs through a mailbox in
symmetric memory: each thread publishes its fragment into the same slot of every
rank's mailbox, polls its own copy until every rank's fragment is there, and
resets the slots it read. Consecutive calls alternate between mailbox
generations held in a device word per rank: thread 0 of each CTA reads it for
the CTA (the other threads take its value after a CTA barrier) and counts the
CTA on an arrival counter, and the last CTA to arrive advances the generation
at its end. A call starts on a rank only after the previous call completed
there (the programmatic-dependent-launch chain).

A model runs that protocol for a few ranks, CTAs, warps and consecutive calls;
a warp stands for its threads (warp 0 for thread 0) and owns one fragment. Each
step of a warp is one access to shared state: the generation and arrival words
(thread 0 reads the generation and arrives in one step, as no other step reads
the counter or writes the generation in between), its CTA's generation in
shared memory, a slot. A step that only waits (the start of a call, a barrier,
a poll that finds every slot full) is merged into the step after it: its
condition stays true once it holds. A multicast store lands on every rank in
one step: the polls wait for it on each rank anyway. The check visits every
state the steps of all warps reach in any order. A run fails when a warp
publishes into a slot that still holds data, reads data of another call, or
when no warp can move before every warp has finished (a hang).

The protocol as the kernel runs it passes. Each defect fails: every thread
reading the generation word itself, one generation instead of two, a missing
reset.

Split calls (the kernel's row split) run the same generations with another
exchange: each CTA's row has an owner rank; a warp of another rank stores its
fragment into its slot of the owner's mailbox and ends without a barrier, the
owner's warp polls the other ranks' slots and resets them. They pass when every
rank owns a row of every call, and fail with one generation, without the
reset, and when a rank owns no row (it never waits for the others, so it
reuses a generation's slots before the owner has read them). The up/mix
all-gather of the split reads (`HcUpMixGather`) runs the one-shot exchange:
every warp stores into every other rank's mailbox and polls its own.
"""

import dataclasses

import pytest


@dataclasses.dataclass(frozen=True)
class Protocol:
    """The protocol of the kernel, or one with a defect."""

    generations: int = 2
    # Every warp reads the generation word itself, rather than thread 0 for
    # the CTA behind a barrier.
    generation_per_warp: bool = False
    reset: bool = True


@dataclasses.dataclass(frozen=True)
class Sizes:
    """Ranks, the warps of each CTA of a rank (one fragment each) and
    consecutive calls."""

    ranks: int
    warps: tuple[int, ...]
    calls: int
    # Split calls: the rank that owns each CTA's row; None for one-shot calls.
    owners: tuple[int, ...] | None = None


# The steps of a warp in a call.
START, BARRIER, PUBLISH, POLL, RESET, END = range(6)
EMPTY = 0


class Violation(Exception):
    """A run of the model broke the exchange."""


def explore(protocol: Protocol, sizes: Sizes) -> int:
    """Visit every reachable state; raise `Violation` at the first broken
    exchange or hang. Returns the number of states."""
    ctas = len(sizes.warps)
    # Per warp of a rank: its CTA, whether it is warp 0, its fragment.
    local = [
        (cta, w == 0) for cta, count in enumerate(sizes.warps) for w in range(count)
    ]
    fragments = len(local)
    warps = sizes.ranks * fragments
    generations = protocol.generations
    program = [START, PUBLISH, POLL, RESET, END]
    if not protocol.generation_per_warp:
        program.insert(1, BARRIER)

    def slot(owner: int, generation: int, source: int, fragment: int) -> int:
        return (
            (owner * generations + generation) * sizes.ranks + source
        ) * fragments + fragment

    def data(call: int, source: int) -> int:
        return 1 + call * sizes.ranks + source

    def owner(fragment: int) -> int | None:
        """The rank that owns the fragment's row in a split call, or None."""
        return None if sizes.owners is None else sizes.owners[local[fragment][0]]

    def sources(rank: int, fragment: int) -> list[int]:
        """The ranks whose slots of `fragment` the warp of `rank` polls
        and resets: every rank in a one-shot call, the others for the row's
        owner in a split call, none for the other ranks."""
        row_owner = owner(fragment)
        if row_owner is None:
            return list(range(sizes.ranks))
        if row_owner != rank:
            return []
        return [r for r in range(sizes.ranks) if r != rank]

    # State: per warp (call, step index, generation, arrived last); per rank
    # (generation word, arrivals); per CTA the generation in shared memory;
    # the mailboxes.
    initial = (
        ((0, 0, 0, False),) * warps,
        ((0, 0),) * sizes.ranks,
        (0,) * (sizes.ranks * ctas),
        (EMPTY,) * (sizes.ranks * generations * sizes.ranks * fragments),
    )

    def successors(state):
        threads, words, shared, mailbox = state
        for index in range(warps):
            call, position, g, last = threads[index]
            if call == sizes.calls:
                continue
            rank, fragment = divmod(index, fragments)
            cta, first = local[fragment]
            step = program[position]
            cta_warps = [
                rank * fragments + f for f in range(fragments) if local[f][0] == cta
            ]
            words_next, shared_next, mailbox_next = words, shared, mailbox
            if step == START:
                # The previous call completed on this rank.
                if any(
                    threads[rank * fragments + f][0] < call for f in range(fragments)
                ):
                    continue
                if first or protocol.generation_per_warp:
                    g = words[rank][0]
                if first:
                    if not protocol.generation_per_warp:
                        shared_next = list(shared)
                        shared_next[rank * ctas + cta] = g
                        shared_next = tuple(shared_next)
                    arrivals = words[rank][1]
                    last = arrivals == ctas - 1
                    words_next = list(words)
                    words_next[rank] = (words[rank][0], arrivals + 1)
                    words_next = tuple(words_next)
            elif step in (BARRIER, END):
                # Every warp of the CTA reached the barrier (BARRIER; END
                # follows the barrier after the sums of squares, which a split
                # call's CTA of another rank's row skips).
                skips = step == END and owner(fragment) not in (None, rank)
                if not skips and any(
                    (threads[w][0], program[threads[w][1]]) < (call, step)
                    for w in cta_warps
                ):
                    continue
                if step == BARRIER:
                    g = shared[rank * ctas + cta]
                elif first and last:
                    words_next = list(words)
                    words_next[rank] = ((g + 1) % generations, 0)
                    words_next = tuple(words_next)
            elif step == PUBLISH:
                mailbox_next = list(mailbox)
                row_owner = owner(fragment)
                if row_owner is None:
                    targets = list(range(sizes.ranks))
                else:
                    targets = [] if row_owner == rank else [row_owner]
                for target_rank in targets:
                    target = slot(target_rank, g, rank, fragment)
                    if mailbox[target] != EMPTY:
                        raise Violation(
                            f"rank {rank} call {call} published into a full slot"
                        )
                    mailbox_next[target] = data(call, rank)
                mailbox_next = tuple(mailbox_next)
            elif step == POLL:
                polled = sources(rank, fragment)
                slots = [slot(rank, g, r, fragment) for r in polled]
                if any(mailbox[s] == EMPTY for s in slots):
                    continue
                for source, s in zip(polled, slots):
                    if mailbox[s] != data(call, source):
                        raise Violation(
                            f"rank {rank} call {call} read data of another call"
                        )
            elif step == RESET and protocol.reset:
                mailbox_next = list(mailbox)
                for r in sources(rank, fragment):
                    mailbox_next[slot(rank, g, r, fragment)] = EMPTY
                mailbox_next = tuple(mailbox_next)
            if position + 1 == len(program):
                thread = (call + 1, 0, g, last)
            else:
                thread = (call, position + 1, g, last)
            threads_next = threads[:index] + (thread,) + threads[index + 1 :]
            yield threads_next, words_next, shared_next, mailbox_next

    seen = {initial}
    stack = [initial]
    while stack:
        state = stack.pop()
        moved = False
        for successor in successors(state):
            moved = True
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)
        if not moved and any(thread[0] < sizes.calls for thread in state[0]):
            raise Violation("no warp can move: the exchange hangs")
    return len(seen)


# Two CTAs of two and one warps over three calls: thread 0 of one CTA can arrive
# while the other warp of its CTA has not read the generation yet, and generation
# 0 comes back. One CTA of one warp over four calls: every generation serves two
# calls.
CTAS_AND_WARPS = Sizes(ranks=2, warps=(2, 1), calls=3)
GENERATION_REUSE = Sizes(ranks=2, warps=(1,), calls=4)
# Split calls: each rank owns one CTA's row (as in the kernel, where every rank
# owns rows / tp rows), over the same shapes; and one where rank 1 owns none.
SPLIT_CTAS_AND_WARPS = Sizes(ranks=2, warps=(2, 1), calls=3, owners=(0, 1))
SPLIT_REUSE = Sizes(ranks=2, warps=(1, 1), calls=4, owners=(0, 1))
SPLIT_UNOWNED = Sizes(ranks=2, warps=(1,), calls=4, owners=(0,))


@pytest.mark.parametrize(
    "sizes",
    [CTAS_AND_WARPS, GENERATION_REUSE, SPLIT_CTAS_AND_WARPS, SPLIT_REUSE],
    ids=["ctas-and-warps", "reuse", "split-ctas-and-warps", "split-reuse"],
)
def test_the_kernel_protocol_holds_in_every_interleaving(sizes: Sizes):
    assert explore(Protocol(), sizes) > 0


@pytest.mark.parametrize(
    "defect, sizes",
    [
        (Protocol(generation_per_warp=True), CTAS_AND_WARPS),
        (Protocol(generations=1), GENERATION_REUSE),
        (Protocol(reset=False), GENERATION_REUSE),
        (Protocol(generations=1), SPLIT_REUSE),
        (Protocol(reset=False), SPLIT_REUSE),
        (Protocol(), SPLIT_UNOWNED),
    ],
    ids=[
        "generation-per-thread",
        "one-generation",
        "no-reset",
        "split-one-generation",
        "split-no-reset",
        "split-rank-without-rows",
    ],
)
def test_a_broken_protocol_fails(defect: Protocol, sizes: Sizes):
    with pytest.raises(Violation):
        explore(defect, sizes)
