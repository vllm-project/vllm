# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The cross-rank exchanges of the specialized decode's logits kernels
(`specialized_decode/logits/`), checked over every interleaving of small
models.

Token choice (`choice.py`). For each row, the row's last CTA on every rank
exchanges the rank's best key through tagged 8-byte records: with its epoch
`e` it stores `(key, tag(e))` into slot `[parity(e)][rank]` of every
peer, polls its own slots `[parity(e)][source]` until each carries
`tag(e)`, and advances `e`. A launch on a rank starts after the previous
one completed there. The model runs that for a few ranks and launches of one
row; each store and each poll is a step, so peers' stores land in any order. A
run fails when a store overwrites a record its owner has not read yet, a poll
accepts a record of another epoch, or no rank can move before all are done (a
hang). Defects it must catch: one slot per source instead of two parities, and
a tag that does not depend on the epoch.

Logits gather (`gather.py`). The boundary kernel's generation protocol
(`test_boundary_protocol.py`) with peer stores: a warp (standing for its
threads, warp 0 for thread 0) stores its pack into every peer's mailbox slot of
its rank, one step per peer, polls its own mailbox's slots of every other rank,
and resets them; thread 0 of each CTA reads the generation for the CTA (the
others take it after a barrier) and arrives; the last CTA to arrive advances
the generation at its end. A run fails when a store lands in a slot that still
holds data, a poll reads data of another call, or the exchange hangs. Defects
it must catch: one generation, a missing reset.

Residency. The models above let every CTA run; a GPU holds only so many CTAs at
once and starts a grid's CTAs in no promised order. The residency models give
every rank `slots` CTA slots: a pending CTA starts, in any order, when a slot
is free, and holds it until it exits. The gather's CTA `c` of `grid` owns
packs `c, c + grid, ...` and publishes them all before it awaits each, in
increasing order, the other ranks' packs; the kernel's grid is at most one CTA
a slot. The token choice's CTA (row, chunk) counts itself on its row; the row's
last one publishes the rank's record and awaits the peers'; the kernel takes
fewer rows than slots, whatever its grid. A run fails when no rank can move
before all are done. Defects it must catch: a gather grid of one CTA per pack
beyond the slots, and a token choice of more rows than slots.
"""

import dataclasses

import pytest

TAG_LIMIT = (1 << 23) - 1


class Violation(Exception):
    """A run of a model broke the exchange."""


def _explore(initial, successors, done) -> int:
    """Visit every reachable state; raise `Violation` at the first broken
    exchange (`successors` raises) or hang. Returns the number of states."""
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
        if not moved and not done(state):
            raise Violation("no rank can move: the exchange hangs")
    return len(seen)


@dataclasses.dataclass(frozen=True)
class Choice:
    """The token choice's exchange as the kernel runs it, or with a defect."""

    parities: int = 2
    epoch_tag: bool = True


def explore_choice(protocol: Choice, ranks: int, launches: int) -> int:
    """The token choice's record exchange for one row (module docstring)."""
    peers = [[p for p in range(ranks) if p != r] for r in range(ranks)]
    # A rank's program in a launch: one store per peer, one poll per peer.
    program = [("store", p) for p in range(ranks - 1)] + [
        ("poll", p) for p in range(ranks - 1)
    ]

    def tag(epoch: int) -> int:
        return 1 + epoch % TAG_LIMIT if protocol.epoch_tag else 1

    def slot(owner: int, parity: int, source: int) -> int:
        return (owner * protocol.parities + parity) * ranks + source

    # State: per rank (epoch, position in the launch); per slot (epoch, tag,
    # read) of the record it holds, or None.
    initial = (((0, 0),) * ranks, (None,) * (ranks * protocol.parities * ranks))

    def successors(state):
        progress, records = state
        for rank in range(ranks):
            epoch, position = progress[rank]
            if epoch == launches:
                continue
            kind, which = program[position]
            parity = epoch % protocol.parities
            records_next = list(records)
            if kind == "store":
                peer = peers[rank][which]
                target = slot(peer, parity, rank)
                held = records[target]
                if held is not None and not held[2]:
                    raise Violation(
                        f"rank {rank} epoch {epoch} overwrote an unread record"
                    )
                records_next[target] = (epoch, tag(epoch), False)
            else:
                source = peers[rank][which]
                target = slot(rank, parity, source)
                held = records[target]
                if held is None or held[1] != tag(epoch):
                    continue  # the poll waits
                if held[0] != epoch:
                    raise Violation(
                        f"rank {rank} epoch {epoch} took a record of epoch {held[0]}"
                    )
                records_next[target] = (held[0], held[1], True)
            if position + 1 == len(program):
                mine = (epoch + 1, 0)
            else:
                mine = (epoch, position + 1)
            progress_next = progress[:rank] + (mine,) + progress[rank + 1 :]
            yield progress_next, tuple(records_next)

    return _explore(
        initial,
        successors,
        lambda state: all(epoch == launches for epoch, _ in state[0]),
    )


@pytest.mark.parametrize("ranks, launches", [(2, 5), (3, 4)])
def test_the_choice_exchange_holds_in_every_interleaving(ranks: int, launches: int):
    assert explore_choice(Choice(), ranks, launches) > 0


@pytest.mark.parametrize(
    "defect",
    [Choice(parities=1), Choice(epoch_tag=False)],
    ids=["one-slot", "epoch-free-tag"],
)
def test_a_broken_choice_exchange_fails(defect: Choice):
    with pytest.raises(Violation):
        explore_choice(defect, 3, 4)


@dataclasses.dataclass(frozen=True)
class Gather:
    """The logits gather's protocol as the kernel runs it, or with a defect."""

    generations: int = 2
    reset: bool = True


EMPTY = 0
START, BARRIER, STORE, POLL, RESET, END = range(6)


def explore_gather(
    protocol: Gather, ranks: int, warps: tuple[int, ...], calls: int
) -> int:
    """The logits gather's exchange (module docstring): `warps` per CTA of a
    rank, one pack each."""
    ctas = len(warps)
    local = [(cta, w == 0) for cta, count in enumerate(warps) for w in range(count)]
    packs = len(local)
    generations = protocol.generations
    program = (
        [(START, 0), (BARRIER, 0)]
        + [(STORE, p) for p in range(ranks - 1)]
        + [(POLL, 0), (RESET, 0), (END, 0)]
    )

    def peers(rank: int) -> list[int]:
        return [p for p in range(ranks) if p != rank]

    def slot(owner: int, generation: int, source: int, pack: int) -> int:
        # The owner's slot of ``source``: (source - owner - 1) mod ranks.
        index = (source - owner - 1) % ranks
        return ((owner * generations + generation) * (ranks - 1) + index) * packs + pack

    def data(call: int, source: int) -> int:
        return 1 + call * ranks + source

    # State: per warp (call, step, generation, arrived last); per rank
    # (generation word, arrivals); per CTA the generation in shared memory;
    # the mailboxes.
    initial = (
        ((0, 0, 0, False),) * (ranks * packs),
        ((0, 0),) * ranks,
        (0,) * (ranks * ctas),
        (EMPTY,) * (ranks * generations * (ranks - 1) * packs),
    )

    def successors(state):
        threads, words, shared, mailbox = state
        for index in range(ranks * packs):
            call, position, g, last = threads[index]
            if call == calls:
                continue
            rank, pack = divmod(index, packs)
            cta, first = local[pack]
            step, which = program[position]
            cta_warps = [rank * packs + f for f in range(packs) if local[f][0] == cta]
            words_next, shared_next, mailbox_next = words, shared, mailbox
            if step == START:
                # The previous call completed on this rank.
                if any(threads[rank * packs + f][0] < call for f in range(packs)):
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
                        (threads[w][0], program[threads[w][1]][0]) < (call, START + 1)
                        for w in cta_warps
                    ):
                        continue
                    g = shared[rank * ctas + cta]
                elif first and last:
                    words_next = list(words)
                    words_next[rank] = ((g + 1) % generations, 0)
                    words_next = tuple(words_next)
            elif step == STORE:
                owner = peers(rank)[which]
                target = slot(owner, g, rank, pack)
                if mailbox[target] != EMPTY:
                    raise Violation(f"rank {rank} call {call} stored into a full slot")
                mailbox_next = list(mailbox)
                mailbox_next[target] = data(call, rank)
                mailbox_next = tuple(mailbox_next)
            elif step == POLL:
                slots = [slot(rank, g, source, pack) for source in peers(rank)]
                if any(mailbox[s] == EMPTY for s in slots):
                    continue
                for source, s in zip(peers(rank), slots):
                    if mailbox[s] != data(call, source):
                        raise Violation(
                            f"rank {rank} call {call} read data of another call"
                        )
            elif step == RESET and protocol.reset:
                mailbox_next = list(mailbox)
                for source in peers(rank):
                    mailbox_next[slot(rank, g, source, pack)] = EMPTY
                mailbox_next = tuple(mailbox_next)
            if position + 1 == len(program):
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


# Two CTAs (two warps and one) over three calls: a CTA's thread 0 can arrive
# while its other warp has not read the generation yet. One CTA of one warp
# over four calls at three ranks: every generation serves two calls and every
# rank stores to two peers.
CTAS_AND_WARPS = dict(ranks=2, warps=(2, 1), calls=3)
GENERATION_REUSE = dict(ranks=3, warps=(1,), calls=4)


@pytest.mark.parametrize(
    "sizes", [CTAS_AND_WARPS, GENERATION_REUSE], ids=["ctas-and-warps", "reuse"]
)
def test_the_gather_protocol_holds_in_every_interleaving(sizes: dict):
    assert explore_gather(Gather(), **sizes) > 0


@pytest.mark.parametrize(
    "defect",
    [Gather(generations=1), Gather(reset=False)],
    ids=["one-generation", "no-reset"],
)
def test_a_broken_gather_protocol_fails(defect: Gather):
    with pytest.raises(Violation):
        explore_gather(defect, **GENERATION_REUSE)


PENDING = -1


def explore_gather_residency(ranks: int, packs: int, grid: int, slots: int) -> int:
    """The gather's CTAs on a GPU of `slots` CTA slots a rank (module
    docstring, Residency): CTA `c` publishes packs `c, c + grid, ...`, then
    awaits them in the same order."""
    owned = [range(cta, packs, grid) for cta in range(grid)]
    programs = [
        [(STORE, pack) for pack in mine] + [(POLL, pack) for pack in mine]
        for mine in owned
    ]

    # State: per rank, every CTA's position in its program (PENDING before it
    # starts, the program's length once it exited); per (rank, pack) whether
    # the rank published the pack.
    initial = (((PENDING,) * grid,) * ranks, (False,) * (ranks * packs))

    def successors(state):
        positions, published = state
        for rank in range(ranks):
            mine = positions[rank]
            running = sum(0 <= at < len(programs[cta]) for cta, at in enumerate(mine))
            for cta, at in enumerate(mine):
                published_next = published
                if at == PENDING:
                    if running == slots:
                        continue  # no free slot
                elif at == len(programs[cta]):
                    continue
                else:
                    step, pack = programs[cta][at]
                    if step == STORE:
                        published_next = list(published)
                        published_next[rank * packs + pack] = True
                        published_next = tuple(published_next)
                    elif not all(
                        published[source * packs + pack]
                        for source in range(ranks)
                        if source != rank
                    ):
                        continue  # the poll waits
                mine_next = mine[:cta] + (at + 1,) + mine[cta + 1 :]
                positions_next = positions[:rank] + (mine_next,) + positions[rank + 1 :]
                yield positions_next, published_next

    return _explore(
        initial,
        successors,
        lambda state: all(
            at == len(programs[cta]) for mine in state[0] for cta, at in enumerate(mine)
        ),
    )


ARRIVE, PUBLISH, AWAIT, EXITED = range(4)


def explore_choice_residency(ranks: int, rows: int, chunks: int, slots: int) -> int:
    """The token choice's CTAs on a GPU of `slots` CTA slots a rank (module
    docstring, Residency): `chunks` CTAs a row; the row's last to count
    itself publishes, then awaits the peers' records of the row."""
    grid = rows * chunks

    # State: per rank, every CTA's step (PENDING before it starts) and every
    # row's count of arrived CTAs; per (rank, row) whether the rank published.
    initial = (
        ((PENDING,) * grid,) * ranks,
        ((0,) * rows,) * ranks,
        (False,) * (ranks * rows),
    )

    def successors(state):
        steps, counts, published = state
        for rank in range(ranks):
            mine = steps[rank]
            running = sum(ARRIVE <= step < EXITED for step in mine)
            for cta, step in enumerate(mine):
                row = cta // chunks
                counts_next, published_next = counts, published
                if step == PENDING:
                    if running == slots:
                        continue  # no free slot
                    step_next = ARRIVE
                elif step == ARRIVE:
                    arrived = counts[rank][row] + 1
                    row_counts = (
                        counts[rank][:row] + (arrived,) + counts[rank][row + 1 :]
                    )
                    counts_next = counts[:rank] + (row_counts,) + counts[rank + 1 :]
                    step_next = PUBLISH if arrived == chunks else EXITED
                elif step == PUBLISH:
                    published_next = list(published)
                    published_next[rank * rows + row] = True
                    published_next = tuple(published_next)
                    step_next = AWAIT
                elif step == AWAIT:
                    if not all(
                        published[source * rows + row]
                        for source in range(ranks)
                        if source != rank
                    ):
                        continue  # the poll waits
                    step_next = EXITED
                else:
                    continue
                mine_next = mine[:cta] + (step_next,) + mine[cta + 1 :]
                steps_next = steps[:rank] + (mine_next,) + steps[rank + 1 :]
                yield steps_next, counts_next, published_next

    return _explore(
        initial,
        successors,
        lambda state: all(step == EXITED for mine in state[0] for step in mine),
    )


# The gather's grid at the bound (two CTAs of several packs on two slots), and
# below it at three ranks; the token choice's grid above the slots with fewer
# rows than slots.
@pytest.mark.parametrize(
    "sizes",
    [
        dict(ranks=2, packs=5, grid=2, slots=2),
        dict(ranks=3, packs=4, grid=2, slots=3),
    ],
    ids=["at-bound", "three-ranks"],
)
def test_the_bounded_gather_grid_finishes_in_every_start_order(sizes: dict):
    assert explore_gather_residency(**sizes) > 0


def test_a_gather_grid_beyond_the_slots_hangs():
    # One CTA per pack, as before the bound: rank 0 starts packs 2 and 3,
    # rank 1 packs 0 and 1, and all four poll for packs nobody can publish.
    with pytest.raises(Violation):
        explore_gather_residency(ranks=2, packs=4, grid=4, slots=2)


@pytest.mark.parametrize(
    "sizes",
    [
        dict(ranks=2, rows=2, chunks=2, slots=3),
        dict(ranks=3, rows=1, chunks=3, slots=2),
    ],
    ids=["grid-above-slots", "three-ranks"],
)
def test_the_bounded_choice_finishes_in_every_start_order(sizes: dict):
    assert explore_choice_residency(**sizes) > 0


def test_a_choice_of_more_rows_than_slots_hangs():
    # Rank 0 starts row 1, rank 1 row 0: each polls for a row the other has
    # no slot left to start.
    with pytest.raises(Violation):
        explore_choice_residency(ranks=2, rows=2, chunks=1, slots=1)
