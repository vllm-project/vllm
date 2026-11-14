# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN record/commit state protocol of the specialized decode
(`runtime/gdn_records.py`), checked on a model of the runner's steps.

The model. Requests own mamba state slots through block tables; a state's
value is the tuple of the token labels folded into it, so every read can be
checked against the truth: the state of the tokens the request has kept so
far. A random sequence of steps (seeded) arrives, prefills, decodes,
finishes, preempts and aborts requests:

- decode steps run as FULL graphs (the executor's, with the records core
  under the `records` protocol; padded to a captured size) or eagerly (a
  stock step, which keeps vLLM's protocol under both); every request
  verifies `q` rows, the sampler accepts `a` of them and the runner
  advances its computed tokens;
- the stock state machinery runs as vLLM's does: in prefix-caching `align`
  mode the pre-copy that moves the running state into a new block column
  (`preprocess_mamba_align_fused_kernel`), the accepted-count scatter, and
  the postprocess that copies the state at a crossed block boundary into the
  boundary's block (`postprocess_mamba_fused_kernel`) and resets the
  accepted count when that block is the running one; prefills save the state
  at every block boundary; a request whose prompt starts with a cached
  block-aligned prefix starts from that block's state (prefix-cache hit);
  freed slots return to a pool and are reused;
- the `records` protocol's verify writes records instead of states, two
  copies into the request's copy slots (`copy_slots`: the first two
  non-source speculative slots that, in `align` mode, are not the slot of
  one of the `boundary_rows` of the computed tokens before the step) and
  the header, which names them, to its own buffer; the commit runs where the
  model state launches it (`postprocess_state`, before the stock
  postprocess) with the decisions of `commit_rows`, `boundary_row` and
  `record_copy`. The commit's CTAs run one after another, each reading the
  records from the copy before it writes, so that a commit writing the slot
  it reads its records from shows.

A run fails when any reader (a verify, a pre-copy, a postprocess copy, a
prefix-cache hit, the commit's records read) reads a value other than the
truth, or a commit finds no copy. The protocol must pass every seed in both
cache modes, and each defect below must fail some seed: folding one row
fewer, a missing or late boundary row, the commit after the stock
postprocess, a missed commit of a padded FULL step, a commit after a stock
step (stale records), commits of the rows past the step's requests, a commit
that reads its source from the destination slot, a step committed twice, a
single copy of the records (which the commit's own writes clobber), a commit
reading a copy in a slot it writes, a verify copy in the source slot, a
verify copy in the boundary row's slot (a verify that ignores the computed
tokens: both copies can then be written), and a verify writing records into
a slot it does not own (another request's).

The model state's own logic (which steps it marks and commits, and in which
order) and `GdnRecords`' capture bookkeeping are checked directly below,
on stand-ins.
"""

import dataclasses
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import model_state
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gdn_records import (
    GdnRecords,
    boundary_row,
    boundary_rows,
    commit_rows,
    copy_slots,
    record_copy,
    records_fit,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

Q = 4  # rows per request: three MTP draft tokens plus one
BLOCK = 6  # mamba block size of the model's align mode
CAPACITY = 4  # requests of the largest FULL graph
GRAPHS = (1, 2, 4)  # captured request counts
DEFECTS = (
    "fold_one_fewer",
    "no_boundary",
    "late_boundary",
    "commit_after_postprocess",
    "skip_padded_commit",
    "commit_stock_step",
    "commit_past_requests",
    "source_from_destination",
    "commit_twice",
    "one_copy",
    "copy_in_written_slot",
    "copy_in_source",
    "copy_in_boundary",
    "slot_not_owned",
)
CTAS = 2  # commit CTAs per request: each reads its records, then writes


class Violation(Exception):
    """A reader saw a state other than the truth."""


@dataclasses.dataclass
class Request:
    """A request of the model: its kept tokens and its block table."""

    rid: int
    history: list[tuple]
    blocks: list[int] = dataclasses.field(default_factory=list)
    accepted: int = 1
    state_idx: int = -1

    @property
    def computed(self) -> int:
        return len(self.history)

    def truth(self, tokens: int | None = None) -> tuple:
        return tuple(self.history[: self.computed if tokens is None else tokens])


class Runner:
    """The runner's steps over one protocol (module docstring)."""

    def __init__(self, protocol: str, align: bool, seed: int, defect: str | None):
        self.protocol, self.align, self.defect = protocol, align, defect
        self.rng = random.Random(seed)
        self.slots: dict[int, tuple] = {}
        self.free: list[int] = list(range(1, 1000))
        self.pinned: dict[int, int] = {}  # slot -> requests that hold it
        self.cache: dict[tuple, int] = {}  # block-aligned token prefix -> slot
        self.records: list[dict] = [{"rows": 0} for _ in range(CAPACITY)]
        self.running: list[Request] = []
        self.parked: list[list[tuple]] = []  # finished or preempted histories
        self.step = 0
        self.next_rid = 0

    # Slots ---------------------------------------------------------------
    def alloc(self) -> int:
        slot = self.free.pop(0)
        for key in [k for k, v in self.cache.items() if v == slot]:
            del self.cache[key]
        self.slots[slot] = ("garbage", slot, self.step)
        self.pinned[slot] = 1
        return slot

    def release(self, req: Request) -> None:
        for slot in req.blocks:
            self.pinned[slot] -= 1
            if not self.pinned[slot]:
                del self.pinned[slot]
                self.free.append(slot)

    def column(self, req: Request, col: int) -> int:
        while len(req.blocks) <= col:
            req.blocks.append(self.alloc())
        return req.blocks[col]

    def read(self, slot: int, truth: tuple, what: str) -> tuple:
        value = self.slots.get(slot)
        if value != truth:
            raise Violation(
                f"step {self.step}: {what} read slot {slot}: not the state of "
                f"its {len(truth)} tokens"
            )
        return truth

    # Requests ------------------------------------------------------------
    def arrive(self) -> None:
        """A new request's prefill (an eager stock step): fresh, a parked
        history resumed, or a parked history's prefix plus fresh tokens."""
        rid, self.next_rid = self.next_rid, self.next_rid + 1
        fresh = [("p", rid, i) for i in range(self.rng.randint(2, 2 * BLOCK + 3))]
        prompt = fresh
        if self.parked and self.rng.random() < 0.6:
            old = self.rng.choice(self.parked)
            prompt = old[: self.rng.randint(1, len(old))] + fresh[:2]
        req = Request(rid, list(prompt))
        start, state = 0, ()
        if self.align:
            # The longest cached prefix that leaves a token to compute.
            for k in range((len(prompt) - 1) // BLOCK, 0, -1):
                key = tuple(prompt[: k * BLOCK])
                if key in self.cache:
                    slot = self.cache[key]
                    if slot in self.free:
                        self.free.remove(slot)
                    self.pinned[slot] = self.pinned.get(slot, 0) + 1
                    req.blocks = [self.column(req, c) for c in range(k - 1)] + [slot]
                    # The prefill starts from the hit block's state.
                    state = self.read(slot, key, "prefix-cache hit")
                    start = k * BLOCK
                    break
            for k in range(start // BLOCK + 1, len(prompt) // BLOCK + 1):
                slot = self.column(req, k - 1)
                self.slots[slot] = tuple(prompt[: k * BLOCK])
                self.cache[tuple(prompt[: k * BLOCK])] = slot
            req.state_idx = (len(prompt) - 1) // BLOCK
            self.slots[self.column(req, req.state_idx)] = req.truth()
        else:
            for col in range(Q):
                self.column(req, col)
            self.slots[req.blocks[0]] = req.truth()
        del state  # the prefill's result is the truth above
        self.running.append(req)

    def leave(self, kind: str) -> None:
        req = self.running.pop(self.rng.randrange(len(self.running)))
        if kind != "abort":
            self.parked.append(list(req.history))
        self.release(req)

    # Decode steps --------------------------------------------------------
    def decode(self, full: bool) -> None:
        batch = list(self.running)
        records = full and self.protocol == "records"
        padded = next(g for g in GRAPHS if g >= len(batch)) if full else len(batch)
        if self.align:
            for req in batch:  # preprocess_state: the pre-copy
                src_col, src_off = req.state_idx, max(req.accepted - 1, 0)
                new = (req.computed + Q - 1) // BLOCK
                if src_col >= 0 and src_col != new:
                    value = self.read(
                        self.column(req, src_col + src_off), req.truth(), "pre-copy"
                    )
                    self.slots[self.column(req, new)] = value
                    req.accepted = 1
                req.state_idx = new
        labels = []
        for b, req in enumerate(batch):  # the verify
            first = req.state_idx if self.align else 0
            source_slot = self.column(req, first + req.accepted - 1)
            source = self.read(source_slot, req.truth(), "verify")
            ids = [self.column(req, first + t) for t in range(Q)]
            rows = [("d", req.rid, self.step, t) for t in range(Q)]
            labels.append(rows)
            if records:
                record = ("records", req.rid, self.step, tuple(rows))
                avoid = []
                if self.align and self.defect != "copy_in_boundary":
                    avoid = boundary_rows(req.computed, Q, BLOCK)
                copies = copy_slots(
                    ids, 0 if self.defect == "copy_in_source" else source_slot, avoid
                )
                if self.defect == "one_copy":
                    copies = copies[:1]
                self.records[b] = {
                    "rows": Q,
                    "source": source_slot,
                    "ids": ids,
                    "copies": list(copies),
                    "record": record,
                }
                if self.defect == "slot_not_owned" and b + 1 < len(batch):
                    # A slot of the next request: the one its verify reads.
                    other = batch[b + 1]
                    other_first = other.state_idx if self.align else 0
                    copies.append(self.column(other, other_first + other.accepted - 1))
                for slot in copies:
                    self.slots[slot] = record
            else:
                for t in range(Q):
                    self.slots[ids[t]] = source + tuple(rows[: t + 1])
        if records:
            for b in range(len(batch), padded):
                self.records[b] = {"rows": 0}
        sampled = [self.rng.randint(1, Q) for _ in batch]
        for req, rows, a in zip(batch, labels, sampled):  # post_update
            req.history += rows[:a]
        commit = records
        if self.defect == "skip_padded_commit" and padded > len(batch):
            commit = False
        if self.defect == "commit_stock_step" and not full:
            commit = True
        if commit and self.defect != "commit_after_postprocess":
            self.commit(batch, sampled)
            if self.defect == "commit_twice":
                self.commit(batch, sampled)
        self.postprocess(batch, sampled)
        if commit and self.defect == "commit_after_postprocess":
            self.commit(batch, sampled)

    def commit(self, batch: list[Request], sampled: list[int]) -> None:
        """`GdnRecords.commit`: the kernel's reads, then its writes."""
        rows = CAPACITY if self.defect == "commit_past_requests" else len(batch)
        for b in range(rows):
            header = self.records[b]
            count = commit_rows(sampled[b] if b < len(batch) else 1, header["rows"])
            if self.defect == "fold_one_fewer":
                count -= 1
            if count <= 0:
                continue
            ids, source = header["ids"], header["source"]
            boundary = -1
            if self.align and b < len(batch) and self.defect != "no_boundary":
                boundary = boundary_row(count, batch[b].computed, BLOCK)
                if boundary >= 0:
                    boundary += self.defect == "late_boundary"
            copy = record_copy(ids, header["copies"], count, boundary)
            if self.defect == "copy_in_written_slot":
                copy = next(slot for slot in ids if slot in header["copies"])
            if copy <= 0:
                raise Violation(f"step {self.step}: request row {b} has no copy")
            if self.defect == "source_from_destination":
                source = ids[count - 1]
            # Each CTA reads its own rows of the source (disjoint from the
            # other CTAs' rows, so read once here) and the records, then
            # writes.
            base = self.slots[source]
            writes = [(count - 1, ids[count - 1])]
            if boundary >= 0:
                writes.append((boundary, ids[boundary]))
            for _ in range(CTAS):
                if self.slots.get(copy) != header["record"]:
                    raise Violation(
                        f"step {self.step}: the commit of request row {b} read "
                        f"slot {copy}: not its verify's records"
                    )
                for row, slot in writes:
                    self.slots[slot] = base + header["record"][3][: row + 1]

    def postprocess(self, batch: list[Request], sampled: list[int]) -> None:
        """The stock postprocess: the accepted-count scatter and the align
        boundary copy (`postprocess_mamba_fused_kernel`)."""
        for req, a in zip(batch, sampled):
            req.accepted = max(a, 1)
            if not self.align:
                continue
            running = req.computed - req.accepted + 1
            aligned = req.computed // BLOCK * BLOCK
            if aligned < running:
                continue
            bias, dest, first = aligned - running, aligned // BLOCK - 1, req.state_idx
            if first == dest:
                req.accepted = 1
            if not (first == dest and bias == 0):
                value = self.read(
                    self.column(req, first + bias), req.truth(aligned), "boundary copy"
                )
                self.slots[self.column(req, dest)] = value
            self.cache[req.truth(aligned)] = self.column(req, dest)

    # The run -------------------------------------------------------------
    def run(self, steps: int) -> None:
        for self.step in range(steps):
            roll = self.rng.random()
            if not self.running or (roll < 0.15 and len(self.running) < CAPACITY + 1):
                if self.running:
                    self.decode(full=False)  # a mixed step: stock for all
                self.arrive()
            elif roll < 0.22:
                self.leave(self.rng.choice(["finish", "preempt", "abort"]))
            elif roll < 0.30 or len(self.running) > CAPACITY:
                self.decode(full=False)
            else:
                self.decode(full=True)
        for req in self.running:  # what the next verify would read
            first = (req.computed + Q - 1) // BLOCK if self.align else 0
            if self.align and first != req.state_idx:
                slot = self.column(req, req.state_idx + req.accepted - 1)
            else:
                slot = self.column(req, first + req.accepted - 1)
            self.read(slot, req.truth(), "final")


SEEDS = range(300)


@pytest.mark.parametrize("align", [False, True])
def test_the_records_protocol_reads_what_vllms_protocol_reads(align):
    for seed in SEEDS:
        Runner("slots", align, seed, None).run(60)
        Runner("records", align, seed, None).run(60)


@pytest.mark.parametrize("defect", DEFECTS)
def test_the_model_catches_each_defect(defect):
    align = defect not in ("skip_padded_commit", "commit_past_requests")
    for seed in SEEDS:
        try:
            Runner("records", align, seed, defect).run(60)
        except Violation:
            return
    pytest.fail(f"no seed caught {defect}")


def test_copy_slots_record_copy_and_records_fit():
    # Slots 7, 8, 9, 6; source 8 (accepted 2).
    slots = [7, 8, 9, 6]
    assert copy_slots(slots, 8, []) == [7, 9]
    assert copy_slots(slots, 8, [0]) == [9, 6]  # row 0 may be the boundary row
    assert copy_slots(slots, 8, [1]) == [7, 9]  # the source's row
    assert copy_slots([7, 8, 0, 0], 8, []) == [7]  # a null slot: one copy
    copies = [7, 9]
    assert record_copy(slots, copies, 2, -1) == 7  # writes 8 (the source)
    assert record_copy(slots, copies, 1, -1) == 9  # writes 7
    assert record_copy(slots, copies, 3, -1) == 7  # writes 9
    assert record_copy(slots, [9, 6], 3, 0) == 6  # writes 9 and 7
    assert record_copy(slots, [7], 1, -1) == 0  # its only copy is written
    assert record_copy(slots, copies, 3, 0) == 0  # a boundary not foreseen
    assert records_fit(4, True, 3) and records_fit(4, True, 16)
    assert records_fit(3, False, 0) and records_fit(4, False, 0)
    assert not records_fit(4, True, 2)  # two rows can be boundary rows
    assert not records_fit(3, True, 16) and not records_fit(2, False, 0)


def test_boundary_rows_name_every_boundary_row():
    # For every computed count before the step and every folded count, the
    # commit's boundary row is one of the verify's boundary rows, and at most
    # one row qualifies when the block size is at least q - 1.
    for block in (Q - 1, Q, BLOCK, 16):
        for before in range(4 * block):
            rows = boundary_rows(before, Q, block)
            assert len(rows) <= 1
            for count in range(1, Q + 1):
                row = boundary_row(count, before + count, block)
                assert row == -1 or row in rows, (block, before, count)
    assert boundary_rows(5, Q, 6) == [0] and boundary_rows(3, Q, 6) == [2]
    assert boundary_rows(2, Q, 6) == []  # row 3 would be slot a - 1 itself


def test_commit_rows_and_boundary_row():
    assert [commit_rows(s, Q) for s in (0, 1, 3, 4, 5)] == [1, 1, 3, 4, 4]
    assert commit_rows(3, 0) == 0 and commit_rows(4, 2) == 2
    # 8 tokens after the step, 4 rows folded: rows cover 5..8 tokens.
    assert boundary_row(4, 8, 6) == 1  # the state after 6 tokens, row 1
    assert boundary_row(4, 8, 8) == -1  # row 3: slot a - 1 holds it
    assert boundary_row(4, 8, 5) == 0
    assert boundary_row(4, 8, 9) == -1  # no boundary crossed
    assert boundary_row(1, 12, 6) == -1


# The model state's marking and commit, on stand-ins ----------------------


class StandInRecords:
    """`GdnRecords`' interface to the model state."""

    def __init__(self, rows: set[int]):
        self.rows = rows
        self.layers = [SimpleNamespace(prefix="gdn.0")]
        self.log: list[tuple] = []
        # The stock metadata build's result (the patched _prepare_attn).
        self.built: dict[str, object] = {"gdn.0": _metadata(2)}

    def records_step(self, rows: int) -> bool:
        return rows in self.rows

    def bind_seq_lens(self, seq_lens: torch.Tensor) -> None:
        self.log.append(("seq_lens", seq_lens))

    def commit(self, requests, sampled, idx_mapping, computed) -> None:
        self.log.append(("commit", requests, idx_mapping, computed))


def _metadata(requests: int, **fields) -> GDNAttentionMetadata:
    values = dict(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=requests,
        num_spec_decode_tokens=Q * requests,
        num_actual_tokens=Q * requests,
    )
    values.update(fields)
    return GDNAttentionMetadata(**values)


@pytest.fixture
def state(monkeypatch):
    """A model state with stand-in records, whose stock metadata build
    returns `state.gdn_records.built` and whose stock postprocess logs its
    call."""
    s = object.__new__(model_state.Qwen4ExpSpecializedDecodeModelState)
    records = StandInRecords({8, 16})
    s.gdn_records = records
    s._pending_commit = None
    monkeypatch.setattr(
        model_state.Qwen4ExpSpecializedDecodeModelState,
        "_prepare_attn",
        lambda self, *args: records.built,
    )
    monkeypatch.setattr(
        MambaHybridModelState,
        "postprocess_state",
        lambda self, *args: self.gdn_records.log.append(("stock", *args)),
    )
    return s


SEQ_LENS = torch.zeros(8, dtype=torch.int32)  # the runner's buffer


def _batch(rows: int, requests: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_tokens_after_padding=rows,
        num_reqs=requests,
        idx_mapping=torch.arange(requests),
        seq_lens=SEQ_LENS[:requests],
    )


def _prepare(state, batch, mode=CUDAGraphMode.FULL, for_capture=False):
    return state.prepare_attn(batch, mode, (), None, [], None, for_capture=for_capture)


def test_the_model_state_commits_records_steps_before_the_stock_postprocess(state):
    batch = _batch(8, 2)
    assert _prepare(state, batch) is state.gdn_records.built
    assert state._pending_commit is batch.idx_mapping
    sampled, computed = torch.ones(2, dtype=torch.int32), torch.zeros(4)
    state.postprocess_state(batch.idx_mapping, sampled, computed)
    log = state.gdn_records.log
    assert [entry[0] for entry in log] == ["seq_lens", "commit", "stock"]
    assert log[0][1] is batch.seq_lens
    assert log[1][1:] == (2, batch.idx_mapping, computed)
    assert state._pending_commit is None


@pytest.mark.parametrize(
    "rows,mode,for_capture,built",
    [
        (12, CUDAGraphMode.FULL, False, None),  # no records graph of 12 rows
        (8, CUDAGraphMode.PIECEWISE, False, None),
        (8, CUDAGraphMode.NONE, False, None),
        (8, CUDAGraphMode.FULL, True, None),  # a capture or dummy run
        (8, CUDAGraphMode.FULL, False, {"draft.0": object()}),  # the drafter's
    ],
)
def test_the_model_state_marks_only_records_replays(
    state, rows, mode, for_capture, built
):
    if built is not None:
        state.gdn_records.built = built
    _prepare(state, _batch(rows, 2), mode, for_capture)
    assert state._pending_commit is None
    state.postprocess_state(torch.arange(2), 1, None)
    # Every target-model step binds its seq_lens (captures included); the
    # drafter's batch has its own input buffers.
    bound = ["seq_lens"] if built is None else []
    assert [entry[0] for entry in state.gdn_records.log] == [*bound, "stock"]


@pytest.mark.parametrize(
    "metadata",
    [_metadata(2, num_prefills=1), _metadata(2, num_decodes=1), _metadata(1), "other"],
)
def test_a_records_replay_of_another_step_kind_raises(state, metadata):
    state.gdn_records.built = {"gdn.0": metadata}
    with pytest.raises(RuntimeError, match="not a pure"):
        _prepare(state, _batch(8, 2))


def test_an_uncommitted_records_step_or_a_foreign_postprocess_raises(state):
    batch = _batch(8, 2)
    _prepare(state, batch)
    with pytest.raises(RuntimeError, match="not committed"):
        _prepare(state, batch)
    state._pending_commit = batch.idx_mapping
    with pytest.raises(RuntimeError, match="not the step's own"):
        state.postprocess_state(torch.arange(2), torch.ones(2), None)
    state._pending_commit = batch.idx_mapping
    with pytest.raises(RuntimeError, match="not the step's own"):
        state.postprocess_state(batch.idx_mapping, 1, None)


# GdnRecords' capture bookkeeping --------------------------------------------


def _records(layers: int) -> GdnRecords:
    """`GdnRecords` without buffers or kernels: its bookkeeping only."""
    records = object.__new__(GdnRecords)
    records.layers = cast(
        "list[QwenGatedDeltaNetAttention]",
        [SimpleNamespace(name=i) for i in range(layers)],
    )
    records._index = {id(attn): i for i, attn in enumerate(records.layers)}
    records.recorded_rows = {}
    records._capture = None
    records.align = True
    records._seq_lens = None
    return records


def test_seq_lens_binding_keeps_the_runners_buffer_and_raises_when_it_moves():
    records = _records(1)
    buffer = torch.zeros(8, dtype=torch.int32)
    records.bind_seq_lens(buffer[:2])
    records.bind_seq_lens(buffer[:4])  # a longer view of the same buffer
    records.bind_seq_lens(buffer[:1])
    assert records.seq_lens is not None and records.seq_lens.numel() == 4
    assert records.seq_lens.data_ptr() == buffer.data_ptr()
    with pytest.raises(RuntimeError, match="seq_lens buffer moved"):
        records.bind_seq_lens(torch.zeros(4, dtype=torch.int32))
    records = _records(1)
    records.align = False  # outside align mode nothing reads it
    records.bind_seq_lens(buffer)
    assert records.seq_lens is None


def test_capture_bookkeeping_names_the_records_sizes():
    records = _records(3)
    for rows in (4, 8):
        records.start_capture()
        for attn in records.layers:
            records.launched(attn)
        records.finish_capture(rows)
    records.finish_capture(16)  # a FULL capture of the stock layers
    # Eager calls (no capture running) are not counted.
    records.launched(records.layers[0])
    records.start_capture()
    for attn in records.layers:
        records.launched(attn)
    records.finish_capture(8)
    assert records.recorded_rows == {4: True, 8: True, 16: False}
    assert [records.records_step(r) for r in (4, 8, 16, 32)] == [
        True,
        True,
        False,
        False,
    ]


def test_capture_bookkeeping_raises_on_partial_or_mixed_graphs():
    records = _records(3)
    records.start_capture()
    records.launched(records.layers[0])
    with pytest.raises(RuntimeError, match="1 of 3 GDN layers"):
        records.finish_capture(4)
    records = _records(2)
    records.start_capture()
    for attn in records.layers:
        records.launched(attn)
    records.finish_capture(8)
    with pytest.raises(RuntimeError, match="with and without"):
        records.finish_capture(8)
