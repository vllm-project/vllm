# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stream order of the overlapped metadata replay
(`runtime/decode_metadata.py`, invariants I1-I4), checked over every
interleaving of the two streams on CPU.

The model runs the module's own code on stand-in streams, events and graphs
that record what it enqueues: `DecodeMetadataGraphs.record` captures a key's
build (the GDN groups' builds, `gdn_ready`, then the PLE and QSA builds),
`CaptureWaits` and `join` place the forward graph's waits, and
`DecodeMetadataGraphs._replay` forks the side stream, copies the request
indices, replays the recording and records `ready`. Around them stand-in
launches stand for the runner's prep, the preprocess, the n-gram kernel, the
layers, the postprocess and the drafts, each with the buffers it reads and
writes. The host enqueues three decode steps before the GPU runs any, as it
does when it runs ahead.

A run executes the two streams' operations in any order that CUDA allows: in
order on each stream, an event wait only after the record it binds to. A
wait binds, when it is enqueued (a graph's waits when the graph is launched),
to the event's last record enqueued before it (CUDA's rule for
`cudaEventRecord` and event wait nodes, see the module docstring). Every
read must see the write it sees when the host program runs in order on one
stream, which is what the stock path does; a run fails when a read sees
another write (a race) or when neither stream can move before both are done
(a hang). In particular step k + 1's replay must not write metadata that step
k's forward graph reads before it: that read would see step k + 1's write.

Defects it must catch: no fork wait on the side stream, no `ready` record, no
`gdn_ready` record node, a wait after the layer that reads the metadata, a
build that writes the GDN metadata after `gdn_ready`, no `record_stream`
of the request indices, no `settle` before a stock build, and waits that
bind to the last record when they run instead of when they are enqueued.
"""

import contextlib
import dataclasses
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Protocol

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import decode_metadata
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode_metadata import (
    CaptureWaits,
    DecodeMetadataGraphs,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

KEY = (2, 2, 8, 8)
STEPS = 3


class Violation(Exception):
    """A run of the model broke the stock order."""


@dataclasses.dataclass(frozen=True)
class Op:
    """One enqueued operation: a launch with the buffers it reads and writes,
    or an event record or wait; `bound` is the program index of the record a
    wait binds to when it is enqueued (None: never recorded)."""

    kind: str
    stream: str
    name: str
    step: int = 0
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    bound: int | None = None


class Recorder:
    """Stand-in streams, events and graphs that record what the module
    enqueues into `program` (or into the graph being captured)."""

    def __init__(self) -> None:
        self.program: list[Op] = []
        self.target: list[Op] | None = None
        self.step = 0
        self.capturing: list[Op] | None = None
        self.forks = 0
        self.record_streams: list[str] = []
        recorder = self

        class Stream:
            def __init__(self, name: str = "side", device=None) -> None:
                self.name = name

            def wait_event(self, event: "Event") -> None:
                recorder.add(Op("wait", self.name, event.name))

            def wait_stream(self, other: "Stream") -> None:
                recorder.forks += 1
                name = f"fork {recorder.forks}"
                recorder.add(Op("record", other.name, name))
                recorder.add(Op("wait", self.name, name))

            def synchronize(self) -> None:
                pass

        class Event:
            def __init__(self, **kwargs) -> None:
                self.name = "unnamed"

            def record(self, stream: Stream | None = None) -> None:
                stream = stream or recorder.current
                recorder.add(Op("record", stream.name, self.name))

        class Graph:
            def __init__(self) -> None:
                self.ops: list[Op] = []

            def replay(self) -> None:
                for op in self.ops:
                    recorder.add(dataclasses.replace(op, stream=recorder.current.name))

        self.Stream, self.Event, self.Graph = Stream, Event, Graph
        self.main = Stream("main")
        self.current = self.main

    def add(self, op: Op) -> None:
        """Enqueue `op` on the program or the graph being captured; a wait
        binds to the event's last record enqueued before it."""
        if self.capturing is not None:
            self.capturing.append(op)
            return
        if self.target is None:
            return
        op = dataclasses.replace(op, step=self.step)
        if op.kind == "wait":
            records = [
                index
                for index, other in enumerate(self.target)
                if other.kind == "record" and other.name == op.name
            ]
            op = dataclasses.replace(op, bound=records[-1] if records else None)
        self.target.append(op)

    def launch(self, name: str, reads: str = "", writes: str = "") -> None:
        """A stand-in kernel on the current stream."""
        self.add(Op("launch", self.current.name, name, 0, *_bufs(reads, writes)))

    @contextlib.contextmanager
    def stream(self, stream) -> Iterator[None]:
        """`torch.cuda.stream`."""
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    @contextlib.contextmanager
    def graph(self, graph, pool=None) -> Iterator[None]:
        """`torch.cuda.graph`: capture into `graph` on a capture stream."""
        self.capturing = graph.ops
        previous, self.current = self.current, self.Stream("capture")
        try:
            yield
        finally:
            self.capturing, self.current = None, previous


def _bufs(reads: str, writes: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Buffer names from space-separated lists."""
    return tuple(reads.split()), tuple(writes.split())


class StandInGdnBuilder(GDNAttentionMetadataBuilder):
    """A GDN group's builder whose build and block-table update launch the
    writes of that group's metadata."""

    def __init__(self, recorder: Recorder, index: int) -> None:
        self.recorder, self.index = recorder, index

    def build(self, *args, **kwargs):
        self.recorder.launch(f"gdn build {self.index}", "R X", f"G{self.index}")
        return f"gdn metadata {self.index}"

    def update_block_table(self, *args, **kwargs):
        self.recorder.launch(f"gdn update {self.index}", "R X G0", f"G{self.index}")
        return f"gdn metadata {self.index}"


class RecordedGraph(Protocol):
    """A graph captured on the recorder: its ops, replayed on the current
    stream."""

    ops: list[Op]

    def replay(self) -> None: ...


@dataclasses.dataclass
class Model:
    """The recorder, the module's metadata graphs over stand-in builders, the
    forward graph and the step's batch."""

    recorder: Recorder
    metadata: DecodeMetadataGraphs
    forward: RecordedGraph
    batch: SimpleNamespace


def _model(monkeypatch, late_gdn_write: bool = False) -> Model:
    """Record a key's build and capture a forward graph with the module's
    code on stand-ins; `late_gdn_write` makes the PLE build also write the
    GDN metadata (a defect)."""
    recorder = Recorder()
    monkeypatch.setattr(torch.cuda, "Stream", recorder.Stream)
    monkeypatch.setattr(torch.cuda, "Event", recorder.Event)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", recorder.Graph)
    monkeypatch.setattr(torch.cuda, "graph", recorder.graph)
    monkeypatch.setattr(torch.cuda, "stream", recorder.stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: recorder.current)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: recorder.capturing is not None,
    )
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorder.record_streams.append(stream.name),
    )
    monkeypatch.setattr(
        decode_metadata, "capture_pool", lambda pool, config: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        decode_metadata,
        "current_platform",
        SimpleNamespace(get_global_graph_pool=lambda: None),
    )
    metadata = DecodeMetadataGraphs(
        SimpleNamespace(), decode_query_len=4, max_num_reqs=4, device="cpu"
    )
    metadata.gdn_ready.name, metadata.ready.name = "gdn_ready", "ready"
    batch = SimpleNamespace(
        num_reqs=KEY[0],
        num_reqs_after_padding=KEY[1],
        idx_mapping=torch.zeros(KEY[0], dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.ones(KEY[1], dtype=torch.int32),
    )
    monkeypatch.setattr(metadata, "_decode_batch", lambda *args: batch)

    class IdxCopy:
        def __getitem__(self, index):
            return SimpleNamespace(
                copy_=lambda source: recorder.launch("idx copy", "idx", "I")
            )

    metadata.idx_mapping = IdxCopy()  # type: ignore[assignment]

    gdn = [StandInGdnBuilder(recorder, index) for index in range(3)]

    def builder(name: str, reads: str, writes: str):
        return SimpleNamespace(
            build=lambda *args, **kwargs: recorder.launch(name, reads, writes)
        )

    ple = builder("ple build", "R X", "P G0" if late_gdn_write else "P")
    qsa = builder("qsa build", "R", "Q")
    groups = [
        [SimpleNamespace(get_metadata_builder=lambda b=b: b, kernel_block_table=None)]
        for b in (*gdn, ple, qsa)
    ]

    def stock_build(batch) -> dict:
        """The stock build's order: the accepted counts and aligned indices,
        then the groups (the first GDN group builds, the others update)."""
        recorder.launch("accepted counts", "A I R", "X")
        built = {"gdn0": gdn[0].build(), "gdn1": gdn[1].update_block_table()}
        built["gdn2"] = gdn[2].update_block_table()
        ple.build()
        qsa.build()
        return built

    # The forward graph of the key's padded size, captured as the executor
    # places its waits: before the first GDN layer, before the first QSA layer.
    forward = recorder.Graph()
    with recorder.graph(forward):
        waits = CaptureWaits(metadata)
        recorder.launch("embedding and first read", "NG")
        waits.before_gdn()
        recorder.launch("GDN layer", "G0 G1 G2")
        recorder.launch("MoE")
        waits.before_all()
        recorder.launch("QSA layer", "P Q")
        waits.finish(KEY[3])
    metadata.record([KEY], None, groups, [], stock_build, 1024)
    assert metadata.overlapped_keys == [KEY]
    return Model(recorder, metadata, forward, batch)


def program(
    model: Model,
    steps: int = STEPS,
    aborted: int | None = None,
    stock_after: bool = False,
    settle: bool = True,
) -> list[Op]:
    """The host program of `steps` replayed decode steps (step `aborted`
    launches no forward graph), then with `stock_after` a stock step, as the
    runner enqueues them before the GPU runs any."""
    recorder, metadata = model.recorder, model.metadata
    recorder.target = recorder.program = []
    recording = metadata._recordings[KEY]
    for step in range(steps):
        recorder.step = step
        if step and "side" in recorder.record_streams:
            # The allocator's reuse of the freed request indices waits for
            # the stream they were recorded on (record_stream).
            recorder.add(Op("wait", "main", f"free {step - 1}"))
        recorder.record_streams.clear()
        recorder.launch("prep", "", "R idx")
        recorder.launch("preprocess", "", "A")
        metadata._replay(recording, model.batch, overlap=True)
        recorder.launch("ngram", "", "NG")
        if step != aborted:
            model.forward.replay()
            recorder.launch("postprocess", "G0", "A")
            recorder.launch("drafts")
        if "side" in recorder.record_streams:
            recorder.add(Op("record", "side", f"free {step}"))
    if stock_after:
        recorder.step = steps
        if settle:
            metadata.settle()
        recorder.launch("stock build", "R A", "G0 G1 G2 P Q")
        recorder.launch("stock forward", "G0 G1 G2 P Q")
    recorder.target = None
    return recorder.program


def explore(ops: list[Op], bind_when_run: bool = False) -> int:
    """Run `ops` in every order the streams allow; raise `Violation` at a
    read that sees another write than the in-order program, or a hang.
    Returns the number of states. With `bind_when_run` a wait binds to the
    event's last record in the whole program (the host is ahead) when it
    runs."""
    expected: dict[int, dict[str, int | None]] = {}
    last: dict[str, int | None] = {}
    for index, op in enumerate(ops):
        expected[index] = {buf: last.get(buf) for buf in op.reads}
        for buf in op.writes:
            last[buf] = index
    streams = sorted({op.stream for op in ops})
    queues = {s: [i for i, op in enumerate(ops) if op.stream == s] for s in streams}
    position = {i: (s, n) for s, q in queues.items() for n, i in enumerate(q)}
    final_record = {
        op.name: index for index, op in enumerate(ops) if op.kind == "record"
    }

    def done(pcs: tuple[int, ...], index: int) -> bool:
        stream, n = position[index]
        return pcs[streams.index(stream)] > n

    start = (tuple(0 for _ in streams), ())
    seen = {start}
    stack = [start]
    while stack:
        pcs, values = stack.pop()
        current: dict[str, int] = dict(values)
        moved = False
        for k, stream in enumerate(streams):
            if pcs[k] == len(queues[stream]):
                continue
            index = queues[stream][pcs[k]]
            op = ops[index]
            if op.kind == "wait":
                bound = final_record.get(op.name) if bind_when_run else op.bound
                if bound is not None and not done(pcs, bound):
                    continue
            for buf in op.reads:
                if current.get(buf) != expected[index][buf]:
                    raise Violation(
                        f"step {op.step} {op.name} on {stream} read {buf} written by "
                        f"{_name(ops, current.get(buf))}, not "
                        f"{_name(ops, expected[index][buf])}"
                    )
            moved = True
            after = dict(current)
            for buf in op.writes:
                after[buf] = index
            state = (
                pcs[:k] + (pcs[k] + 1,) + pcs[k + 1 :],
                tuple(sorted(after.items())),
            )
            if state not in seen:
                seen.add(state)
                stack.append(state)
        if not moved and any(pcs[k] < len(queues[s]) for k, s in enumerate(streams)):
            raise Violation(f"no stream can move at {dict(zip(streams, pcs))}: hang")
    return len(seen)


def _name(ops: list[Op], index: int | None) -> str:
    """A writer's description."""
    return "nothing" if index is None else f"step {ops[index].step} {ops[index].name}"


def test_the_overlapped_steps_keep_the_stock_order(monkeypatch):
    model = _model(monkeypatch)
    ops = program(model)

    assert explore(ops) > len(ops)
    side = [op for op in ops if op.stream == "side" and op.step == 0]
    assert [(op.kind, op.name) for op in side] == [
        ("wait", "fork 1"),
        ("launch", "idx copy"),
        ("launch", "accepted counts"),
        ("launch", "gdn build 0"),
        ("launch", "gdn update 1"),
        ("launch", "gdn update 2"),
        ("record", "gdn_ready"),
        ("launch", "ple build"),
        ("launch", "qsa build"),
        ("record", "ready"),
        ("record", "free 0"),
    ]
    # Each step's forward graph waits bind to that step's records.
    for op in ops:
        if op.kind == "wait" and op.name in ("gdn_ready", "ready"):
            assert op.bound is not None
            assert op.stream == "main" and ops[op.bound].step == op.step


def test_a_step_without_a_forward_graph_keeps_the_order(monkeypatch):
    """A replayed step that launches no forward graph: the next step's reuse
    of its request indices waits for the side stream (`record_stream`), and
    so does a stock build after it (`settle`)."""
    explore(program(_model(monkeypatch), aborted=1))
    explore(program(_model(monkeypatch), steps=2, aborted=1, stock_after=True))


def _drop(ops: list[Op], match: Callable[[Op], bool]) -> list[Op]:
    """`ops` without the first operation of step 1 that `match` selects,
    waits rebound to the records still enqueued before them."""
    index = next(i for i, op in enumerate(ops) if op.step == 1 and match(op))
    kept = ops[:index] + ops[index + 1 :]
    return _rebind(kept)


def _rebind(ops: list[Op]) -> list[Op]:
    """`ops` with every wait bound to its event's last record before it."""
    out: list[Op] = []
    for op in ops:
        if op.kind == "wait":
            records = [
                i for i, o in enumerate(out) if o.kind == "record" and o.name == op.name
            ]
            op = dataclasses.replace(op, bound=records[-1] if records else None)
        out.append(op)
    return out


def _wait_after_reader(model: Model) -> list[Op]:
    """The forward graph's GDN wait moved after the GDN layer."""
    ops = model.forward.ops
    wait = next(op for op in ops if op.kind == "wait" and op.name == "gdn_ready")
    ops.remove(wait)
    at = next(i for i, op in enumerate(ops) if op.name == "GDN layer")
    ops.insert(at + 1, wait)
    return program(model)


DEFECTS: dict[str, Callable] = {
    "no fork wait on the side stream": lambda m: _drop(
        program(m), lambda op: op.kind == "wait" and op.name.startswith("fork")
    ),
    "no ready record": lambda m: _drop(
        program(m), lambda op: op.kind == "record" and op.name == "ready"
    ),
    "no gdn_ready record node": lambda m: _drop(
        program(m), lambda op: op.kind == "record" and op.name == "gdn_ready"
    ),
    "wait after the layer that reads": _wait_after_reader,
    "no record_stream": lambda m: _rebind(
        [op for op in program(m, aborted=1) if not op.name.startswith("free")]
    ),
}


@pytest.mark.parametrize("defect", DEFECTS)
def test_the_check_catches_a_broken_order(monkeypatch, defect):
    model = _model(monkeypatch)
    explore(program(model))
    with pytest.raises(Violation, match="read"):
        explore(DEFECTS[defect](_model(monkeypatch)))


def test_the_check_catches_a_late_gdn_metadata_write(monkeypatch):
    with pytest.raises(Violation, match="GDN layer .* read G0"):
        explore(program(_model(monkeypatch, late_gdn_write=True)))


def test_the_check_catches_a_stock_build_without_settle(monkeypatch):
    ops = program(_model(monkeypatch), 2, aborted=1, stock_after=True, settle=False)
    with pytest.raises(Violation, match="read"):
        explore(ops)


def test_waits_bound_when_they_run_would_hang(monkeypatch):
    ops = program(_model(monkeypatch))
    with pytest.raises(Violation, match="hang"):
        explore(ops, bind_when_run=True)
