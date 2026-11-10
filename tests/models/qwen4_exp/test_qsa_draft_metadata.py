# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place draft decode metadata of the Qwen4Exp QSA indexer caches.

Model Runner V2 records one CUDA graph for all draft decode steps when every
draft attention backend can refresh its metadata in place. The QSA indexer
builder's refresh must leave, at every replay, the rows that a fresh `build`
of the replayed batch writes, padding rows included, and must stay ordered
with the draft model's reads of those rows.
"""

import ast
import inspect
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.qwen4_exp.common import qsa_indexer
from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadataBuilder
from vllm.models.qwen4_exp.common.qsa_indexer import (
    QSAIndexerBackend,
    QSAIndexerMetadata,
    QSAIndexerMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

COMPRESS_RATIO = 4
BLOCK_SIZE = 64
# The raw-key ring of a depth-3 speculator: round_up(4 + 3, 4).
RING_CAPACITY = 8
BLOCK_TABLE_COLUMNS = 16
MAX_NUM_SEQS = 16
MAX_NUM_TOKENS = 64
NULL_BLOCK = 0
# Written over the metadata rows before each refresh, so a row the refresh
# skips cannot match the reference by chance.
SENTINEL = 1 << 20

ROW_FIELDS = (
    "token_to_req",
    "logical_positions",
    "visible_blocks",
    "slot_mapping",
    "k_work_metadata",
)


def _make_spec(kind: str) -> CircularBufferSpec | MLAAttentionSpec:
    if kind == "raw":
        return CircularBufferSpec(
            block_size=RING_CAPACITY,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
        )
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=COMPRESS_RATIO if kind == "compressed" else 1,
    )


def _make_builder(kind: str, device: str = "cuda") -> QSAIndexerMetadataBuilder:
    vllm_config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_TOKENS, max_num_seqs=MAX_NUM_SEQS
        ),
    )
    return QSAIndexerMetadataBuilder(
        _make_spec(kind),
        ["indexer_cache"],
        vllm_config,  # type: ignore[arg-type]
        torch.device(device),
    )


@dataclass
class DraftStep:
    """One draft decode step of the live requests: their sequence lengths and
    common slots (-1 where the runner maps none)."""

    seq_lens: list[int]
    slots: list[int]


class RunnerBuffers:
    """The runner's persistent tensors for one padded draft decode shape, which
    a FULL graph reads at fixed addresses."""

    def __init__(self, num_reqs_padded: int) -> None:
        device = torch.device("cuda")
        self.num_reqs_padded = num_reqs_padded
        self.query_start_loc = torch.zeros(
            num_reqs_padded + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32, device=device)
        self.slot_mapping = torch.full(
            (num_reqs_padded,), -1, dtype=torch.int64, device=device
        )
        # Stale block ids in the padding rows: the null block must not be what
        # keeps padding rows inert.
        self.block_table = torch.randint(
            1,
            1 << 12,
            (num_reqs_padded, BLOCK_TABLE_COLUMNS),
            dtype=torch.int32,
            device=device,
        )
        self.query_start_loc_cpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)

    def set(self, step: DraftStep) -> CommonAttentionMetadata:
        """Write `step` as the runner does: one query token per live request,
        padding requests with no tokens and zero sequence length."""
        num_live = len(step.seq_lens)
        assert len(step.slots) == num_live <= self.num_reqs_padded
        padding = self.num_reqs_padded - num_live
        self.query_start_loc_cpu.copy_(
            torch.arange(self.num_reqs_padded + 1).clamp_max(num_live)
        )
        self.query_start_loc.copy_(self.query_start_loc_cpu)
        self.seq_lens.copy_(torch.tensor(step.seq_lens + [0] * padding))
        self.slot_mapping.copy_(torch.tensor(step.slots + [-1] * padding))
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc,
            query_start_loc_cpu=self.query_start_loc_cpu.clone(),
            seq_lens=self.seq_lens,
            num_reqs=self.num_reqs_padded,
            num_actual_tokens=self.num_reqs_padded,
            max_query_len=1,
            max_seq_len=max(step.seq_lens),
            block_table_tensor=self.block_table,
            slot_mapping=self.slot_mapping,
        )


def _poison(metadata: QSAIndexerMetadata, kind: str) -> None:
    for name in ROW_FIELDS:
        if name == "slot_mapping" and kind == "uncompressed":
            # The runner's own slot mapping, which the builder only references.
            continue
        getattr(metadata, name).fill_(SENTINEL)


def _assert_rows_equal(
    actual: dict[str, torch.Tensor], expected: QSAIndexerMetadata
) -> None:
    for name in ROW_FIELDS:
        want = getattr(expected, name)
        assert torch.equal(actual[name], want), (
            f"{name}: {actual[name].tolist()} != {want.tolist()}"
        )


def _draft_steps(num_reqs_padded: int) -> list[DraftStep]:
    """Batches one graph replays: first all of the captured requests, then fewer
    live requests over two draft steps, then all of them again. The lengths
    cross compression group boundaries."""

    def step(num_live: int, base: int) -> DraftStep:
        seq_lens = [base + 37 * req for req in range(num_live)]
        slots = [1000 + 3 * req for req in range(num_live)]
        if num_live > 1:
            slots[1] = -1  # a live token the runner maps to no slot
        return DraftStep(seq_lens, slots)

    num_live = max(1, num_reqs_padded - 3)
    return [
        step(num_reqs_padded, 128),
        step(num_live, 201),
        step(num_live, 202),
        step(num_reqs_padded, 203),
    ]


def test_qsa_draft_backends_update_metadata_in_place() -> None:
    """Model Runner V2 fuses the draft decode steps only if every draft
    backend, the QSA main cache's and the indexer caches', refreshes its
    metadata in place."""
    assert Qwen4ExpQSAMetadataBuilder.supports_draft_decode_metadata_update
    indexer_builder = QSAIndexerBackend.get_builder_cls()
    assert indexer_builder.supports_draft_decode_metadata_update is HAS_TRITON


@requires_qsa_kernels
@pytest.mark.parametrize("kind", ["raw", "compressed", "uncompressed"])
@pytest.mark.parametrize("num_reqs_padded", [1, 4, 8, 16])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "cuda_graph"])
def test_qsa_draft_decode_update_matches_build(
    kind: str, num_reqs_padded: int, graph: bool
) -> None:
    steps = _draft_steps(num_reqs_padded)
    buffers = RunnerBuffers(num_reqs_padded)
    builder = _make_builder(kind)
    # Built once, as for a CUDA graph capture of the padded shape.
    metadata = builder.build(0, buffers.set(steps[0]))
    if kind == "compressed":
        assert metadata.k_work_metadata.numel() > 0

    rows = {name: getattr(metadata, name) for name in ROW_FIELDS}
    cuda_graph: torch.cuda.CUDAGraph | None = None
    # Read in the same graph right after the refresh, as the draft model's
    # kernels read the rows.
    snapshot = {name: torch.empty_like(rows[name]) for name in ROW_FIELDS}
    if graph:
        builder.update_draft_decode_metadata(metadata)  # compile outside capture
        torch.accelerator.synchronize()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            builder.update_draft_decode_metadata(metadata)
            for name in ROW_FIELDS:
                snapshot[name].copy_(rows[name])

    for step in steps:
        common = buffers.set(step)
        _poison(metadata, kind)
        if cuda_graph is None:
            builder.update_draft_decode_metadata(metadata)
        else:
            cuda_graph.replay()
        torch.accelerator.synchronize()
        expected = _make_builder(kind).build(0, common)
        _assert_rows_equal(rows, expected)
        if cuda_graph is not None:
            _assert_rows_equal(snapshot, expected)


# The refresh's ordering in the draft decode graph, checked by simulation
# (QSAIndexerMetadataBuilder.update_draft_decode_metadata states the
# argument). Model Runner V2's fused draft loop runs with stand-ins that
# record each kernel it enqueues as a Node, in stream order; the refresh's
# accesses come from the metadata kernel's source. completed_before() then
# applies CUDA's rules: an edge into a node without the PDL attribute starts
# it after its predecessor completed and its stores are visible; an edge into
# a PDL node starts it once the predecessor triggers (its first
# launch_dependents, else its exit, which is not completion) and only the
# node's own wait waits for the predecessor to complete.
Access = tuple[str, str]  # (wait | trigger | load | store, buffer)
WAIT: Access = ("wait", "")
TRIGGER: Access = ("trigger", "")
# The indexer caches of a QSA layer whose refreshes the draft step records.
GROUPS = ("raw", "compressed")


@dataclass(frozen=True)
class Node:
    """A kernel of the draft decode graph: whether it is launched with the
    PDL attribute, and its global accesses and PDL controls in program
    order."""

    name: str
    pdl: bool
    accesses: tuple[Access, ...]


def kernel_accesses() -> list[Access]:
    """The metadata kernel's loads, stores (named by pointer parameter),
    waits and triggers in source order."""
    kernel = qsa_indexer._build_qsa_metadata_kernel.fn
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    accesses: list[Access] = []
    for call in sorted(calls, key=lambda call: (call.lineno, call.col_offset)):
        function = ast.unparse(call.func)
        if function.endswith(".gdc_wait"):
            accesses.append(WAIT)
        elif function.endswith(".gdc_launch_dependents"):
            accesses.append(TRIGGER)
        elif function in ("tl.load", "tl.store"):
            pointer = call.args[0]
            while isinstance(pointer, ast.BinOp):
                pointer = pointer.left
            assert isinstance(pointer, ast.Name)
            accesses.append((function.removeprefix("tl."), pointer.id))
    return accesses


def completed_before(nodes: list[Node]) -> list[list[frozenset[str]]]:
    """For each access of each node, the nodes that have completed, their
    stores visible, before the access runs."""
    seen: list[list[frozenset[str]]] = []
    done: frozenset[str] = frozenset()  # complete once the previous node is
    released: frozenset[str] = frozenset()  # complete once it triggers
    for node in nodes:
        now = released if node.pdl else done
        trigger: frozenset[str] | None = None
        seen.append([])
        for access in node.accesses:
            if access == WAIT and node.pdl:
                now = now | done
            if access == TRIGGER and trigger is None:
                trigger = now
            seen[-1].append(now)
        released = now if trigger is None else trigger
        done = now | {node.name}
    return seen


def violations(nodes: list[Node]) -> list[str]:
    """Each access to a buffer that an earlier node stores, or a store to a
    buffer an earlier node accesses, without the earlier node complete."""
    found: list[str] = []
    for index, (node, seen) in enumerate(zip(nodes, completed_before(nodes))):
        for (kind, buffer), done in zip(node.accesses, seen):
            if kind not in ("load", "store"):
                continue
            for earlier in nodes[:index]:
                conflict = any(
                    other == buffer and "store" in (kind, other_kind)
                    for other_kind, other in earlier.accesses
                )
                if conflict and earlier.name not in done:
                    found.append(
                        f"{node.name} {kind}s {buffer} while {earlier.name} "
                        "may still run"
                    )
    return found


def launched_early(nodes: list[Node]) -> list[Node]:
    """`nodes` with every node but the refreshes launched with PDL and
    triggering at entry, the earliest that a node keeping the PDL contract
    lets its successor start; a node without a wait gets one before its
    accesses."""
    early = []
    for node in nodes:
        if not node.name.startswith("refresh"):
            accesses = node.accesses
            if WAIT not in accesses:
                accesses = (WAIT, *accesses)
            node = replace(node, pdl=True, accesses=(TRIGGER, *accesses))
        early.append(node)
    return early


def check_draft_steps(nodes: list[Node]) -> list[str]:
    """The order violations of the recorded draft steps, and of the same
    steps with every other node launched early: the refresh must not depend
    on how the runner launches the nodes around it, beyond `fill_`
    executing no wait."""
    return [f"as recorded: {found}" for found in violations(nodes)] + [
        f"with every other node launched early: {found}"
        for found in violations(launched_early(nodes))
    ]


class DraftStepRecorder:
    """Stand-ins for the runner's slot mappings, step counter and draft
    model that record what the fused draft loop enqueues. The draft forward
    reads every refreshed row; the draft input update advances the positions
    and sequence lengths that the next step reads."""

    def __init__(self, kernel: list[Access]) -> None:
        self.kernel = kernel
        self.nodes: list[Node] = []
        self.step = 0

    def compute_slot_mappings(self, *args: object) -> None:
        stores = (("store", f"{group} common_slot_mapping") for group in GROUPS)
        self.nodes.append(
            Node(
                f"compute slot mappings {self.step + 1}",
                False,
                (("load", "query_start_loc"), ("load", "positions"), *stores),
            )
        )

    def launch_refresh(self, *buffers: str | None, **kwargs: object) -> None:
        """`_launch_qsa_metadata_kernel`, whose leading arguments are the
        kernel's pointers in order."""
        kernel = qsa_indexer._build_qsa_metadata_kernel.fn
        pointers = dict(zip(inspect.signature(kernel).parameters, buffers))
        assert all(pointer.endswith("_ptr") for pointer in pointers)
        accesses = tuple(
            access if access in (WAIT, TRIGGER) else (access[0], pointers[access[1]])
            for access in self.kernel
            if access in (WAIT, TRIGGER) or pointers[access[1]] is not None
        )
        group = str(pointers["token_to_req_ptr"]).split()[0]
        self.nodes.append(Node(f"refresh {group} {self.step + 1}", True, accesses))

    def fill(self, step: int) -> None:
        """`current_draft_step.fill_`: a PyTorch kernel, without a wait."""
        self.step = step
        store = ("store", "current_draft_step")
        self.nodes.append(Node(f"fill_ {step}", False, (store,)))

    def decode(self, *args: object) -> None:
        rows = [
            f"{group} {field}"
            for group in GROUPS
            for field in ROW_FIELDS
            if field != "k_work_metadata" or group == "compressed"
        ]
        inputs = [f"{group} common_slot_mapping" for group in GROUPS]
        inputs += ["seq_lens", "positions", "current_draft_step"]
        reads = tuple(("load", buffer) for buffer in rows + inputs)
        self.nodes.append(Node(f"draft forward {self.step}", True, (WAIT, *reads)))
        self.nodes.append(
            Node(
                f"draft input update {self.step}",
                False,
                (
                    ("load", "current_draft_step"),
                    ("store", "positions"),
                    ("store", "seq_lens"),
                ),
            )
        )


def record_draft_steps(monkeypatch, kernel: list[Access] | None = None) -> list[Node]:
    """What Model Runner V2's fused draft loop enqueues for three draft
    tokens (fused steps 1 and 2) with a raw and a compressed indexer cache,
    whose refreshes run the real builders' update."""
    recorder = DraftStepRecorder(kernel_accesses() if kernel is None else kernel)
    monkeypatch.setattr(
        qsa_indexer, "_launch_qsa_metadata_kernel", recorder.launch_refresh
    )
    builders = {group: _make_builder(group, "cpu") for group in GROUPS}
    # Stand-in metadata whose tensors are the names of the buffers.
    attn_metadata = {
        group: SimpleNamespace(
            num_prefills=0,
            query_start_loc="query_start_loc",
            seq_lens="seq_lens",
            **{
                field: f"{group} {field}"
                for field in (*ROW_FIELDS, "common_slot_mapping", "block_table")
            },
        )
        for group in GROUPS
    }
    # AttentionGroup.update_draft_decode_metadata, one group per cache.
    attn_groups = [
        SimpleNamespace(
            update_draft_decode_metadata=lambda metadata, group=group: builders[
                group
            ].update_draft_decode_metadata(metadata[group])
        )
        for group in GROUPS
    ]
    runner = SimpleNamespace(
        idx_mapping=torch.zeros(2, dtype=torch.int32),
        input_buffers=SimpleNamespace(
            positions=torch.zeros(2, dtype=torch.int64),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
        ),
        advance_draft_positions=True,
        block_tables=SimpleNamespace(
            compute_slot_mappings=recorder.compute_slot_mappings, cp_size=1
        ),
        attn_groups=[attn_groups],
        current_draft_step=SimpleNamespace(fill_=recorder.fill),
        _decode=recorder.decode,
    )
    runner._update_draft_decode_metadata = MethodType(
        DraftModelSpeculator._update_draft_decode_metadata, runner
    )
    AutoRegressiveSpeculator._fused_multi_step_decode(
        runner, 2, 2, attn_metadata, None, None, CUDAGraphMode.FULL, 3
    )
    return recorder.nodes


requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="reads a Triton kernel")


@requires_triton
def test_the_draft_steps_order_the_refresh(monkeypatch):
    nodes = record_draft_steps(monkeypatch)

    assert [(node.name, node.pdl) for node in nodes[:6]] == [
        ("compute slot mappings 1", False),
        ("refresh raw 1", True),
        ("refresh compressed 1", True),
        ("fill_ 1", False),
        ("draft forward 1", True),
        ("draft input update 1", False),
    ]
    assert len(nodes) == 12
    assert check_draft_steps(nodes) == []


def _wait_after_first_store(kernel: list[Access]) -> list[Access]:
    kernel = [access for access in kernel if access != WAIT]
    first_store = next(i for i, (kind, _) in enumerate(kernel) if kind == "store")
    return kernel[: first_store + 1] + [WAIT] + kernel[first_store + 1 :]


def _launched_with_pdl(prefix: str) -> Callable[[list[Node]], list[Node]]:
    return lambda nodes: [
        replace(node, pdl=True) if node.name.startswith(prefix) else node
        for node in nodes
    ]


def _forward_reads_before_its_wait(nodes: list[Node]) -> list[Node]:
    return [
        replace(node, accesses=(*node.accesses[1:], WAIT))
        if node.name.startswith("draft forward")
        else node
        for node in nodes
    ]


# Defects injected into the metadata kernel or the recorded steps, each with a
# violation the check must report. The refresh's wait placement matters once
# the nodes before it can start early: as recorded, compute_slot_mappings,
# launched without PDL, already orders the refresh. Likewise fill_, launched
# without PDL, keeps a forward that reads before its wait correct as recorded.
DEFECTS: dict[
    str,
    tuple[
        Callable[[list[Access]], list[Access]],
        Callable[[list[Node]], list[Node]],
        str,
    ],
] = {
    "fill_ launched with PDL": (
        list,
        _launched_with_pdl("fill_"),
        (
            "as recorded: draft forward 1 loads compressed token_to_req while "
            "refresh compressed 1 may still run"
        ),
    ),
    "the refresh waiting after its first store": (
        _wait_after_first_store,
        list,
        (
            "with every other node launched early: refresh raw 2 stores raw "
            "token_to_req while draft forward 1 may still run"
        ),
    ),
    # The draft input update executes no wait, so with PDL it no longer
    # waits for the forward before it.
    "a step 1 reader running when the step 2 refresh writes": (
        list,
        _launched_with_pdl("draft input update"),
        (
            "as recorded: refresh raw 2 stores raw token_to_req while draft "
            "forward 1 may still run"
        ),
    ),
    "the draft forward reading the rows before its wait": (
        list,
        _forward_reads_before_its_wait,
        (
            "with every other node launched early: draft forward 1 loads "
            "compressed token_to_req while refresh compressed 1 may still run"
        ),
    ),
}


@requires_triton
@pytest.mark.parametrize("defect", DEFECTS)
def test_the_order_check_catches_a_broken_refresh_order(monkeypatch, defect):
    in_kernel, in_steps, expected = DEFECTS[defect]
    nodes = in_steps(record_draft_steps(monkeypatch, in_kernel(kernel_accesses())))

    assert expected in check_draft_steps(nodes)


@requires_triton
def test_no_trigger_position_breaks_the_refresh_order(monkeypatch):
    """The compressed cache's work tiles trigger before their stores. A
    trigger publishes no store, so the order holds wherever it sits, even
    before the wait: readers see the rows only through completion."""
    kernel = kernel_accesses()
    trigger = kernel.index(TRIGGER)
    assert any(kind == "store" for kind, _ in kernel[trigger:])
    for position in range(trigger):
        moved = kernel[:trigger] + kernel[trigger + 1 :]
        moved.insert(position, TRIGGER)
        assert check_draft_steps(record_draft_steps(monkeypatch, moved)) == []


@triton.jit
def _store_one_kernel(out_ptr, launch_pdl: tl.constexpr):
    tl.store(out_ptr, 1)


def _capture_refreshes(successor: Callable[[], object]) -> None:
    """Capture two draft steps that refresh a raw and a compressed indexer
    cache and run `successor` after each step's refreshes."""
    buffers = RunnerBuffers(4)
    common = buffers.set(_draft_steps(4)[0])
    builders = {group: _make_builder(group) for group in GROUPS}
    metadata = {group: builders[group].build(0, common) for group in GROUPS}
    for group in GROUPS:  # compile outside the capture
        builders[group].update_draft_decode_metadata(metadata[group])
    torch.accelerator.synchronize()
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        for _ in range(2):
            buffers.slot_mapping.add_(1)
            for group in GROUPS:
                builders[group].update_draft_decode_metadata(metadata[group])
            successor()


requires_capture_check = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not HAS_TRITON
    or not qsa_indexer._can_check_captures(),
    reason="the capture check needs CUDA, Triton and cuda-python",
)


@requires_capture_check
def test_the_capture_check_accepts_a_successor_that_waits_for_the_refresh():
    step = torch.zeros((), dtype=torch.int64, device="cuda")
    _capture_refreshes(lambda: step.fill_(1))


@requires_capture_check
@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_arch_support_pdl(),
    reason="needs PDL",
)
def test_the_capture_check_rejects_a_pdl_successor():
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    _store_one_kernel[(1,)](out, launch_pdl=True)  # compile outside the capture

    with pytest.raises(RuntimeError, match="programmatic edge"):
        _capture_refreshes(lambda: _store_one_kernel[(1,)](out, launch_pdl=True))


@requires_capture_check
def test_the_capture_check_rejects_a_refresh_off_the_capturing_stream(monkeypatch):
    monkeypatch.setattr(
        qsa_indexer, "_launch_qsa_metadata_kernel", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError, match="capturing stream"):
        _capture_refreshes(lambda: None)


def test_the_capture_check_probes_once_and_warns_without_bindings(monkeypatch):
    """The builder probes the capture check's bindings when it is built,
    outside any capture; without them the check is off, with one warning."""
    probes: list[bool] = []
    monkeypatch.setattr(qsa_indexer, "_can_check_captures", lambda: probes.append(True))
    _make_builder("raw", "cpu")
    assert probes == [True]
    monkeypatch.undo()

    warnings: list[str] = []
    monkeypatch.setattr(qsa_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setitem(sys.modules, "cuda.bindings", None)  # the import fails
    monkeypatch.setattr(
        qsa_indexer.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    qsa_indexer._can_check_captures.cache_clear()
    try:
        assert not qsa_indexer._can_check_captures()
        assert not qsa_indexer._can_check_captures()
    finally:
        qsa_indexer._can_check_captures.cache_clear()
    assert len(warnings) == 1 and "capture-time check" in warnings[0]
