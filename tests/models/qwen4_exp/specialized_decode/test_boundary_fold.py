# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The folded routed finalize of Qwen4Exp's specialized decode: FlashInfer's
TRT-LLM NVFP4 experts stop after GEMM2 and the boundary kernel
(`SublayerBoundary.moe_folded`) runs their top-10 reduction.

- Bits. FlashInfer's MoE on random NVFP4 experts, deferred and then
  `moe_folded`, against the same call finalized and then
  `SublayerBoundary.moe`: equal bits at 4, 16, 32, 64, 118, 119 and 128
  rows (both finalize kernels; the kernel itself, though steps below 32 rows
  do not fold), with every expert local (TP 1) or a quarter of them (an
  expert-parallel rank of four), eagerly and in a CUDA graph, at TP 1 and TP 4.
- Gate. `Boundaries.moe_output` folds only in steps of `FOLD_MIN_ROWS`
  (32) rows or more: at 4 and 16 rows the experts run their finalize without
  the post-GEMM2 hook, from 32 to 128 rows the hook engages.
- Capture checks. In a capture, `DeferredFinalize` and `require_follows`
  pass on the deferred call followed by the boundary kernel, and raise on a
  finalized call or a kernel between GEMM2 and the boundary kernel.
- Order. `runtime/boundary.py` states who waits on what and that the GEMM2
  rows, routing weights and slots are not reused before the boundary kernel
  has read them. A simulation checks both on what `Boundaries.moe_output`
  and `Boundaries.read` enqueue and release over three MoE layers, with the
  boundary kernel's accesses read from its source, and reports the injected
  defects: the boundary reading before its wait, a kernel after it without a
  wait, and the rows released before the boundary kernel is enqueued.
"""

import ast
import gc
import inspect
import textwrap
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import ray
import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock

from ....utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

STREAMS = 4
HIDDEN = 2560
WIDTH = STREAMS * HIDDEN
EPS = 1e-6
EXPERTS = 512
TOP_K = 10
INTERMEDIATE = 640
EP_SIZE = 4
LOCAL_EXPERTS = EXPERTS // EP_SIZE
# 118 and 119 rows straddle FlashInfer's switch between its two finalize kernels.
ROWS = (4, 16, 32, 64, 118, 119, 128)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode's boundary kernel requires SM100",
)


def _experts(device: torch.device) -> dict[str, torch.Tensor]:
    """Random NVFP4 experts: packed FP4 bytes, E4M3 block scales in [0.25, 1)
    and per-expert output scales. Any bytes are valid weights, so FlashInfer's
    weight shuffle is left out."""
    generator = torch.Generator(device=device).manual_seed(0)

    def randint(low: int, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(
            low, high, shape, dtype=torch.uint8, generator=generator, device=device
        )

    e, h, i = EXPERTS, HIDDEN, INTERMEDIATE
    scale = torch.full((e,), 1 / 64, dtype=torch.float32, device=device)
    return {
        "gemm1_weights": randint(0, 256, e, 2 * i, h // 2),
        "gemm1_weights_scale": randint(0x28, 0x38, e, 2 * i, h // 16).view(
            torch.float8_e4m3fn
        ),
        "gemm2_weights": randint(0, 256, e, h, i // 2),
        "gemm2_weights_scale": randint(0x28, 0x38, e, h, i // 16).view(
            torch.float8_e4m3fn
        ),
        "output1_scale_scalar": scale,
        "output1_scale_gate_scalar": scale.clone(),
        "output2_scale_scalar": scale.clone(),
    }


@dataclass
class MoeInputs:
    """A MoE call's inputs: the NVFP4 activation (as the stock prepare step
    quantizes it, row-major scales) and the router logits."""

    values: torch.Tensor
    scales: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def random(cls, rows: int, seed: int, device: torch.device) -> "MoeInputs":
        """Inputs of `rows` tokens."""
        from vllm import _custom_ops as ops

        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(rows, HIDDEN, generator=generator, device=device)
        logits = torch.randn(rows, EXPERTS, generator=generator, device=device)
        one = torch.ones(1, dtype=torch.float32, device=device)
        values, scales = ops.scaled_fp4_quant(
            x.to(torch.bfloat16), one, is_sf_swizzled_layout=False
        )
        return cls(values, scales, logits.to(torch.bfloat16))

    def copy_(self, other: "MoeInputs") -> None:
        """Overwrite with `other`'s values, for a graph replay."""
        self.values.copy_(other.values)
        self.scales.copy_(other.scales)
        self.logits.copy_(other.logits)


def _moe(
    experts: dict[str, torch.Tensor], inputs: MoeInputs, rank: int | None, deferred
):
    """FlashInfer's TRT-LLM NVFP4 MoE as the stock experts call it, on every
    expert (`rank` None) or on an expert-parallel rank's quarter: the
    finalized output, or with `deferred` the GEMM2 rows, routing weights and
    slots (given the finalized-shape `output`, as `deferred_finalize`
    does)."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    first, count = (
        (0, EXPERTS) if rank is None else (rank * LOCAL_EXPERTS, LOCAL_EXPERTS)
    )
    local = {name: tensor[first : first + count] for name, tensor in experts.items()}
    rows = inputs.values.shape[0]
    output = torch.empty(
        rows, HIDDEN, dtype=torch.bfloat16, device=inputs.values.device
    )
    result = trtllm_fp4_block_scale_moe(
        routing_logits=inputs.logits,
        routing_bias=None,
        hidden_states=inputs.values,
        hidden_states_scale=inputs.scales.view(torch.float8_e4m3fn).reshape(rows, -1),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_bias=None,
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=first,
        local_num_experts=count,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize,
        do_finalize=not deferred,
        output=output,
        tune_max_num_tokens=256,
        **local,
    )
    if not deferred:
        return output
    rows_out, weights, slots = result
    return rows_out, weights.view(rows, TOP_K), slots.view(rows, TOP_K)


def _site(rows: int, seed: int, device: torch.device, zero: bool):
    """The boundary's other operands: streams, injection logits, norm weight,
    shared expert output and its gate logits; with `zero` all zeros, so that
    the new streams are the routed output itself."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        value = torch.randn(*shape, generator=generator, device=device) * scale
        return (value * (not zero)).to(torch.bfloat16)

    return {
        "streams": randn(rows, WIDTH),
        "injection": randn(rows, STREAMS),
        "norm": randn(WIDTH, scale=0.1) + (0.5 if zero else 0.0),
        "shared": randn(rows, HIDDEN),
        "logit": randn(rows, 1),
    }


def _stock(boundary, experts, inputs, site, rank):
    """Finalize, then the boundary kernel on the routed output."""
    routed = _moe(experts, inputs, rank, deferred=False)
    fused = boundary.moe(
        routed,
        site["shared"],
        site["logit"],
        site["streams"],
        site["injection"],
        site["norm"],
        EPS,
    )
    return routed, fused


def _folded(boundary, experts, inputs, site, rank, check: bool = False):
    """The deferred call, then the boundary kernel with the folded finalize;
    with `check`, under the runtime's capture checks."""
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.boundary import (
        DeferredFinalize,
        require_follows,
    )

    rows_out, weights, slots = _moe(experts, inputs, rank, deferred=True)
    hook = DeferredFinalize()
    if check:
        hook(UnfinalizedMoEOutput(rows_out, weights, slots), rows_out)
        assert hook.gemm2_node is not None
    fused = boundary.moe_folded(
        rows_out,
        weights,
        slots,
        site["shared"],
        site["logit"],
        site["streams"],
        site["injection"],
        site["norm"],
        EPS,
    )
    if check:
        require_follows(hook.gemm2_node)
    return fused


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _compare(boundary, experts, rows, rank, zero, graph):
    """The folded and the stock paths on the same inputs: equal bits, and
    with `zero` operands the new streams equal the finalized output."""
    device = torch.device("cuda")
    inputs = MoeInputs.random(rows, rows, device)
    site = _site(rows, rows + 1, device, zero)
    if graph:
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            routed, stock = _stock(boundary, experts, inputs, site, rank)
            folded = _folded(boundary, experts, inputs, site, rank, check=True)
        inputs.copy_(MoeInputs.random(rows, rows + 2, device))
        cuda_graph.replay()
    else:
        routed, stock = _stock(boundary, experts, inputs, site, rank)
        folded = _folded(boundary, experts, inputs, site, rank)
    torch.accelerator.synchronize()
    for ours, theirs in zip(folded, stock):
        _assert_bits_equal(ours, theirs)
    if zero:
        assert torch.equal(folded[0][:, :HIDDEN], routed)
    if rank is not None:
        # Some tokens have experts on other ranks (-1 slots).
        slots = _moe(experts, inputs, rank, deferred=True)[2]
        assert bool((slots < 0).any()) and bool((slots >= 0).any())


@pytest.fixture(scope="module")
def experts() -> dict[str, torch.Tensor]:
    return _experts(torch.device("cuda"))


@requires_sm100
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("zero", [False, True], ids=["random", "zero"])
@pytest.mark.parametrize("rank", [None, 0, 3], ids=["tp1", "ep-rank0", "ep-rank3"])
@pytest.mark.parametrize("rows", ROWS)
def test_folded_finalize_matches_stock(experts, rows, rank, zero, graph):
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary.all_reduce_hc import (
        finalize_groups,
    )

    assert finalize_groups(rows) == (4 if rows <= 118 else 1)
    boundary = SublayerBoundary(None, 256, torch.device("cuda"))
    _compare(boundary, experts, rows, rank, zero, graph)


@requires_sm100
def test_capture_checks_reject_a_finalize_or_a_kernel_in_between(experts):
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.boundary import (
        DeferredFinalize,
        require_follows,
    )

    device = torch.device("cuda")
    boundary = SublayerBoundary(None, 256, device)
    inputs = MoeInputs.random(16, 0, device)
    site = _site(16, 1, device, zero=False)
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        # A finalized call ends with the finalize kernel, not GEMM2.
        routed = _moe(experts, inputs, None, deferred=False)
        unfinalized = UnfinalizedMoEOutput(routed, routed, routed)
        with pytest.raises(RuntimeError, match="did not end with GEMM2"):
            DeferredFinalize()(unfinalized, routed)
        # A kernel between GEMM2 and the boundary kernel.
        rows_out, weights, slots = _moe(experts, inputs, None, deferred=True)
        hook = DeferredFinalize()
        hook(UnfinalizedMoEOutput(rows_out, weights, slots), rows_out)
        site["shared"].mul_(1)
        boundary.moe_folded(
            rows_out,
            weights,
            slots,
            site["shared"],
            site["logit"],
            site["streams"],
            site["injection"],
            site["norm"],
            EPS,
        )
        assert hook.gemm2_node is not None
        with pytest.raises(RuntimeError, match="right after the routed experts"):
            require_follows(hook.gemm2_node)


@ray.remote(num_gpus=1, max_calls=1)
def _tp_fold(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary import (
        SublayerBoundary,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        boundary = SublayerBoundary(group, 64, device)
        experts = _experts(device)
        for rows in (4, 16, 32, 64):
            for graph in (False, True):
                _compare(boundary, experts, rows, rank, zero=False, graph=graph)


@requires_sm100
def test_tp4_folded_finalize_matches_stock(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _tp_fold)


# The order of the folded finalize (runtime/boundary.py, points 1 and 2),
# checked by simulation as in tests/models/qwen4_exp/test_qsa_draft_metadata.py.
# Stand-ins for the MoE runner, the boundary kernel and the read's projections
# record each kernel that the real Boundaries.moe_output and Boundaries.read
# enqueue as a Node, in stream order, and each buffer's allocation and release
# in host order (releases from the stand-in buffers' finalizers, so they follow
# the real code's references). The boundary kernel's accesses come from its
# source. completed_before() applies CUDA's rules: an edge into a node without
# the PDL attribute starts it after its predecessor completed; an edge into a
# PDL node starts it once the predecessor triggers (its first
# launch_dependents, else its exit, which is not completion), and only the
# node's own wait waits for the predecessor to complete. A buffer allocated
# after another was released may take its memory.
Access = tuple[str, str]  # (wait | trigger | load | store, buffer)
WAIT: Access = ("wait", "")
TRIGGER: Access = ("trigger", "")
# The boundary kernel's global buffers; its mailbox and protocol words keep
# their own order (all_reduce_hc.py).
KERNEL_BUFFERS = (
    "partial",
    "shared",
    "gate",
    "expert_weights",
    "expert_slots",
    "streams",
    "injection",
    "norm",
    "streams_out",
    "normed",
)
# The kernel's global memory helpers and the argument holding the address.
GLOBAL_ACCESSES = {
    "load_x4": ("load", 0),
    "load_x4_if": ("load", 0),
    "load_bf16": ("load", 0),
    "load_s32": ("load", 0),
    "copy_async_16": ("load", 1),
    "store_x4": ("store", 0),
}


@dataclass(frozen=True)
class Node:
    """A kernel of the main stream: whether it is launched with the PDL
    attribute, and its global accesses and PDL controls in program order."""

    name: str
    pdl: bool
    accesses: tuple[Access, ...]


def _function_accesses(function: Callable, names: dict[str, str]) -> list[Access]:
    """The global accesses, waits and triggers of a kernel method in source
    order, its buffers named by `names` (parameter to buffer); a call of
    `self._fold` contributes `_fold`'s accesses."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary.all_reduce_hc import (  # noqa: E501
        BoundaryKernel,
    )

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    # Locals that address a buffer: ``row = streams + offset``.
    bases = dict(names)

    def base(node: ast.expr) -> str | None:
        while isinstance(node, ast.BinOp):
            node = node.left
        return bases.get(node.id) if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and base(node.value) is not None:
                bases[target.id] = base(node.value)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    accesses: list[Access] = []
    for call in sorted(calls, key=lambda call: (call.lineno, call.col_offset)):
        callee = ast.unparse(call.func)
        if callee.endswith("griddepcontrol_wait"):
            accesses.append(WAIT)
        elif callee.endswith("griddepcontrol_launch_dependents"):
            accesses.append(TRIGGER)
        elif callee == "self._fold":
            parameters = list(inspect.signature(BoundaryKernel._fold).parameters)[1:]
            arguments = [base(argument) for argument in call.args]
            accesses += _function_accesses(
                BoundaryKernel._fold,
                {p: a for p, a in zip(parameters, arguments) if a is not None},
            )
        elif callee in GLOBAL_ACCESSES:
            kind, index = GLOBAL_ACCESSES[callee]
            buffer = base(call.args[index])
            if buffer is not None:
                accesses.append((kind, buffer))
    return accesses


def kernel_accesses() -> list[Access]:
    """The boundary kernel's accesses by parameter name (every variant: both
    sides of each compile-time branch, the earliest trigger included)."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary.all_reduce_hc import (  # noqa: E501
        BoundaryKernel,
    )

    return _function_accesses(
        BoundaryKernel.kernel, {name: name for name in KERNEL_BUFFERS}
    )


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


Event = tuple[str, "str | Node"]  # (alloc | free | launch, buffer or node)


def launched(item: "str | Node") -> Node:
    """The node of a launch event."""
    assert isinstance(item, Node)
    return item


def may_share(events: list[Event]) -> Callable[[str, str], bool]:
    """Whether two buffers may share memory: the same buffer, or one
    allocated after the other was released."""
    position = {(kind, item): index for index, (kind, item) in enumerate(events)}

    def after_release(first: str, second: str) -> bool:
        released = position.get(("free", first))
        allocated = position.get(("alloc", second))
        return released is not None and allocated is not None and allocated > released

    return lambda a, b: a == b or after_release(a, b) or after_release(b, a)


def violations(events: list[Event]) -> list[str]:
    """Each access to memory that an earlier node stores, or a store to memory
    an earlier node accesses, without the earlier node complete; and each
    node that stores memory it also reads under another buffer."""
    nodes = [launched(item) for kind, item in events if kind == "launch"]
    share = may_share(events)
    found: list[str] = []
    for index, (node, seen) in enumerate(zip(nodes, completed_before(nodes))):
        for (kind, buffer), done in zip(node.accesses, seen):
            if kind not in ("load", "store"):
                continue
            for other_kind, other in node.accesses:
                if other != buffer and kind == "store" and share(buffer, other):
                    found.append(
                        f"{node.name} stores {buffer}, which may share memory "
                        f"with {other} that it {other_kind}s"
                    )
            for earlier in nodes[:index]:
                conflict = [
                    other
                    for other_kind, other in earlier.accesses
                    if other_kind in ("load", "store")
                    and "store" in (kind, other_kind)
                    and share(buffer, other)
                ]
                if conflict and earlier.name not in done:
                    found.append(
                        f"{node.name} {kind}s {buffer} while {earlier.name} "
                        "may still run"
                    )
    return found


def launched_early(events: list[Event]) -> list[Event]:
    """`events` with every node but the boundary kernels launched with PDL
    and triggering at entry, the earliest that a node keeping the PDL contract
    lets its successor start; a node without a wait gets one before its
    accesses."""
    early: list[Event] = []
    for kind, item in events:
        if kind == "launch" and not launched(item).name.startswith("boundary"):
            node = launched(item)
            accesses = node.accesses
            if WAIT not in accesses:
                accesses = (WAIT, *accesses)
            item = replace(node, pdl=True, accesses=(TRIGGER, *accesses))
        early.append((kind, item))
    return early


def check_order(events: list[Event]) -> list[str]:
    """The order violations of the recorded events, and of the same events
    with every node but the boundary kernels launched early."""
    return [f"as recorded: {found}" for found in violations(events)] + [
        f"with every other node launched early: {found}"
        for found in violations(launched_early(events))
    ]


class Buffer:
    """A stand-in tensor: a named buffer whose release the recorder logs. Its
    shape is a step's that folds (`Boundaries.moe_output` reads the rows)."""

    shape = (64, HIDDEN)

    def __init__(self, name: str, recorder: "Recorder") -> None:
        self.name = name
        recorder.events.append(("alloc", name))
        weakref.finalize(self, recorder.events.append, ("free", name))


class Recorder:
    """Stand-ins for the MoE layers' runners, the boundary kernel and the
    reads' projections: what the decode loop enqueues on the main stream and
    allocates, at TP 1 (MoE boundaries only)."""

    # The current MoE layer's GateLogits (record_layers sets it per layer).
    # Any: its ``logits`` take a Buffer stand-in here, not a tensor.
    _gate_logits: Any

    def __init__(self, kernel: list[Access]) -> None:
        self.kernel = kernel
        self.events: list[Event] = []

    def launch(self, name: str, pdl: bool, accesses: tuple[Access, ...]) -> None:
        self.events.append(("launch", Node(name, pdl, accesses)))

    def experts(self, layer: int, hook_owner: SimpleNamespace):
        """`MoERunner._forward_impl` of MoE layer `layer`: the router and
        FlashInfer's routing, GEMM1 and GEMM2 kernels (each waiting first, as
        FlashInfer's PDL kernels do), the post-GEMM2 hook with the unfinalized
        output (`_run_post_gemm2_hook`), and the shared expert's output on
        the side stream."""

        def forward_impl(x: Buffer, *args: object) -> tuple[Buffer, Buffer]:
            from vllm.model_executor.layers.fused_moe.moe_output import (
                UnfinalizedMoEOutput,
            )

            logits = Buffer(f"router logits {layer}", self)
            self.launch(
                f"router {layer}",
                True,
                (WAIT, ("load", x.name), ("store", logits.name)),
            )
            weights = Buffer(f"expert weights {layer}", self)
            slots = Buffer(f"expert slots {layer}", self)
            self.launch(
                f"routing {layer}",
                True,
                (
                    WAIT,
                    ("load", logits.name),
                    ("store", weights.name),
                    ("store", slots.name),
                ),
            )
            hidden = Buffer(f"gemm1 output {layer}", self)
            self.launch(
                f"gemm1 {layer}",
                True,
                (WAIT, ("load", x.name), ("load", slots.name), ("store", hidden.name)),
            )
            rows = Buffer(f"expert rows {layer}", self)
            self.launch(
                f"gemm2 {layer}",
                True,
                (
                    WAIT,
                    ("load", hidden.name),
                    ("load", slots.name),
                    ("store", rows.name),
                ),
            )
            del logits, hidden
            output = Buffer(f"unwritten output {layer}", self)
            hook_owner.post_gemm2_hook(
                UnfinalizedMoEOutput(rows, weights, slots), output
            )
            # The shared expert and its gate (the runner's GateLogits stand-in)
            # ran on the side stream, which the main stream joined.
            gate = Buffer(f"gate logit {layer}", self)
            shared = Buffer(f"shared {layer}", self)
            self._gate_logits.logits = gate
            return shared, output

        return forward_impl

    def moe_folded(
        self, rows, weights, slots, shared, gate, streams, injection, norm, eps, split
    ):
        """`SublayerBoundary.moe_folded`: allocates its outputs while its
        inputs are referenced (`_launch`), then launches the kernel (the
        replicated call: the model's steps do not split)."""
        assert not split
        layer = rows.name.split()[-1]
        streams_out = Buffer(f"streams out {layer}", self)
        normed = Buffer(f"xn {layer}", self)
        names = {
            "partial": rows.name,
            "shared": shared.name,
            "gate": gate.name,
            "expert_weights": weights.name,
            "expert_slots": slots.name,
            "streams": streams.name,
            "injection": injection.name,
            "norm": norm.name,
            "streams_out": streams_out.name,
            "normed": normed.name,
        }
        accesses = tuple(
            access if access in (WAIT, TRIGGER) else (access[0], names[access[1]])
            for access in self.kernel
        )
        self.launch(f"boundary {layer}", True, accesses)
        return streams_out, normed

    def project(self, hc: object, xn: Buffer) -> tuple[Buffer, Buffer]:
        """The read's projections (`hc_down`, then `hc_up_mix`)."""
        layer = xn.name.split()[-1]
        down = Buffer(f"down output {layer}", self)
        self.launch(
            f"down projection {layer}",
            True,
            (WAIT, ("load", xn.name), ("store", down.name)),
        )
        block_input = Buffer(f"block input {layer}", self)
        self.launch(
            f"up mix {layer}",
            True,
            (WAIT, ("load", down.name), ("store", block_input.name)),
        )
        return block_input, down

    def attention(self, layer: int, block_input: Buffer, streams: Buffer):
        """The attention block and the stock read after it, at TP 1."""
        output = Buffer(f"attention output {layer}", self)
        self.launch(
            f"attention {layer}",
            True,
            (WAIT, ("load", block_input.name), ("store", output.name)),
        )
        new_streams = Buffer(f"streams {layer}", self)
        moe_input = Buffer(f"moe input {layer}", self)
        injection = Buffer(f"injection {layer}", self)
        self.launch(
            f"read {layer}",
            True,
            (
                WAIT,
                ("load", output.name),
                ("load", streams.name),
                ("store", new_streams.name),
                ("store", moe_input.name),
                ("store", injection.name),
            ),
        )
        return new_streams, moe_input, injection


LAYERS = 3


def record_layers(monkeypatch, kernel: list[Access] | None = None) -> list[Event]:
    """What the decode loop (`SpecializedDecode._run_layers` at TP 1)
    enqueues and releases over three MoE layers with the folded finalize, the
    real `Boundaries.moe_output` and `Boundaries.read` included."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import boundary

    recorder = Recorder(kernel_accesses() if kernel is None else kernel)
    monkeypatch.setattr(boundary, "folding_experts", lambda mlp: mlp.routed_experts)
    mlps = []
    for layer in range(LAYERS):
        routed_experts = SimpleNamespace(post_gemm2_hook=None)
        shared_expert = SimpleNamespace(expert_gate=object())
        mlp = SimpleNamespace(
            shared_expert=shared_expert,
            routed_experts=routed_experts,
            experts=SimpleNamespace(
                shared_experts=SimpleNamespace(_layer=shared_expert),
                _forward_impl=recorder.experts(layer, routed_experts),
            ),
        )
        mlps.append(mlp)
    boundaries = boundary.Boundaries(
        SimpleNamespace(moe_folded=recorder.moe_folded), False, mlps, [True] * LAYERS
    )
    hc = SimpleNamespace(
        hc_norm=SimpleNamespace(weight=SimpleNamespace(name="norm")),
        config=SimpleNamespace(rms_norm_eps=EPS),
    )
    # The loop's carried state, as _run_layers names it.
    hidden_states = Buffer("streams", recorder)
    block_input = Buffer("block input", recorder)
    block_output: boundary.PendingBlock | None = None
    injection = None
    for position in range(LAYERS):
        if isinstance(block_output, boundary.PendingBlock):
            hidden_states, block_input, injection = boundaries.read(
                hc, hidden_states, block_output, injection, recorder.project
            )
        hidden_states, block_input, injection = recorder.attention(
            position, block_input, hidden_states
        )
        moe_layer = boundaries.moe_layers[position]
        assert moe_layer is not None
        recorder._gate_logits = moe_layer[1]
        pending = boundaries.moe_output(position, block_input, True)
        assert pending is not None and pending.expert_weights is not None
        block_output = pending
    hidden_states, block_input, injection = boundaries.read(
        hc, hidden_states, block_output, injection, recorder.project
    )
    del hidden_states, block_input, injection, block_output, pending
    gc.collect()
    return recorder.events


def test_the_folded_finalize_keeps_its_order(monkeypatch):
    events = record_layers(monkeypatch)
    names = [launched(item).name for kind, item in events if kind == "launch"]
    assert names[:8] == [
        "attention 0",
        "read 0",
        "router 0",
        "routing 0",
        "gemm1 0",
        "gemm2 0",
        "boundary 0",
        "down projection 0",
    ]
    kernel = kernel_accesses()
    # The fold reads the rows, weights and slots after the wait.
    wait = kernel.index(WAIT)
    for buffer in ("partial", "expert_weights", "expert_slots"):
        assert ("load", buffer) in kernel[wait:]
    # Each layer's rows are released only after its boundary was enqueued,
    # and a later layer may take their memory.
    position = {
        (kind, item if kind != "launch" else launched(item).name): index
        for index, (kind, item) in enumerate(events)
    }
    for layer in range(LAYERS):
        rows = f"expert rows {layer}"
        assert position["alloc", rows] < position["launch", f"gemm2 {layer}"]
        assert position["free", rows] > position["launch", f"boundary {layer}"]
    assert may_share(events)("expert rows 0", "expert rows 2")
    assert check_order(events) == []


def _fold_before_wait(kernel: list[Access]) -> list[Access]:
    kernel = [access for access in kernel if access != WAIT]
    first_store = next(i for i, (kind, _) in enumerate(kernel) if kind == "store")
    return kernel[:first_store] + [WAIT] + kernel[first_store:]


def _without_wait(prefix: str) -> Callable[[list[Event]], list[Event]]:
    def drop_wait(item: "str | Node") -> Node:
        node = launched(item)
        return replace(node, accesses=tuple(a for a in node.accesses if a != WAIT))

    return lambda events: [
        (kind, drop_wait(item))
        if kind == "launch" and launched(item).name.startswith(prefix)
        else (kind, item)
        for kind, item in events
    ]


def _released_after_gemm2(events: list[Event]) -> list[Event]:
    free = ("free", "expert rows 0")
    events = [event for event in events if event != free]
    gemm2 = next(
        i
        for i, (kind, item) in enumerate(events)
        if kind == "launch" and launched(item).name == "gemm2 0"
    )
    return events[: gemm2 + 1] + [free] + events[gemm2 + 1 :]


# Defects injected into the boundary kernel or the recorded events, each with
# a violation the check must report.
DEFECTS: dict[
    str,
    tuple[
        Callable[[list[Access]], list[Access]],
        Callable[[list[Event]], list[Event]],
        str,
    ],
] = {
    "the boundary folding before its wait": (
        _fold_before_wait,
        list,
        "as recorded: boundary 0 loads expert rows 0 while gemm2 0 may still run",
    ),
    "the projection after the boundary without a wait": (
        list,
        _without_wait("down projection 0"),
        "as recorded: gemm2 2 stores expert rows 2 while boundary 0 may still run",
    ),
    "the rows released before the boundary is enqueued": (
        list,
        _released_after_gemm2,
        (
            "as recorded: boundary 0 stores streams out 0, which may share memory "
            "with expert rows 0 that it loads"
        ),
    ),
}


@pytest.mark.parametrize("defect", DEFECTS)
def test_the_order_check_catches_a_broken_fold(monkeypatch, defect):
    in_kernel, in_events, expected = DEFECTS[defect]
    events = in_events(record_layers(monkeypatch, in_kernel(kernel_accesses())))

    assert expected in check_order(events)


@pytest.mark.parametrize("rows", ROWS)
def test_the_fold_engages_from_32_rows(monkeypatch, rows):
    """Steps below `FOLD_MIN_ROWS` keep FlashInfer's finalize: the experts run
    without the post-GEMM2 hook, through FlashInfer's own entry point, and the
    pending block holds their finalized output for the unfolded MoE boundary.
    From 32 rows the hook and the finalized-shape stand-in engage and the block
    holds the unfinalized rows. Either way the experts are restored after."""
    import flashinfer.fused_moe as fused_moe

    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import boundary

    assert boundary.FOLD_MIN_ROWS == 32
    monkeypatch.setattr(boundary, "folding_experts", lambda mlp: mlp.routed_experts)
    flashinfer_entry = fused_moe.trtllm_fp4_block_scale_moe
    routed_experts = SimpleNamespace(post_gemm2_hook=None)
    shared_expert = SimpleNamespace(expert_gate=object())
    routed = torch.full((rows, HIDDEN), 2.0)
    seen: list[tuple[object, bool]] = []

    def forward_impl(x: torch.Tensor, *args: object):
        hook = routed_experts.post_gemm2_hook
        seen.append((hook, fused_moe.trtllm_fp4_block_scale_moe is flashinfer_entry))
        gate_logits.logits = torch.zeros(rows, 1)
        shared = torch.ones(rows, HIDDEN)
        if hook is None:
            return shared, routed
        output = torch.empty(rows, HIDDEN)
        hook(
            UnfinalizedMoEOutput(
                torch.zeros(rows * TOP_K, HIDDEN),
                torch.zeros(rows, TOP_K),
                torch.zeros(rows, TOP_K, dtype=torch.int32),
            ),
            output,
        )
        return shared, output

    mlp = SimpleNamespace(
        shared_expert=shared_expert,
        routed_experts=routed_experts,
        experts=SimpleNamespace(
            shared_experts=SimpleNamespace(_layer=shared_expert),
            _forward_impl=forward_impl,
        ),
    )
    boundaries = boundary.Boundaries(
        SimpleNamespace(), False, [cast("Qwen4ExpSparseMoeBlock", mlp)], [True]
    )
    moe_layer = boundaries.moe_layers[0]
    assert moe_layer is not None
    gate_logits = moe_layer[1]

    pending = boundaries.moe_output(0, torch.zeros(rows, HIDDEN), True)

    assert pending is not None and len(seen) == 1
    hook, through_flashinfer = seen[0]
    if rows < 32:
        assert hook is None and through_flashinfer
        assert pending.partial is routed
        assert pending.expert_weights is None and pending.expert_slots is None
        assert pending.gemm2_node is None
    else:
        assert isinstance(hook, boundary.DeferredFinalize)
        assert not through_flashinfer
        assert pending.expert_weights is not None
        assert pending.expert_slots is not None
    assert routed_experts.post_gemm2_hook is None
    assert fused_moe.trtllm_fp4_block_scale_moe is flashinfer_entry
