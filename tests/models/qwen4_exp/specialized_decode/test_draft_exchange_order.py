# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exchange order of the MTP draft layer's boundary kernel in Qwen4Exp's
specialized decode (`runtime/draft_front.py`, "Exchange order", point 2),
checked on the CPU.

At TP > 1 the boundary kernel's ranks exchange their partial outputs through a
mailbox with two generations. A call reads the generation word after its
grid-dependency wait, and its last CTA stores the next generation at its end
(`boundary/all_reduce_hc.py`). The exchange holds only if each call reads
the word after the instance's previous call on the rank completed. The draft
layer's instance makes up to two calls a pass, the attention block's and the
MoE block's. Between the MoE block's call and the next pass's attention block's
call run the final mixer's read, the draft head and the token choice, then the
next pass's metadata, embedding, embedding all-reduce, front, first read and
QSA kernels.

A model lists the decode stream's kernels over four draft passes as the
`Node`s of `test_boundary_fold.py`. The first pass has its row gather, the
next two are later passes and the fourth is the next step's first pass. Each
node records whether it is launched with programmatic dependent launch (PDL),
and its waits, triggers and generation-word accesses in program order:

- The boundary kernel, the read kernels (`hc_down`, `hc_up_mix`) and the
  token choice are read from their source.
- Every other kernel up to the token choice reads its predecessor's output. It
  is modeled as a PDL kernel that triggers at entry and then waits, the
  earliest trigger such a kernel could have. Real kernels are no earlier:
  PyTorch's and cuBLAS's do not use PDL, and the planned GEMMs trigger after
  their stores.
- The kernels after the token choice are the worst case: PDL kernels that
  trigger at entry and never wait.

The shared expert's kernels on the side stream are left out. The side stream
forks from the decode stream and joins it through events. Those edges only
add predecessors to a node, so they can delay a call but never start it
earlier.

`check_order` applies CUDA's rules for PDL to these nodes, as recorded and
with every node but the boundary kernels triggering at entry. It finds no
call that reads or stores the generation word while an earlier call may
still run, with the read kernels or the stock reads (above 64 rows), and with
or without the MoE block's call. It reports each injected defect:

- an up mix without its wait, in the MoE block's read or the final mixer's;
- a token choice that triggers its dependents at entry;
- a call that reads the generation word before its wait.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import replace

import pytest

from .test_boundary_fold import TRIGGER, WAIT, Access, Event, Node, check_order

GENERATION = "generation"
# The first pass of a step, two later passes, and the next step's first pass.
PASSES = (True, False, False, True)


def _controls(function: Callable, generation: bool = True) -> tuple[Access, ...]:
    """The waits and triggers of a CuTe DSL kernel method in source order
    (both sides of each compile-time branch), and (`generation`) its loads
    and stores of the boundary kernel's generation word."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    accesses: list[Access] = []
    for call in sorted(calls, key=lambda call: (call.lineno, call.col_offset)):
        callee = ast.unparse(call.func)
        if callee.endswith("griddepcontrol_wait"):
            accesses.append(WAIT)
        elif callee.endswith("griddepcontrol_launch_dependents"):
            accesses.append(TRIGGER)
        elif (
            generation and call.args and "GENERATION_WORD" in ast.unparse(call.args[0])
        ):
            kind = "store" if callee.startswith("store") else "load"
            accesses.append((kind, GENERATION))
    return tuple(accesses)


def boundary_accesses() -> tuple[Access, ...]:
    """`BoundaryKernel.kernel`'s controls and generation-word accesses."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.boundary.all_reduce_hc import (  # noqa: E501
        BoundaryKernel,
    )

    return _controls(BoundaryKernel.kernel)


def read_accesses() -> tuple[tuple[Access, ...], tuple[Access, ...]]:
    """The controls of `hc_down`'s and `hc_up_mix`'s kernels. The up/mix
    kernel's all-gather variant (`HcUpMixGather`, the target's row split at
    TP4) reads and stores a generation word of its own mailbox, not the
    boundary kernel's; the draft passes run the variant without it."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc.down import HcDown
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc.up_mix import HcUpMix

    return _controls(HcDown.kernel), _controls(HcUpMix.kernel, generation=False)


def choice_accesses() -> tuple[Access, ...]:
    """The token choice kernel's controls."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.choice import (
        TokenChoiceKernel,
    )

    return _controls(TokenChoiceKernel.kernel)


def _contract(name: str) -> Node:
    """A kernel that reads its predecessor's output, at the earliest trigger
    the PDL contract allows: it triggers at entry, then waits."""
    return Node(name, True, (TRIGGER, WAIT))


def _no_wait(name: str) -> Node:
    """A kernel after the token choice, at its worst: it triggers at entry and
    never waits."""
    return Node(name, True, (TRIGGER,))


def draft_passes(
    kernel_reads: bool = True,
    moe_boundary: bool = True,
    boundary: tuple[Access, ...] | None = None,
    choice: tuple[Access, ...] | None = None,
) -> list[Event]:
    """The decode stream's kernels over `PASSES`, as launch events."""
    boundary = boundary_accesses() if boundary is None else boundary
    choice = choice_accesses() if choice is None else choice
    down, up_mix = read_accesses()

    def project(read: str) -> list[Node]:
        """A read's projections, after its combine-norm or boundary call."""
        if kernel_reads:
            return [
                Node(f"down {read}", True, down),
                Node(f"up mix {read}", True, up_mix),
            ]
        return [_contract(f"stock down {read}"), _contract(f"stock up mix {read}")]

    nodes: list[Node] = []
    for index, first in enumerate(PASSES):
        nodes += [
            _no_wait(f"{name} {index}")
            for name in (
                "metadata",
                "embedding",
                "embedding all-reduce",
                "front norm",
                "front GEMM",
                "first read",
                "QSA",
                "o_proj partial",
            )
        ]
        nodes.append(Node(f"boundary {index} attention", True, boundary))
        nodes += project(f"MoE read {index}")
        nodes += [
            _contract(f"{name} {index}")
            for name in ("router", "quantization", "routing", "gemm1", "gemm2")
        ]
        if moe_boundary:
            nodes.append(_contract(f"finalize {index}"))
            nodes.append(Node(f"boundary {index} MoE", True, boundary))
        else:
            nodes += [
                _contract(f"finalize {index}"),
                _contract(f"MoE all-reduce {index}"),
                _contract(f"combine-norm {index}"),
            ]
        nodes += project(f"final read {index}")
        if first:
            nodes.append(_contract(f"row gather {index}"))
        nodes.append(_contract(f"head {index}"))
        nodes.append(Node(f"token choice {index}", True, choice))
    return [("launch", node) for node in nodes]


def test_the_kernels_keep_the_order_the_exchange_needs():
    """The source: the boundary kernel reads the generation word after its
    wait and stores the next one at its end; the read kernels trigger, then
    wait; the token choice waits and never triggers."""
    assert boundary_accesses() == (
        TRIGGER,
        WAIT,
        ("load", GENERATION),
        TRIGGER,
        ("store", GENERATION),
    )
    assert read_accesses() == ((TRIGGER, WAIT), (TRIGGER, WAIT))
    assert choice_accesses() == (WAIT,)


@pytest.mark.parametrize("moe_boundary", [True, False], ids=["moe-call", "moe-stock"])
@pytest.mark.parametrize("kernel_reads", [True, False], ids=["kernel-reads", "stock"])
def test_each_call_reads_the_generation_after_the_previous_call(
    kernel_reads, moe_boundary
):
    events = draft_passes(kernel_reads, moe_boundary)
    calls = [item.name for _, item in events if isinstance(item, Node)]
    assert sum(name.startswith("boundary") for name in calls) == len(PASSES) * (
        2 if moe_boundary else 1
    )
    assert check_order(events) == []


def _without_wait(name: str) -> Callable[[list[Event]], list[Event]]:
    """Drop the wait of the node `name`."""

    def drop(events: list[Event]) -> list[Event]:
        return [
            (kind, replace(item, accesses=tuple(a for a in item.accesses if a != WAIT)))
            if isinstance(item, Node) and item.name == name
            else (kind, item)
            for kind, item in events
        ]

    return drop


def _identity(events: list[Event]) -> list[Event]:
    return events


# Defects injected into the kernels or the recorded events, each with the
# violation the check must report.
DEFECTS: dict[str, tuple[dict[str, tuple[Access, ...]], Callable, str]] = {
    "the MoE block's up mix without its wait": (
        {},
        _without_wait("up mix MoE read 1"),
        (
            "as recorded: boundary 1 MoE loads generation while boundary 1 attention "
            "may still run"
        ),
    ),
    "the final mixer's up mix without its wait": (
        {},
        _without_wait("up mix final read 1"),
        (
            "as recorded: boundary 2 attention loads generation while boundary 1 MoE "
            "may still run"
        ),
    ),
    "a token choice that triggers its dependents at entry": (
        {"choice": (TRIGGER, WAIT)},
        _identity,
        (
            "as recorded: boundary 1 attention loads generation while boundary 0 MoE "
            "may still run"
        ),
    ),
    "a call that reads the generation word before its wait": (
        {"boundary": (TRIGGER, ("load", GENERATION), WAIT, ("store", GENERATION))},
        _identity,
        (
            "as recorded: boundary 0 MoE loads generation while boundary 0 attention "
            "may still run"
        ),
    ),
}


@pytest.mark.parametrize("defect", DEFECTS)
def test_the_order_check_catches_a_broken_chain(defect):
    kernels, in_events, expected = DEFECTS[defect]
    found = check_order(in_events(draft_passes(**kernels)))
    assert any(violation.startswith(expected) for violation in found), found
