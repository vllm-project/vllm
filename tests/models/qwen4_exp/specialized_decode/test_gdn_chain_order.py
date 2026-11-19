# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The order of the staged GDN layer of Qwen4Exp's specialized decode
(`runtime/gdn.py`, "Order of the staged layer"), checked on the CPU.

- Order. A simulation runs the real `GdnLayers.__call__` over three layers
  of a TP 4 step of 4 rows, between the sublayer boundaries and reads that
  surround them, with stand-ins that record each kernel enqueued on the main
  stream and each buffer's allocation and release. The projection kernel's
  accesses (`GdnProjection.kernel`) and the boundary kernel's come from
  their source; the core's from its stated contract (`gdn/core.py`). The
  check of `test_boundary_fold.py` (CUDA's rules for programmatic dependent
  launch, and memory reuse after a release) finds no violation, as recorded or
  with every node but the boundary kernels triggering at entry, and reports
  each injected defect: the in projection loading its activations before its
  wait, the core or the next core without a wait, the out projection storing
  before its wait, and the layer input released before the in projection is
  enqueued.
- Choice. `staged_projection_unsupported` accepts the TP 4 layer of
  Qwen3.8-Flash-Next on 152 SMs and names what TP 1, TP 2 and quantized or
  biased projections miss; `GdnLayers.staged` takes steps of up to 16 rows;
  without a partial output (no attention boundary) the layer's own output
  projection runs.
"""

import ast
import gc
import inspect
import textwrap
import weakref
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from .test_boundary_fold import (
    TRIGGER,
    WAIT,
    Access,
    Event,
    Node,
    check_order,
    kernel_accesses,
    launched,
    may_share,
)

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn import Schedule

Q = 4  # rows per request
LAYERS = 3
EPS = 1e-6
# The projection kernel's global memory helpers and the argument holding the
# address.
PROJECTION_ACCESSES = {
    "copy_bulk_hint": ("load", 1),
    "load_x4": ("load", 0),
    "store_bf16": ("store", 0),
}
PROJECTION_BUFFERS = ("x", "w0", "w1", "out0", "out1")
# The boundary kernel's buffers that a MoE block's boundary reads only.
MOE_ONLY = ("shared", "gate", "expert_weights", "expert_slots")


def projection_accesses() -> list[Access]:
    """`GdnProjection.kernel`'s global accesses, waits and triggers in source
    order, by parameter name."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn.proj import (
        GdnProjection,
    )

    tree = ast.parse(textwrap.dedent(inspect.getsource(GdnProjection.kernel)))
    bases = {name: name for name in PROJECTION_BUFFERS}

    def base(node: ast.expr) -> str | None:
        while isinstance(node, ast.BinOp):
            node = node.left
        return bases.get(node.id) if isinstance(node, ast.Name) else None

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    accesses: list[Access] = []
    for call in sorted(calls, key=lambda call: (call.lineno, call.col_offset)):
        callee = ast.unparse(call.func)
        if callee.endswith("griddepcontrol_wait"):
            accesses.append(WAIT)
        elif callee.endswith("griddepcontrol_launch_dependents"):
            accesses.append(TRIGGER)
        elif callee in PROJECTION_ACCESSES:
            kind, index = PROJECTION_ACCESSES[callee]
            buffer = base(call.args[index])
            if buffer is not None:
                accesses.append((kind, buffer))
    return accesses


# The core's accesses as gdn/core.py states them: before its wait the step's
# metadata, its loaded weights and this layer's caches, then the trigger;
# after it the projections' outputs, then its writes.
CORE_ACCESSES: tuple[Access, ...] = (
    ("load", "slots"),
    ("load", "accepted"),
    ("load", "cu_seqlens"),
    ("load", "conv_weight"),
    ("load", "a_log"),
    ("load", "dt_bias"),
    ("load", "norm_weight"),
    ("load", "conv_state"),
    ("load", "state"),
    TRIGGER,
    WAIT,
    ("load", "qkvz"),
    ("load", "ba"),
    ("store", "state"),
    ("store", "conv_state"),
    ("store", "out"),
)


class Buffer:
    """A stand-in tensor: a named buffer whose release the recorder logs."""

    def __init__(self, name: str, recorder: "Recorder", rows: int = Q) -> None:
        self.name = name
        self.rows = rows
        self.device = torch.device("cpu")
        recorder.events.append(("alloc", name))
        weakref.finalize(self, recorder.events.append, ("free", name))

    def size(self, dim: int | None = None) -> int:
        return self.rows if dim == 0 else Q

    def view(self, *shape: int) -> "Buffer":
        return self

    def __getitem__(self, index: object) -> "Buffer":
        return self


class Recorder:
    """Stand-ins for the kernels around and inside the staged GDN layer: what
    the decode loop enqueues on the main stream at TP 4, and allocates."""

    def __init__(self, projection: list[Access], core: tuple[Access, ...]) -> None:
        self.projection_kernel = projection
        self.core_kernel = core
        self.boundary_kernel = [
            access
            for access in kernel_accesses()
            if access in (WAIT, TRIGGER) or access[1] not in MOE_ONLY
        ]
        self.events: list[Event] = []
        self.layer = 0

    def launch(
        self, name: str, kernel: list[Access] | tuple[Access, ...], names: dict
    ) -> None:
        accesses = tuple(
            access if access in (WAIT, TRIGGER) else (access[0], names[access[1]])
            for access in kernel
            if access in (WAIT, TRIGGER) or access[1] in names
        )
        self.events.append(("launch", Node(name, True, accesses)))

    def projection(self, x: Buffer, weights: list[Buffer]) -> list[Buffer]:
        """`gdn_projection`: allocates its outputs, then launches."""
        kind = "in" if len(weights) == 2 else "out"
        if kind == "in":
            outs = [
                Buffer(f"qkvz {self.layer}", self),
                Buffer(f"ba {self.layer}", self),
            ]
        else:
            outs = [Buffer(f"partial {self.layer}", self)]
        names = {"x": x.name, "w0": weights[0].name, "out0": outs[0].name}
        if kind == "in":
            names.update(w1=weights[1].name, out1=outs[1].name)
        self.launch(f"{kind} projection {self.layer}", self.projection_kernel, names)
        return outs

    def core(self, qkvz, ba, conv_weight, conv_state, state, a_log, dt_bias,
             norm_weight, slots, accepted, cu_seqlens, requests, out, eps,
             schedule):  # fmt: skip
        """`gdn_decode_core`."""
        names = {
            "qkvz": qkvz.name,
            "ba": ba.name,
            "conv_weight": conv_weight.name,
            "conv_state": conv_state.name,
            "state": state.name,
            "a_log": a_log.name,
            "dt_bias": dt_bias.name,
            "norm_weight": norm_weight.name,
            "slots": slots.name,
            "accepted": accepted.name,
            "cu_seqlens": cu_seqlens.name,
            "out": out.name,
        }
        self.launch(f"core {self.layer}", self.core_kernel, names)
        return out

    def read(
        self, layer: int, streams: Buffer, partial: Buffer, injection: Buffer
    ) -> tuple[Buffer, Buffer, Buffer]:
        """The read after a block (`Boundaries.read` with `hc_project`):
        the boundary kernel, then the down projection and the up mix."""
        streams_out = Buffer(f"streams {layer}", self)
        xn = Buffer(f"xn {layer}", self)
        names = {
            "partial": partial.name,
            "streams": streams.name,
            "injection": injection.name,
            "norm": "norm",
            "streams_out": streams_out.name,
            "normed": xn.name,
        }
        self.launch(f"boundary {layer}", self.boundary_kernel, names)
        down = Buffer(f"down {layer}", self)
        self.launch(
            f"down {layer}",
            (("load", "w"), TRIGGER, WAIT, ("load", "xn"), ("store", "down")),
            {"w": "w_down", "xn": xn.name, "down": down.name},
        )
        block_input = Buffer(f"block input {layer}", self)
        self.launch(
            f"up mix {layer}",
            (
                ("load", "w"),
                TRIGGER,
                WAIT,
                ("load", "down"),
                ("load", "xn"),
                ("store", "out"),
            ),
            {"w": "w_up", "down": down.name, "xn": xn.name, "out": block_input.name},
        )
        # The injection logits are columns of the down projection's output.
        return streams_out, block_input, down


def _attention(recorder: Recorder, layer: int) -> SimpleNamespace:
    """The parts of a TP 4 `QwenGatedDeltaNetAttention` the layer reads."""

    def buffer(name: str) -> Buffer:
        return Buffer(f"{name} {layer}", recorder)

    return SimpleNamespace(
        in_proj_qkvz=SimpleNamespace(weight=buffer("w_qkvz")),
        in_proj_ba=SimpleNamespace(weight=buffer("w_ba")),
        out_proj=SimpleNamespace(weight=buffer("w_out"), reduce_results=False),
        kv_cache=[buffer("conv state"), buffer("state")],
        conv1d=SimpleNamespace(weight=buffer("conv weight")),
        A_log=buffer("a_log"),
        dt_bias=buffer("dt_bias"),
        norm=SimpleNamespace(weight=buffer("norm weight")),
        layer_norm_epsilon=EPS,
        num_k_heads=16,
        tp_size=4,
        value_dim=48 * 128,
    )


def _staged_layers(monkeypatch, recorder: Recorder, attns: list[SimpleNamespace]):
    """A `GdnLayers` over `attns` with the recorder's core and projection
    (on the CPU: no stream, no capture), and a one-request step's metadata."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import gdn

    monkeypatch.setattr(gdn, "GDNAttentionMetadata", SimpleNamespace)
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    layers = gdn.GdnLayers(
        attns, Q, 4, recorder.core,
        lambda heads, requests: cast("Schedule", requests), None,
        projection=recorder.projection,
    )  # fmt: skip
    layers.out = Buffer("core output", recorder, rows=4 * Q)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        spec_state_indices_tensor=Buffer("slots", recorder),
        spec_query_start_loc=Buffer("cu_seqlens", recorder),
        num_accepted_tokens=Buffer("accepted", recorder),
    )
    return layers, metadata


def record_layers(
    monkeypatch,
    projection: list[Access] | None = None,
    core: tuple[Access, ...] = CORE_ACCESSES,
) -> list[Event]:
    """What the decode loop enqueues and releases over three staged GDN layers
    of one request, the real `GdnLayers.__call__` included."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.boundary import (
        PendingBlock,
    )

    recorder = Recorder(
        projection_accesses() if projection is None else projection, core
    )
    attns = [_attention(recorder, layer) for layer in range(LAYERS)]
    layers, metadata = _staged_layers(monkeypatch, recorder, attns)
    # The loop's carried state, as SpecializedDecode._run_layers names it.
    hidden_states = Buffer("streams", recorder)
    injection = Buffer("injection", recorder)
    block_output = PendingBlock(Buffer("partial before", recorder))
    for layer in range(LAYERS):
        recorder.layer = layer
        hidden_states, block_input, injection = recorder.read(
            layer, hidden_states, block_output.partial, injection
        )
        output = layers(attns[layer], block_input, metadata)
        block_output = PendingBlock(output)
    hidden_states, block_input, injection = recorder.read(
        LAYERS, hidden_states, block_output.partial, injection
    )
    del hidden_states, block_input, injection, block_output, output
    gc.collect()
    return recorder.events


def test_the_staged_layer_keeps_its_order(monkeypatch):
    events = record_layers(monkeypatch)
    names = [launched(item).name for kind, item in events if kind == "launch"]
    assert names[:7] == [
        "boundary 0",
        "down 0",
        "up mix 0",
        "in projection 0",
        "core 0",
        "out projection 0",
        "boundary 1",
    ]
    projection = projection_accesses()
    wait = projection.index(WAIT)
    # Weights before the wait, the activations and every store after it.
    assert projection[:wait] == [("load", "w0"), ("load", "w1"), TRIGGER]
    assert ("load", "x") in projection[wait:]
    assert {a for a in projection[wait:] if a[0] == "store"} == {
        ("store", "out0"),
        ("store", "out1"),
    }
    # Each layer's input is released only after its in projection was
    # enqueued, and a later layer's buffers may take its memory.
    position = {
        (kind, item if kind != "launch" else launched(item).name): index
        for index, (kind, item) in enumerate(events)
    }
    for layer in range(LAYERS):
        assert (
            position["free", f"block input {layer}"]
            > position["launch", f"in projection {layer}"]
        )
    assert may_share(events)("block input 0", "qkvz 2")
    assert check_order(events) == []


def _move_before_wait(access: Access) -> Callable[[list[Access]], list[Access]]:
    def move(kernel: list[Access]) -> list[Access]:
        kernel = [a for a in kernel if a != access]
        return [access, *kernel]

    return move


def _without_wait(name: str) -> Callable[[list[Event]], list[Event]]:
    return lambda events: [
        (
            kind,
            replace(
                launched(item),
                accesses=tuple(a for a in launched(item).accesses if a != WAIT),
            ),
        )
        if kind == "launch" and launched(item).name == name
        else (kind, item)
        for kind, item in events
    ]


def _released_after(buffer: str, launch: str) -> Callable[[list[Event]], list[Event]]:
    def release(events: list[Event]) -> list[Event]:
        free = ("free", buffer)
        events = [event for event in events if event != free]
        index = next(
            i
            for i, (kind, item) in enumerate(events)
            if kind == "launch" and launched(item).name == launch
        )
        return events[: index + 1] + [free] + events[index + 1 :]

    return release


def _identity(x):
    return x


# Defects injected into the projection kernel or the recorded events, each
# with a violation the check must report.
DEFECTS: dict[
    str,
    tuple[
        Callable[[list[Access]], list[Access]],
        Callable[[list[Event]], list[Event]],
        str,
    ],
] = {
    "the in projection loading its activations before its wait": (
        _move_before_wait(("load", "x")),
        _identity,
        "as recorded: in projection 0 loads block input 0 while up mix 0 may still run",
    ),
    "the out projection storing before its wait": (
        _move_before_wait(("store", "out0")),
        _identity,
        "as recorded: out projection 0 stores partial 0 while",
    ),
    "the core without a wait": (
        _identity,
        _without_wait("core 0"),
        "as recorded: core 0 loads qkvz 0 while in projection 0 may still run",
    ),
    "the next core without a wait": (
        _identity,
        _without_wait("core 1"),
        "as recorded: core 1 stores core output while out projection 0 may still run",
    ),
    "the layer input released before the in projection is enqueued": (
        _identity,
        _released_after("block input 0", "up mix 0"),
        (
            "as recorded: in projection 0 stores qkvz 0, which may share memory "
            "with block input 0 that it loads"
        ),
    ),
}


@pytest.mark.parametrize("defect", DEFECTS)
def test_the_order_check_catches_a_broken_staged_layer(monkeypatch, defect):
    in_kernel, in_events, expected = DEFECTS[defect]
    events = in_events(record_layers(monkeypatch, in_kernel(projection_accesses())))

    found = check_order(events)
    assert any(violation.startswith(expected) for violation in found), found


def _layer(tp: int, dtype: torch.dtype = torch.bfloat16, bias: bool = False):
    """A Qwen3.8-Flash-Next GDN layer's projection weights at `tp`, on the
    meta device."""
    kh, hv = 16 // tp, 48 // tp

    def linear(n: int, k: int) -> SimpleNamespace:
        weight = torch.empty(n, k, dtype=dtype, device="meta")
        return SimpleNamespace(
            weight=weight, bias=torch.empty(n, device="meta") if bias else None
        )

    return SimpleNamespace(
        tp_size=tp,
        in_proj_qkvz=linear(2 * (kh + hv) * 128, 2560),
        in_proj_ba=linear(2 * hv, 2560),
        out_proj=linear(2560, hv * 128),
    )


def test_the_staged_projections_serve_tp4():
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gdn import (
        staged_projection_unsupported,
    )

    assert staged_projection_unsupported([_layer(4)], 152) == ""
    tp1 = staged_projection_unsupported([_layer(1)], 152)
    assert "tensor parallelism" in tp1 and "one CTA per SM" in tp1
    assert "at most one CTA per SM (258 CTAs, 152 SMs)" in (
        staged_projection_unsupported([_layer(2)], 152)
    )
    assert "unquantized BF16" in staged_projection_unsupported(
        [_layer(4, torch.float8_e4m3fn)], 152
    )
    assert "unquantized BF16" in staged_projection_unsupported(
        [_layer(4, bias=True)], 152
    )


def test_steps_of_up_to_16_rows_run_the_staged_projections(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import gdn

    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: None)
    attn = SimpleNamespace(
        num_k_heads=16, tp_size=4, value_dim=6144, A_log=torch.empty(0)
    )
    args = ([attn], Q, 16, lambda *a: None, lambda h, r: r, None)
    staged = gdn.GdnLayers(*args, projection=lambda x, w: [])
    plain = gdn.GdnLayers(*args)
    assert [staged.staged(rows) for rows in (1, 4, 16, 17, 64)] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert not any(plain.staged(rows) for rows in (1, 4, 16))


class _ReducingOutProjection:
    """An output projection that all-reduces (`reduce_results`): records
    its calls."""

    def __init__(self, weight: Buffer) -> None:
        self.weight = weight
        self.reduce_results = True
        self.calls = 0

    def __call__(self, x: Buffer) -> tuple[str, None]:
        self.calls += 1
        return "reduced", None


def test_without_a_partial_output_the_layer_runs_its_own_out_projection(
    monkeypatch,
):
    """At TP > 1 without an attention boundary the output projection must
    all-reduce: the layer calls its own, after the staged in projection."""
    recorder = Recorder(projection_accesses(), CORE_ACCESSES)
    attn = _attention(recorder, 0)
    attn.out_proj = _ReducingOutProjection(attn.out_proj.weight)
    layers, metadata = _staged_layers(monkeypatch, recorder, [attn])

    output = layers(attn, Buffer("block input", recorder), metadata)

    assert output == "reduced" and attn.out_proj.calls == 1
    names = [launched(item).name for kind, item in recorder.events if kind == "launch"]
    assert names == ["in projection 0", "core 0"]
