# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chakra execution trace capture, normalization and comparison.

A capture on the `meta` device is only trustworthy if it matches what the model
actually executes, so this module records a Chakra execution trace (ET) of a
forward pass and can diff a `meta` trace against a real-device one.

The two traces are normalized first. An ET node carries run-specific bookkeeping
-- node, tensor and storage ids, devices, process metadata -- so only operator
names, operand shapes and operand types are compared. Operators whose internals
are device-specific are then treated as black boxes, their descendants dropped
from both traces, which leaves the sequence that must agree: see [`_is_opaque`]
[vllm.profiler.op_capture.trace._is_opaque].
"""

import difflib
import json
import multiprocessing
import os
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from torch.profiler import ExecutionTraceObserver

from vllm.engine.arg_utils import EngineArgs
from vllm.profiler.op_capture.capture import BatchSpec, capture_model_ops
from vllm.profiler.op_capture.meta_ops import LEAF_NAMESPACES, NATIVE_PREFIXES

_NON_OP_PREFIXES = ("[pytorch|", "##", "[param|")


@contextmanager
def capture_execution_trace(path: Path) -> Iterator[None]:
    """Record a Chakra execution trace of everything in the block to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    observer = ExecutionTraceObserver()
    observer.register_callback(str(path))
    observer.start()
    try:
        yield
    finally:
        observer.stop()
        observer.unregister_callback()


@dataclass(frozen=True)
class TraceOp:
    """One operator node from an execution trace, stripped of run-specific ids.

    Tensor ids, storage ids and devices live in a node's input *values*, which
    are dropped; only the shapes and types, which must agree across devices, are
    kept.
    """

    name: str
    input_shapes: str
    input_types: str
    output_shapes: str

    def __str__(self) -> str:
        return f"{self.name}{self.input_shapes} -> {self.output_shapes}"


def _is_operator(name: str) -> bool:
    return "::" in name and not name.startswith(_NON_OP_PREFIXES)


def _is_opaque(name: str) -> bool:
    """Whether an operator's internals are not comparable across devices.

    Two kinds of operator have device-specific internals. Shape inference on the
    meta device runs decompositions of native operators that real kernels do not,
    and a compiled kernel allocates its own scratch buffers, which a meta run
    never reaches. Both show up as descendants of the operator responsible, so
    both are removed by treating that operator as a black box -- while the
    operator node itself, with its shapes and dtypes, is still compared.
    """
    return (
        name.startswith(NATIVE_PREFIXES) or name.partition("::")[0] in LEAF_NAMESPACES
    )


def _under_opaque_op(node_id: int, nodes: dict[int, dict[str, Any]]) -> bool:
    """Whether an operator node was issued from inside an opaque operator."""
    seen = set()
    parent = nodes[node_id].get("ctrl_deps")
    while parent in nodes and parent not in seen:
        seen.add(parent)
        if _is_opaque(nodes[parent].get("name", "")):
            return True
        parent = nodes[parent].get("ctrl_deps")
    return False


def load_trace(path: Path) -> list[TraceOp]:
    """Read the operator nodes of an execution trace, in execution order.

    Operators issued from inside an opaque one are dropped, leaving a sequence
    that is comparable across devices. Node ids are assigned in issue order, so
    sorting by id recovers the sequence without depending on how the observer
    flushed the file.

    Args:
        path: Execution trace written by [`capture_execution_trace`]
            [vllm.profiler.op_capture.trace.capture_execution_trace].

    Returns:
        The operators in execution order.

    """
    with path.open() as file:
        nodes = {node["id"]: node for node in json.load(file)["nodes"]}
    ops = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        if not _is_operator(node.get("name", "")) or _under_opaque_op(node_id, nodes):
            continue
        ops.append(
            TraceOp(
                name=node["name"],
                input_shapes=json.dumps(node["inputs"]["shapes"]),
                input_types=json.dumps(node["inputs"]["types"]),
                output_shapes=json.dumps(node["outputs"]["shapes"]),
            )
        )
    return ops


@dataclass
class TraceDiff:
    """Result of comparing two normalized execution traces."""

    matched: int
    """Operators that line up between the two traces."""
    only_in_meta: list[TraceOp]
    only_in_real: list[TraceOp]
    meta_total: int
    real_total: int

    @property
    def equal(self) -> bool:
        return not self.only_in_meta and not self.only_in_real


def compare_traces(meta_path: Path, real_path: Path) -> TraceDiff:
    """Compare a `meta` execution trace against a real-device one.

    Args:
        meta_path: Trace captured with `device="meta"`.
        real_path: Trace captured on hardware for the same model and batch.

    Returns:
        The aligned difference between the two normalized traces.

    """
    meta_ops = load_trace(meta_path)
    real_ops = load_trace(real_path)

    only_in_meta: list[TraceOp] = []
    only_in_real: list[TraceOp] = []
    matched = 0
    matcher = difflib.SequenceMatcher(a=meta_ops, b=real_ops, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            continue
        only_in_meta.extend(meta_ops[i1:i2])
        only_in_real.extend(real_ops[j1:j2])

    return TraceDiff(
        matched=matched,
        only_in_meta=only_in_meta,
        only_in_real=only_in_real,
        meta_total=len(meta_ops),
        real_total=len(real_ops),
    )


def compare_devices(
    model: str,
    real_device: str,
    *,
    batch: BatchSpec | None = None,
    directory: str | os.PathLike | None = None,
    engine_args: EngineArgs | None = None,
) -> TraceDiff:
    """Check a `meta` capture against hardware, for one model and batch shape.

    Each capture runs in its own process, because vLLM caches layers across
    models -- the rotary embedding, for one -- so a meta-device model's buffers
    would be handed to the real one.

    Args:
        model: Model id or local path.
        real_device: Device to capture the reference trace on, e.g. `"cuda"`.
        batch: Batch shape to run on both devices.
        directory: Where to keep the two traces; a temporary one by default.
        engine_args: Base engine args, applied to both captures.

    Returns:
        The aligned difference between the two normalized traces, equal when the
        meta capture reproduced the real one.

    Raises:
        RuntimeError: If either capture process failed.

    """
    context = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        # One hash seed for both, so they iterate sets alike: the order
        # multimodal inputs reach the device in, for one, follows it.
        seed = os.environ.get("PYTHONHASHSEED", "0")
        stack.enter_context(patch.dict(os.environ, PYTHONHASHSEED=seed))
        if directory is None:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
        paths = {
            "meta": Path(directory) / "meta.et.json",
            real_device: Path(directory) / "real.et.json",
        }
        for device, path in paths.items():
            process = context.Process(
                target=capture_model_ops,
                args=(model,),
                kwargs={
                    "device": device,
                    "batch": batch,
                    "trace_path": path,
                    "engine_args": engine_args,
                },
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Capturing {model} on {device} failed with exit code "
                    f"{process.exitcode}"
                )
        return compare_traces(paths["meta"], paths[real_device])


def format_diff(diff: TraceDiff, limit: int = 20) -> str:
    """Render a trace comparison as a report.

    Args:
        diff: Comparison to render.
        limit: Most residual operators to list per side.

    Returns:
        A multi-line report, ending in a verdict line.

    """
    lines = [
        "Execution trace comparison (meta vs real device)",
        f"  operators: {diff.meta_total} meta, {diff.real_total} real, "
        f"{diff.matched} aligned",
    ]
    for label, residual in (
        ("only in meta", diff.only_in_meta),
        ("only in real", diff.only_in_real),
    ):
        if not residual:
            continue
        lines.append(f"  {label} ({len(residual)}):")
        lines.extend(f"    {op}" for op in residual[:limit])
        if len(residual) > limit:
            lines.append(f"    ... {len(residual) - limit} more")
    lines.append("MATCH" if diff.equal else "MISMATCH")
    return "\n".join(lines)
