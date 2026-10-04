# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Human-readable renderings of an operator capture."""

import json
import os
from collections import Counter
from collections.abc import Sequence, Set
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path

from vllm.profiler.op_capture.capture import BatchSpec, OpCapture, SelectionMetadata
from vllm.profiler.op_capture.recorder import RecordedOp

_PLATFORM_CAVEAT = (
    "Valid only for the platform above: attention backend, KV cache layout, "
    "dtype and block sizes are chosen per platform, so the operators below do "
    "not carry over to other hardware."
)


def format_selection(selection: SelectionMetadata) -> str:
    """Render the selection metadata a capture is only valid under."""
    backends = ", ".join(sorted(set(selection.attention_backends.values())))
    if selection.num_query_heads is None:
        heads = "unknown (no layer reports head counts)"
    else:
        heads = (
            f"{selection.num_query_heads} query, {selection.num_kv_heads} kv, "
            f"head size {selection.head_size}"
        )
        if selection.tensor_parallel_size > 1:
            heads += " (per rank)"
    layers = str(selection.num_attention_layers)
    if len(selection.layer_kinds) > 1:
        kinds = ", ".join(f"{count} {name}" for name, count in selection.layer_kinds)
        layers += f" ({kinds})"
    rows = {
        "platform": selection.platform,
        "device": selection.device,
        "dtype": selection.dtype,
        "quantization": selection.quantization or "none",
        "attention backend": backends or "none",
        "kv cache layout": selection.kv_cache_layout,
        "kv cache dtype": selection.kv_cache_dtype,
        "block size": f"{selection.block_size} (kernel {selection.kernel_block_sizes})",
        "attention layers": layers,
        "heads": heads,
    }
    if selection.tensor_parallel_size > 1:
        rows["tensor parallel"] = f"{selection.tensor_parallel_size} ranks"
    if windows := Counter(selection.sliding_windows.values()):
        rows["sliding window"] = ", ".join(
            f"{window} ({count} layers)" for window, count in sorted(windows.items())
        )
    if selection.moe_experts:
        rows["moe experts"] = ", ".join(selection.moe_experts)
    width = max(len(key) for key in rows)
    lines = ["Selection metadata"]
    lines += [f"  {key:<{width}}  {value}" for key, value in rows.items()]
    lines += ["", _PLATFORM_CAVEAT]
    return "\n".join(lines)


def format_summary(capture: OpCapture) -> str:
    """Render operator counts, total and per decoder layer, most frequent first."""
    counts = Counter(op.name for op in capture.ops)
    selection = capture.selection
    # Not `num_attention_layers`: that counts KV-cache-only layers too, which
    # would divide a sparse-attention model's counts by four times its depth.
    layers = max(selection.num_model_layers or selection.num_attention_layers, 1)
    width = max((len(name) for name in counts), default=1)
    lines = [
        f"Operator summary ({len(capture.ops)} calls, {len(counts)} distinct)",
        f"  {'operator':<{width}}  {'count':>6}  {'per layer':>9}",
    ]
    for name, count in counts.most_common():
        lines.append(f"  {name:<{width}}  {count:>6}  {count / layers:>9.2f}")
    return "\n".join(lines)


def format_gaps(capture: OpCapture) -> str | None:
    """Render what would keep the model from running on this platform.

    Returns:
        The section, or None when the capture found no gaps.

    """
    lines = []
    placeholder_ops = capture.placeholder_ops
    if capture.failure is not None:
        failure = capture.failure
        where = failure.module or "top level"
        if module_type := capture.module_types.get(failure.module):
            where = f"{where} [{module_type}]"
        lines += [
            f"  Forward pass stopped in {where}, after {len(capture.ops)} ops:",
            f"    {failure.error}",
        ]
        if failure.location:
            lines.append(f"    at {failure.location}")
        if placeholder_ops:
            lines.append(
                f"    may be a knock-on effect of the placeholder for "
                f"{placeholder_ops[0].name}, if it is shape-related"
            )
    if capture.missing_kernels:
        platform = capture.selection.platform
        lines.append(f"  No {platform} kernel registered (would fail on device):")
        lines += [f"    {name}" for name in capture.missing_kernels]
    if body_errors := {op.name: op.body_error for op in capture.body_error_ops}:
        lines.append(
            "  Custom ops whose kernel failed on the meta device, outputs faked "
            "(the ops they would have dispatched next are missing):"
        )
        lines += [f"    {name}: {error}" for name, error in body_errors.items()]
    if placeholders := Counter(op.name for op in placeholder_ops):
        lines.append(
            "  Output shapes unknown, placeholders used (add to OVERRIDES; "
            "shapes after the first are guesses):"
        )
        lines += [f"    {name} (x{count})" for name, count in placeholders.items()]
    if capture.materialized:
        platform = capture.selection.platform
        lines.append(
            f"  Placed on {platform} despite the meta device (explicit `device=` "
            f"in model code), {len(capture.materialized)} tensors, e.g.:"
        )
        lines += [f"    {name}" for name in capture.materialized[:3]]
    if not lines:
        return None
    return "\n".join(["Platform gaps", *lines])


@dataclass
class _Node:
    """One module in the call tree, holding the ops it issued directly."""

    name: str
    module_type: str = ""
    children: dict[str, "_Node"] = field(default_factory=dict)
    ops: list[RecordedOp] = field(default_factory=list)

    def fingerprint(self) -> tuple:
        """Structure of this subtree, ignoring which repeat index it is."""
        return (
            self.module_type,
            tuple(op.name for op in self.ops),
            tuple(child.fingerprint() for child in self.children.values()),
        )


def _build_tree(capture: OpCapture) -> _Node:
    root = _Node(name="")
    # Consecutive ops share a module, and a module's node is the same every time,
    # so each distinct path is walked once rather than once per op.
    nodes = {"": root}
    for op in capture.ops:
        node = nodes.get(op.module)
        if node is None:
            node = root
            path = ""
            for part in filter(None, op.module.split(".")):
                path = f"{path}.{part}" if path else part
                child = node.children.get(part)
                if child is None:
                    child = node.children[part] = _Node(name=part)
                    child.module_type = capture.module_types.get(path, "")
                node = child
            nodes[op.module] = node
        node.ops.append(op)
    return root


def _render(node: _Node, indent: str, lines: list[str], show_shapes: bool) -> None:
    for op in node.ops:
        detail = op.signature if show_shapes else op.name
        marker = "*" if op.is_custom else " "
        lines.append(f"{indent}{'  ' * op.depth}{marker} {detail}")

    for _, group in groupby(node.children.values(), key=_Node.fingerprint):
        child, *rest = group
        repeats = 1 + len(rest)
        suffix = f" (x{repeats} identical)" if repeats > 1 else ""
        label = child.name
        if child.module_type:
            label = f"{label} [{child.module_type}]"
        lines.append(f"{indent}{label}{suffix}")
        _render(child, indent + "  ", lines, show_shapes)


def format_tree(capture: OpCapture, show_shapes: bool = True) -> str:
    """Render the capture as a module tree, collapsing repeated layers.

    Sibling modules with identical subtrees -- the decoder layers -- are printed
    once with a repeat count, so a 60-layer model reads as easily as a 2-layer
    one. Custom ops are marked with `*`.

    Args:
        capture: Capture to render.
        show_shapes: Include argument shapes and dtypes for each operator.

    Returns:
        The indented tree.

    """
    lines = [f"Operator tree for {capture.model}"]
    _render(_build_tree(capture), "  ", lines, show_shapes)
    return "\n".join(lines)


def format_report(capture: OpCapture, show_shapes: bool = False) -> str:
    """Render the full report: selection metadata, gaps, tree, and summary."""
    sections = [
        format_selection(capture.selection),
        format_gaps(capture),
        format_tree(capture, show_shapes=show_shapes),
        format_summary(capture),
    ]
    if capture.trace_path is not None:
        sections.append(f"Chakra execution trace: {capture.trace_path}")
    return "\n\n".join(section for section in sections if section)


def _batch_label(batch: BatchSpec) -> str:
    label = (
        f"{batch.num_reqs} reqs, {batch.num_tokens} tokens, "
        f"{batch.num_computed_tokens} computed per req"
    )
    if batch.num_mm_items:
        label += f", {batch.num_mm_items} mm items"
    return label


def format_batches(captures: Sequence[OpCapture]) -> str:
    """Render which operators only some of several batches reach.

    Args:
        captures: Captures of the same model on different batches.

    Returns:
        Per-batch totals, then each operator missing from at least one batch
        with the batches that do reach it.

    """
    names = [{op.name for op in capture.ops} for capture in captures]
    lines = ["Batches"]
    for index, (capture, reached) in enumerate(zip(captures, names)):
        lines.append(
            f"  [{index}] {_batch_label(capture.batch)}: "
            f"{len(capture.ops)} calls, {len(reached)} distinct"
        )
    reached_by_all = names[0].intersection(*names[1:])
    partial = sorted(set().union(*names) - reached_by_all)
    if partial:
        width = max(len(name) for name in partial)
        lines.append("  Reached by only some batches:")
        for name in partial:
            where = ", ".join(
                f"[{index}]" for index, reached in enumerate(names) if name in reached
            )
            lines.append(f"    {name:<{width}}  {where}")
    return "\n".join(lines)


def _annotation(
    op: RecordedOp, selection: SelectionMetadata, attention_layers: Set[str]
) -> str:
    if not op.is_custom or op.module not in attention_layers:
        return ""
    window = selection.sliding_windows.get(op.module)
    return f" [sliding_window={window}]" if window else " [full]"


def write_capture_files(capture: OpCapture, directory: str | os.PathLike) -> Path:
    """Write a capture as files, for diffing captures or feeding other tools.

    Writes `report.txt` (`format_report` with shapes), `ops.txt` (distinct
    operators, sorted), `ops.sequence.txt` (every operator in order, indented
    by nesting depth, with shapes and module; attention layers' custom ops are
    tagged `[sliding_window=N]` or `[full]`) and `capture.json` (batch,
    selection metadata and gaps).

    Args:
        capture: Capture to write.
        directory: Directory to write into, created if missing.

    Returns:
        The directory.

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selection = capture.selection
    distinct = sorted({op.name for op in capture.ops})
    # A hybrid model has hundreds of attention layers, and every op asks.
    attention_layers = frozenset(selection.attention_layers)
    (directory / "report.txt").write_text(format_report(capture, show_shapes=True))
    (directory / "ops.txt").write_text("".join(f"{name}\n" for name in distinct))
    (directory / "ops.sequence.txt").write_text(
        "".join(
            f"{'  ' * op.depth}{op.signature} -> {', '.join(op.outputs) or '()'}"
            f"  @{op.module or '<top>'}"
            f"{_annotation(op, selection, attention_layers)}\n"
            for op in capture.ops
        )
    )
    summary = {
        "model": capture.model,
        "rank": capture.rank,
        "batch": asdict(capture.batch),
        "selection": asdict(selection),
        "ops": len(capture.ops),
        "distinct_ops": len(distinct),
        "failure": None if capture.failure is None else asdict(capture.failure),
        "missing_kernels": capture.missing_kernels,
        "placeholder_ops": sorted({op.name for op in capture.placeholder_ops}),
        "body_errors": {op.name: op.body_error for op in capture.body_error_ops},
        "materialized": capture.materialized,
    }
    (directory / "capture.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory
