# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Human-readable renderings of an operator capture."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby

from vllm.profiler.op_capture.capture import OpCapture, SelectionMetadata
from vllm.profiler.op_capture.recorder import RecordedOp

_PLATFORM_CAVEAT = (
    "Valid only for the platform above: attention backend, KV cache layout, "
    "dtype and block sizes are chosen per platform, so the operators below do "
    "not carry over to other hardware."
)


def format_selection(selection: SelectionMetadata) -> str:
    """Render the selection metadata a capture is only valid under."""
    backends = ", ".join(sorted(set(selection.attention_backends.values())))
    rows = {
        "platform": selection.platform,
        "device": selection.device,
        "dtype": selection.dtype,
        "quantization": selection.quantization or "none",
        "attention backend": backends or "none",
        "kv cache layout": selection.kv_cache_layout,
        "kv cache dtype": selection.kv_cache_dtype,
        "block size": f"{selection.block_size} (kernel {selection.kernel_block_sizes})",
        "attention layers": selection.num_attention_layers,
        "heads": f"{selection.num_query_heads} query, "
        f"{selection.num_kv_heads} kv, head size {selection.head_size}",
    }
    width = max(len(key) for key in rows)
    lines = ["Selection metadata"]
    lines += [f"  {key:<{width}}  {value}" for key, value in rows.items()]
    lines += ["", _PLATFORM_CAVEAT]
    return "\n".join(lines)


def format_summary(capture: OpCapture) -> str:
    """Render operator counts, total and per attention layer, most frequent first."""
    counts = Counter(op.name for op in capture.ops)
    layers = max(capture.selection.num_attention_layers, 1)
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
        if placeholder_ops := capture.placeholder_ops:
            lines.append(
                f"    may be a knock-on effect of the placeholder for "
                f"{placeholder_ops[0].name}, if it is shape-related"
            )
    if capture.missing_kernels:
        platform = capture.selection.platform
        lines.append(f"  No {platform} kernel registered (would fail on device):")
        lines += [f"    {name}" for name in capture.missing_kernels]
    if placeholders := Counter(op.name for op in capture.placeholder_ops):
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
    for op in capture.ops:
        node = root
        path = ""
        for part in filter(None, op.module.split(".")):
            path = f"{path}.{part}" if path else part
            node = node.children.setdefault(part, _Node(name=part))
            node.module_type = capture.module_types.get(path, node.module_type)
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
