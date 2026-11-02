# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generates quantization capability tables for the docs.

Like ``generate_attention_backends.py``, this parses source with ``ast`` instead
of importing it, so the docs build does not need CUDA, ROCm, or ``torch``.

Scheme names come from ``QUANT_KEY_NAMES`` and their definitions from the
``QuantKey`` assignments in ``quant_utils.py``.

Each scheme needs a hand-written note in ``SCHEME_NOTES``. The build fails on a
missing or orphaned note.
"""

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generated_content import fill_markers  # noqa: E402

logger = logging.getLogger("mkdocs")

REPO_ROOT = Path(__file__).parent.parent.parent.parent

QUANT_CONFIG_FILE = REPO_ROOT / "vllm" / "config" / "quantization.py"
QUANT_UTILS_FILE = (
    REPO_ROOT
    / "vllm"
    / "model_executor"
    / "layers"
    / "quantization"
    / "utils"
    / "quant_utils.py"
)

FORMATS_PAGE = "features/quantization/formats.md"

# Display names keyed by source expression, so the table shows what a checkpoint
# config says rather than an internal constant. Unknown expressions fail the build.
DATA_DTYPE_NAMES: dict[str, str] = {
    "FP8_DTYPE": "FP8 (E4M3)",
    "FP4_DTYPE": "FP4 (E2M1)",
    "INT4_DTYPE": "INT4",
    "INT8_DTYPE": "INT8",
    "torch.int8": "INT8",
    "torch.uint8": "UINT8",
    "scalar_types.uint4": "UINT4",
    "scalar_types.uint4b8": "UINT4",
    "scalar_types.uint8b128": "UINT8",
}

SCALE_DTYPE_NAMES: dict[str, str] = {
    "FP8_DTYPE": "E4M3",
    "MXFP_SCALE_DTYPE": "E8M0",
    "torch.float": "fp32",
    "torch.float32": "fp32",
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.uint8": "E8M0",
    "torch.float8_e8m0fnu": "E8M0",
}

# GroupShape static members, by attribute name.
GROUP_SHAPE_ALIASES: dict[str, tuple[int, int]] = {
    "PER_TENSOR": (-1, -1),
    "PER_TOKEN": (1, -1),
    "PER_CHANNEL": (-1, 1),
}

# One entry per key of QUANT_KEY_NAMES.
SCHEME_NOTES: dict[str, str] = {
    "fp8_per_tensor_static": (
        "One fp32 scale per weight tensor. Cheapest FP8 to apply, but a single "
        "outlier compresses the whole tensor."
    ),
    "fp8_per_tensor_dynamic": (
        "One fp32 scale per activation tensor, computed per forward pass. Used "
        "for activations, not weights."
    ),
    "fp8_per_token": (
        "One fp32 scale per activation row. Dynamic per-token activations, "
        "usually paired with per-channel weights."
    ),
    "fp8_per_channel_static": (
        "One fp32 scale per output channel. The usual FP8 weight recipe "
        "(llm-compressor's FP8_DYNAMIC weight side)."
    ),
    "fp8_per_block_static": (
        "One fp32 scale per 128x128 weight block. Robust to outliers; needs "
        "block-scaled FP8 kernels."
    ),
    "fp8_per_block_dynamic": (
        "One fp32 scale per 1x128 activation block, computed per forward pass. "
        "The activation half of block-scaled FP8."
    ),
    "mxfp8": (
        "OCP MX: E4M3 data with an E8M0 scale per 32 elements. W8A8 needs "
        "SM100+; older GPUs run W8A16."
    ),
    "mxfp4": (
        "OCP MX: E2M1 data with an E8M0 scale per 32 elements. Block size is "
        "fixed by the standard; dimensions must be multiples of 32."
    ),
    "nvfp4_per_token": (
        "E2M1 data with an E4M3 scale per 16 elements plus one fp32 scale per "
        "tensor, and per-token activation scales. Needs SM100+."
    ),
    "int8_per_channel_static": (
        "INT8 with one fp32 scale per output channel. The classic INT8 W8A8 "
        "weight side."
    ),
    "int8_per_group_32": (
        "Dynamic per-32-element INT8 activations. Used for the RDNA3/RDNA3.5 "
        "W4A8 GEMV path, not a general-purpose scheme."
    ),
    "int4_per_group_32": (
        "INT4 weights with a per-32 group scale. Applied at load time when "
        "re-quantizing MXFP4 experts on gfx942; not selectable as an online "
        "scheme."
    ),
}


@dataclass(frozen=True)
class GroupShapeSpec:
    """A parsed ``GroupShape``: ``row``/``col`` as in ``quant_utils.py``."""

    row: int
    col: int


@dataclass(frozen=True)
class ScaleSpec:
    """A parsed ``ScaleDesc``."""

    dtype: str
    static: bool
    group: GroupShapeSpec


@dataclass(frozen=True)
class QuantKeySpec:
    """A parsed ``QuantKey``."""

    dtype: str
    scale: ScaleSpec
    scale2: ScaleSpec | None
    symmetric: bool


def _source(node: ast.expr) -> str:
    """Render an expression back to source, e.g. ``torch.float32``."""
    return ast.unparse(node)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _constant(node: ast.expr) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"expected a literal, got: {_source(node)}")


class QuantKeyParser:
    """Resolves module-level ``QuantKey``/``ScaleDesc`` assignments by name."""

    def __init__(self, tree: ast.Module) -> None:
        self._assignments: dict[str, ast.expr] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if isinstance(target, ast.Name):
                self._assignments[target.id] = value

    def resolve_name(self, name: str) -> QuantKeySpec:
        """Resolve a ``QuantKey`` assignment such as ``kFp8StaticTensorSym``."""
        if name not in self._assignments:
            raise ValueError(
                f"{name!r} is used as a quantization key but never assigned in "
                f"{QUANT_UTILS_FILE.name}"
            )
        key = self._quant_key(self._assignments[name], frozenset({name}))
        return key

    def _resolve(self, node: ast.expr, seen: frozenset[str]) -> object:
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "ScaleDesc":
                return self._scale(node)
            if name == "GroupShape":
                return GroupShapeSpec(
                    int(_constant(node.args[0])), int(_constant(node.args[1]))
                )
            raise ValueError(f"unsupported call in a QuantKey: {_source(node)}")
        if isinstance(node, ast.Attribute) and node.attr in GROUP_SHAPE_ALIASES:
            row, col = GROUP_SHAPE_ALIASES[node.attr]
            return GroupShapeSpec(row, col)
        if isinstance(node, ast.Name) and node.id in self._assignments:
            if node.id in seen:
                raise ValueError(f"cyclic quantization key reference: {node.id}")
            return self._resolve(self._assignments[node.id], seen | {node.id})
        raise ValueError(f"unsupported quantization key expression: {_source(node)}")

    def _group(self, node: ast.expr) -> GroupShapeSpec:
        group = self._resolve(node, frozenset())
        if not isinstance(group, GroupShapeSpec):
            raise ValueError(f"expected a GroupShape, got: {_source(node)}")
        return group

    def _scale(self, call: ast.Call) -> ScaleSpec:
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        dtype_node = kwargs.get("dtype") or call.args[0]
        static_node = kwargs.get("static") or call.args[1]
        group_node = kwargs.get("group_shape") or call.args[2]
        return ScaleSpec(
            dtype=_source(dtype_node),
            static=bool(_constant(static_node)),
            group=self._group(group_node),
        )

    def _quant_key(self, node: ast.expr, seen: frozenset[str]) -> QuantKeySpec:
        if not isinstance(node, ast.Call) or _call_name(node) != "QuantKey":
            raise ValueError(f"expected a QuantKey(...) call, got: {_source(node)}")
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}

        dtype_node = kwargs.get("dtype") or node.args[0]
        scale_node = kwargs.get("scale") or node.args[1]
        scale = self._resolve(scale_node, seen)
        if not isinstance(scale, ScaleSpec):
            raise ValueError(f"expected a ScaleDesc, got: {_source(scale_node)}")

        # QuantKey(dtype, scale, scale2=None, symmetric=True): accept both the
        # positional and keyword spellings used across the codebase.
        scale2_node = kwargs.get("scale2")
        if scale2_node is None and len(node.args) > 2:
            scale2_node = node.args[2]
        scale2: ScaleSpec | None = None
        if scale2_node is not None:
            resolved = self._resolve(scale2_node, seen)
            if not isinstance(resolved, ScaleSpec):
                raise ValueError(f"expected a ScaleDesc, got: {_source(scale2_node)}")
            scale2 = resolved

        symmetric_node = kwargs.get("symmetric")
        if symmetric_node is None and len(node.args) > 3:
            symmetric_node = node.args[3]
        symmetric = True if symmetric_node is None else bool(_constant(symmetric_node))

        return QuantKeySpec(
            dtype=_source(dtype_node),
            scale=scale,
            scale2=scale2,
            symmetric=symmetric,
        )


def parse_quant_key_names() -> list[tuple[str, str]]:
    """Return ``(user-facing name, QuantKey symbol)`` from QUANT_KEY_NAMES."""
    tree = ast.parse(QUANT_CONFIG_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id != "QUANT_KEY_NAMES":
            continue
        if not isinstance(value, ast.Dict):
            raise ValueError("QUANT_KEY_NAMES must be a dict literal")
        pairs: list[tuple[str, str]] = []
        for key_node, symbol_node in zip(value.keys, value.values, strict=True):
            if key_node is None or not isinstance(symbol_node, ast.Name):
                raise ValueError("QUANT_KEY_NAMES keys/values must be name literals")
            pairs.append((str(_constant(key_node)), symbol_node.id))
        return pairs
    raise ValueError(f"QUANT_KEY_NAMES not found in {QUANT_CONFIG_FILE}")


def parse_quant_keys() -> QuantKeyParser:
    """Return a parser over the ``QuantKey`` assignments in ``quant_utils.py``."""
    return QuantKeyParser(ast.parse(QUANT_UTILS_FILE.read_text()))


def _data_dtype_name(expr: str) -> str:
    try:
        return DATA_DTYPE_NAMES[expr]
    except KeyError:
        raise ValueError(
            f"Unknown data dtype {expr!r} in a user-facing quantization key. "
            "Add it to DATA_DTYPE_NAMES in generate_quantization_matrix.py."
        ) from None


def _scale_dtype_name(expr: str) -> str:
    try:
        return SCALE_DTYPE_NAMES[expr]
    except KeyError:
        raise ValueError(
            f"Unknown scale dtype {expr!r} in a user-facing quantization key. "
            "Add it to SCALE_DTYPE_NAMES in generate_quantization_matrix.py."
        ) from None


def _granularity(group: GroupShapeSpec) -> str:
    if (group.row, group.col) == (-1, -1):
        return "per-tensor"
    if (group.row, group.col) == (1, -1):
        return "per-token"
    if (group.row, group.col) == (-1, 1):
        return "per-channel"
    return f"{group.row}×{group.col} block"


def _scale_cell(key: QuantKeySpec) -> str:
    cell = _scale_dtype_name(key.scale.dtype)
    if key.scale2 is not None:
        cell += f" + {_scale_dtype_name(key.scale2.dtype)} global"
    return cell


def _granularity_cell(key: QuantKeySpec) -> str:
    cell = _granularity(key.scale.group)
    if key.scale2 is not None:
        cell += f" + {_granularity(key.scale2.group)}"
    return cell


def _validate_notes(names: list[str]) -> None:
    missing = sorted(set(names) - SCHEME_NOTES.keys())
    stale = sorted(SCHEME_NOTES.keys() - set(names))
    if missing or stale:
        problems = []
        if missing:
            problems.append(f"no note for {missing}")
        if stale:
            problems.append(f"note for removed scheme(s) {stale}")
        raise ValueError(
            f"{FORMATS_PAGE}: {'; '.join(problems)}. Update SCHEME_NOTES in "
            "docs/mkdocs/gen_files/generate_quantization_matrix.py."
        )
    for name, note in SCHEME_NOTES.items():
        if "|" in note:
            raise ValueError(f"note for {name!r} contains '|', which breaks tables")


def render_scheme_table(names: list[str], keys: dict[str, QuantKeySpec]) -> str:
    """Render the scheme table filled into ``gen:quantization-schemes``."""
    _validate_notes(names)
    rows = [
        "| Scheme | Data | Scale | Granularity | Symmetric | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in names:
        if name not in keys:
            raise ValueError(f"{name!r} has no referent in quant_utils.py")
        key = keys[name]
        rows.append(
            f"| `{name}` "
            f"| {_data_dtype_name(key.dtype)} "
            f"| {_scale_cell(key)} "
            f"| {_granularity_cell(key)} "
            f"| {'yes' if key.symmetric else 'no'} "
            f"| {SCHEME_NOTES[name]} |"
        )
    return "\n".join(rows)


def build_blocks() -> dict[str, str]:
    """Build the generated table blocks keyed by their ``gen:`` marker name."""
    pairs = parse_quant_key_names()
    parser = parse_quant_keys()
    names = [name for name, _ in pairs]
    keys = {name: parser.resolve_name(symbol) for name, symbol in pairs}
    return {"quantization-schemes": render_scheme_table(names, keys)}


logger.info("Generating quantization documentation tables")
fill_markers(FORMATS_PAGE, build_blocks())
