# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generates the quantization reference tables for the docs.

Like `generate_attention_backends.py`, this parses source with `ast` instead of
importing it, so the docs build does not need CUDA, ROCm, or `torch`.

Only registries written as literals are read: method, shorthand and scheme
names, the per-platform allow-lists, and the linear kernel registries.
Whether a kernel runs on a device is decided by code (`is_supported`,
`can_implement`) and is deliberately not rendered.

Adding an entry to a registry must never require an edit here: a value this
script does not recognize is rendered as written in the source. The build fails
only when a registry it looks up by name is missing or is no longer a literal.
"""

import ast
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generated_content import fill_markers  # noqa: E402

logger = logging.getLogger("mkdocs")

REPO_ROOT = Path(__file__).parent.parent.parent.parent
VLLM_DIR = REPO_ROOT / "vllm"
QUANT_DIR = VLLM_DIR / "model_executor" / "layers" / "quantization"

QUANT_CONFIG_FILE = VLLM_DIR / "config" / "quantization.py"
QUANT_UTILS_FILE = QUANT_DIR / "utils" / "quant_utils.py"
QUANT_METHODS_FILE = QUANT_DIR / "__init__.py"
ONLINE_METHODS_FILE = QUANT_DIR / "online" / "base.py"
LINEAR_KERNELS_FILE = VLLM_DIR / "model_executor" / "kernels" / "linear" / "__init__.py"
PLATFORMS_DIR = VLLM_DIR / "platforms"

FORMATS_PAGE = "features/quantization/formats.md"
KERNELS_PAGE = "features/quantization/kernels.md"

# The maps below only improve display. An entry missing from any of them falls
# back to the name used in the source.

# Dtype expressions of `quant_utils.py`. Torch has no 4-bit float or E8M0
# dtype, so the constant names carry the meaning.
DATA_DTYPE_NAMES: dict[str, str] = {
    "FP8_DTYPE": "FP8 (E4M3)",
    "FP4_DTYPE": "FP4 (E2M1)",
    "INT4_DTYPE": "INT4",
    "INT8_DTYPE": "INT8",
    "torch.int8": "INT8",
    "scalar_types.uint4": "UINT4",
    "scalar_types.float6_e3m2f": "FP6 (E3M2)",
    "scalar_types.float6_e2m3f": "FP6 (E2M3)",
}
SCALE_DTYPE_NAMES: dict[str, str] = {
    "FP8_DTYPE": "E4M3",
    "MXFP_SCALE_DTYPE": "E8M0",
    "torch.float8_e4m3fn": "E4M3",
    "torch.float8_e8m0fnu": "E8M0",
    # Scales stored as bytes are E8M0 exponents.
    "torch.uint8": "E8M0",
    "torch.float32": "fp32",
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
}

PLATFORM_NAMES: dict[str, str] = {"ROCM": "ROCm"}
PLATFORM_ORDER = ("CUDA", "ROCm", "XPU", "CPU")

# Guide for each `QuantizationConfig` class, relative to the quantization docs.
CONFIG_DOCS: dict[str, str] = {
    "AutoAWQConfig": "auto_awq.md",
    "AutoGPTQConfig": "gptqmodel.md",
    "CompressedTensorsConfig": "llm_compressor/README.md",
    "INCConfig": "inc.md",
    "ModelOptFp8Config": "modelopt.md",
    "ModelOptInt8Config": "modelopt.md",
    "ModelOptMixedPrecisionConfig": "modelopt.md",
    "ModelOptMxFp8Config": "modelopt.md",
    "ModelOptNvFp4Config": "modelopt.md",
    "OnlineQuantizationConfig": "online.md",
    "QuarkConfig": "quark.md",
    "TorchAOConfig": "torchao.md",
}

# Short names for the linear kernel registries.
LINEAR_FAMILY_TITLES: dict[str, str] = {
    "_POSSIBLE_FP8_KERNELS": "FP8",
    "_POSSIBLE_FP8_BLOCK_KERNELS": "FP8 block",
    "_POSSIBLE_WFP8A16_KERNELS": "FP8 W8A16",
    "_POSSIBLE_INT8_KERNELS": "INT8",
    "_POSSIBLE_KERNELS": "WNA16",
    "_POSSIBLE_MXFP8_KERNELS": "MXFP8",
    "_POSSIBLE_MXFP6_KERNELS": "MXFP6",
    "_POSSIBLE_MXFP4_KERNELS": "MXFP4",
    "_POSSIBLE_NVFP4_KERNELS": "NVFP4",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


class Source:
    """One parsed source file, with errors that name it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(), filename=str(path))
        self.assigned = _assignments(self.tree.body)

    def error(self, message: str) -> ValueError:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return ValueError(f"{rel}: {message}; update {Path(__file__).name}")

    def require(self, name: str) -> ast.expr:
        if name not in self.assigned:
            raise self.error(f"{name} not found")
        return self.assigned[name]

    def dict_items(self, name: str, node: ast.expr) -> Iterator[tuple[str, ast.expr]]:
        """Items of a dict literal with string keys."""
        if not isinstance(node, ast.Dict):
            raise self.error(f"{name} is not a dict literal")
        for key, value in zip(node.keys, node.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                raise self.error(f"{name} has a key that is not a string literal")
            yield key.value, value

    def strings(self, name: str, node: ast.expr) -> list[str]:
        """A list, tuple or set literal of strings, optionally wrapped in a call
        such as `frozenset(...)`."""
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            node = node.args[0]
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if len(values) == len(node.elts) and all(
                isinstance(value, str) for value in values
            ):
                return values
        raise self.error(f"{name} is not a literal collection of strings")

    def elements(self, name: str, node: ast.expr) -> list[str]:
        """Names of the classes in a list, tuple or set literal."""
        if isinstance(node, ast.Call) and not node.args and not node.keywords:
            return []
        if not isinstance(node, ast.List | ast.Tuple | ast.Set) or any(
            isinstance(element, ast.Starred) for element in node.elts
        ):
            raise self.error(f"{name} is not a literal collection")
        return [_short(element) for element in node.elts]


def _assignments(body: list[ast.stmt]) -> dict[str, ast.expr]:
    """Map names to their assigned expression. The last assignment wins, as it
    does at runtime."""
    assigned: dict[str, ast.expr] = {}
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            assigned[target.id] = value
    return assigned


def _short(node: ast.expr) -> str:
    """An expression as written, without the module prefix of a name."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return node.attr if isinstance(node, ast.Attribute) else ast.unparse(node)


def _as_call(node: ast.expr | None, name: str) -> ast.Call | None:
    """`node` if it is a call to `name`, as `name(...)` or `module.name(...)`."""
    if isinstance(node, ast.Call) and _short(node.func) == name:
        return node
    return None


def _call_args(call: ast.Call, params: tuple[str, ...]) -> dict[str, ast.expr]:
    """Bind positional and keyword arguments of `call` to `params`."""
    bound = dict(zip(params, call.args))
    bound.update({kw.arg: kw.value for kw in call.keywords if kw.arg})
    return bound


def _is_none(node: ast.expr | None) -> bool:
    return node is None or (isinstance(node, ast.Constant) and node.value is None)


def _code(text: str) -> str:
    return f"`{text}`"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join(
        "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
        for cells in lines
    )


def _platform_sort_key(platform: str) -> tuple[int, str]:
    if platform in PLATFORM_ORDER:
        return PLATFORM_ORDER.index(platform), ""
    return len(PLATFORM_ORDER), platform


# ---------------------------------------------------------------------------
# Quantization keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleSpec:
    """A parsed `ScaleDesc`. `dtype` is the source expression."""

    dtype: str
    static: bool
    group: tuple[int, int]


@dataclass(frozen=True)
class QuantKeySpec:
    """A parsed `QuantKey`. `dtype` is the source expression."""

    dtype: str
    scale: ScaleSpec
    scale2: ScaleSpec | None
    symmetric: bool


class QuantKeys:
    """Resolves the module-level `QuantKey` assignments of `quant_utils.py`."""

    def __init__(self) -> None:
        source = Source(QUANT_UTILS_FILE)
        self._assigned = source.assigned
        # `GroupShape.PER_TENSOR = GroupShape(-1, -1)` and friends.
        self._group_aliases: dict[str, ast.expr] = {}
        for node in source.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (
                    isinstance(target, ast.Attribute)
                    and ast.unparse(target.value) == "GroupShape"
                ):
                    self._group_aliases[target.attr] = node.value

    def resolve(self, node: ast.expr) -> QuantKeySpec | None:
        """Parse a reference to a `QuantKey`, or return None for one that is not
        written as a literal `QuantKey(...)` in `quant_utils.py`."""
        try:
            call = _as_call(self._deref(node), "QuantKey")
            if call is None:
                return None
            args = _call_args(call, ("dtype", "scale", "scale2", "symmetric"))
            scale2, symmetric = args.get("scale2"), args.get("symmetric")
            return QuantKeySpec(
                dtype=ast.unparse(args["dtype"]),
                scale=self._scale(args["scale"]),
                scale2=None if _is_none(scale2) else self._scale(scale2),
                symmetric=_is_none(symmetric) or bool(ast.literal_eval(symmetric)),
            )
        except (KeyError, ValueError):
            return None

    def _deref(self, node: ast.expr) -> ast.expr:
        for _ in range(len(self._assigned) + 1):
            name = _short(node)
            if isinstance(node, ast.Call) or name not in self._assigned:
                break
            node = self._assigned[name]
        return node

    def _scale(self, node: ast.expr | None) -> ScaleSpec:
        call = None if node is None else _as_call(self._deref(node), "ScaleDesc")
        if call is None:
            raise ValueError("not a ScaleDesc literal")
        args = _call_args(call, ("dtype", "static", "group_shape"))
        return ScaleSpec(
            dtype=ast.unparse(args["dtype"]),
            static=bool(ast.literal_eval(args["static"])),
            group=self._group(args["group_shape"]),
        )

    def _group(self, node: ast.expr) -> tuple[int, int]:
        if isinstance(node, ast.Attribute) and node.attr in self._group_aliases:
            node = self._group_aliases[node.attr]
        call = _as_call(self._deref(node), "GroupShape")
        if call is None:
            raise ValueError("not a GroupShape literal")
        args = _call_args(call, ("row", "col"))
        return int(ast.literal_eval(args["row"])), int(ast.literal_eval(args["col"]))


def _dtype_name(expr: str, names: dict[str, str]) -> str:
    return names.get(expr) or _code(expr.rsplit(".", 1)[-1])


def _shared_by(scale: ScaleSpec) -> str:
    """What shares one scale value."""
    row, col = scale.group
    if (row, col) == (-1, -1):
        return "tensor"
    if (row, col) == (-1, 1):
        return "output channel"
    if (row, col) == (1, -1):
        return "row" if scale.static else "token"
    if row == 1 and col > 0:
        return f"{col} elements"
    return f"{row}×{col} block"


def _scale_text(scale: ScaleSpec) -> str:
    return f"{_dtype_name(scale.dtype, SCALE_DTYPE_NAMES)} per {_shared_by(scale)}"


def _data_text(key: QuantKeySpec) -> str:
    text = _dtype_name(key.dtype, DATA_DTYPE_NAMES)
    return text if key.symmetric else f"{text}, asymmetric"


def _key_text(key: QuantKeySpec) -> str:
    """One-line description of a key that has no name of its own."""
    parts = [_data_text(key), _scale_text(key.scale)]
    if key.scale2 is not None:
        parts.append(_scale_text(key.scale2))
    return ", ".join(parts)


def _computed(scale: ScaleSpec) -> str:
    return "at load" if scale.static else "per forward pass"


@dataclass(frozen=True)
class Scheme:
    """A name accepted on the `weight` and `activation` fields."""

    name: str
    source: str
    key: QuantKeySpec | None
    # Key used instead when the name is given as a weight.
    weight_key: QuantKeySpec | None
    # Layer kinds with an online method that quantizes weights to this scheme.
    online: list[str]


def _online_weight_kinds(keys: QuantKeys) -> dict[QuantKeySpec, list[str]]:
    source = Source(ONLINE_METHODS_FILE)
    kinds: dict[QuantKeySpec, list[str]] = {}
    for registry, kind in (
        ("_ONLINE_LINEAR_METHODS", "linear"),
        ("_ONLINE_MOE_METHODS", "MoE"),
    ):
        table = source.require(registry)
        if not isinstance(table, ast.Dict):
            raise source.error(f"{registry} is not a dict literal")
        for key_node, method in zip(table.keys, table.values, strict=True):
            key = None if key_node is None else keys.resolve(key_node)
            if key is None:
                continue
            # Methods for the lm_head are registered with the linear ones.
            is_lm_head = "lmhead" in _short(method).lower()
            kinds.setdefault(key, []).append("lm_head" if is_lm_head else kind)
    return kinds


def parse_schemes(keys: QuantKeys) -> list[Scheme]:
    source = Source(QUANT_CONFIG_FILE)
    weight_names = dict(
        source.dict_items(
            "_WEIGHT_QUANT_KEY_NAMES", source.require("_WEIGHT_QUANT_KEY_NAMES")
        )
    )
    online = _online_weight_kinds(keys)
    schemes = []
    for name, node in source.dict_items(
        "QUANT_KEY_NAMES", source.require("QUANT_KEY_NAMES")
    ):
        key = keys.resolve(node)
        weight_key = keys.resolve(weight_names[name]) if name in weight_names else None
        as_weight = weight_key or key
        schemes.append(
            Scheme(
                name=name,
                source=_short(node),
                key=key,
                weight_key=weight_key,
                online=online.get(as_weight, []) if as_weight else [],
            )
        )
    if schemes and not any(scheme.key for scheme in schemes):
        raise Source(QUANT_UTILS_FILE).error(
            "no QUANT_KEY_NAMES entry resolves to a QuantKey(dtype, scale, scale2, "
            "symmetric) of ScaleDesc(dtype, static, group_shape)"
        )
    return schemes


def render_schemes(schemes: list[Scheme]) -> str:
    rows = []
    for scheme in schemes:
        key = scheme.key
        if key is None:
            rows.append([_code(scheme.name), _code(scheme.source), "–", "–", "–"])
            continue
        if scheme.weight_key is None:
            computed = _computed(key.scale)
        else:
            computed = (
                f"{_computed(scheme.weight_key.scale)} (weight), "
                f"{_computed(key.scale)} (activation)"
            )
        scale = _scale_text(key.scale)
        if key.scale2 is not None:
            scale += f", then {_scale_text(key.scale2)} ({_computed(key.scale2)})"
        rows.append(
            [
                _code(scheme.name),
                _data_text(key),
                scale,
                computed,
                ", ".join(scheme.online) or "–",
            ]
        )
    return _table(
        ["Name", "Data", "Scale", "Scale computed", "Online weight for"], rows
    )


# ---------------------------------------------------------------------------
# `--quantization` values
# ---------------------------------------------------------------------------


def parse_platform_allow_lists() -> dict[str, set[str] | None]:
    """The `--quantization` values each platform accepts.

    None is a platform without a `supported_quantization` list, which accepts
    every value.
    """
    interface = PLATFORMS_DIR / "interface.py"
    allow_lists: dict[str, set[str] | None] = {}
    found_default = False
    for path in sorted(PLATFORMS_DIR.glob("*.py")):
        source = Source(path)
        for cls in source.tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            assigned = _assignments(cls.body)
            allowed = assigned.get("supported_quantization")
            if path == interface:
                found_default |= allowed is not None
                continue
            if isinstance(allowed, ast.List) and not allowed.elts:
                allowed = None
            if allowed is None and "_enum" not in assigned:
                continue
            name = cls.name.removesuffix("Platform")
            if "_enum" in assigned:
                enum = _short(assigned["_enum"])
                name = PLATFORM_NAMES.get(enum, enum)
            allow_lists[name] = (
                None
                if allowed is None
                else set(source.strings(f"{cls.name}.supported_quantization", allowed))
            )
    if not found_default:
        raise Source(interface).error("Platform.supported_quantization not found")
    return dict(sorted(allow_lists.items(), key=lambda i: _platform_sort_key(i[0])))


def _accepted_on(names: list[str], allow_lists: dict[str, set[str] | None]) -> str:
    """Platforms that accept at least one of `names`."""
    platforms = [
        platform
        for platform, allowed in allow_lists.items()
        if allowed is None or allowed.intersection(names)
    ]
    return ", ".join(platforms) or "–"


@dataclass(frozen=True)
class LayerSpec:
    """The `QuantSpec` of an online shorthand. `weight` is None when the weight
    is not a named `QuantKey` literal; `weight_source` is then shown instead."""

    weight: QuantKeySpec | None
    weight_source: str
    hadamard_size: str | None


@dataclass(frozen=True)
class Shorthand:
    name: str
    linear: LayerSpec | None
    moe: LayerSpec | None
    # Also a checkpoint method name.
    deferred: bool


def parse_shorthands(keys: QuantKeys) -> list[Shorthand]:
    source = Source(QUANT_CONFIG_FILE)
    deferred = source.strings(
        "_DEFERRED_ONLINE_SHORTHANDS", source.require("_DEFERRED_ONLINE_SHORTHANDS")
    )

    def layer_spec(node: ast.expr | None) -> LayerSpec | None:
        if node is None or _is_none(node):
            return None
        call = _as_call(node, "QuantSpec")
        if call is None:
            return LayerSpec(None, _short(node), None)
        args = _call_args(call, ("weight", "activation", "hadamard_size"))
        weight, hadamard = args.get("weight"), args.get("hadamard_size")
        if weight is None or _is_none(weight):
            return None
        return LayerSpec(
            weight=keys.resolve(weight),
            weight_source=_short(weight),
            hadamard_size=None if _is_none(hadamard) else ast.unparse(hadamard),
        )

    shorthands = []
    for name, value in source.dict_items(
        "_ONLINE_SHORTHANDS", source.require("_ONLINE_SHORTHANDS")
    ):
        call = _as_call(value, "QuantizationConfigArgs")
        if call is None or any(kw.arg is None for kw in call.keywords):
            raise source.error(
                f"_ONLINE_SHORTHANDS[{name!r}] is not a literal "
                "QuantizationConfigArgs(...)"
            )
        args = _call_args(call, ("linear", "moe"))
        shorthands.append(
            Shorthand(
                name=name,
                linear=layer_spec(args.get("linear")),
                moe=layer_spec(args.get("moe")),
                deferred=name in deferred,
            )
        )
    return shorthands


def render_shorthands(
    shorthands: list[Shorthand],
    schemes: list[Scheme],
    allow_lists: dict[str, set[str] | None],
) -> str:
    weight_names: dict[QuantKeySpec, str] = {}
    for scheme in schemes:
        key = scheme.weight_key or scheme.key
        if key is not None:
            weight_names.setdefault(key, scheme.name)

    def cell(spec: LayerSpec | None) -> str:
        if spec is None:
            return "–"
        if spec.weight is None:
            text = _code(spec.weight_source)
        elif spec.weight in weight_names:
            text = _code(weight_names[spec.weight])
        else:
            text = _key_text(spec.weight)
        if spec.hadamard_size is not None:
            text += f" with a {spec.hadamard_size}-wide Hadamard rotation"
        return text

    rows = [
        [
            _code(shorthand.name),
            cell(shorthand.linear),
            cell(shorthand.moe),
            _accepted_on([shorthand.name], allow_lists),
        ]
        for shorthand in shorthands
    ]
    table = _table(["Name", "Linear weights", "MoE weights", "Platforms"], rows)
    deferred = [_code(s.name) for s in shorthands if s.deferred]
    if not deferred:
        return table
    return (
        f"{table}\n\n"
        f"Also a method name: {', '.join(deferred)}. On a checkpoint that is "
        "already quantized, these select the method of that name instead."
    )


@dataclass(frozen=True)
class Method:
    """Method names that share one `QuantizationConfig` class."""

    config: str
    names: list[str]


def parse_methods() -> tuple[list[Method], set[str]]:
    """Return the methods and the deprecated method names."""
    source = Source(QUANT_METHODS_FILE)
    deprecated = set(
        source.strings(
            "DEPRECATED_QUANTIZATION_METHODS",
            source.require("DEPRECATED_QUANTIZATION_METHODS"),
        )
    )
    function = next(
        (
            node
            for node in source.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_quantization_config"
        ),
        None,
    )
    if function is None:
        raise source.error("get_quantization_config not found")

    # Classes behind local variables (`deepseek_config = ...`) and import aliases.
    # A variable assigned per platform is shown by class name only: the two
    # `DeepseekV4FP8Config` classes of `deepseek_v4` and `deepseek_v41` are one row.
    variables: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}
    table: ast.expr | None = None
    for node in ast.walk(function):
        if isinstance(node, ast.ImportFrom):
            aliases.update({a.asname: a.name for a in node.names if a.asname})
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id == "method_to_config":
            table = value
        elif isinstance(value, ast.Name):
            variables.setdefault(target.id, []).append(value.id)
    if table is None:
        raise source.error("method_to_config not found in get_quantization_config")

    by_config: dict[str, list[str]] = {}
    for name, value in source.dict_items("method_to_config", table):
        symbol = _short(value)
        classes = {aliases.get(c, c) for c in variables.get(symbol, [symbol])}
        by_config.setdefault(" / ".join(sorted(classes)), []).append(name)
    methods = [Method(config, names) for config, names in by_config.items()]
    return methods, deprecated


def render_methods(
    methods: list[Method], deprecated: set[str], allow_lists: dict[str, set[str] | None]
) -> str:
    rows = []
    for method in methods:
        names = [
            _code(name) + (" (deprecated)" if name in deprecated else "")
            for name in method.names
        ]
        page = CONFIG_DOCS.get(method.config)
        rows.append(
            [
                ", ".join(names),
                _code(method.config),
                _accepted_on(method.names, allow_lists),
                f"[Guide]({page})" if page else "–",
            ]
        )
    return _table(["Names", "Implementation", "Platforms", "Guide"], rows)


# ---------------------------------------------------------------------------
# Linear kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearFamily:
    """One linear kernel registry: kernels per platform, in priority order."""

    registry: str
    # Keys of `linear_backend_per_quant` read by the selector of this registry.
    quant_keys: list[str]
    kernels: dict[str, list[str]]

    @property
    def title(self) -> str:
        return LINEAR_FAMILY_TITLES.get(self.registry) or _code(self.registry)


def _string_choices(node: ast.expr | None) -> list[str]:
    """The strings of a literal, or of both branches of a conditional of
    literals. Any other expression gives none."""
    if isinstance(node, ast.IfExp):
        body, orelse = _string_choices(node.body), _string_choices(node.orelse)
        return body + orelse if body and orelse else []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return []


def _per_quant_keys(tree: ast.Module, registries: set[str]) -> dict[str, list[str]]:
    """Map each registry to the `quantization=` keys its selector passes.

    A selector names its key as a string literal in a `quantization=` argument,
    directly or through a local variable, either in the call that receives the
    registry or elsewhere in a function that uses a single registry.
    """
    keys: dict[str, list[str]] = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        # Locals that are only ever assigned string literals.
        local: dict[str, list[str]] = {}
        used: list[str] = []
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    choices = _string_choices(node.value)
                    known = local.setdefault(target.id, choices)
                    local[target.id] = known + choices if known and choices else []
            elif isinstance(node, ast.Name) and node.id in registries:
                if node.id not in used:
                    used.append(node.id)
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            value = next(
                (kw.value for kw in call.keywords if kw.arg == "quantization"), None
            )
            if isinstance(value, ast.Name):
                found = local.get(value.id, [])
            else:
                found = _string_choices(value)
            in_call = [
                node.id
                for node in ast.walk(call)
                if isinstance(node, ast.Name) and node.id in registries
            ]
            for registry in in_call or (used if len(used) == 1 else []):
                merged = keys.setdefault(registry, [])
                merged.extend(key for key in found if key not in merged)
    return keys


def parse_linear_families() -> tuple[list[LinearFamily], dict[str, list[str]]]:
    """Return the kernel registries and the `--linear-backend` to kernels map."""
    source = Source(LINEAR_KERNELS_FILE)

    # A registry is a module-level dict of `PlatformEnum` members to lists.
    registries: dict[str, dict[str, list[str]]] = {}
    for name, value in source.assigned.items():
        if not isinstance(value, ast.Dict):
            continue
        is_platform = [
            isinstance(key, ast.Attribute) and _short(key.value) == "PlatformEnum"
            for key in value.keys
        ]
        if not any(is_platform):
            continue
        if not all(is_platform):
            raise source.error(f"{name} has a key that is not a PlatformEnum member")
        if not all(isinstance(v, ast.List | ast.Tuple | ast.Set) for v in value.values):
            continue
        per_platform: dict[str, list[str]] = {}
        for platform, kernels in zip(value.keys, value.values, strict=True):
            assert isinstance(platform, ast.Attribute)
            names = source.elements(f"{name}[{platform.attr}]", kernels)
            if names:
                label = PLATFORM_NAMES.get(platform.attr, platform.attr)
                per_platform[label] = names
        registries[name] = dict(
            sorted(per_platform.items(), key=lambda item: _platform_sort_key(item[0]))
        )
    if not registries:
        raise source.error("no kernel registry keyed by PlatformEnum found")

    backends = {
        backend: source.elements(f"_LINEAR_BACKEND_KERNEL_MAP[{backend!r}]", kernels)
        for backend, kernels in source.dict_items(
            "_LINEAR_BACKEND_KERNEL_MAP", source.require("_LINEAR_BACKEND_KERNEL_MAP")
        )
    }

    quant_keys = _per_quant_keys(source.tree, set(registries))
    families = [
        LinearFamily(name, quant_keys.get(name, []), kernels)
        for name, kernels in registries.items()
        if kernels
    ]
    return families, backends


def render_linear_backends(
    families: list[LinearFamily], backends: dict[str, list[str]]
) -> str:
    rows = []
    for backend, kernels in backends.items():
        covered = [
            family
            for family in families
            if any(k in kernels for ks in family.kernels.values() for k in ks)
        ]
        platforms = sorted(
            {
                platform
                for family in covered
                for platform, ks in family.kernels.items()
                if any(k in kernels for k in ks)
            },
            key=_platform_sort_key,
        )
        rows.append(
            [
                _code(backend),
                ", ".join(platforms) or "–",
                ", ".join(family.title for family in covered) or "–",
            ]
        )
    return _table(["Backend", "Platforms", "Quantization types"], rows)


def render_linear_families(families: list[LinearFamily]) -> str:
    rows = [
        [
            family.title,
            ", ".join(_code(key) for key in family.quant_keys) or "–",
            ", ".join(family.kernels),
        ]
        for family in families
    ]
    return _table(
        ["Quantization type", "`linear_backend_per_quant` key", "Platforms"], rows
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_blocks() -> dict[str, dict[str, str]]:
    """Build the generated blocks per page, keyed by their `gen:` marker name."""
    keys = QuantKeys()
    schemes = parse_schemes(keys)
    shorthands = parse_shorthands(keys)
    allow_lists = parse_platform_allow_lists()
    methods, deprecated = parse_methods()
    families, backends = parse_linear_families()
    return {
        FORMATS_PAGE: {
            "quantization-methods": render_methods(methods, deprecated, allow_lists),
            "quantization-shorthands": render_shorthands(
                shorthands, schemes, allow_lists
            ),
            "quantization-schemes": render_schemes(schemes),
        },
        KERNELS_PAGE: {
            "linear-backends": render_linear_backends(families, backends),
            "linear-families": render_linear_families(families),
        },
    }


# mkdocs-gen-files runs this file with `runpy.run_path`.
if __name__ == "<run_path>":
    logger.info("Generating quantization documentation tables")
    for page, blocks in build_blocks().items():
        fill_markers(page, blocks)
