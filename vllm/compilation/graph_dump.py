# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import enum
import itertools
import json
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
import torch.fx as fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._ops import OpOverload, OpOverloadPacket

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationConfig

logger = init_logger(__name__)

_dump_indices: dict[Path, itertools.count] = {}


@dataclasses.dataclass
class GraphDumpContext:
    gm: fx.GraphModule | None = None
    dump_path: Path | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    stage_prefix: str = ""


_graph_dump_context = GraphDumpContext()


@contextmanager
def graph_dump_context(
    gm: fx.GraphModule | None = None,
    dump_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    stage_prefix: str | None = None,
) -> Generator[None, None, None]:
    global _graph_dump_context
    previous = _graph_dump_context
    merged_metadata = dict(previous.metadata)
    if metadata:
        merged_metadata.update(metadata)
    _graph_dump_context = GraphDumpContext(
        gm=gm if gm is not None else previous.gm,
        dump_path=dump_path if dump_path is not None else previous.dump_path,
        metadata=merged_metadata,
        stage_prefix=stage_prefix
        if stage_prefix is not None
        else previous.stage_prefix,
    )
    try:
        yield
    finally:
        _graph_dump_context = previous


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted((json_safe(v) for v in value), key=str)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compilation_config_metadata(cc: "CompilationConfig") -> dict[str, Any]:
    metadata = {
        field.name: getattr(cc, field.name)
        for field in dataclasses.fields(cc)
        if field.name != "static_forward_context"
    }
    metadata["compile_ranges"] = [str(r) for r in cc.get_compile_ranges()]
    metadata["static_forward_context"] = {
        "keys": sorted(str(k) for k in cc.static_forward_context)
    }
    return metadata


def collect_graph_metadata(
    vllm_config: "VllmConfig | None" = None, **extra: Any
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if vllm_config is not None:
        cc = vllm_config.compilation_config
        model_config = vllm_config.model_config
        device_config = vllm_config.device_config
        metadata.update(
            {
                "model": getattr(model_config, "model", None),
                "model_dtype": getattr(model_config, "dtype", None),
                "device": getattr(device_config, "device", None),
                "compilation_config": compilation_config_metadata(cc),
                "rank": vllm_config.parallel_config.rank,
                "data_parallel_index": vllm_config.parallel_config.data_parallel_index,
            }
        )

    try:
        from vllm.compilation.passes.inductor_pass import get_pass_context

        metadata["pass_context"] = vars(get_pass_context()).copy()
    except (AssertionError, ImportError):
        pass

    metadata.update(extra)
    return json_safe(metadata)


def module_type_name(module_type: Any) -> str:
    if hasattr(module_type, "__name__"):
        return module_type.__name__
    name = str(module_type)
    if "." in name:
        return name.rsplit(".", 1)[-1].strip("'>")
    return name


def format_path_part(part: str) -> str:
    part = part.removeprefix("L['self'].").removeprefix("self.")
    if part == "L['self']":
        part = ""
    pieces: list[str] = []
    for token in part.split("."):
        if token.isdigit() and pieces:
            pieces[-1] += f"[{token}]"
        elif token:
            pieces.append(token)
    return ".".join(pieces)


def module_stack(node: fx.Node) -> list[tuple[str, str]]:
    stack = node.meta.get("nn_module_stack") or {}
    result: list[tuple[str, str]] = []
    prev_path = ""
    for key, value in stack.items():
        if isinstance(value, tuple) and len(value) >= 2:
            path, module_type = value[:2]
        else:
            path, module_type = key, value
        path = str(path)
        display = path
        if prev_path and path.startswith(prev_path + "."):
            display = path[len(prev_path) + 1 :]
        result.append((format_path_part(display), module_type_name(module_type)))
        prev_path = path
    return result


def source_context(node: fx.Node) -> str:
    stack = node.meta.get("source_fn_stack") or []
    for entry in reversed(stack):
        name = entry[0] if isinstance(entry, tuple) and entry else entry
        name = str(name)
        if name and "torch" not in name:
            return f"  # via {name}"
    return ""


def format_structured_graph(gm: fx.GraphModule) -> str:
    lines: list[str] = []
    previous_stack: list[tuple[str, str]] = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue

        stack = module_stack(node)
        divergence = 0
        for old, new in zip(previous_stack, stack):
            if old != new:
                break
            divergence += 1

        for depth, (name, module_type) in enumerate(stack[divergence:], divergence):
            label = f"{name}: {module_type}" if name else module_type
            lines.append(f"{'  ' * depth}{label}")

        node_text = node.format_node() or str(node)
        lines.append(f"{'  ' * len(stack)}{node_text}{source_context(node)}")
        previous_stack = stack

    return "\n".join(lines) + ("\n" if lines else "")


def default_overload(target: Any) -> OpOverload | None:
    if isinstance(target, OpOverload):
        return target
    if isinstance(target, OpOverloadPacket):
        return target.default
    return None


def collect_vllm_ir_metadata(gm: fx.GraphModule) -> list[dict[str, Any]]:
    from vllm.ir.op import IrOp

    metadata: list[dict[str, Any]] = []
    for node in gm.graph.nodes:
        op = default_overload(node.target)
        if op is None or op.namespace != "vllm_ir":
            continue

        item: dict[str, Any] = {"node": node.name, "op": op._opname}
        ir_op = IrOp.registry.get(op._opname)
        if ir_op is None:
            item["provider"] = None
            item["error"] = "unknown vllm_ir op"
            metadata.append(item)
            continue

        def fake_value(arg: Any) -> Any:
            if isinstance(arg, fx.Node):
                if "val" in arg.meta:
                    return arg.meta["val"]
                return arg.meta["example_value"]
            return arg

        try:
            fake_args = fx.map_arg(node.args, fake_value)
            item["provider"] = ir_op.dispatch(*fake_args).provider
        except Exception as exc:
            item["provider"] = None
            item["error"] = str(exc)
        metadata.append(item)
    return json_safe(metadata)


def safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    return safe or "graph"


def next_base_path(dump_dir: Path, name: str) -> Path:
    safe = safe_name(name)
    while True:
        counter = _dump_indices.setdefault(dump_dir, itertools.count())
        index = next(counter)
        base = dump_dir / f"{index:04d}_{os.getpid()}_{safe}"
        if not any(
            base.with_suffix(suffix).exists()
            for suffix in (".structured.txt", ".raw.py", ".metadata.json")
        ):
            return base


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(text)
            tmp_path = Path(f.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def append_index(
    dump_dir: Path, base: Path, name: str, metadata: dict[str, Any]
) -> None:
    index_path = dump_dir / "index.md"

    def cell(value: Any) -> str:
        return str(value or "-").replace("\n", " ").replace("|", "\\|")

    ir = ", ".join(
        f"{item['op']}:{item['provider']}" if item.get("provider") else item["op"]
        for item in metadata.get("vllm_ir") or []
        if item.get("op")
    )
    row = [
        base.name.split("_", 1)[0],
        name,
        metadata.get("function_name")
        or metadata.get("parent_function_name")
        or metadata.get("pass_name"),
        ir,
        f"[structured]({base.with_suffix('.structured.txt').name})",
        f"[raw]({base.with_suffix('.raw.py').name})",
        f"[metadata]({base.with_suffix('.metadata.json').name})",
    ]
    header = (
        "# vLLM Graph Dumps\n\n"
        "Read `*.structured.txt` first for layer/module nesting, "
        "`*.raw.py` for the raw FX graph, and `*.metadata.json` for "
        "vLLM compile context.\n\n"
        "| # | stage | context | vllm_ir | structured | raw | metadata |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else header
    row_text = "| " + " | ".join(cell(item) for item in row) + " |\n"
    atomic_write_text(index_path, existing + row_text)


def write_graph_dump(
    name: str,
    gm: fx.GraphModule,
    metadata: dict[str, Any] | None = None,
    dump_path: Path | None = None,
) -> None:
    if dump_path is None:
        logger.debug("%s", lazy_format_graph_code(name, gm))
        return

    dump_path.mkdir(parents=True, exist_ok=True)
    base = next_base_path(dump_path, name)
    metadata = json_safe(dict(metadata or {}))
    metadata.update({"name": name, "vllm_ir": collect_vllm_ir_metadata(gm)})

    atomic_write_text(base.with_suffix(".structured.txt"), format_structured_graph(gm))
    atomic_write_text(
        base.with_suffix(".raw.py"), str(lazy_format_graph_code(name, gm))
    )
    atomic_write_text(
        base.with_suffix(".metadata.json"),
        json.dumps(json_safe(metadata), indent=2, sort_keys=True),
    )
    append_index(dump_path, base, name, metadata)


def dump_graph(stage: str) -> None:
    context = _graph_dump_context
    if context.gm is None:
        return
    name = f"{context.stage_prefix}.{stage}" if context.stage_prefix else stage
    write_graph_dump(
        name,
        context.gm,
        dict(context.metadata, stage=stage),
        context.dump_path,
    )
