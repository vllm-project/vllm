# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
import torch.fx as fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._ops import OpOverload, OpOverloadPacket

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_dump_indices: dict[Path, itertools.count] = {}


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def enabled_passes(pass_config: Any) -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(pass_config)
        if isinstance(getattr(pass_config, field.name), bool)
        and getattr(pass_config, field.name)
    ]


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
                "mode": str(cc.mode),
                "backend": cc.backend,
                "custom_ops": cc.custom_ops,
                "debug_dump_path": vllm_config.compile_debug_dump_path(),
                "splitting_ops": cc.splitting_ops,
                "compile_sizes": cc.compile_sizes,
                "compile_ranges_endpoints": cc.compile_ranges_endpoints,
                "compile_ranges": [str(r) for r in cc.get_compile_ranges()],
                "cudagraph_mode": str(cc.cudagraph_mode),
                "use_inductor_graph_partition": cc.use_inductor_graph_partition,
                "enabled_passes": enabled_passes(cc.pass_config),
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

    metadata = []
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
    counter = _dump_indices.setdefault(dump_dir, itertools.count())
    while True:
        index = next(counter)
        base = dump_dir / f"{index:04d}_{safe}"
        if not any(
            base.with_suffix(suffix).exists()
            for suffix in (".structured.txt", ".raw.py", ".metadata.json")
        ):
            return base


def append_index(
    dump_dir: Path, base: Path, name: str, metadata: dict[str, Any]
) -> None:
    index_path = dump_dir / "index.md"
    if not index_path.exists():
        index_path.write_text(
            "# vLLM Graph Dumps\n\n"
            "Read `*.structured.txt` first for layer/module nesting, "
            "`*.raw.py` for the raw FX graph, and `*.metadata.json` for "
            "vLLM compile context.\n\n"
            "| # | stage | context | vllm_ir | structured | raw | metadata |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )

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
    with index_path.open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(cell(item) for item in row) + " |\n")


def dump_graph(
    name: str,
    gm: fx.GraphModule,
    dump_path: Path | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if dump_path is None:
        return

    dump_path.mkdir(parents=True, exist_ok=True)
    base = next_base_path(dump_path, name)
    metadata = dict(metadata or {})
    metadata.update({"name": name, "vllm_ir": collect_vllm_ir_metadata(gm)})

    base.with_suffix(".structured.txt").write_text(
        format_structured_graph(gm), encoding="utf-8"
    )
    base.with_suffix(".raw.py").write_text(
        str(lazy_format_graph_code(name, gm)), encoding="utf-8"
    )
    base.with_suffix(".metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_index(dump_path, base, name, metadata)
