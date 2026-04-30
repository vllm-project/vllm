# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import itertools
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.fx as fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._ops import OpOverload, OpOverloadPacket

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_dump_index = itertools.count()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _enabled_passes(pass_config: Any) -> list[str]:
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
        metadata.update(
            {
                "model": getattr(vllm_config.model_config, "model", None),
                "mode": str(cc.mode),
                "backend": cc.backend,
                "custom_ops": cc.custom_ops,
                "splitting_ops": cc.splitting_ops,
                "compile_sizes": cc.compile_sizes,
                "compile_ranges_endpoints": cc.compile_ranges_endpoints,
                "compile_ranges": [str(r) for r in cc.get_compile_ranges()],
                "cudagraph_mode": str(cc.cudagraph_mode),
                "use_inductor_graph_partition": cc.use_inductor_graph_partition,
                "enabled_passes": _enabled_passes(cc.pass_config),
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
    return _json_safe(metadata)


def _module_type_name(module_type: Any) -> str:
    if hasattr(module_type, "__name__"):
        return module_type.__name__
    name = str(module_type)
    if "." in name:
        return name.rsplit(".", 1)[-1].strip("'>")
    return name


def _format_path_part(part: str) -> str:
    part = part.removeprefix("L['self'].").removeprefix("self.")
    if part == "L['self']":
        part = ""
    pieces = []
    for token in part.split("."):
        if token.isdigit() and pieces:
            pieces[-1] += f"[{token}]"
        elif token:
            pieces.append(token)
    return ".".join(pieces)


def _module_stack(node: fx.Node) -> list[tuple[str, str]]:
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
        result.append((_format_path_part(display), _module_type_name(module_type)))
        prev_path = path
    return result


def _source_context(node: fx.Node) -> str:
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

        stack = _module_stack(node)
        divergence = 0
        for old, new in zip(previous_stack, stack):
            if old != new:
                break
            divergence += 1

        for depth, (name, module_type) in enumerate(stack[divergence:], divergence):
            label = f"{name}: {module_type}" if name else module_type
            lines.append(f"{'  ' * depth}{label}")

        node_text = node.format_node() or str(node)
        lines.append(f"{'  ' * len(stack)}{node_text}{_source_context(node)}")
        previous_stack = stack

    return "\n".join(lines) + ("\n" if lines else "")


def _default_overload(target: Any) -> OpOverload | None:
    if isinstance(target, OpOverload):
        return target
    if isinstance(target, OpOverloadPacket):
        return target.default
    return None


def collect_vllm_ir_metadata(gm: fx.GraphModule) -> list[dict[str, Any]]:
    from vllm.ir.op import IrOp

    metadata = []
    for node in gm.graph.nodes:
        op = _default_overload(node.target)
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
    return _json_safe(metadata)


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    return safe or "graph"


def _next_base_path(dump_dir: Path, name: str) -> Path:
    safe = _safe_name(name)
    while True:
        index = next(_dump_index)
        base = dump_dir / f"{index:04d}_{safe}"
        paths = [
            base.with_suffix(".structured.txt"),
            base.with_suffix(".raw.py"),
            base.with_suffix(".metadata.json"),
        ]
        if not any(path.exists() for path in paths):
            return base


def dump_graph(
    name: str,
    gm: fx.GraphModule,
    dump_path: Path | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if dump_path is None:
        return

    dump_path.mkdir(parents=True, exist_ok=True)
    base = _next_base_path(dump_path, name)
    metadata = dict(metadata or {})
    metadata.update({"name": name, "vllm_ir": collect_vllm_ir_metadata(gm)})

    base.with_suffix(".structured.txt").write_text(
        format_structured_graph(gm), encoding="utf-8"
    )
    base.with_suffix(".raw.py").write_text(
        str(lazy_format_graph_code(name, gm)), encoding="utf-8"
    )
    base.with_suffix(".metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
