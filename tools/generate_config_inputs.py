# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Project scheduler declarations into ordinary Python input and runtime fields."""

import argparse
import ast
import importlib.util
import subprocess
import sys
from collections import defaultdict
from dataclasses import MISSING, InitVar
from pathlib import Path
from types import UnionType
from typing import ClassVar, Literal, Union, get_args, get_origin

BEGIN = "    # BEGIN GENERATED SchedulerFields inputs"
END = "    # END GENERATED SchedulerFields inputs"
RUNTIME_BEGIN = "    # BEGIN GENERATED SchedulerFields runtime"
RUNTIME_END = "    # END GENERATED SchedulerFields runtime"


def type_source(annotation):
    origin = get_origin(annotation)
    if isinstance(annotation, InitVar):
        return "InitVar[" + type_source(annotation.type) + "]"
    if origin is ClassVar:
        return "ClassVar[" + type_source(get_args(annotation)[0]) + "]"
    if annotation is type(None):
        return "None"
    if origin in (Union, UnionType):
        return " | ".join(type_source(arg) for arg in get_args(annotation))
    if origin is Literal:
        return "Literal[" + ", ".join(repr(arg) for arg in get_args(annotation)) + "]"
    if origin is type:
        return "type[" + type_source(get_args(annotation)[0]) + "]"
    if isinstance(annotation, type) and annotation.__module__ == "builtins":
        return annotation.__name__
    raise ValueError(f"Unsupported input annotation: {annotation!r}")


def strip_generated(text):
    result = []
    inside = False
    for line in text.splitlines(keepends=True):
        if line.rstrip() == BEGIN:
            if inside:
                raise ValueError("Nested generated input section")
            inside = True
        elif line.rstrip() == END:
            if not inside:
                raise ValueError("Unmatched generated input section")
            inside = False
        elif not inside:
            result.append(line)
    if inside:
        raise ValueError("Unclosed generated input section")
    return "".join(result)


def load_declarations(source):
    package_name = "_vllm_config_specs_build"
    package_dir = source / "vllm/config_specs"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load config declarations")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    try:
        spec.loader.exec_module(package)
        scheduler_spec = importlib.util.find_spec(f"{package_name}.scheduler")
        if scheduler_spec is None or scheduler_spec.loader is None:
            raise RuntimeError("Unable to load scheduler declarations")
        scheduler = importlib.util.module_from_spec(scheduler_spec)
        sys.modules[scheduler_spec.name] = scheduler
        scheduler_spec.loader.exec_module(scheduler)
        return package.cli_default, package.cli_fields, scheduler.SchedulerFields
    finally:
        sys.modules.pop(f"{package_name}.scheduler", None)
        sys.modules.pop(package_name, None)


def render(source, text):
    cli_default, cli_fields, SchedulerFields = load_declarations(source)

    text = strip_generated(text)
    tree = ast.parse(text)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineArgs"
    )
    existing = {
        node.target.id: node
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    declared = {}
    following = defaultdict(list)
    for field, declaration in cli_fields(SchedulerFields):
        name = declaration.dest or field.name
        if not name.isidentifier() or name in declared or name in existing:
            raise ValueError(f"Conflicting input field: {name}")
        annotation = field.type
        if cli_default(SchedulerFields, field.name) is None:
            annotation = annotation | None
        expression = f'cli_default(SchedulerFields, "{field.name}")'
        # New inputs do not shift existing base/subclass positional parameters.
        if declaration.python_after is None:
            expression = f"dataclasses.field(default={expression}, kw_only=True)"
        declared[name] = f"    {name}: {type_source(annotation)} = {expression}\n"
        following[declaration.python_after].append(name)

    for anchor in following:
        if anchor is not None and anchor not in existing and anchor not in declared:
            raise ValueError(f"Missing Python input anchor: {anchor}")
    emitted = set()

    def expand(anchor):
        lines = []
        for name in following.get(anchor, ()):
            if name in emitted:
                raise ValueError(f"Repeated Python input: {name}")
            emitted.add(name)
            lines.append(declared[name])
            lines.extend(expand(name))
        return lines

    def end_of_field(node):
        index = cls.body.index(node)
        if index + 1 < len(cls.body):
            following_node = cls.body[index + 1]
            if (
                isinstance(following_node, ast.Expr)
                and isinstance(following_node.value, ast.Constant)
                and isinstance(following_node.value.value, str)
            ):
                return following_node.end_lineno
        return node.end_lineno

    insertions = defaultdict(list)
    for name, node in existing.items():
        insertions[end_of_field(node)].extend(expand(name))
    insertions[end_of_field(list(existing.values())[-1])].extend(expand(None))
    if emitted != set(declared):
        raise ValueError(
            f"Cyclic Python input anchors: {sorted(set(declared) - emitted)}"
        )
    lines = text.splitlines(keepends=True)
    for line, content in sorted(insertions.items(), reverse=True):
        if content:
            lines[line:line] = ["\n", BEGIN + "\n", *content, END + "\n", "\n"]
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("ruff")),
            "format",
            "--stdin-filename",
            "vllm/engine/arg_utils.py",
            "-",
        ],
        input="".join(lines),
        text=True,
        capture_output=True,
        cwd=source,
        check=True,
        timeout=30,
    )
    return result.stdout


def render_runtime(source, text):
    _, _, SchedulerFields = load_declarations(source)

    if text.count(RUNTIME_BEGIN) != 1 or text.count(RUNTIME_END) != 1:
        raise ValueError("Expected one generated runtime section")
    before, body = text.split(RUNTIME_BEGIN)
    _, after = body.split(RUNTIME_END)
    lines = []
    for field in SchedulerFields.__dataclass_fields__.values():
        unknown = field.metadata.keys() - {"doc", "cli", "ge", "lt"}
        if unknown or field.default_factory is not MISSING:
            raise ValueError(f"Unsupported runtime field: {field.name}")
        if not field.repr or not field.compare or field.hash is not None:
            raise ValueError(f"Unsupported dataclass options: {field.name}")
        options = {
            key: value for key, value in field.metadata.items() if key in {"ge", "lt"}
        }
        if not field.init:
            options["init"] = False
        if field.kw_only is True:
            options["kw_only"] = True
        line = f"    {field.name}: {type_source(field.type)}"
        if options:
            if field.default is not MISSING:
                options = {"default": field.default, **options}
            arguments = ", ".join(f"{key}={value!r}" for key, value in options.items())
            line += f" = Field({arguments})"
        elif field.default is not MISSING:
            line += f" = {field.default!r}"
        lines.append(line + "\n")
        if doc := field.metadata.get("doc"):
            doc = doc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            doc = "\n".join(
                ("    " if index and line else "") + line
                for index, line in enumerate(doc.split("\n"))
            )
            lines.append('    """' + doc + '"""\n')
        lines.append("\n")
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("ruff")),
            "format",
            "--stdin-filename",
            "vllm/config/scheduler.py",
            "-",
        ],
        input=before + RUNTIME_BEGIN + "\n" + "".join(lines) + RUNTIME_END + after,
        text=True,
        capture_output=True,
        cwd=source,
        check=True,
        timeout=30,
    )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    targets = {
        source / "vllm/engine/arg_utils.py": render,
        source / "vllm/config/scheduler.py": render_runtime,
    }
    updates = {}
    for target, renderer in targets.items():
        actual = target.read_text()
        expected = renderer(source, actual)
        if actual != expected:
            updates[target] = expected
    if updates and not args.write:
        parser.exit(
            1,
            "Stale config projections; run tools/generate_config_inputs.py --write\n",
        )
    for target, expected in updates.items():
        target.write_text(expected)
        print(f"Updated {target.relative_to(source)}.")
    if not updates:
        print("Python input and runtime projections are current.")


if __name__ == "__main__":
    main()
