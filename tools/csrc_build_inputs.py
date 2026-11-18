# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write what setup.py reads of pyproject.toml and vllm/envs.py.

The CUDA and ROCm Dockerfiles copy this output, not the original files, into
the stage that compiles the native extensions. COPY --from keys on content, so
an edit to an unrelated env var, a comment, a docstring or a lint rule leaves
the output unchanged and the compiled layer cached.

Output: the project and setuptools tables of pyproject.toml; the variables
setup.py names with the definitions they use (ast.unparse drops comments); an
empty vllm/__init__.py, which only has to exist.

Usage: tools/csrc_build_inputs.py SRC_DIR OUT_DIR, where SRC_DIR holds
setup.py, pyproject.toml and envs.py.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path


def prune_pyproject(text: str) -> str:
    out, keep = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("["):
            keep = re.match(r"\[(project|tool\.setuptools)", line) is not None
        if keep:
            out.append(line)
    return "".join(out)


def _names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef):
        return {node.name}
    targets = getattr(node, "targets", [getattr(node, "target", None)])
    return {t.id for t in targets if isinstance(t, ast.Name)}


def prune_envs(envs_src: str, setup_src: str) -> str:
    body = ast.parse(envs_src).body
    defs = {name: node for node in body for name in _names(node)}
    table = defs["environment_variables"].value
    assert isinstance(table, ast.Dict)
    words = set(re.findall(r"\w+", setup_src))
    kept = [
        (k, v)
        for k, v in zip(table.keys, table.values)
        if isinstance(k, ast.Constant) and k.value in words
    ]
    table.keys = [k for k, _ in kept]
    table.values = [v for _, v in kept]

    need: set[str] = set()
    todo = ["__getattr__", *re.findall(r"\benvs\.(\w+)", setup_src)]
    while todo:  # the module-level definitions those use, transitively
        name = todo.pop()
        if name in defs and name not in need:
            need.add(name)
            todo += [x.id for x in ast.walk(defs[name]) if isinstance(x, ast.Name)]
    keep = [
        node
        for node in body
        if isinstance(node, (ast.Import, ast.ImportFrom)) or _names(node) & need
    ]
    for node in keep:  # docstring edits must not recompile either
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.Module(keep, []))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("envs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(src: Path, out: Path) -> None:
    sys.dont_write_bytecode = True  # a .pyc in OUT_DIR would differ every build
    setup_src = (src / "setup.py").read_text()
    (out / "vllm").mkdir(parents=True, exist_ok=True)
    (out / "vllm" / "__init__.py").write_text("")
    (out / "pyproject.toml").write_text(
        prune_pyproject((src / "pyproject.toml").read_text())
    )
    pruned_path = out / "vllm" / "envs.py"
    pruned_path.write_text(prune_envs((src / "envs.py").read_text(), setup_src))

    # The pruning must not change what setup.py reads.
    orig, pruned = _load(src / "envs.py"), _load(pruned_path)
    for name in sorted(set(re.findall(r"\benvs\.(\w+)", setup_src))):
        if name in orig.environment_variables:
            assert getattr(orig, name) == getattr(pruned, name), f"envs.{name}"
        elif hasattr(orig, name):  # helpers such as envs.is_set
            assert hasattr(pruned, name), f"envs.{name}"


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
