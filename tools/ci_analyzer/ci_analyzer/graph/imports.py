# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AST import graph over indexed packages plus examples/ script nodes.

Beyond plain and relative imports, it resolves literal importlib/__import__
calls, pytest's sibling-first resolution for non-package test dirs, and the
conftest/__init__ auto-load chain every test file depends on. examples/*.py parse
as script nodes with dir-local sibling resolution. Non-literal dynamic imports
become DynamicSite for the policy layer and produce no edge here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..repo import SKIP_DIRS, ModuleIndex

DYNAMIC_IMPORT_FUNCS = {"import_module", "__import__", "resolve_obj_by_qualname"}

# Shortest string constant that can carry routing signal. Below it a literal is a
# flag value ("1", "v1"), not a key. Single home because it defines what the
# dispatch fanout bar is able to measure: an uncollected literal scores zero
# there, clearing a bar it was never weighed against (dispatch._route).
LITERAL_MIN_LEN = 3
_LITERAL_MAX_LEN = 120


def routable_literal(value: object) -> bool:
    """A string constant specific enough to route by and short enough to be a
    key rather than prose."""
    return (
        isinstance(value, str)
        and LITERAL_MIN_LEN <= len(value) < _LITERAL_MAX_LEN
        and "\n" not in value
    )


@dataclass
class DynamicSite:
    file: str
    lineno: int
    func: str


@dataclass
class ImportGraph:
    imports: dict[str, set[str]] = field(default_factory=dict)
    dynamic_sites: list[DynamicSite] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    # String constants per tests/ file, for string-keyed wall parsers
    # (model archs / HF ids are matched against these, never guessed).
    string_literals: dict[str, set[str]] = field(default_factory=dict)
    # Function-local imports: (src, dst) pairs deferred until the wall
    # parsers declare which targets they string-key (finalize_lazy_edges).
    lazy_imports: set[tuple[str, str]] = field(default_factory=set)
    # Lazy edges dropped because their target is claimed; audited by
    # ci-validate dropped-edges. Test/benchmark-origin edges are never here.
    dropped_lazy: list[tuple[str, str]] = field(default_factory=list)
    # Recorded for the wall parsers (filled during the same visitor pass):
    # method name -> files calling <obj>.method(); only get_*/import_kernels.
    method_calls: dict[str, set[str]] = field(default_factory=dict)
    # (file, resolved_base_file, alias) for `from X import alias` where alias
    # is NOT a submodule: lazy-export tables and class imports resolve here.
    from_import_aliases: list[tuple[str, str, str]] = field(default_factory=list)
    # file -> enclosing class names (or "") containing a literal
    # ["vllm", ...] / [-m vllm...] spawn argv.
    spawn_sites: dict[str, set[str]] = field(default_factory=dict)
    # bare imports resolvable to BOTH a sibling and an indexed module:
    # (file, name, sibling_target, other_target). Sibling wins (pytest
    # prepend), the ambiguity is reported (Pants: refuse to guess silently).
    ambiguities: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Edges that execute only when an engine boots (platform worker_cls
    # qualname assignments): traversed by default, but selection gates the
    # tests reached ONLY through them to engine-starting tests.
    gated_edges: set[tuple[str, str]] = field(default_factory=set)
    # Eager registration imports demoted out of selection closures: an
    # importer (a model runner) imports a plugin at module top but only
    # constructs it behind a config-key guard, so the plugin runs only when
    # that key is selected. The raw edge stays in `imports` (plain_reverse and
    # import audits keep the truth); selection skips it and routes the
    # plugin's coverage through string keys instead (add_demotion_edges).
    demoted_edges: set[tuple[str, str]] = field(default_factory=set)
    # tests/ files whose test functions take an engine fixture (vllm_runner)
    engine_fixture_files: set[str] = field(default_factory=set)
    _reverse: dict[str, set[str]] | None = None

    def add_edge(self, src: str, dst: str) -> None:
        if dst != src:
            self.imports.setdefault(src, set()).add(dst)
        self._reverse = None

    @property
    def reverse(self) -> dict[str, set[str]]:
        if self._reverse is None:
            rev: dict[str, set[str]] = {}
            for src, dsts in self.imports.items():
                for dst in dsts:
                    rev.setdefault(dst, set()).add(src)
            self._reverse = rev
        return self._reverse

    def add_gated_edge(self, src: str, dst: str) -> None:
        self.add_edge(src, dst)
        self.gated_edges.add((src, dst))

    def reverse_closure(self, files: set[str], include_gated: bool = True) -> set[str]:
        """All files that (transitively) import any of `files`."""
        seen = set(files)
        stack = list(files)
        while stack:
            cur = stack.pop()
            for src in self.reverse.get(cur, ()):
                if (src, cur) in self.demoted_edges:
                    continue
                if not include_gated and (src, cur) in self.gated_edges:
                    continue
                if src not in seen:
                    seen.add(src)
                    stack.append(src)
        return seen

    def finalize_lazy_edges(self, string_keyed: set[str]) -> None:
        """Materialize deferred function-local import edges.

        A lazy edge into a string-keyed target (a file, or anything under a
        claimed package prefix ending in /) is DROPPED: the wall parser
        already routes that target's coverage through leaf test edges, and
        keeping the import edge would route every lazy registry (quant
        method map, model lookup) into near-run-all closures. EXCEPT when the
        importer is itself a test/benchmark file: a leaf consumer lazily
        importing a claimed target is a genuine behavioral dependency costing
        one leaf, not a registry fan-out -- dropping it silently unlinked
        tests from the backends they exercise. Everything else gets its
        conservative edge back: a lazy import no parser claims must still
        count.
        """
        prefixes = tuple(p for p in string_keyed if p.endswith("/"))
        for src, dst in sorted(self.lazy_imports):
            if (dst in string_keyed or dst.startswith(prefixes)) and not src.startswith(
                ("tests/", "benchmarks/")
            ):
                self.dropped_lazy.append((src, dst))
                continue
            self.add_edge(src, dst)
        self.lazy_imports.clear()


def build_graph(repo: Path, index: ModuleIndex) -> ImportGraph:
    graph = ImportGraph()
    for file in index.file_to_module:
        _parse_file(repo, file, index, graph)
    _add_examples_nodes(repo, index, graph)
    _add_conftest_edges(index, graph)
    return graph


def _parse_file(repo: Path, file: str, index: ModuleIndex, graph: ImportGraph) -> None:
    try:
        tree = ast.parse((repo / file).read_text(), filename=file)
    except (SyntaxError, UnicodeDecodeError, OSError):
        graph.parse_errors.append(file)
        return
    module = index.file_to_module.get(file, "")
    _FileVisitor(repo, file, module, index, graph).visit(tree)


class _FileVisitor(ast.NodeVisitor):
    """Import walk with executability awareness.

    - `if TYPE_CHECKING:` subtrees are skipped: those imports never run.
    - Imports inside function bodies are LAZY: recorded for
      finalize_lazy_edges, not edged directly. Class bodies execute at
      import time, so only function depth counts.
    - Literal dynamic-import calls stay direct edges even inside functions
      (small, and usually the actual dispatch target).
    """

    def __init__(self, repo, file, module, index, graph):
        self.repo = repo
        self.file = file
        self.index = index
        self.graph = graph
        self.package = _package_of(module, file)
        self.bare_dir = _bare_import_dir(repo, file)
        self.func_depth = 0
        self.class_stack: list[str] = []
        self.collect_literals = file.startswith(("tests/", "benchmarks/"))
        self.literals: set[str] = set()

    def visit(self, node):
        super().visit(node)
        if node.__class__ is ast.Module and self.collect_literals:
            self.graph.string_literals[self.file] = self.literals

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        name = (
            test.id
            if isinstance(test, ast.Name)
            else test.attr
            if isinstance(test, ast.Attribute)
            else None
        )
        if name == "TYPE_CHECKING":
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node) -> None:
        if self.collect_literals and any(
            a.arg == "vllm_runner" for a in (*node.args.args, *node.args.kwonlyargs)
        ):
            self.graph.engine_fixture_files.add(self.file)
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        sink = self._sink()
        for alias in node.names:
            _resolve_absolute(
                alias.name, self.index, sink, self.file, self.bare_dir, self.repo
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        _resolve_from(
            node,
            self.package,
            self.index,
            self._sink(),
            self.file,
            self.bare_dir,
            self.repo,
        )
        # Relative levels resolve too: a test doing `from ...utils import
        # RemoteOpenAIServer` binds the same name as the absolute form, and
        # missing it drops the test out of engine_starting_tests, where the
        # worker-seam gate then subtracts it.
        base_module = _from_base(node, self.package)
        if base_module:
            base_file = self.index.resolve(base_module)
            if base_file:
                for alias in node.names:
                    if alias.name != "*" and not self.index.resolve(
                        f"{base_module}.{alias.name}"
                    ):
                        self.graph.from_import_aliases.append(
                            (self.file, base_file, alias.name)
                        )

    def visit_Call(self, node: ast.Call) -> None:
        _resolve_call(node, self.index, self.graph, self.file)
        if isinstance(node.func, ast.Attribute) and (
            node.func.attr.startswith("get_") or node.func.attr == "import_kernels"
        ):
            self.graph.method_calls.setdefault(node.func.attr, set()).add(self.file)
        if self.collect_literals:
            self._collect_join_literal(node)
        self.generic_visit(node)

    def _collect_join_literal(self, node: ast.Call) -> None:
        # os.path.join("prompts", "example.txt") builds a fixture path whose
        # slash never appears in one string literal, so _looks_like_path
        # (assets.py) is blind to it. Synthesize the "/"-joined string-constant
        # components; the asset parser resolves it against the test's own dir
        # and the repo root, so a leading non-constant dir anchor is harmless.
        # str.join takes a single iterable arg, so >=2 args excludes it.
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "join"):
            return
        if len(node.args) < 2:
            return
        parts = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        if len(parts) >= 2:
            joined = "/".join(parts)
            if len(joined) < 200 and "\n" not in joined:
                self.literals.add(joined)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Path(_TEST_DIR) / "prompts" / "example.txt": the same hidden-slash
        # shape via pathlib. Collect the trailing chain of `/ "const"` operands
        # (dividing by a str is only valid for Path, so this cannot misfire on
        # arithmetic) and synthesize the joined tail.
        if self.collect_literals and isinstance(node.op, ast.Div):
            parts: list[str] = []
            cur: ast.AST = node
            while (
                isinstance(cur, ast.BinOp)
                and isinstance(cur.op, ast.Div)
                and isinstance(cur.right, ast.Constant)
                and isinstance(cur.right.value, str)
            ):
                parts.append(cur.right.value)
                cur = cur.left
            if len(parts) >= 2:
                joined = "/".join(reversed(parts))
                if len(joined) < 200 and "\n" not in joined:
                    self.literals.add(joined)
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> None:
        consts = [
            e.value
            for e in node.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if consts and (
            consts[0] == "vllm"
            or any(
                c == "-m" and i + 1 < len(consts) and consts[i + 1].startswith("vllm")
                for i, c in enumerate(consts)
            )
        ):
            owner = self.class_stack[-1] if self.class_stack else ""
            self.graph.spawn_sites.setdefault(self.file, set()).add(owner)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if self.collect_literals and routable_literal(node.value):
            self.literals.add(node.value)

    def _sink(self):
        if self.func_depth == 0:
            return self.graph
        return _LazySink(self.graph, self.file)


class _LazySink:
    """Graph facade routing edges into the deferred lazy set."""

    def __init__(self, graph: ImportGraph, file: str):
        self._graph = graph
        self._file = file

    def add_edge(self, src: str, dst: str) -> None:
        if dst != src:
            self._graph.lazy_imports.add((src, dst))

    @property
    def ambiguities(self):
        """The resolvers take this sink as their `graph` and report bare-name
        clashes on it. Deferring the EDGE does not defer the clash, which is a
        fact about the name either way."""
        return self._graph.ambiguities


def _package_of(module: str, file: str) -> str:
    if file.endswith("__init__.py"):
        return module
    return module.rsplit(".", 1)[0] if "." in module else ""


def _bare_import_dir(repo: Path, file: str) -> str | None:
    """For tests/ files in a dir with no __init__.py, bare imports resolve
    to siblings first (pytest prepend semantics)."""
    if not file.startswith("tests/"):
        return None
    parent = (repo / file).parent
    if (parent / "__init__.py").is_file():
        return None
    return parent.relative_to(repo).as_posix()


def _resolve_absolute(
    name: str,
    index: ModuleIndex,
    graph: ImportGraph,
    file: str,
    bare_dir: str | None,
    repo: Path,
) -> None:
    if bare_dir and "." not in name:
        sibling = f"{bare_dir}/{name}.py"
        if (repo / sibling).is_file():
            graph.add_edge(file, sibling)
            # Confusable: the same bare name also names an importable module
            # (`utils` -> tests/utils.py). Sibling wins under pytest prepend;
            # the clash is surfaced, never silently guessed away.
            other = index.resolve(name) or index.resolve(f"tests.{name}")
            if other and other != sibling:
                graph.ambiguities.append((file, name, sibling, other))
            return
    # `import a.b.c` executes every package __init__ on the way down.
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        target = index.resolve(".".join(parts[:depth]))
        if target:
            graph.add_edge(file, target)


def _from_base(node: ast.ImportFrom, package: str) -> str | None:
    """The dotted module an `from X import ...` reads, with relative levels
    resolved against the importer's package. None when the level walks above
    the package root."""
    if not node.level:
        return node.module or None
    parts = package.split(".") if package else []
    cut = len(parts) - (node.level - 1)
    if cut < 0:
        return None
    base = ".".join(parts[:cut])
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base or None


def _resolve_from(
    node: ast.ImportFrom,
    package: str,
    index: ModuleIndex,
    graph: ImportGraph,
    file: str,
    bare_dir: str | None,
    repo: Path,
) -> None:
    if node.level:
        resolved = _from_base(node, package)
        if resolved is None:
            return
        base = resolved
    else:
        base = node.module or ""
        if bare_dir and base and "." not in base:
            sibling = f"{bare_dir}/{base}.py"
            if (repo / sibling).is_file():
                graph.add_edge(file, sibling)
                other = index.resolve(base) or index.resolve(f"tests.{base}")
                if other and other != sibling:
                    graph.ambiguities.append((file, base, sibling, other))
                return
    if base:
        _resolve_absolute(base, index, graph, file, None, repo)
    for alias in node.names:
        if alias.name == "*":
            continue
        target = index.resolve(f"{base}.{alias.name}" if base else alias.name)
        if target:
            graph.add_edge(file, target)


def _resolve_call(
    node: ast.Call, index: ModuleIndex, graph: ImportGraph, file: str
) -> None:
    func = node.func
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    if name not in DYNAMIC_IMPORT_FUNCS or not node.args:
        return
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        target_name = arg.value
        resolved = index.resolve(target_name)
        if (
            resolved is None
            and name == "resolve_obj_by_qualname"
            and "." in target_name
        ):
            resolved = index.resolve(target_name.rsplit(".", 1)[0])
        if resolved:
            graph.add_edge(file, resolved)
        return
    graph.dynamic_sites.append(DynamicSite(file, node.lineno, name))


def _add_examples_nodes(repo: Path, index: ModuleIndex, graph: ImportGraph) -> None:
    base = repo / "examples"
    if not base.is_dir():
        return
    for path in base.rglob("*.py"):
        if SKIP_DIRS.intersection(path.parts):
            continue
        file = path.relative_to(repo).as_posix()
        try:
            tree = ast.parse(path.read_text(), filename=file)
        except (SyntaxError, UnicodeDecodeError, OSError):
            graph.parse_errors.append(file)
            continue
        local_dir = path.parent
        literals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _example_edge(alias.name, local_dir, repo, index, graph, file)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                _example_edge(node.module, local_dir, repo, index, graph, file)
            elif isinstance(node, ast.Constant) and routable_literal(node.value):
                # Example scripts are leaf consumers like tests: their string
                # constants (argparse choices, model names) feed leaf routing.
                literals.add(node.value)
        if literals:
            graph.string_literals[file] = literals


def _example_edge(name, local_dir, repo, index, graph, file):
    sibling = local_dir / (name.split(".")[0] + ".py")
    if "." not in name and sibling.is_file():
        graph.add_edge(file, sibling.relative_to(repo).as_posix())
        return
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        target = index.resolve(".".join(parts[:depth]))
        if target:
            graph.add_edge(file, target)


def _add_conftest_edges(index: ModuleIndex, graph: ImportGraph) -> None:
    # Package __init__.py files ride the same mechanism: importing any test
    # module inside a package executes every ancestor __init__.py, so tests
    # depend on them exactly like on ancestor conftests.
    auto_loaded = {
        f
        for f in index.file_to_module
        if f.endswith(("/conftest.py", "/__init__.py")) and f.startswith("tests/")
    }
    by_dir: dict[str, list[str]] = {}
    for c in auto_loaded:
        by_dir.setdefault(c.rsplit("/", 1)[0] + "/", []).append(c)
    for file in index.file_to_module:
        if not file.startswith("tests/"):
            continue
        basename = file.rsplit("/", 1)[-1]
        if not (basename.startswith("test_") or basename == "conftest.py"):
            continue
        for dir_prefix, loaded in by_dir.items():
            if file.startswith(dir_prefix):
                for target in loaded:
                    if file != target:
                        graph.add_edge(file, target)
