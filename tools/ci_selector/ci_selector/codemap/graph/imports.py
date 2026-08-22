# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The import graph, read from source.

Beyond plain and relative imports it follows dynamic imports whose target is a
readable string, pytest's sibling-first rule for test dirs with no __init__,
and the conftest chain every test depends on. Example scripts parse as nodes
too, since steps run them directly. A dynamic import whose target cannot be
read becomes a DynamicSite and gets no edge; the policy layer decides what that
means.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ...handwritten import PACKAGE_ROOTS
from ..repo import SKIP_DIRS, ModuleIndex

DYNAMIC_IMPORT_FUNCS = {"import_module", "__import__", "resolve_obj_by_qualname"}
LAZY_LOADER_CLASS = "LazyLoader"
ENGINE_FIXTURE = "vllm_runner"

EXAMPLES_DIR = "examples"
PYTHON_GLOB = "*.py"

LAZY_LOADER_MODULE_ARG = 2  # LazyLoader(name, globals(), module)
LITERAL_MIN_LEN = 3  # shorter is noise ("py", "id")
LITERAL_MAX_LEN = 120  # longer is prose or a command line


def routable_literal(value: object) -> bool:
    """Specific enough to route by, short enough to be a key not prose."""
    return (
        isinstance(value, str)
        and LITERAL_MIN_LEN <= len(value) < LITERAL_MAX_LEN
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
    # String constants per test file, matched against registered names.
    string_literals: dict[str, set[str]] = field(default_factory=dict)
    # Function-local imports, held until the parsers say which targets they
    # route by string key. See finalize_lazy_edges.
    lazy_imports: set[tuple[str, str]] = field(default_factory=set)
    # Lazy edges dropped because a parser claims the target. Never holds one
    # that started at a test or benchmark.
    dropped_lazy: list[tuple[str, str]] = field(default_factory=list)
    # method name -> files calling it, for the platform parser. get_* only.
    method_calls: dict[str, set[str]] = field(default_factory=dict)
    # `from X import alias` where alias is not a submodule. Lazy-export
    # tables and class imports resolve through this.
    from_import_aliases: list[tuple[str, str, str]] = field(default_factory=list)
    # file -> classes holding a literal argv that spawns vllm.
    spawn_sites: dict[str, set[str]] = field(default_factory=dict)
    # Bare imports that could mean a sibling OR an indexed module. The
    # sibling wins, as pytest does it, and the clash is reported rather than
    # guessed at silently.
    ambiguities: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Edges only taken when an engine boots. Walked by default, but a test
    # reached ONLY through one counts only if it boots an engine.
    gated_edges: set[tuple[str, str]] = field(default_factory=set)
    # Registration imports cut out of selection: the importer pulls a plugin
    # in at module top but only builds it behind a config guard. The real edge
    # stays in `imports`; selection skips it and routes by config key instead.
    demoted_edges: set[tuple[str, str]] = field(default_factory=set)
    # Test files whose tests take the engine fixture.
    engine_fixture_files: set[str] = field(default_factory=set)
    # Files a parser read a table out of. Derived, so one that stops matching
    # drops its file and the dynamic import there goes unclassified, loudly.
    table_files: set[str] = field(default_factory=set)
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

    def forward_closure(self, files: set[str], include_gated: bool = True) -> set[str]:
        """Everything `files` imports, transitively.

        `reverse_closure` turned around. Selection asks who reaches a file;
        staleness asks what a test reads. Skips the same two edge classes for
        the same reasons: a demoted edge is config-gated rather than real, and
        a gated edge is only conditionally taken.
        """
        seen = set(files)
        stack = list(files)
        while stack:
            cur = stack.pop()
            for dst in self.imports.get(cur, ()):
                if (cur, dst) in self.demoted_edges:
                    continue
                if not include_gated and (cur, dst) in self.gated_edges:
                    continue
                if dst not in seen:
                    seen.add(dst)
                    stack.append(dst)
        return seen

    def finalize_lazy_edges(self, string_keyed: set[str]) -> None:
        """Turn the deferred function-local imports into real edges.

        A lazy edge into a claimed target is dropped, because a parser already
        routes that target by string key and keeping the edge would pull every
        lazy registry into a near-run-all closure. One exception: when the
        importer is a test or benchmark, the edge stays. That is one leaf
        depending on one backend, not a registry fanning out, and dropping it
        used to unlink tests from the backends they exercise. Everything else
        keeps its edge, since a lazy import no parser claims still counts.
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
    consts = _module_string_consts(tree)
    _FileVisitor(repo, file, module, index, graph, consts).visit(tree)


def _module_string_consts(tree: ast.Module) -> dict[str, str]:
    """Module-level `NAME = "literal"` bindings. An import handed one of these
    is as readable as one handed the literal."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            out[target.id] = value.value
    return out


class _FileVisitor(ast.NodeVisitor):
    """Import walk that tracks whether an import actually runs.

    `if TYPE_CHECKING:` is skipped, since those never execute. An import inside
    a function is lazy and gets deferred to finalize_lazy_edges; class bodies
    run at import time, so only function depth counts. A dynamic import with a
    readable target stays a direct edge even inside a function, because it is
    usually the real dispatch.
    """

    def __init__(self, repo, file, module, index, graph, consts=None):
        self.repo = repo
        self.file = file
        self.index = index
        self.graph = graph
        self.consts = consts or {}
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
            a.arg == ENGINE_FIXTURE for a in (*node.args.args, *node.args.kwonlyargs)
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
        _resolve_call(node, self.index, self.graph, self.file, self.consts)
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
    """In a test dir with no __init__.py, pytest resolves a bare import to a
    sibling first."""
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
    """The module a `from X import ...` reads, with relative levels resolved
    against the importer's package. None if it walks above the root."""
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
    node: ast.Call,
    index: ModuleIndex,
    graph: ImportGraph,
    file: str,
    consts: dict[str, str],
) -> None:
    func = node.func
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    if name == LAZY_LOADER_CLASS:
        _resolve_lazy_loader(node, index, graph, file, consts)
        return
    if name not in DYNAMIC_IMPORT_FUNCS or not node.args:
        return
    target_name = _module_string(node.args[0], consts)
    if target_name is None:
        graph.dynamic_sites.append(DynamicSite(file, node.lineno, name))
        return
    _link_module(
        target_name,
        index,
        graph,
        file,
        node.lineno,
        name,
        parent_fallback=name == "resolve_obj_by_qualname",
    )


def _resolve_lazy_loader(
    node: ast.Call,
    index: ModuleIndex,
    graph: ImportGraph,
    file: str,
    consts: dict[str, str],
) -> None:
    """LazyLoader defers an import to first attribute access, but the file
    holding it is still the consumer, so it gets the edge like any import.
    Without this the in-repo ones had no importer at all."""
    if len(node.args) <= LAZY_LOADER_MODULE_ARG:
        graph.dynamic_sites.append(DynamicSite(file, node.lineno, LAZY_LOADER_CLASS))
        return
    target_name = _module_string(node.args[LAZY_LOADER_MODULE_ARG], consts)
    if target_name is None:
        graph.dynamic_sites.append(DynamicSite(file, node.lineno, LAZY_LOADER_CLASS))
        return
    _link_module(target_name, index, graph, file, node.lineno, LAZY_LOADER_CLASS, False)


def _module_string(arg: ast.expr, consts: dict[str, str]) -> str | None:
    """The module an import names, when we can read it: a literal, or a name
    bound to one at module level."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        return consts.get(arg.id)
    return None


def _link_module(
    target_name: str,
    index: ModuleIndex,
    graph: ImportGraph,
    file: str,
    lineno: int,
    func: str,
    parent_fallback: bool,
) -> None:
    resolved = index.resolve(target_name)
    if resolved is None and parent_fallback and "." in target_name:
        resolved = index.resolve(target_name.rsplit(".", 1)[0])
    if resolved:
        graph.add_edge(file, resolved)
        return
    # Nothing resolved. A top-level package that is not one of ours proves the
    # import leaves the repo, so there is no edge to miss and nothing to flag.
    # A package that IS ours means the path is broken or moved -- a real hole,
    # and it used to disappear here without a word.
    if target_name.split(".", 1)[0] in PACKAGE_ROOTS:
        graph.dynamic_sites.append(DynamicSite(file, lineno, func))


def _add_examples_nodes(repo: Path, index: ModuleIndex, graph: ImportGraph) -> None:
    base = repo / EXAMPLES_DIR
    if not base.is_dir():
        return
    for path in base.rglob(PYTHON_GLOB):
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
