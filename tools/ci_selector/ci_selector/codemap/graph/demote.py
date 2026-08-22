# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cutting import edges that only exist to register a plugin.

A model runner imports every plugin class at module top but builds one only
behind a config guard like `method == "eagle3"`. That import registers the
plugin, it does not depend on it. Left as an edge, changing one proposer reaches
nearly every engine test through the worker seam. So the edge is cut and the
member is routed by its config key instead, the same way the model registry and
parser engine work.

Only safe when every real use of the member sits behind a guard whose keys we
could read in full. Anything we could not read keeps the edge, which
over-selects. The guards come from config predicates read out of vLLM, never a
hand list.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ...handwritten import PLATFORM_GUARD_LITERALS
from ..repo import ModuleIndex, is_test_basename
from .factories import _leaf_consumer, _leaf_edges, _path_leaf_edges
from .imports import LITERAL_MIN_LEN, ImportGraph, _from_base, _package_of

CONFIG_DIR = "vllm/config/"
CONFIG_KEY_MAX_TEST_FILES = 32  # above this a word is ordinary ("auto": 170 files)


@dataclass
class DispatchParse:
    # (importer_file, member_file) -> the config-key literals that gate it
    demotions: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    claims: set[str] = field(default_factory=set)
    edges_added: int = 0


def _str_const(node: ast.AST) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


@dataclass(frozen=True)
class Flags:
    """Config-derived boolean flags, kept in two namespaces on purpose.

    `self.use_data_parallel` and a bare `use_data_parallel` are different
    bindings. Sharing one map let any local of the same name read as the config
    guard, and a member got demoted on a literal the guard never implied."""

    attrs: dict[str, set[str]] = field(default_factory=dict)
    names: dict[str, set[str]] = field(default_factory=dict)

    def rebound(self, bound: set[str], local: dict[str, set[str]]) -> Flags:
        """Enter a function scope: names it binds shadow the outer ones (and
        stay unresolved unless the local value itself reduces to literals)."""
        kept = {k: v for k, v in self.names.items() if k not in bound}
        return Flags(self.attrs, {**kept, **local})


def _unit_literals(
    node: ast.AST, helpers: dict[str, set[str]], flags: Flags
) -> set[str] | None:
    """Config-key literals of a single (non-boolean) predicate, else None."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op, right = node.ops[0], node.comparators[0]
        if isinstance(op, ast.Eq):
            # `__name__ == "__main__"` is script dispatch, not a config key.
            for side in (node.left, right):
                if isinstance(side, ast.Name) and side.id == "__name__":
                    return None
            lit = _str_const(right)
            if lit is None:
                lit = _str_const(node.left)
            if lit is None or lit in PLATFORM_GUARD_LITERALS:
                return None
            return {lit}
        if isinstance(op, ast.In) and isinstance(right, (ast.Tuple, ast.List, ast.Set)):
            lits = {c for e in right.elts if (c := _str_const(e)) is not None}
            # One platform member spoils the whole set: routing by the rest
            # would under-select the platform case.
            if not lits or lits & PLATFORM_GUARD_LITERALS:
                return None
            return lits
        return None
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if name in helpers:
            return set(helpers[name])
    if isinstance(node, ast.Attribute) and node.attr in flags.attrs:
        return set(flags.attrs[node.attr])
    if isinstance(node, ast.Name) and node.id in flags.names:
        return set(flags.names[node.id])
    return None


def _test_literals(
    node: ast.AST, helpers: dict[str, set[str]], flags: Flags
) -> set[str] | None:
    """The config keys a guard guarantees when its branch runs, else None.

    With AND, one config operand is enough, since the branch running implies it
    held. With OR, every operand must be one, or the branch says nothing. `not`
    never guarantees anything."""
    if isinstance(node, ast.BoolOp):
        parts = [_test_literals(v, helpers, flags) for v in node.values]
        if isinstance(node.op, ast.And):
            cfg = [p for p in parts if p is not None]
            return set().union(*cfg) if cfg else None
        if any(p is None for p in parts):
            return None
        return set().union(*parts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return None
    return _unit_literals(node, helpers, flags)


def _predicate_literals(
    func: ast.FunctionDef, helpers: dict[str, set[str]], flags: Flags
) -> set[str] | None:
    """A predicate method reduces to config literals only if EVERY return path
    yields them; a bare `return True` makes it unresolvable (keep broad)."""
    out: set[str] = set()
    seen = False
    for n in ast.walk(func):
        if isinstance(n, ast.Return) and n.value is not None:
            seen = True
            lits = _test_literals(n.value, helpers, flags)
            if lits is None:
                return None
            out |= lits
    return out if (seen and out) else None


def _config_helper_literals(repo: Path, index: ModuleIndex) -> dict[str, set[str]]:
    """method name -> literals, for simple predicate methods under vllm/config/
    (`def use_eagle(self): return self.method in (...)`)."""
    helpers: dict[str, set[str]] = {}
    for file in index.file_to_module:
        if not file.startswith(CONFIG_DIR):
            continue
        try:
            tree = ast.parse((repo / file).read_text(), filename=file)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lits = _predicate_literals(node, {}, Flags())
                if lits:
                    helpers.setdefault(node.name, set()).update(lits)
    return helpers


def _flag_literals(tree: ast.AST, helpers: dict[str, set[str]]) -> dict[str, set[str]]:
    """`self.<flag> = <config> == "lit"` assignments, mapped to their literals.

    Only trustworthy when the attribute is stored exactly once. A second store
    anywhere means `if self.f:` no longer implies the config key, so the
    attribute is refused and the edge stays broad."""
    binds: Counter[str] = Counter()
    values: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            binds[node.attr] += 1
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        lits = _test_literals(node.value, helpers, Flags())
        if lits:
            values.setdefault(target.attr, set()).update(lits)
    return {attr: lits for attr, lits in values.items() if binds[attr] == 1}


def _own_scope_nodes(func: ast.AST):
    """The nodes in this function's own scope, not any nested one. Names bound
    inside a nested body are separate and must not leak out."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _local_flag_literals(
    func: ast.AST, helpers: dict[str, set[str]], flags: Flags
) -> tuple[dict[str, set[str]], set[str]]:
    """Returns (readable local flags, every name this scope binds).

    A local like `is_ngram_gpu = ... and use_ngram_gpu()` gates a later call.
    Readable only when the name is bound exactly once here and its value
    reduces to config literals; anything else keeps the edge broad.

    The bound set comes back separately because an unreadable binding still has
    to shadow. A parameter sharing an outer flag's name must make the guard
    unreadable, not inherit its meaning."""
    binds: Counter[str] = Counter()
    values: dict[str, ast.AST] = {}
    disq: set[str] = set()
    for n in _own_scope_nodes(func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            binds[n.id] += 1
        elif isinstance(n, ast.arg):
            disq.add(n.arg)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            disq.update(n.names)
        elif (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            values[n.targets[0].id] = n.value
    bound = set(binds) | disq
    out: dict[str, set[str]] = {}
    for name, node in values.items():
        if binds[name] != 1 or name in disq:
            continue
        lits = _test_literals(node, helpers, flags)
        if lits:
            out[name] = lits
    return out, bound


def _toplevel_imports(
    tree: ast.Module, index: ModuleIndex, package: str = ""
) -> dict[str, str]:
    """Module-scope `from MOD import NAME` to the file it names, for targets
    inside the repo. Relative levels resolve against `package`; missing them
    would hide a relatively-bound name from the rest of the pass."""
    out: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.ImportFrom):
            continue
        base = _from_base(stmt, package)
        if not base:
            continue
        for alias in stmt.names:
            if alias.name == "*":
                continue
            member = index.resolve(f"{base}.{alias.name}") or index.resolve(base)
            if member and member.startswith("vllm/"):
                out[alias.asname or alias.name] = member
    return out


def _dotted_files(dotted: str, index: ModuleIndex) -> set[str]:
    """Files executed by importing `dotted`: the module plus every parent
    package __init__ on the way down (mirrors _resolve_absolute)."""
    parts = dotted.split(".")
    out: set[str] = set()
    for depth in range(1, len(parts) + 1):
        target = index.resolve(".".join(parts[:depth]))
        if target and target.startswith("vllm/"):
            out.add(target)
    return out


class _Collector(ast.NodeVisitor):
    """Records, per tracked symbol, whether every value-position use is
    guarded and which literals gate it; discovers guarded LOCAL imports (the
    runner's `elif method == "ngram": from ... import NgramProposer`) and the
    import channels that must block demotion outright.

    Annotation positions are skipped: the runner annotates
    `self.drafter: (Eagle | Ngram | ...)` under a non-config guard, and
    counting it would wrongly block demotion.

    Base-class positions ARE sites (class creation runs the base): an
    unguarded subclass keeps the edge, a guarded one (parser_manager's
    local-import subclass) is evidence like a guarded construction.

    Pure type positions (isinstance/issubclass classinfo, cast type arg) are
    recorded separately in `type_uses`: they never run the class, so they
    never block, but a GUARDED one is evidence for a member the importer only
    type-checks (the base proposer's `isinstance(model, (Eagle3..., ...))`)
    and any type use marks a member routable by its own registration key."""

    def __init__(
        self,
        toplevel: dict[str, str],
        helpers: dict[str, set[str]],
        flags: Flags,
        index: ModuleIndex,
        extra_symbols: frozenset[str] | set[str] = frozenset(),
        package: str = "",
    ):
        self.toplevel = toplevel
        self.symbols = set(toplevel) | set(extra_symbols)
        self.helpers = helpers
        self.flags = flags
        self.index = index
        self.package = package
        self.guard_stack: list[set[str] | None] = []
        self.func_depth = 0
        # Per symbol, one entry per construction site.
        self.constructs: dict[str, list[tuple[bool, set[str]]]] = {
            s: [] for s in self.symbols
        }
        # Per symbol, uses that only name the type (isinstance, cast). These
        # never block; a guarded one counts as evidence.
        self.type_uses: dict[str, list[tuple[bool, set[str]]]] = {
            s: [] for s in self.symbols
        }
        # Symbols used any other way, or rebound. The class escapes, so we
        # cannot prove every run is guarded and the edge stays broad.
        self.escaped: set[str] = set()
        # Guarded local imports. The second pass treats these names like
        # top-level ones.
        self.local_names: dict[str, tuple[str, set[str]]] = {}
        # Literals from guarded imports. An import done purely for its side
        # effect still counts as evidence.
        self.local_evidence: dict[str, set[str]] = {}
        # Members reached through a channel the guard analysis cannot follow:
        # `import x.y` bindings, unguarded local imports, ambiguous names,
        # parent packages run at module scope. Never demoted from this file.
        self.blocked_members: set[str] = set()

    def _active(self) -> tuple[bool, set[str]]:
        live = [g for g in self.guard_stack if g is not None]
        return bool(live), set().union(*live)

    def _record_type_pos(self, expr: ast.AST) -> None:
        active = self._active()
        for nm in ast.walk(expr):
            if isinstance(nm, ast.Name) and nm.id in self.symbols:
                self.type_uses[nm.id].append(active)

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
            # Type-only subtree: nothing in it runs.
            for child in node.orelse:
                self.visit(child)
            return
        # The test itself runs under the OUTER guards, not this one, so a
        # member used there is a real value position. Skipping it would hide a
        # member whose only unguarded use is `if member_call():`.
        self.visit(test)
        self.guard_stack.append(_test_literals(test, self.helpers, self.flags))
        for child in node.body:
            self.visit(child)
        self.guard_stack.pop()
        self.guard_stack.append(None)  # orelse is not the positive branch
        for child in node.orelse:
            self.visit(child)
        self.guard_stack.pop()

    def visit_FunctionDef(self, node) -> None:
        # Decorators and defaults run at def time, under the guards around the
        # `def`, so a member passed to one counts here.
        for d in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
            if d is not None:
                self.visit(d)
        # The body runs at call time and can be called from anywhere, so the
        # guards around the `def` do not apply. Visit it with a clear stack;
        # only guards inside the body count.
        saved_flags = self.flags
        saved_guards = self.guard_stack
        local, bound = _local_flag_literals(node, self.helpers, self.flags)
        if local or bound:
            self.flags = self.flags.rebound(bound, local)
        self.guard_stack = []
        self.func_depth += 1
        for child in node.body:
            self.visit(child)
        self.func_depth -= 1
        self.guard_stack = saved_guards
        self.flags = saved_flags

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Decorators run when the class is created, under the enclosing
        # guards, so a member passed to one escapes there and must be visited.
        # The body runs then too, so it keeps those guards.
        for d in node.decorator_list:
            self.visit(d)
        # A base class runs at class creation, so it is a real use: unguarded
        # keeps the edge broad, guarded counts as evidence.
        active = self._active()
        for expr in (*node.bases, *(kw.value for kw in node.keywords)):
            for nm in ast.walk(expr):
                if isinstance(nm, ast.Name) and nm.id in self.symbols:
                    self.constructs[nm.id].append(active)
        for child in node.body:
            self.visit(child)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_arg(self, node: ast.arg) -> None:
        pass

    def visit_Import(self, node: ast.Import) -> None:
        # `import x.y` binds a module whose attribute uses we do not follow,
        # so nothing it executes can be demoted.
        for alias in node.names:
            self.blocked_members.update(_dotted_files(alias.name, self.index))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _from_base(node, self.package)
        if not base:
            return
        members: dict[str, str] = {}
        for alias in node.names:
            if alias.name == "*":
                continue
            member = self.index.resolve(f"{base}.{alias.name}") or self.index.resolve(
                base
            )
            if member and member.startswith("vllm/"):
                members[alias.asname or alias.name] = member
        guarded, lits = self._active()
        if guarded:
            for name, member in members.items():
                self.local_evidence.setdefault(member, set()).update(lits)
                self._bind_local(name, member, lits)
            return
        chain = _dotted_files(base, self.index)
        if self.func_depth == 0:
            # At module scope, parent packages run on every import of this
            # file, which no guard covers. Tracked bindings are exempt because
            # their uses are what the analysis reads.
            tracked = {m for name, m in members.items() if self.toplevel.get(name) == m}
            self.blocked_members.update(chain - tracked)
        else:
            # Unguarded lazy import: an unanalyzable runtime channel.
            self.blocked_members.update(chain | set(members.values()))

    def _bind_local(self, name: str, member: str, lits: set[str]) -> None:
        if name in self.toplevel:
            if self.toplevel[name] != member:
                self.blocked_members.update({member, self.toplevel[name]})
            return
        prev = self.local_names.get(name)
        if prev is None:
            self.local_names[name] = (member, set(lits))
        elif prev[0] != member:
            self.blocked_members.update({member, prev[0]})
        else:
            prev[1].update(lits)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else None
        # isinstance/issubclass/cast reference the class as a TYPE, not a
        # construction: they never run the plugin's code, so they must not
        # block demotion. Record the classinfo as a type position (Ext A/C)
        # and visit only the value operand.
        if name in ("isinstance", "issubclass") and node.args:
            if len(node.args) >= 2:
                self._record_type_pos(node.args[1])
            self.visit(node.args[0])
            return
        if name == "cast" and len(node.args) >= 2:
            self._record_type_pos(node.args[0])
            for a in node.args[1:]:
                self.visit(a)
            for kw in node.keywords:
                self.visit(kw.value)
            return
        if name in self.symbols:
            # S(...) is a construction; the guard on it decides demotion. Visit
            # only the ARGS -- visiting the callee Name would mark it escaped.
            self.constructs[name].append(self._active())
            for a in (*node.args, *(k.value for k in node.keywords)):
                self.visit(a)
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # S.attr / S.classmethod() runs S's code at this site, so the guard
        # here decides demotion just like a construction -- not an escape.
        base = node.value
        if isinstance(base, ast.Name) and base.id in self.symbols:
            self.constructs[base.id].append(self._active())
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Load reached only for non-construction, non-type positions (see
        # visit_Call / the annotation and base-class skips): the class
        # escapes. Store/Del rebinds the name (`E = None` fallback), making
        # later sites unattributable -> also an escape.
        if node.id in self.symbols and isinstance(
            node.ctx, (ast.Load, ast.Store, ast.Del)
        ):
            self.escaped.add(node.id)


def leaf_literal_fanout(graph: ImportGraph) -> Counter[str]:
    """literal -> count of leaf-consumer files whose string literals hold it."""
    fanout: Counter[str] = Counter()
    for leaf_file, literals in graph.string_literals.items():
        if _leaf_consumer(leaf_file):
            fanout.update(literals)
    return fanout


def _route(
    graph: ImportGraph,
    parse: DispatchParse,
    member: str,
    literals: set[str],
    fanout: Counter[str],
) -> None:
    # Drop generic words from BOTH routing channels: the exact-literal
    # _leaf_edges and the literal-derived tokens of _path_leaf_edges. The
    # member's stem and mirror-dir tokens are specific and always kept.
    kept = {lit for lit in literals if fanout[lit] <= CONFIG_KEY_MAX_TEST_FILES}
    # fanout only counts literals the collector recorded, so a shorter one scores
    # zero and clears the bar unweighed. Exact matching survives that (leaves
    # record the same lengths, so it matches nothing), but as a PATH substring
    # "1" hits a fifth of the tree, and that fake coverage outranks the
    # empty-closure fail-open the member should have gotten.
    tokens = {t.lower() for t in kept if len(t) >= LITERAL_MIN_LEN} | {
        Path(member).stem.lower()
    }
    if member.count("/") >= 2:
        tokens.add(member.rsplit("/", 2)[-2].lower())  # mirror-dir floor
    parse.edges_added += _leaf_edges(graph, kept, member)
    parse.edges_added += _path_leaf_edges(graph, tokens, member)


def _demote(
    graph: ImportGraph,
    parse: DispatchParse,
    importer: str,
    member: str,
    literals: set[str],
    fanout: Counter[str],
) -> None:
    if member == importer or (importer, member) in graph.demoted_edges:
        return
    graph.demoted_edges.add((importer, member))
    parse.demotions[(importer, member)] = literals
    _route(graph, parse, member, literals, fanout)


def add_demotion_edges(
    repo: Path,
    index: ModuleIndex,
    graph: ImportGraph,
    preclaimed: frozenset[str] = frozenset(),
) -> DispatchParse:
    parse = DispatchParse()
    helpers = _config_helper_literals(repo, index)
    helper_calls = tuple(f"{h}(" for h in helpers)
    fanout = leaf_literal_fanout(graph)
    # A type-only importer of a preclaimed member (Ext C) may contain neither
    # "==" nor a helper call, so the cheap text prefilter must also admit any
    # file that imports one.
    preclaimed_files = frozenset(p for p in preclaimed if not p.endswith("/"))
    for file in index.file_to_module:
        if not file.startswith("vllm/"):
            continue
        try:
            text = (repo / file).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if (
            "==" not in text
            and not any(m in text for m in (" in (", " in [", " in {"))
            and not any(h in text for h in helper_calls)
            and not (graph.imports.get(file, set()) & preclaimed_files)
        ):
            continue
        try:
            tree = ast.parse(text, filename=file)
        except SyntaxError:
            continue
        package = _package_of(index.file_to_module[file], file)
        for member, lits in _member_demotions(
            tree, index, helpers, preclaimed, package
        ).items():
            _demote(graph, parse, file, member, lits, fanout)
    _revert_starved(graph, parse)
    _claim_severed(graph, parse)
    return parse


def _revert_starved(graph: ImportGraph, parse: DispatchParse) -> None:
    """A member left with no test coverage after routing must not stay demoted,
    so its importers get their edges back and normal closure covers it. Leaf
    edges already added stay, since they only add. Work out the starved set
    first, then revert in one batch: a revert only adds coverage, so order
    cannot under-select. Runs before _claim_severed, so the claims pass sees
    the final set.
    The demotion-time twin of the starved check in tests/helpers.py; it fires
    zero times at HEAD, a dormant safety net for the leaf-fanout bar."""
    starved = [
        member
        for member in sorted({m for _, m in parse.demotions})
        if not any(is_test_basename(f) for f in graph.reverse_closure({member}))
    ]
    for member in starved:
        for pair in [p for p in parse.demotions if p[1] == member]:
            graph.demoted_edges.discard(pair)
            del parse.demotions[pair]


def _member_demotions(
    tree: ast.Module,
    index: ModuleIndex,
    helpers: dict[str, set[str]],
    preclaimed: frozenset[str] | set[str] = frozenset(),
    package: str = "",
) -> dict[str, set[str]]:
    """One importer module -> {member file: gating literals} to demote."""
    imports = _toplevel_imports(tree, index, package)
    flags = Flags(attrs=_flag_literals(tree, helpers))
    col = _Collector(imports, helpers, flags, index, package=package)
    col.visit(tree)
    if col.local_names:
        # Second pass tracks the guarded-local names like top-level symbols:
        # function bodies execute in call order, not source order, so a
        # single pass can miss a use preceding the import.
        col = _Collector(
            imports,
            helpers,
            flags,
            index,
            extra_symbols=set(col.local_names),
            package=package,
        )
        col.visit(tree)
    bound: dict[str, set[str]] = {}
    for name, member in imports.items():
        bound.setdefault(member, set()).add(name)
    for name, (member, _) in col.local_names.items():
        bound.setdefault(member, set()).add(name)
    # The decision is per MEMBER: every symbol bound to the file must pass,
    # and their guard literals union (per-symbol demotion silently lost the
    # second symbol's literals on the already-demoted pair).
    out: dict[str, set[str]] = {}
    for member in sorted(bound):
        names = bound[member]
        if member in col.blocked_members or names & col.escaped:
            continue
        sites = [s for n in names for s in col.constructs.get(n, [])]
        if not all(guarded for guarded, _ in sites):
            continue
        evidence = set(col.local_evidence.get(member, set()))
        tuses = [t for n in names for t in col.type_uses.get(n, [])]
        if not sites:
            # Ext A: with no value-position site, a guarded type position is
            # the only gate (the base proposer only isinstance-checks the
            # model), so fold its literals into evidence. When a construct
            # site exists it stays authoritative -- type literals are ignored
            # so existing literal sets are unperturbed.
            evidence |= set().union(*(lit for g, lit in tuses if g))
        if sites or evidence:
            lits = set().union(evidence, *(lit for _, lit in sites))
            if lits:
                out[member] = lits
        elif tuses and member in preclaimed:
            # Ext C: every use is a pure type position and the member already
            # routes by its own typed registration key (enum/registry/quant/
            # parser). A type reference never creates an instance; the only
            # runtime that does is the keyed path, which carries its own
            # coverage. Empty literals -> routing rides stem + mirror-dir.
            out[member] = set()
    return out


def _claim_severed(graph: ImportGraph, parse: DispatchParse) -> None:
    """Claim only members whose every unconditional vllm importer edge was
    demoted (directly or via an importer that is itself severed; the fixpoint
    handles member->member chains like dflash -> eagle).

    Routing leaf edges (additive) apply to every demoted member, but a CLAIM
    is subtractive -- it reroutes coverage and drops lazy edges into the
    member -- so a member still imported plainly elsewhere keeps its true
    broad closure and must not be claimed."""
    severed = {member for _, member in parse.demotions}
    changed = True
    while changed:
        changed = False
        for member in sorted(severed):
            for src in graph.reverse.get(member, ()):
                if not src.startswith("vllm/"):
                    continue
                if (src, member) in graph.demoted_edges or src in severed:
                    continue
                severed.discard(member)
                changed = True
                break
    for member in sorted(severed):
        parse.claims.add(member)
        if member.endswith("/__init__.py"):
            parse.claims.add(member[: -len("__init__.py")])
