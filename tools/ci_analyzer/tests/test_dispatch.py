# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-guarded plugin demotion: synthetic units, detector fixtures, and
live-checkout selection cases.

The rule (graph/dispatch.py): a plugin imported at module top but only used
behind a config-key guard is demoted out of selection closures and re-routed
by key. Under-selection is the cardinal sin here, so the fixtures pin the
places a naive detector would wrongly demote (aliasing, unconditional use).
"""

import ast

import pytest
from ci_analyzer.curated import PLATFORM_GUARD_LITERALS
from ci_analyzer.graph.dispatch import (
    _CONFIG_KEY_MAX_TEST_FILES,
    DispatchParse,
    Flags,
    _claim_severed,
    _Collector,
    _config_helper_literals,
    _demote,
    _flag_literals,
    _local_flag_literals,
    _member_demotions,
    _predicate_literals,
    _revert_starved,
    _route,
    _test_literals,
    leaf_literal_fanout,
)
from ci_analyzer.graph.imports import ImportGraph
from ci_analyzer.repo import ModuleIndex

# ---- synthetic ImportGraph: the demotion primitive ------------------------


def _wired() -> ImportGraph:
    """test_a -> runner -> eagle (the amplifier path); a sibling base edge
    step3p5 -> eagle carrying its own test."""
    g = ImportGraph()
    g.add_edge("tests/test_a.py", "runner.py")
    g.add_edge("runner.py", "eagle.py")
    g.add_edge("tests/test_spec.py", "step3p5.py")
    g.add_edge("step3p5.py", "eagle.py")
    return g


def test_demoted_edge_cuts_amplifier_keeps_sibling():
    g = _wired()
    assert "tests/test_a.py" in g.reverse_closure({"eagle.py"})
    g.demoted_edges.add(("runner.py", "eagle.py"))
    closure = g.reverse_closure({"eagle.py"})
    assert "tests/test_a.py" not in closure  # amplifier path cut
    assert "tests/test_spec.py" in closure  # base-class path preserved


def test_demoted_edge_skipped_in_both_gated_modes():
    g = _wired()
    g.demoted_edges.add(("runner.py", "eagle.py"))
    for include_gated in (True, False):
        assert "tests/test_a.py" not in g.reverse_closure(
            {"eagle.py"}, include_gated=include_gated
        )


def test_demotion_leaves_raw_import_edge_intact():
    """plain_reverse / import audits must still see the true edge."""
    g = _wired()
    g.demoted_edges.add(("runner.py", "eagle.py"))
    assert "eagle.py" in g.imports["runner.py"]
    assert "runner.py" in g.reverse["eagle.py"]


# ---- leaf-fanout bar: generic literals refused for demotion routing --------


def _leafy(literal_counts: dict[str, int]) -> ImportGraph:
    """A graph whose string_literals give each literal the requested leaf
    fanout (that many distinct test files carry it)."""
    g = ImportGraph()
    n = 0
    for lit, count in literal_counts.items():
        for _ in range(count):
            g.string_literals[f"tests/l/test_{n}.py"] = {lit}
            n += 1
    return g


def test_literal_below_the_collector_floor_refused_for_path_routing():
    """A literal too short to be collected never reaches the fanout counter, so
    it scores 0 and clears the bar without ever being weighed. Exact matching
    survives that (leaves record the same lengths, so it matches nothing), but
    as a path substring "1" hit a fifth of the tree."""
    g = _leafy({"specific": 1})
    g.string_literals["tests/v1/test_p.py"] = {"unrelated"}
    member = "vllm/foo/bar.py"
    parse = DispatchParse()
    _route(g, parse, member, {"1", "specific"}, leaf_literal_fanout(g))
    srcs = g.reverse.get(member, set())
    assert "tests/v1/test_p.py" not in srcs, "unweighed literal still routes by path"
    assert any(s.startswith("tests/l/") for s in srcs), "kept literal stopped routing"


def test_generic_literal_refused_for_leaf_routing():
    over = _CONFIG_KEY_MAX_TEST_FILES + 1
    g = _leafy({"generic": over, "specific": 1})
    # a leaf whose PATH (not literals) carries the refused word
    g.string_literals["tests/generic_area/test_p.py"] = {"unrelated"}
    # a leaf whose PATH carries the member stem (a kept token)
    g.string_literals["tests/x/test_bar_thing.py"] = {"unrelated"}
    member = "vllm/foo/bar.py"
    parse = DispatchParse()
    _route(g, parse, member, {"generic", "specific"}, leaf_literal_fanout(g))
    srcs = g.reverse.get(member, set())
    # kept literal attaches; every generic-literal leaf is refused
    assert any(
        s.startswith("tests/l/") and g.string_literals[s] == {"specific"} for s in srcs
    )
    assert not any(g.string_literals[s] == {"generic"} for s in srcs)
    # the refused word is stripped from the path channel too
    assert "tests/generic_area/test_p.py" not in srcs
    # but the member stem survives as a path token
    assert "tests/x/test_bar_thing.py" in srcs


def test_starved_member_demotion_reverted():
    g = _leafy({"generic": _CONFIG_KEY_MAX_TEST_FILES + 1})
    g.add_edge("tests/test_x.py", "vllm/imp.py")
    g.add_edge("vllm/imp.py", "vllm/m.py")
    parse = DispatchParse()
    _demote(g, parse, "vllm/imp.py", "vllm/m.py", {"generic"}, leaf_literal_fanout(g))
    assert ("vllm/imp.py", "vllm/m.py") in g.demoted_edges
    assert "tests/test_x.py" not in g.reverse_closure({"vllm/m.py"})  # starved
    _revert_starved(g, parse)
    assert ("vllm/imp.py", "vllm/m.py") not in parse.demotions
    assert "tests/test_x.py" in g.reverse_closure({"vllm/m.py"})  # coverage regained


def test_revert_runs_before_claims_fixpoint():
    """A member reverted for starvation must not be claimed; a healthy demoted
    member in the same parse still is."""
    g = _leafy({"generic": _CONFIG_KEY_MAX_TEST_FILES + 1, "keep": 1})
    g.add_edge("tests/test_a.py", "vllm/impa.py")
    g.add_edge("vllm/impa.py", "vllm/starved.py")  # generic-only -> starved
    g.add_edge("vllm/impb.py", "vllm/healthy.py")  # routed via "keep" leaf
    fanout = leaf_literal_fanout(g)
    parse = DispatchParse()
    _demote(g, parse, "vllm/impa.py", "vllm/starved.py", {"generic"}, fanout)
    _demote(g, parse, "vllm/impb.py", "vllm/healthy.py", {"keep"}, fanout)
    _revert_starved(g, parse)
    _claim_severed(g, parse)
    assert "vllm/starved.py" not in parse.claims
    assert "vllm/healthy.py" in parse.claims


# ---- detector: guard literal extraction -----------------------------------


def _expr(src: str) -> ast.AST:
    return ast.parse(src, mode="eval").body


@pytest.mark.parametrize(
    "src, expected",
    [
        ('m == "eagle3"', {"eagle3"}),
        ('"eagle3" == m', {"eagle3"}),
        ('m in ("eagle", "mtp")', {"eagle", "mtp"}),
        ('m == "a" and other', {"a"}),  # AND: the config conjunct suffices
        ('m == "a" and m2 == "b"', {"a", "b"}),
        ('m == "a" or m == "b"', {"a", "b"}),  # OR over config keys = membership
    ],
)
def test_test_literals_positive(src, expected):
    assert _test_literals(_expr(src), {}, Flags()) == expected


@pytest.mark.parametrize(
    "src",
    [
        "flag",  # bare truthiness, no literal
        'not m == "a"',  # negation yields no positive key
        'm == "a" or debug',  # OR with a non-config operand is unsafe
        "m == other",  # compared to a non-constant
        'device == "cuda"',  # platform dispatch, not a config key
        'backend in ("cuda", "triton")',  # one platform member poisons all
        '__name__ == "__main__"',  # script dispatch
        '"__main__" == __name__',
    ],
)
def test_test_literals_negative(src):
    assert _test_literals(_expr(src), {}, Flags()) is None


def test_platform_conjunct_still_extracts_config_key():
    assert _test_literals(_expr('d == "cuda" and m == "eagle"'), {}, Flags()) == {
        "eagle"
    }


def test_helper_and_flag_resolution():
    helpers = {"use_eagle": {"eagle", "eagle3", "mtp"}}
    flags = Flags(attrs={"is_pooling_model": {"pooling"}})
    assert _test_literals(_expr("cfg.use_eagle()"), helpers, flags) == {
        "eagle",
        "eagle3",
        "mtp",
    }
    assert _test_literals(_expr("self.is_pooling_model"), helpers, flags) == {"pooling"}


def test_predicate_literals_needs_every_return_config():
    fn = ast.parse('def f(self):\n return self.method in ("a", "b")').body[0]
    assert _predicate_literals(fn, {}, Flags()) == {"a", "b"}
    # a bare `return True` path makes the helper unresolvable (keep broad)
    leaky = ast.parse("def f(self):\n if x: return True\n return self.m == 'a'").body[0]
    assert _predicate_literals(leaky, {}, Flags()) is None


def _fn(src: str):
    return ast.parse(src).body[0]


def test_local_flag_resolves_when_bound_once():
    fn = _fn('def f(cfg):\n x = cfg.method == "a"\n return x')
    assert _local_flag_literals(fn, {}, Flags())[0] == {"x": {"a"}}


def test_local_flag_reassignment_refused():
    fn = _fn('def f(cfg):\n x = cfg.method == "a"\n x = g()\n return x')
    assert _local_flag_literals(fn, {}, Flags())[0] == {}


def test_local_flag_parameter_refused():
    fn = _fn('def f(cfg, x):\n x = cfg.method == "a"\n return x')
    assert _local_flag_literals(fn, {}, Flags())[0] == {}


def test_local_binding_shadows_outer_flag():
    """A name this scope binds refers to the local, so the outer flag must stop
    answering for it -- resolving from the local value when that reduces, and
    unresolvable (keep broad) when it does not."""
    fn = _fn('def f(cfg):\n x = cfg.method == "a"\n return x')
    outer = Flags(names={"x": {"b"}})
    local, bound = _local_flag_literals(fn, {}, outer)
    assert local == {"x": {"a"}} and "x" in bound
    assert outer.rebound(bound, local).names["x"] == {"a"}

    opaque = _fn("def f(cfg, x):\n return x")
    local, bound = _local_flag_literals(opaque, {}, outer)
    assert local == {} and "x" in bound
    assert "x" not in outer.rebound(bound, local).names


def test_self_attr_flag_does_not_answer_a_bare_name():
    """`self.use_data_parallel` and a bare `use_data_parallel` are different
    bindings; one map answering both made a colliding parameter or local
    resolve as the config guard and demote on a literal it never implied."""
    flags = Flags(attrs={"use_data_parallel": {"data"}})
    assert _test_literals(_expr("self.use_data_parallel"), {}, flags) == {"data"}
    assert _test_literals(_expr("use_data_parallel"), {}, flags) is None


def test_local_flag_nested_scope_does_not_leak():
    fn = _fn('def f(cfg):\n def g():\n  y = cfg.method == "a"\n return 0')
    assert _local_flag_literals(fn, {}, Flags())[0] == {}


def test_self_flag_single_assignment_resolves():
    tree = ast.parse(
        "class R:\n    def __init__(self, cfg):\n"
        '        self.is_eagle = cfg.method == "eagle"\n'
    )
    assert _flag_literals(tree, {}) == {"is_eagle": {"eagle"}}


def test_self_flag_multiple_assignments_refused():
    """Config-gated in __init__ but unconditionally set True in reset(): the
    guard `if self.is_eagle:` no longer implies the key, so refuse it."""
    tree = ast.parse(
        "class R:\n    def __init__(self, cfg):\n"
        '        self.is_eagle = cfg.method == "eagle"\n'
        "    def reset(self):\n        self.is_eagle = True\n"
    )
    assert _flag_literals(tree, {}) == {}


# ---- detector: use classification (the under-selection guards) -------------


def _analyze(src: str, symbols, helpers=None, flags=None):
    toplevel = {s: f"vllm/{s.lower()}.py" for s in symbols}
    c = _Collector(toplevel, helpers or {}, flags or Flags(), ModuleIndex())
    c.visit(ast.parse(src))
    return c


def test_guarded_construction_is_demotable():
    src = 'if cfg.method == "eagle":\n    d = Eagle(a, b)\n'
    c = _analyze(src, {"Eagle"})
    assert "Eagle" not in c.escaped
    assert c.constructs["Eagle"] == [(True, {"eagle"})]


def test_annotation_and_isinstance_do_not_block():
    src = (
        "d: Eagle | Ngram = None\n"
        "if isinstance(d, (Eagle, Ngram)):\n"
        "    pass\n"
        'if cfg.method == "eagle":\n'
        "    d = Eagle(x)\n"
    )
    c = _analyze(src, {"Eagle"})
    assert "Eagle" not in c.escaped  # type positions are not uses
    assert c.constructs["Eagle"] == [(True, {"eagle"})]


def test_unconditional_construction_keeps_broad():
    c = _analyze("d = Eagle(x)\n", {"Eagle"})
    assert c.constructs["Eagle"] == [(False, set())]


def test_aliasing_escape_keeps_broad():
    """`cls = Eagle` under a guard then an unguarded `cls()` would run Eagle
    unguarded; the bare reference must mark it escaped."""
    src = 'if cfg.method == "eagle":\n    cls = Eagle\ncls()\n'
    c = _analyze(src, {"Eagle"})
    assert "Eagle" in c.escaped


def test_member_access_is_guarded_use_not_escape():
    src = 'if cfg.method == "eagle":\n    Eagle.build(x)\n'
    c = _analyze(src, {"Eagle"})
    assert "Eagle" not in c.escaped
    assert c.constructs["Eagle"] == [(True, {"eagle"})]


def test_isinstance_classinfo_is_type_use_not_site():
    src = "if isinstance(m, (Eagle,)):\n    pass\n"
    c = _analyze(src, {"Eagle"})
    assert "Eagle" not in c.escaped
    assert c.constructs["Eagle"] == []
    assert c.type_uses["Eagle"] == [(False, set())]  # test scanned under outer guard


def test_guarded_isinstance_classinfo_records_the_guard():
    src = 'if cfg.method == "eagle":\n    assert isinstance(m, (Eagle,))\n'
    c = _analyze(src, {"Eagle"})
    assert c.constructs["Eagle"] == []
    assert c.type_uses["Eagle"] == [(True, {"eagle"})]


def test_base_class_is_a_site():
    src = "class Sub(Eagle):\n    pass\n"
    c = _analyze(src, {"Eagle"})
    assert "Eagle" not in c.escaped
    assert c.constructs["Eagle"] == [(False, set())]  # unguarded base = a site


def test_decorator_argument_escapes():
    """A member passed to a decorator runs unconditionally at import, so the
    registration edge must never demote (the minimax @register_processor shape)."""
    src = "@registry.register(Eagle, info=Eagle)\nclass M:\n    pass\n"
    c = _analyze(src, {"Eagle"})
    assert "Eagle" in c.escaped


def test_if_test_construction_is_a_site():
    """A member constructed in the if-test runs before either branch, under the
    outer guards -- a value site, not invisible."""
    c = _analyze("if Eagle():\n    pass\n", {"Eagle"})
    assert c.constructs["Eagle"] == [(False, set())]


def test_if_test_bare_name_escapes():
    c = _analyze("if Eagle:\n    pass\n", {"Eagle"})
    assert "Eagle" in c.escaped


def test_nested_def_body_not_guarded_by_enclosing_if():
    """A construction inside a def defined under a config guard runs at CALL
    time, not def time: the enclosing guard must not make it look guarded."""
    src = 'if cfg.method == "eagle":\n    def make():\n        return Eagle(x)\n'
    c = _analyze(src, {"Eagle"})
    assert c.constructs["Eagle"] == [(False, set())]


# ---- per-member decision (dict = {member file: gating literals}) -----------


def _decide(src: str, modules: dict[str, str], preclaimed=frozenset()):
    index = ModuleIndex()
    for mod, file in modules.items():
        index.add(mod, file)
    return _member_demotions(ast.parse(src), index, {}, preclaimed)


UNI = {"vllm.uni": "vllm/uni.py"}


def test_two_symbols_one_member_union_literals():
    src = (
        "from vllm.uni import A, B\n"
        'if cfg.kind == "alpha":\n    x = A(1)\n'
        'if cfg.kind == "beta":\n    y = B(2)\n'
    )
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha", "beta"}}


def test_second_symbol_escape_blocks_member():
    """Per-symbol demotion would demote on A and silently suppress B's
    unguarded escape through the same (importer, member) pair."""
    src = 'from vllm.uni import A, B\nif cfg.kind == "alpha":\n    x = A(1)\ny = B\n'
    assert _decide(src, UNI) == {}


def test_guarded_local_import_demotes_on_import_site():
    src = 'def f(cfg):\n    if cfg.kind == "alpha":\n        from vllm.uni import U\n'
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha"}}


def test_guarded_local_import_blocked_by_module_level_plain_import():
    src = (
        "import vllm.uni\n"
        "def f(cfg):\n"
        '    if cfg.kind == "alpha":\n'
        "        from vllm.uni import U\n"
        "        return U()\n"
    )
    assert _decide(src, UNI) == {}


def test_guarded_local_import_blocked_by_toplevel_escape():
    """The abstract.py shape: a module-level import whose symbol escapes must
    not be suppressed by a guarded local re-import of the same member."""
    src = (
        "from vllm.uni import U\n"
        "DEFAULT = U\n"
        "def f(cfg):\n"
        '    if cfg.kind == "alpha":\n'
        "        from vllm.uni import U\n"
        "        return U()\n"
    )
    assert _decide(src, UNI) == {}


def test_guarded_local_alias_escape_keeps_broad():
    src = (
        "def f(cfg):\n"
        '    if cfg.kind == "alpha":\n'
        "        from vllm.uni import U\n"
        "        cls = U\n"
        "    return cls()\n"
    )
    assert _decide(src, UNI) == {}


def test_unguarded_lazy_import_blocks_member():
    src = (
        "def a(cfg):\n"
        '    if cfg.kind == "alpha":\n'
        "        from vllm.uni import U\n"
        "        U()\n"
        "def b():\n"
        "    from vllm.uni import U\n"
        "    U()\n"
    )
    assert _decide(src, UNI) == {}


def test_guarded_isinstance_is_evidence_when_never_constructed():
    """Ext A: the base proposer only isinstance-checks the model under a
    method guard; that guarded type position is the demotion evidence."""
    src = (
        "from vllm.uni import U\n"
        'if cfg.method == "alpha":\n'
        "    assert isinstance(m, (U,))\n"
    )
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha"}}


def test_guarded_cast_is_evidence_when_never_constructed():
    src = (
        "from vllm.uni import U\n"
        "from typing import cast\n"
        'if cfg.method == "alpha":\n'
        "    x = cast(U, m)\n"
    )
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha"}}


def test_unguarded_type_only_use_stays_registration_only():
    """No guard and not preclaimed: nothing gates it, keep broad."""
    src = "from vllm.uni import U\nassert isinstance(m, (U,))\n"
    assert _decide(src, UNI) == {}


def test_type_use_literals_ignored_when_a_construct_site_exists():
    """Fold-only-when-no-sites: a construction under one key stays
    authoritative; a family-wide guarded isinstance must not widen it."""
    src = (
        "from vllm.uni import U\n"
        'if cfg.method == "alpha":\n'
        "    U(x)\n"
        'if cfg.method == "beta":\n'
        "    assert isinstance(m, (U,))\n"
    )
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha"}}


def test_annotation_never_evidence_even_under_guard():
    src = 'from vllm.uni import U\nif cfg.method == "alpha":\n    x: U = None\n'
    assert _decide(src, UNI) == {}


def test_unguarded_base_class_blocks_demotion():
    src = (
        "from vllm.uni import U\n"
        "class Sub(U):\n    pass\n"
        'if cfg.method == "alpha":\n    U(x)\n'
    )
    assert _decide(src, UNI) == {}


def test_guarded_base_class_is_evidence():
    """The parser_manager shape: a subclass inside the guarded branch is as
    safe as a guarded construction."""
    src = (
        "def f(cfg):\n"
        '    if cfg.kind == "alpha":\n'
        "        from vllm.uni import U\n"
        "        class Sub(U):\n"
        "            pass\n"
    )
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha"}}


def test_local_flag_gates_construction_end_to_end():
    """Ext B: a function-local `is_x = <cfg predicate>` gates a later use."""
    src = (
        "from vllm.uni import U\n"
        "def f(cfg):\n"
        '    ok = cfg.method == "alpha"\n'
        "    if ok:\n"
        "        U()\n"
    )
    assert _decide(src, UNI) == {"vllm/uni.py": {"alpha"}}


def test_membership_guard_demotes():
    """A member gated purely by `in (...)` membership demotes on the union of
    its keys (the file the prefilter now admits without a bare `==`)."""
    src = 'from vllm.uni import U\nif cfg.method in ("a", "b"):\n    U()\n'
    assert _decide(src, UNI) == {"vllm/uni.py": {"a", "b"}}


def test_type_only_use_of_preclaimed_member_demotes_empty():
    """Ext C: every use is a type position and the member routes by its own
    typed key elsewhere -> demote with no literals (routing rides the stem and
    mirror-dir tokens plus the member's registration key)."""
    src = "from vllm.uni import U\nassert isinstance(m, (U,))\n"
    assert _decide(src, UNI, preclaimed={"vllm/uni.py"}) == {"vllm/uni.py": set()}


def test_type_only_preclaimed_with_escape_stays_broad():
    """A bare reference is a value use, not a type position, so even a
    preclaimed member keeps its broad edge (the triton_attn list shape)."""
    src = "from vllm.uni import U\nassert isinstance(m, (U,))\nx = [U]\n"
    assert _decide(src, UNI, preclaimed={"vllm/uni.py"}) == {}


def test_construct_site_beats_ext_c_empty_literals():
    """A guarded construction gives real literals; Ext C's empty-literal path
    is only for the no-value-site case."""
    src = (
        "from vllm.uni import U\n"
        'if cfg.k == "a":\n    U()\n'
        "assert isinstance(m, (U,))\n"
    )
    assert _decide(src, UNI, preclaimed={"vllm/uni.py"}) == {"vllm/uni.py": {"a"}}


def test_parent_package_never_demoted_bound_member_still_is():
    """A module-scope from-import executes parent __init__ files: they must
    never be demotable from this file, while the bound member itself stays
    demotable (its uses ARE the analysis)."""
    modules = {
        "vllm.uni": "vllm/uni/__init__.py",
        "vllm.uni.mod": "vllm/uni/mod.py",
    }
    src = (
        "from vllm.uni.mod import U\n"
        'if cfg.kind == "alpha":\n    x = U(1)\n'
        "def f(cfg):\n"
        '    if cfg.kind == "beta":\n'
        "        from vllm.uni import V\n"
    )
    out = _decide(src, modules)
    assert "vllm/uni/mod.py" in out
    assert "vllm/uni/__init__.py" not in out


# ---- live-checkout end-to-end cases ----------------------------------------

SPEC_PROPOSERS = (
    "vllm/v1/spec_decode/eagle.py",
    "vllm/v1/spec_decode/dflash.py",
    "vllm/v1/spec_decode/gemma4.py",
    "vllm/v1/spec_decode/draft_model.py",
    "vllm/v1/spec_decode/medusa.py",
    "vllm/v1/spec_decode/suffix_decoding.py",
)
RUNNER = "vllm/v1/worker/gpu_model_runner.py"


@pytest.fixture(scope="module")
def fg(full):
    return full


def _tests(fg, path):
    return {f for f in fg.graph.reverse_closure({path}) if f.startswith("tests/")}


def test_spec_proposers_demoted_from_runner(fg):
    for proposer in SPEC_PROPOSERS:
        assert (RUNNER, proposer) in fg.graph.demoted_edges, proposer
        assert len(_tests(fg, proposer)) < 300, (proposer, len(_tests(fg, proposer)))


def test_eagle_routes_to_spec_tests(fg):
    """Its closure is the spec family, not the ~1400 engine tests, and the
    real eagle test is still selected."""
    tests = _tests(fg, "vllm/v1/spec_decode/eagle.py")
    assert any("spec_decode" in t for t in tests)
    assert len(tests) < 300


def test_pooling_runner_demoted(fg):
    member = "vllm/v1/worker/gpu/pool/pooling_runner.py"
    assert any(m == member for _, m in fg.graph.demoted_edges)
    assert len(_tests(fg, member)) < 400


def test_hubs_stay_broad(fg):
    # A genuine hub may be demoted from ONE guarded importer (per-edge, e.g.
    # a router that touches _custom_ops only under a key), but it stays broad
    # via every other unconditional importer -- demotion never shrinks it.
    for hub in ("vllm/config/__init__.py", "vllm/_custom_ops.py"):
        assert len(_tests(fg, hub)) > 1000, hub


def test_unconditional_feature_not_demoted(fg):
    # elastic EP is CONSTRUCTED unconditionally (only its methods are
    # flag-gated) -> symbol-level, out of scope: never demoted, stays broad.
    ee = "vllm/distributed/elastic_ep/elastic_execute.py"
    assert not any(m == ee for _, m in fg.graph.demoted_edges)
    assert len(_tests(fg, ee)) > 1000


def test_shared_base_not_claimed(fg):
    """Claiming llm_base_proposer would drop the cpu_model_runner monkeypatch
    lazy edge -> CPU under-selection. It must stay unclaimed and broad."""
    base = "vllm/v1/spec_decode/llm_base_proposer.py"
    assert base not in fg.all_claims()
    assert not any(m == base for _, m in fg.graph.demoted_edges)


def test_every_demoted_member_routes_to_tests(fg):
    """The guardrail invariant: no demotion starves its member to fail-open."""
    from ci_analyzer.validate.demoted_plugins import starved_members

    assert starved_members(fg) == []


def test_uniproc_unconditional_edge_not_suppressed(fg):
    """abstract.py imports uniproc_executor at module level and the symbol
    escapes; a guarded local re-import must not demote the pair (the
    pair-keyed skip would suppress the unconditional edge too)."""
    pair = ("vllm/v1/executor/abstract.py", "vllm/v1/executor/uniproc_executor.py")
    assert pair not in fg.graph.demoted_edges


def test_platform_guard_literals_cover_the_filesystem_platforms(repo):
    """PLATFORM_GUARD_LITERALS is dispatch's REFUSAL set: a guard comparing
    against one of these words is platform dispatch, so the member keeps its
    broad edge. The set SHRINKING is the under-selection risk, because a
    platform-dispatched import would start demoting and route by platform word.

    Asserting the demotion output holds none of them (what this replaced) can
    never fail: dispatch.py filters on this same set, so it restates the filter
    and passes for any contents, including empty. Asking the job YAML is
    circular too, since family_of_device() reads FAMILY_DEVICE_PREFIXES whose
    keys ARE most of the set.

    The floor comes from the filesystem instead: a word naming BOTH a
    vllm/platforms/<word>.py module and a requirements/<word>.txt install
    target is a platform word by construction. The intersection is itself the
    exclusion rule, so no hand-list is needed -- common/dev/docs/lint have no
    platform module, __init__/interface/zen_cpu have no requirements file.

    What it does NOT catch: it floors five of the twelve words. Dropping one of
    the other seven (aiter, amd, ascend, hip, hpu, npu, pallas) passes here, as
    does a new platform word that arrives without both files. It stops the set
    collapsing, not drifting."""
    platforms = {p.stem for p in (repo / "vllm/platforms").glob("*.py")}
    requirements = {p.stem for p in (repo / "requirements").glob("*.txt")}
    derived = platforms & requirements
    # Without this, a moved directory empties the intersection and `set() <=
    # anything` passes -- the same vacuity this test replaced.
    assert len(derived) >= 4, f"derivation broke, not the table: {sorted(derived)}"
    assert derived <= PLATFORM_GUARD_LITERALS, (
        "platform words missing from the curated taxonomy: "
        f"{sorted(derived - PLATFORM_GUARD_LITERALS)}; dispatch would read a "
        "guard on them as a config key and route platform-dispatched imports "
        "by platform word (under-selection)"
    )


def test_no_main_literal_demotions(fg):
    """Not a twin of the platform check: dispatch only refuses a "__main__"
    comparison whose other side is the bare NAME `__name__`, so
    `mod.__name__ == "__main__"`, `name == "__main__"` and any `in (...)`
    membership holding it all reach the literal set. Script dispatch is not
    config dispatch: routing a member by "__main__" would attach it to every
    leaf whose source carries the word."""
    assert fg.dispatch.demotions, "no demotions parsed; the assert below is vacuous"
    assert "__main__" not in set().union(*fg.dispatch.demotions.values())


def test_hub_members_not_claimed(fg):
    """A demoted member still imported plainly elsewhere keeps its broad
    closure; claiming it would drop live lazy edges (severed fixpoint)."""
    for hub in (
        "vllm/distributed/parallel_state.py",
        "vllm/platforms/__init__.py",
        "vllm/_custom_ops.py",
        "vllm/multimodal/inputs.py",
    ):
        assert hub not in fg.all_claims(), hub


def test_severed_members_claimed(fg):
    for member in (
        "vllm/v1/spec_decode/eagle.py",
        "vllm/v1/worker/gpu/pool/pooling_runner.py",
        "vllm/transformers_utils/configs/eagle.py",
    ):
        assert member in fg.dispatch.claims, member


def test_demoted_members_reach_example_scripts(fg):
    """The examples-step recall route: leaf-consumer literals wire the step's
    script into the demoted member's closure."""
    closure = fg.graph.reverse_closure({"vllm/v1/spec_decode/eagle.py"})
    assert "examples/features/speculative_decoding/spec_decode_offline.py" in closure
    pool = fg.graph.reverse_closure({"vllm/v1/worker/gpu/pool/pooling_runner.py"})
    assert any(f.startswith("examples/") for f in pool)


def test_decorator_registered_member_not_demoted(fg):
    """mm_preprocess's classes go to minimax model files' module-level
    `@MULTIMODAL_REGISTRY.register_processor(...)`, which runs at import: the
    decorator arg is a value use, so the member must not be demoted."""
    member = "vllm/models/minimax_m3/common/mm_preprocess.py"
    assert not any(m == member for _, m in fg.graph.demoted_edges)
    assert member not in fg.dispatch.claims
    assert len(_tests(fg, member)) > 1000


def test_spec_proposer_literal_pinned(fg):
    """dflash's guard literal must stay exactly {dflash}; a widened upstream
    guard shape that inflated it to the whole spec family would silently
    misroute."""
    assert fg.dispatch.demotions.get((RUNNER, "vllm/v1/spec_decode/dflash.py")) == {
        "dflash"
    }


def test_no_leaf_origin_lazy_drops_and_flashinfer_edge(fg):
    assert not [
        p for p in fg.graph.dropped_lazy if p[0].startswith(("tests/", "benchmarks/"))
    ]
    assert "vllm/v1/attention/backends/flashinfer.py" in fg.graph.imports.get(
        "tests/kernels/attention/test_flashinfer.py", set()
    )


def test_config_helpers_include_spec_predicates(fg, repo):
    """Self-adapting guard: if the SpeculativeConfig predicate helpers are
    renamed/removed the spec proposers silently revert to broad."""
    helpers = _config_helper_literals(repo, fg.index)
    assert "eagle" in helpers.get("use_eagle", set())


# ---- Ext A/B: speculator models + ngram_gpu (guarded type positions) -------

SPECULATOR_MODELS = (
    "vllm/model_executor/models/llama_eagle3.py",
    "vllm/model_executor/models/deepseek_eagle3.py",
    "vllm/model_executor/models/qwen3_eagle3.py",
    "vllm/model_executor/models/qwen3_dflash.py",
    "vllm/model_executor/models/laguna_dflash.py",
)
BASE_PROPOSER = "vllm/v1/spec_decode/llm_base_proposer.py"


def test_speculator_models_demoted_from_base_proposer(fg):
    for m in SPECULATOR_MODELS:
        assert (BASE_PROPOSER, m) in fg.graph.demoted_edges, m
        assert len(_tests(fg, m)) < 300, (m, len(_tests(fg, m)))


def test_speculator_models_claimed_except_multi_importer(fg):
    """The 4 routed solely through the base proposer are claimed; qwen3_dflash
    is also subclassed by qwen3_dspark/gemma4_dspark, so it stays broadly
    imported and unclaimed (its own closure keeps it covered)."""
    for m in SPECULATOR_MODELS:
        if m.endswith("qwen3_dflash.py"):
            assert m not in fg.dispatch.claims, m
        else:
            assert m in fg.dispatch.claims, m


def test_ngram_gpu_demoted_and_claimed(fg):
    ng = "vllm/v1/spec_decode/ngram_proposer_gpu.py"
    assert fg.dispatch.demotions.get((RUNNER, ng)) == {"ngram_gpu"}
    assert ng in fg.dispatch.claims
    assert len(_tests(fg, ng)) < 100


def test_type_use_collateral_hubs_stay_broad(fg):
    """Ext A demotes a hub from one guarded type-use importer, but it stays
    broad via other unconditional importers and is never claimed."""
    for hub in (
        "vllm/inputs/__init__.py",
        "vllm/utils/math_utils.py",
        "vllm/logging_utils/__init__.py",
        "vllm/model_executor/models/interfaces_base.py",
        "vllm/compilation/breakable_cudagraph.py",
    ):
        assert len(_tests(fg, hub)) > 1000, hub
        assert hub not in fg.all_claims(), hub


def test_base_class_subclass_edges_not_demoted(fg):
    """A guarded base-as-site blocks demoting the real subclass edge.
    kda_metadata subclasses gdn_attn, qwen3_dspark subclasses qwen3_dflash --
    demoting either would cut a genuine dependency."""
    for importer, member in (
        (
            "vllm/models/kimi_k3/nvidia/kda_metadata.py",
            "vllm/v1/attention/backends/gdn_attn.py",
        ),
        (
            "vllm/model_executor/models/qwen3_dspark.py",
            "vllm/model_executor/models/qwen3_dflash.py",
        ),
    ):
        assert member in fg.graph.imports.get(importer, set()), (
            f"{importer} no longer imports {member}: specimen drifted"
        )
        assert (importer, member) not in fg.graph.demoted_edges


# ---- Ext C: type-only imports of independently-keyed members ---------------

ATTN_BACKENDS = (
    "vllm/v1/attention/backends/mamba2_attn.py",
    "vllm/v1/attention/backends/gdn_attn.py",
    "vllm/v1/attention/backends/linear_attn.py",
)


def test_enum_keyed_attention_backends_demoted(fg):
    """They are AttentionBackendEnum-claimed yet were imported only in
    isinstance/annotation positions; Ext C demotes those type-only edges."""
    for m in ATTN_BACKENDS:
        assert any(x == m for _, x in fg.graph.demoted_edges), m
        assert len(_tests(fg, m)) < 200, (m, len(_tests(fg, m)))


def test_mamba2_claimed_gdn_linear_kept_broad_importer(fg):
    """mamba2's importers are all type-only -> claimed. gdn (kda_metadata
    subclasses it) and linear (bailing's unguarded lazy import) keep a broad
    importer, so the severed fixpoint leaves them unclaimed."""
    assert "vllm/v1/attention/backends/mamba2_attn.py" in fg.dispatch.claims
    for m in (
        "vllm/v1/attention/backends/gdn_attn.py",
        "vllm/v1/attention/backends/linear_attn.py",
    ):
        assert m not in fg.dispatch.claims, m


def test_mamba_base_not_demoted_but_collapses_transitively(fg):
    """The shared base is never a demoted member (subclass edges are real),
    but it scopes down once its subclasses demote."""
    base = "vllm/v1/attention/backends/mamba_attn.py"
    assert not any(x == base for _, x in fg.graph.demoted_edges)
    assert len(_tests(fg, base)) < 200


def test_triton_attn_stays_broad_via_list_escape(fg):
    """triton_attn is stored in a list literal in llm_base_proposer (a value
    escape), so it is NOT type-only and must keep its broad closure."""
    tri = "vllm/v1/attention/backends/triton_attn.py"
    assert not any(x == tri for _, x in fg.graph.demoted_edges)
    assert len(_tests(fg, tri)) > 1000
