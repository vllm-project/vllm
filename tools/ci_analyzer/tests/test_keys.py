# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered-key routing: true-positive pins (the no-loss proof for typed
matching) + matcher units."""


def _steps_for(state, key):
    return {sid for sid, ks in state.keys.step_keys.items() if key in ks}


def test_register_key_value_contexts_pinned(state):
    """PR #50326 anchor: NixlConnector selects the PD/accuracy e2e fleet via
    quoted/assigned config values; must survive any matcher change."""
    nixl = _steps_for(state, "NixlConnector")
    assert len(nixl) >= 15, sorted(nixl)
    assert any("nixlconnector-pd-accuracy" in s for s in nixl)


def test_parser_key_flag_contexts_pinned(state):
    """Premise on the raw searchable text (the flag context still exists),
    assertion on the matcher output."""
    k = state.keys
    sid = "vllm_ci:MRCR Eval Small Models"
    assert "--reasoning-parser qwen3" in k.searchable[sid], (
        "MRCR eval no longer pins qwen3 by flag: update specimen"
    )
    assert "qwen3" in k.step_keys[sid]

    sid = "vllm_ci:moe-refactor-integration-test-b200-temporary"
    assert "deepseek_v4" in k.searchable[sid], "specimen moved"
    assert "deepseek_v4" in k.step_keys[sid]


def test_parser_flag_re_flags_exist_in_vllm(repo):
    """Rot guard: every flag in _PARSER_FLAG_RE must still exist in vLLM.
    A phantom flag (the never-existent --renderer) silently degrades typed matching."""
    import subprocess

    from ci_analyzer.keys import _PARSER_FLAG_RE

    body = _PARSER_FLAG_RE.removeprefix("(?:").removesuffix(")")
    flags = [a for a in body.split("|") if a.startswith("--")]
    assert flags, "no -- flags parsed from _PARSER_FLAG_RE"
    missing = [
        flag
        for flag in flags
        if subprocess.run(["grep", "-rq", "--", flag, str(repo / "vllm")]).returncode
        != 0
    ]
    assert not missing, f"parser flags absent from vLLM (rot): {missing}"


def test_hf_id_env_pin(state):
    assert (
        "deepseek-ai/DeepSeek-V2-Lite-Chat"
        in state.keys.step_keys["vllm_ci:2-node-test-4-gpus"]
    )


def test_register_value_context_matcher():
    from ci_analyzer.keys import _typed_pattern

    pat = _typed_pattern("NixlConnector", "register")
    assert pat.search('"kv_connector":"NixlConnector"')
    assert pat.search("KV_CONNECTOR=${KV_CONNECTOR:-NixlConnector}")
    assert pat.search('\\"NixlConnector\\"')
    assert pat.search("kv_connector=NixlConnector")
    ipc = _typed_pattern("ipc", "register")
    assert not ipc.search("docker run --ipc=host img")
    assert not ipc.search("some words about ipc handles")


def test_parser_flag_context_matcher():
    from ci_analyzer.keys import _typed_pattern

    pat = _typed_pattern("granite", "parser")
    assert pat.search("--tool-call-parser granite -x")
    assert pat.search('"tool_call_parser": "granite"')
    assert not pat.search("MODEL_NAMES=ibm-granite/granite-4.0-h-tiny")
    assert not pat.search("run tests/tool_use with granite fixtures")


def test_comment_lines_do_not_route_keys():
    from ci_analyzer.keys import _strip_comment_lines

    text = "# Default: TRITON_ATTN on ROCm\npytest v1/attention"
    assert "TRITON_ATTN" not in _strip_comment_lines(text)
    assert "pytest v1/attention" in _strip_comment_lines(text)


def test_scalar_literals_never_become_keys(state):
    """`1` and `true` come from an env truthiness test and match `-tp=1`,
    `sleep 1`, `|| true` all over the pipeline. Rejecting them by type is what
    replaced a step-fanout bar that could not tell them from a popular genuine
    key, and severed the genuine one."""
    from ci_analyzer.keys import _is_scalar_literal

    assert all(_is_scalar_literal(k) for k in ("1", "true", "0", "null"))
    assert not any(_is_scalar_literal(k) for k in ("mtp", "pooling", "fp8", "auto"))
    assert {"1", "true"} <= set(state.keys.refused)
    assert not any(_is_scalar_literal(k) for k in state.keys.key_mechanism)


def test_wide_dispatch_key_survives(state):
    """No key is dropped for naming too many steps. `mtp` is the specimen: it
    routes the spec-decode PD-accuracy and lm-eval steps, which reach it by
    config string and not by import, so losing it is silent under-selection."""
    from collections import Counter

    counts = Counter(key for ks in state.keys.step_keys.values() for key in ks)
    assert counts["mtp"] > 20, counts["mtp"]
    assert state.keys.key_mechanism.get("mtp") == "dispatch"
    routed = state.keys.steps_naming({"mtp"})
    assert any("pd-accuracy" in s for s in routed), sorted(routed)[:5]
    assert any("lm-eval" in s for s in routed), sorted(routed)[:5]


def test_dropped_edges_separates_refused_from_unrouted():
    """The old predicate asked whether a literal existed anywhere in the index,
    so one owned by another file counted as routing this member, and a deleted
    one just slid the member into the skipped bucket."""
    from types import SimpleNamespace

    from ci_analyzer.keys import KeyIndex
    from ci_analyzer.validate.dropped_edges import key_selection_gaps

    keys = KeyIndex()
    keys.keyed_modules["vllm/a.py"] = {"alpha"}
    keys.refused["beta"] = "non-string scalar"
    state = SimpleNamespace(
        full=SimpleNamespace(
            dispatch=SimpleNamespace(
                demotions={("vllm/imp.py", "vllm/a.py"): {"alpha", "beta", "gamma"}},
                claims={"vllm/a.py"},
            )
        ),
        keys=keys,
        auto_step_ids=set(),
    )
    _gaps, unrouted, checked = key_selection_gaps(state)
    assert unrouted == [("vllm/a.py", "gamma")]
    assert checked == 0


def test_fanout_bar_loss_needs_a_registered_key_with_no_route():
    """The bar drops registered keys as well as English words ("pooling" is one
    at HEAD), so a drop is only a loss when the key index says the literal
    routes somewhere and nothing survives to reach it."""
    from types import SimpleNamespace

    from ci_analyzer.curated import CONFIG_KEY_MAX_TEST_FILES as BAR
    from ci_analyzer.graph.imports import ImportGraph
    from ci_analyzer.keys import KeyIndex
    from ci_analyzer.validate.dropped_edges import fanout_dropped_literals

    graph = ImportGraph()
    for lit in ("english", "keyed", "routed"):
        for n in range(BAR + 1):  # every literal is over the bar
            graph.string_literals[f"tests/{lit}/test_{n}.py"] = {lit}

    keys = KeyIndex()
    keys.key_mechanism["keyed"] = "dispatch"  # registered, routes no auto step
    keys.key_mechanism["routed"] = "dispatch"
    keys.step_keys = {"vllm_ci:s": {"routed"}}
    state = SimpleNamespace(
        full=SimpleNamespace(
            graph=graph,
            dispatch=SimpleNamespace(
                demotions={
                    ("vllm/imp.py", "vllm/m.py"): {"english", "keyed", "routed"}
                },
            ),
        ),
        keys=keys,
        auto_step_ids={"vllm_ci:s"},
    )
    losses, dropped = fanout_dropped_literals(state)
    assert dropped == 3
    assert losses == [("vllm/m.py", "keyed")]


def test_key_routing_is_belt_over_graph_coverage(state):
    """Fallback: with key routing disabled entirely, a parser file still
    selects the steps covering its own test files via the import graph."""
    import dataclasses

    from ci_analyzer.keys import KeyIndex
    from ci_analyzer.select import select

    bare_keys = KeyIndex(searchable=dict(state.keys.searchable))
    bare = dataclasses.replace(state, keys=bare_keys)
    sel = select(bare, ["vllm/tool_parsers/granite_tool_parser.py"])
    assert not sel.run_all
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) - always, "graph channel must still select"


def test_short_key_word_boundary_units():
    import regex as re
    from ci_analyzer.keys import match_keys

    patterns = {
        "inc": re.compile(r"\binc\b"),
        "fp8": re.compile(r"\bfp8\b"),
    }
    hits = match_keys(set(), patterns, {}, set(), "use --include=x and fp8 quant")
    assert hits == {"fp8"}, "inc must not match inside include"
    assert "inc" in match_keys(set(), patterns, {}, set(), "backend inc selected")
    assert not match_keys(set(), patterns, {}, set(), "xfp8 only"), (
        "fp8 must not match inside a word"
    )


def test_substring_key_with_metachars_safe():
    from ci_analyzer.keys import match_keys

    hits = match_keys({"org/model+x"}, {}, {}, set(), "eval run org/model+x here")
    assert hits == {"org/model+x"}


def test_target_literal_routes_without_haystack_hit():
    """Key absent from the command haystack still routes on the step's own
    target test file carrying it as a string literal."""
    from ci_analyzer.keys import match_keys

    hits = match_keys({"org/model+x"}, {}, {}, {"org/model+x"}, "unrelated cmd")
    assert hits == {"org/model+x"}


def test_matching_thresholds_pinned():
    from ci_analyzer.keys import _RAW_KEY_MIN_LEN, _SUBSTRING_KEY_MIN_LEN

    assert _SUBSTRING_KEY_MIN_LEN == 12
    # 18 real archs are 8-11 chars slash-free; raising the raw bar drops
    # them from table-diff head-side routing (under-selection direction).
    assert _RAW_KEY_MIN_LEN == 8


def test_parser_key_env_default_context():
    from ci_analyzer.keys import _typed_pattern

    pat = _typed_pattern("openai", "parser")
    assert pat.search('TOOL_CALL_PARSER="${BFCL_TOOL_CALL_PARSER:-openai}"')
    assert not pat.search("pytest entrypoints/openai -v")
