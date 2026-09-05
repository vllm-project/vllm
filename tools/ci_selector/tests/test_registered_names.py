# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered-key routing: true-positive pins (the no-loss proof for typed
matching) + matcher units."""

import pytest
from helpers import drift_message


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
    sid = "vllm_ci::nvidia: (H200 MIG 35GB) MRCR Eval Small Models"
    assert "--reasoning-parser qwen3" in k.searchable[sid], (
        "MRCR eval no longer pins qwen3 by flag: update specimen"
    )
    assert "qwen3" in k.step_keys[sid]

    # A second flag, so the two specimens cover both rather than one twice.
    sid = "vllm_ci:lm-eval-spec-decode-4xb200"
    assert "--tool-call-parser kimi_k3" in k.searchable[sid], (
        "spec-decode eval no longer pins kimi_k3 by flag: update specimen"
    )
    assert "kimi_k3" in k.step_keys[sid]


@pytest.mark.drift
def test_parser_flag_re_flags_exist_in_vllm(vllm_repo):
    """Rot guard: every flag in _PARSER_FLAG_RE must still exist in vLLM.
    A phantom flag (the never-existent --renderer) silently degrades typed matching."""
    import subprocess

    from ci_selector.codemap.registered_names import _PARSER_FLAG_RE

    cost = (
        "These flags are how a job command like `--kv-connector NixlConnector` "
        "routes to the code that registers that name. A flag we watch for that "
        "vLLM does not have matches nothing, so those jobs lose their typed "
        "edge and fall back to broader matching."
    )
    body = _PARSER_FLAG_RE.removeprefix("(?:").removesuffix(")")
    # Every alternative, not just the `--` spellings. The snake_case forms are
    # what match config and env contexts, and filtering them out left half the
    # pattern unchecked: a renamed config field moved nothing.
    flags = [a for a in body.split("|") if a]
    assert len(flags) >= 4, drift_message(
        f"Only {len(flags)} alternatives could be read out of _PARSER_FLAG_RE, "
        "so this guard is checking almost nothing.",
        cost,
        "the pattern changed shape: update the split in this test to match "
        "_PARSER_FLAG_RE in ci_selector/codemap/registered_names.py",
    )
    missing = [
        flag
        for flag in flags
        if subprocess.run(
            ["grep", "-rq", "--", flag, str(vllm_repo / "vllm")]
        ).returncode
        != 0
    ]
    assert not missing, drift_message(
        f"_PARSER_FLAG_RE watches flags that no longer appear in vLLM: {missing}",
        cost,
        "the flag was renamed upstream: update _PARSER_FLAG_RE in "
        "ci_selector/codemap/registered_names.py",
        "the flag was removed: drop it from _PARSER_FLAG_RE",
    )


def test_hf_id_env_pin(state):
    assert (
        "deepseek-ai/DeepSeek-V2-Lite-Chat"
        in state.keys.step_keys["vllm_ci:2-node-test-4-gpus"]
    )


def test_register_value_context_matcher():
    from ci_selector.codemap.registered_names import _typed_pattern

    pat = _typed_pattern("NixlConnector", "register")
    assert pat.search('"kv_connector":"NixlConnector"')
    assert pat.search("KV_CONNECTOR=${KV_CONNECTOR:-NixlConnector}")
    assert pat.search('\\"NixlConnector\\"')
    assert pat.search("kv_connector=NixlConnector")
    ipc = _typed_pattern("ipc", "register")
    assert not ipc.search("docker run --ipc=host img")
    assert not ipc.search("some words about ipc handles")


def test_parser_flag_context_matcher():
    from ci_selector.codemap.registered_names import _typed_pattern

    pat = _typed_pattern("granite", "parser")
    assert pat.search("--tool-call-parser granite -x")
    assert pat.search('"tool_call_parser": "granite"')
    assert not pat.search("MODEL_NAMES=ibm-granite/granite-4.0-h-tiny")
    assert not pat.search("run tests/tool_use with granite fixtures")


def test_comment_lines_do_not_route_keys():
    from ci_selector.codemap.registered_names import _strip_comment_lines

    text = "# Default: TRITON_ATTN on ROCm\npytest v1/attention"
    assert "TRITON_ATTN" not in _strip_comment_lines(text)
    assert "pytest v1/attention" in _strip_comment_lines(text)


def test_scalar_literals_never_become_keys(state):
    """`1` and `true` come from an env truthiness test and match `-tp=1`,
    `sleep 1`, `|| true` all over the pipeline. Rejecting them by type is what
    replaced a step-fanout bar that could not tell them from a popular genuine
    key, and severed the genuine one."""
    from ci_selector.codemap.registered_names import _is_scalar_literal

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

    from ci_selector.codemap.registered_names import KeyIndex
    from helpers import key_selection_gaps

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

    from ci_selector.codemap.graph.demote import CONFIG_KEY_MAX_TEST_FILES as BAR
    from ci_selector.codemap.graph.imports import ImportGraph
    from ci_selector.codemap.registered_names import KeyIndex
    from helpers import fanout_dropped_literals

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

    from ci_selector.codemap.classify import select
    from ci_selector.codemap.registered_names import KeyIndex

    bare_keys = KeyIndex(searchable=dict(state.keys.searchable))
    bare = dataclasses.replace(state, keys=bare_keys)
    sel = select(bare, ["vllm/tool_parsers/granite_tool_parser.py"])
    assert not sel.run_all
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) - always, "graph channel must still select"


def test_short_key_word_boundary_units():
    import regex as re
    from ci_selector.codemap.registered_names import match_keys

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
    from ci_selector.codemap.registered_names import match_keys

    hits = match_keys({"org/model+x"}, {}, {}, set(), "eval run org/model+x here")
    assert hits == {"org/model+x"}


def test_target_literal_routes_without_haystack_hit():
    """Key absent from the command haystack still routes on the step's own
    target test file carrying it as a string literal."""
    from ci_selector.codemap.registered_names import match_keys

    hits = match_keys({"org/model+x"}, {}, {}, {"org/model+x"}, "unrelated cmd")
    assert hits == {"org/model+x"}


def test_matching_thresholds_pinned():
    from ci_selector.codemap.registered_names import (
        RAW_KEY_MIN_LEN,
        SUBSTRING_KEY_MIN_LEN,
    )

    assert SUBSTRING_KEY_MIN_LEN == 12
    # 18 real archs are 8-11 chars slash-free; raising the raw bar drops
    # them from table-diff head-side routing (under-selection direction).
    assert RAW_KEY_MIN_LEN == 8


def test_parser_key_env_default_context():
    from ci_selector.codemap.registered_names import _typed_pattern

    pat = _typed_pattern("openai", "parser")
    assert pat.search('TOOL_CALL_PARSER="${BFCL_TOOL_CALL_PARSER:-openai}"')
    assert not pat.search("pytest entrypoints/openai -v")


def test_colliding_parser_key_keeps_all_module_files(state):
    """kimi_k3 registers as a tool parser AND a tokenizer mode; the merged
    parser_entries dict is last-wins (tokenizers parse after tool_parsers),
    which pointed the key at vllm/tokenizers/hf.py alone and silently cut the
    parser file's key route. Every registering file must keep the key."""
    parser_file = "vllm/tool_parsers/kimi_k3_tool_parser.py"
    assert "kimi_k3" in state.keys.for_file(parser_file), (
        "kimi_k3 no longer keys its parser file; if the collision is gone "
        "pick a new colliding specimen before deleting this test"
    )
    assert "kimi_k3" in state.keys.for_file("vllm/tokenizers/hf.py")


def test_directory_targets_fold_test_literals(state):
    """A step whose pytest target is a DIRECTORY carries the keys its test
    files pin. entrypoints-integration-api-server-generate runs
    `pytest tool_use` and declares only bare vllm/ (the B4 shape); without
    the fold its step_keys were empty and every parser key lost the belt
    route that survives an import edge going missing."""
    sid = "vllm_ci:entrypoints-integration-api-server-generate"
    lits = state.full.graph.string_literals
    assert "kimi_k3" in lits.get("tests/tool_use/test_kimi_k3_tool_parser.py", set()), (
        "specimen moved: pick another parser key pinned by a test under a "
        "directory target"
    )
    assert "kimi_k3" in state.keys.step_keys.get(sid, set())


def test_directory_fold_excludes_non_test_files(state):
    """tests/tool_use/utils.py pins the whole parser matrix; helper literals
    must not join the fold. A helper names parsers the step may never run,
    and an imported helper is already the graph's job to cover."""
    sid = "vllm_ci:entrypoints-integration-api-server-generate"
    assert "internlm" in state.full.graph.string_literals.get(
        "tests/tool_use/utils.py", set()
    ), "specimen moved: utils.py no longer pins internlm"
    assert "internlm" not in state.keys.step_keys.get(sid, set())


def test_tool_parser_file_routes_to_steps_running_its_tests(state):
    """The belt over graph coverage, end to end: changed parser file -> its
    registered key -> the step whose directory target holds the test pinning
    that key. Holds even if the test's import edge disappears."""
    keys = state.keys.for_file("vllm/tool_parsers/kimi_k3_tool_parser.py")
    steps = state.keys.steps_naming(keys)
    assert "vllm_ci:entrypoints-integration-api-server-generate" in steps


def test_tool_parser_key_routing_floor(state):
    """Detection floor for the whole parser-key channel: a moved registry
    table or a broken fold must read as loud failure, not as quietly-empty
    routing. 47 tool-parser entries and 44/57 merged keys routed at pinning
    time; the bars sit far below that so ordinary churn passes."""
    from ci_selector.codemap.graph.factories import TOOL_PARSER_INIT

    counts = state.full.factories.parser_table_counts
    assert counts.get(TOOL_PARSER_INIT, 0) > 40, counts
    entries = state.full.factories.parser_entries
    routed = {key for key in entries if state.keys.steps_naming({key})}
    assert len(routed) >= 30, (
        f"only {len(routed)}/{len(entries)} parser keys reach any step; "
        "the key-routing belt has collapsed"
    )


# A flag whose name ends in `parser`, and the trailing character that proves it
# ended there. --tool-parser-plugin names a plugin module, not a parser.
PARSER_FLAG_SHAPE = r"--[a-z0-9-]*parser([^a-z0-9-]|$)"


@pytest.mark.drift
def test_no_parser_selecting_flag_is_unwatched(vllm_repo):
    """The other direction, which the rot guard above cannot see.

    A flag we watch that vLLM dropped matches nothing and is caught above. A
    flag vLLM added that we do not watch is the expensive one: those jobs never
    get a typed edge to the parser they name, so a change to that parser stops
    selecting them and nothing anywhere goes red.
    """
    import subprocess

    import regex as re
    from ci_selector.codemap.registered_names import PARSER_SELECTING_FLAGS

    # The whole tree, not just cli_args.py: only --tool-call-parser is spelled
    # there, so scanning that one file left the other flags unwatched by this.
    # `parser` must end the flag, since --tool-parser-plugin names a plugin
    # module rather than selecting a parser.
    hits = subprocess.run(
        # -e, or grep reads the pattern's leading -- as an unknown option and
        # exits 2 with no output. The floor below is what would actually catch
        # that; -e keeps the failure from being a confusing one.
        ["grep", "-rhoE", "-e", PARSER_FLAG_SHAPE, str(vllm_repo / "vllm")],
        capture_output=True,
        text=True,
    ).stdout
    found = {re.sub(r"[^a-z-]+$", "", line) for line in hits.split() if line}
    assert len(found) >= 2, drift_message(
        f"Only {len(found)} parser-selecting flags were found in vllm/: {found}.",
        "This guard reads the tree to notice a flag we do not watch. Finding "
        "none finds no gaps, which looks exactly like watching all of them.",
        "the flags moved or changed shape: update the pattern in this test",
    )
    unwatched = sorted(found - set(PARSER_SELECTING_FLAGS))
    assert not unwatched, drift_message(
        f"vLLM has parser-selecting flags we do not watch: {unwatched}",
        "Typed matching is how `--reasoning-parser qwen3` in a job command "
        "routes that job to the parser it names. An unwatched flag leaves "
        "those jobs unrouted, so editing the parser stops selecting them.",
        "add the flag, and its snake_case config spelling, to "
        "PARSER_SELECTING_FLAGS in ci_selector/codemap/registered_names.py",
    )
