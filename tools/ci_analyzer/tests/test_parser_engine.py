# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified parser-engine registry (PR #48947): each vllm/parser/<name>.py is
claimed and stem-keyed so a change to it selects the steps that exercise that
parser, not the api_server-hub near-run-all blanket.

Soundness guarantee (static-floor seal): claiming an engine only drops LAZY
edges into it, so every AUTO-run step that runs a test STATICALLY importing the
engine must survive. Runtime-only coverage (integration tests with no static
import, e.g. tests/tool_use/mistral/) is pinned by the mistral case below and
re-verified adversarially out-of-band when the leaf-key channels change.
"""

from ci_analyzer.select import select


def _auto_steps(state):
    return {s.step_id for p in state.pipelines for s in p.steps if not s.manual_only}


def _static_importer_tests(state, target):
    """Test files that transitively import `target` via module-level edges."""
    pr = state.full.plain_reverse
    seen, stack = {target}, [target]
    while stack:
        node = stack.pop()
        for src in pr.get(node, ()):
            if src not in seen:
                seen.add(src)
                stack.append(src)
    return {f for f in seen if f.startswith("tests/")}


def _steps_running(state, test_files):
    hit = set()
    for p in state.pipelines:
        for sid, st in p.targets.items():
            for t in st.targets:
                covered = (
                    t.path in test_files
                    if t.path.endswith(".py")
                    else any(f.startswith(t.path.rstrip("/") + "/") for f in test_files)
                )
                if covered:
                    hit.add(sid)
                    break
    return hit


def test_engine_set_derived_and_claimed(state):
    """The derivation finds the concrete parsers, not the base class or the
    package init; if the registry restructures, the specimens below move."""
    keyed = state.full.factories.parser_engine_entries
    assert len(keyed) >= 10, sorted(keyed)
    assert {"mistral", "qwen3", "harmony"} <= set(keyed)
    assert "__init__" not in keyed and "abstract_parser" not in keyed
    assert set(keyed.values()) <= state.full.factories.claims
    for stem, module in keyed.items():
        assert module == f"vllm/parser/{stem}.py"


def test_static_floor_no_under_selection(state):
    """The load-bearing seal: for every engine, every AUTO step running a test
    that statically imports it survives the tightened selection."""
    auto = _auto_steps(state)
    for module in sorted(state.full.factories.parser_engine_entries.values()):
        floor = _steps_running(state, _static_importer_tests(state, module)) & auto
        sel = select(state, [module])
        assert not sel.run_all, module
        missing = floor - set(sel.selected)
        assert not missing, (module, sorted(missing))


def test_mistral_tightened_but_keeps_parser_coverage(state):
    """mistral has zero static importers, so its coverage is entirely stem-keyed.
    It must drop below the 252 hub blanket yet keep every mistral-parser-test step,
    including model-generation, whose test_mistral.py constructs MistralToolParser."""
    sel = select(state, ["vllm/parser/mistral.py"])
    assert not sel.run_all
    selected = set(sel.selected)
    assert len(selected) < 120, len(selected)  # was 252 (near-run-all)
    for step in (
        "vllm_ci:Rust Frontend Tool Use",
        "vllm_ci:entrypoints-integration-api-server-generate",
        "vllm_ci:async-engine-inputs-utils-worker-config-cpu",
        "vllm_ci:language-models-tests-standard",
    ):
        assert step in selected, step
