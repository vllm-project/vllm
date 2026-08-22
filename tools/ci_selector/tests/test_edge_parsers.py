# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge parsers (registry flip, factories, platform, spawn, assets).

Asserted names are re-derived from the vllm_repo where cheap; a specific known
entry is first proven to exist in the source, so vllm_repo drift fails as
"update fixture" rather than as parser skew.
"""

import pytest
import regex as re
from ci_selector.codemap.classify import select
from helpers import HW, drift_message


@pytest.fixture(scope="module")
def fg(full):
    return full


def _test_files(graph, target):
    closure = graph.reverse_closure({target})
    return {
        f
        for f in closure
        if f.startswith("tests/") and f.rsplit("/", 1)[-1].startswith("test_")
    }


def test_single_model_closure_scoped_after_registry(fg):
    """String-keyed registry + lazy finalization scope a single-model file's
    blast radius to its real derived-model tests: dispatch Ext A demotes the
    base proposers' type-only edges, severing their one static bridge to the
    worker seam. minimax_m3 stays broad via the warmup chain (witness below),
    a deferred capability-gated cluster and a layering finding for maintainers."""
    for target in (
        "vllm/model_executor/models/mllama4.py",
        "vllm/model_executor/models/qwen3.py",
    ):
        tf = _test_files(fg.graph, target)
        assert 0 < len(tf) < 400, (target, len(tf))
    for wide, witness in (
        (
            "vllm/models/minimax_m3/__init__.py",
            "vllm/model_executor/warmup/minimax_m3_msa_warmup.py",
        ),
    ):
        closure = fg.graph.reverse_closure({wide})
        assert witness in closure, (
            f"{wide}'s engine-core chain vanished: if the layering was fixed "
            "upstream, move it back into the scoped list above"
        )


def test_registry_pass_through_qualnames(fg, vllm_repo):
    src = (vllm_repo / "vllm/model_executor/models/registry.py").read_text()
    oracle = set(re.findall(r'"(vllm\.[\w.]+)"', src))
    assert oracle, "no pass-through entries left; update fixture"
    covered = {mod for mod, _cls in fg.registry.entries.values()}
    assert oracle & covered, "pass-through qualname entries were dropped"


def test_kv_connector_register_calls(fg, vllm_repo):
    factory_src = (
        vllm_repo / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    ).read_text()
    names = re.findall(r'register_connector\(\s*"(\w+)"', factory_src)
    assert names, "no register_connector calls found; update fixture"
    for name in names:
        assert name in fg.factories.register_entries, name


def test_reasoning_and_tool_parser_tables(fg):
    assert "deepseek_r1" in fg.factories.parser_entries
    assert fg.factories.parser_entries["deepseek_r1"].startswith("vllm/reasoning/")
    assert any(
        t.startswith("vllm/tool_parsers/") for t in fg.factories.parser_entries.values()
    )


@pytest.mark.drift
def test_lazy_parser_tables_counted_per_table(fg):
    """Every lazy table must parse entries of its own. The merged
    parser_entries dict cannot show this: parser names collide across the four
    tables, so it is last-wins and a dead tokenizers anchor leaves its size
    unchanged."""
    counts = fg.factories.parser_table_counts
    assert len(counts) >= 4, counts
    assert not [a for a, n in counts.items() if not n], drift_message(
        "These lazy parser tables parsed nothing: "
        f"{sorted(a for a, n in counts.items() if not n)}",
        "Each one routes a family of plugins (reasoning parsers, tool parsers, "
        "renderers, tokenizers). An empty table means none of them can be "
        "reached by name, so tests that select one are not picked up.",
        "the table was renamed or moved in vLLM: update the anchor path or "
        "table name in " + HW,
        "the table changed shape: update add_lazy_parser_table_edges in "
        "ci_selector/codemap/graph/factories.py",
    )
    assert sum(counts.values()) > len(fg.factories.parser_entries), (
        "parser names no longer collide across the tables; if that is real the "
        "merged emptiness row would suffice -- re-check before deleting"
    )


def test_attention_enum_backends(fg):
    assert "FLASH_ATTN" in fg.factories.enum_entries
    assert "flash_attn" in fg.factories.enum_entries["FLASH_ATTN"]


@pytest.mark.drift
def test_qualname_enum_scan_finds_every_registry_not_just_the_attention_one(fg):
    """The parser is aimed at a SHAPE, so a second registry with the same shape
    must be read without anyone adding its path anywhere. Per-file counts,
    because member names collide across registries (FLASH_ATTN is in both) and
    the merged dict would not shrink if one registry went dead."""
    counts = fg.factories.enum_table_counts
    for registry in (
        "vllm/v1/attention/backends/registry.py",
        "vllm/v1/attention/backends/mla/prefill/registry.py",
    ):
        assert counts.get(registry), drift_message(
            f"The qualname-enum scan read no members out of {registry}. "
            f"It found: {sorted(counts)}",
            "Those enum values are how backends are loaded at runtime. Unread, "
            "the backend files have no importer and a change to one selects "
            "nothing.",
            "the enum changed shape: update add_qualname_enum_edges in "
            "ci_selector/codemap/graph/factories.py",
            "the registry moved: the scan finds files by shape, so check the "
            "enum still has string members that resolve to real modules",
        )


def test_mla_prefill_backends_are_reachable(fg):
    """The hole this scan closed: the MLA prefill registry loads its backends
    through resolve_obj_by_qualname, and nothing read the table, so the edges
    to the backend files did not exist."""
    prefill = "vllm/v1/attention/backends/mla/prefill/"
    claimed = {c for c in fg.factories.claims if c.startswith(prefill)}
    assert {f"{prefill}flash_attn.py", f"{prefill}flashinfer.py"} <= claimed, claimed


def test_module_attrs_edges(fg, vllm_repo):
    # Stored value is the module part only; an unstripped ":LLM" would make
    # every resolve fail silently (the bug this test previously masked).
    assert fg.factories.module_attrs.get("LLM") == "entrypoints.llm"
    assert fg.factories.module_attr_resolved > 0
    llm_file = "vllm/entrypoints/llm.py"

    specimen = "tests/v1/cudagraph/test_cudagraph_mode.py"
    src = (vllm_repo / specimen).read_text()
    assert "from vllm import" in src and "LLM" in src, (
        f"{specimen} no longer imports LLM from the vllm package: pick a new specimen"
    )
    assert "vllm.entrypoints.llm" not in src, (
        f"{specimen} now imports entrypoints.llm directly: the specimen no "
        "longer proves the module-attr edge; pick a new one"
    )
    assert specimen in fg.graph.reverse_closure({llm_file})

    # `from vllm import LLM` importers far outnumber explicit entrypoints.llm ones.
    direct = {
        f for f in fg.graph.reverse.get(llm_file, set()) if f.startswith("tests/")
    }
    assert len(direct) > 20, (
        f"only {len(direct)} tests/ importers of {llm_file}: module-attr "
        "edges look dead again"
    )


def test_pkgutil_helion_dir(fg):
    assert "vllm/kernels/helion/ops/" in fg.factories.pkgutil_dirs
    init = "vllm/kernels/helion/ops/__init__.py"
    assert any(
        dst.startswith("vllm/kernels/helion/ops/") and dst != init
        for dst in fg.graph.imports.get(init, ())
    )


def test_platform_punica_tracing(fg):
    targets = fg.platform.candidates.get("get_punica_wrapper", set())
    assert any("punica" in t for t in targets), fg.platform.candidates.keys()
    consumers = fg.platform.consumers.get("get_punica_wrapper", set())
    assert "vllm/lora/punica_wrapper/punica_selector.py" in consumers


def test_spawn_leaf_edges_not_helper(fg):
    entry = fg.spawn.entrypoint_file
    assert entry and entry.startswith("vllm/entrypoints/cli/")
    importers = fg.graph.reverse.get(entry, set())
    spawning_tests = {f for f in importers if f.startswith("tests/")}
    assert len(spawning_tests) > 50  # the RemoteOpenAIServer fleet
    assert "tests/utils.py" not in fg.graph.imports or entry not in (
        fg.graph.imports.get("tests/utils.py", set())
    ), "helper file must not be edged (amplifier)"


def test_asset_jinja_selects_tool_use(fg):
    tf = _test_files(fg.graph, "examples/tool_chat_template_hermes.jinja")
    assert any(f.startswith("tests/tool_use/") for f in tf), tf


def test_asset_edge_end_to_end_selection(state):
    sel = select(state, ["examples/tool_chat_template_hermes.jinja"])
    assert "vllm_ci" not in sel.run_all
    assert any(
        "tool" in s or "entrypoints" in s or "rust" in s for s in sel.selected
    ), sorted(sel.selected)


def test_conftest_server_fixture_suites_engine_starting(fg, vllm_repo):
    """Suites whose engine boot lives in a conftest server fixture count as
    engine-starting; the root-conftest exclude keeps the gate meaningful."""
    for conftest in (
        "tests/entrypoints/openai/responses/conftest.py",
        "tests/tool_use/conftest.py",
    ):
        src = (vllm_repo / conftest).read_text()
        assert "RemoteOpenAIServer" in src, (
            f"{conftest} no longer defines a server fixture: update specimen"
        )
    eng = fg.engine_starting_tests()
    assert any(f.startswith("tests/entrypoints/openai/responses/test_") for f in eng)
    assert any(f.startswith("tests/tool_use/") and "/test_" in f for f in eng)
    from ci_selector.codemap.repo import test_file_catalog

    catalog = test_file_catalog(vllm_repo)
    assert len(eng & set(catalog)) < 0.5 * len(catalog), (
        "engine-starting set covers most of the suite: a world-scale "
        "conftest is amplifying; extend CONFTESTS_NOT_ENGINE_STARTING"
    )


@pytest.mark.drift
def test_root_conftest_exclusion_is_still_load_bearing(fg):
    """Anti-vacuity for the one exclusion that can silently stop mattering.

    The exclusion only does anything while the named conftest really does
    import an engine entrypoint. If it stops, the entry is dead weight, and a
    dead entry pre-approves whatever that conftest imports next.
    """
    from ci_selector.handwritten import (
        CONFTESTS_NOT_ENGINE_STARTING,
        ENGINE_ENTRY_MODULES,
    )

    entry_files = {f for m in ENGINE_ENTRY_MODULES if (f := fg.index.resolve(m))}
    for conftest in CONFTESTS_NOT_ENGINE_STARTING:
        imports = fg.graph.imports.get(conftest)
        assert imports is not None, drift_message(
            f"{conftest} is excluded from the engine-starting gate but is not "
            "in the graph at all.",
            "The exclusion protects nothing, and the gate it protects decides "
            "worker-seam reachability for the whole suite.",
            f"the conftest moved: update CONFTESTS_NOT_ENGINE_STARTING in {HW}",
            f"it is gone: delete the entry from {HW}",
        )
        assert imports & entry_files, drift_message(
            f"{conftest} no longer imports an engine entrypoint, so excluding "
            "it changes nothing.",
            "The entry exists because that conftest booting an engine would "
            "make the entire suite look engine-starting. A dead entry keeps "
            "that exemption standing for whatever it imports next.",
            f"delete the entry from CONFTESTS_NOT_ENGINE_STARTING in {HW}; the "
            "gate then discriminates on its own",
        )


def test_vllm_runner_fixture_channel_populated(fg, vllm_repo):
    """engine_starting_tests has two independent channels. The test above only
    exercises the conftest one, so a rename of the vllm_runner fixture would
    empty this one with the suite green -- and an empty channel over-subtracts
    at the worker seam."""
    fixture_files = fg.graph.engine_fixture_files
    assert len(fixture_files) > 50, len(fixture_files)
    assert fixture_files <= fg.engine_starting_tests()
    specimen = next(iter(sorted(fixture_files)))
    assert "vllm_runner" in (vllm_repo / specimen).read_text(), (
        f"{specimen} no longer takes the fixture: detection drifted"
    )


def test_relative_spawner_import_is_engine_starting(fg, vllm_repo):
    """`from ...utils import RemoteOpenAIServer` binds the same name as the
    absolute form; missing it drops the test out of the seam gate."""
    import ast

    relative = []
    for f in fg.graph.engine_fixture_files | set(fg.engine_starting_tests()):
        if not f.startswith("tests/"):
            continue
        try:
            tree = ast.parse((vllm_repo / f).read_text())
        except (OSError, SyntaxError):
            continue
        if any(
            isinstance(n, ast.ImportFrom)
            and n.level
            and any(a.name == "RemoteOpenAIServer" for a in n.names)
            for n in ast.walk(tree)
        ):
            relative.append(f)
    assert relative, "no test imports a spawner relatively any more: specimen drifted"
    assert set(relative) <= fg.engine_starting_tests(), sorted(
        set(relative) - fg.engine_starting_tests()
    )[:5]


@pytest.mark.drift
def test_lazy_export_scan_finds_every_table_not_just_the_named_ones(fg):
    """The parser is aimed at a SHAPE (module-level __getattr__ plus an
    all-string dict), so every lazy-export table is read without anyone typing
    its path. Two of these were previously invisible and their dynamic imports
    were vouched for by a hand list instead."""
    counts = fg.factories.lazy_table_counts
    assert set(counts) == {
        "vllm/transformers_utils/configs/__init__.py",
        "vllm/transformers_utils/processors/__init__.py",
        "vllm/utils/humming.py",
        "vllm/models/inkling/amd/ops/__init__.py",
        "vllm/models/inkling/nvidia/ops/__init__.py",
    }, drift_message(
        f"The lazy-export scan found a different set of tables: {sorted(counts)}",
        "These tables are how `from X import Name` reaches a module the static "
        "graph cannot see. A missed table means those modules lose their "
        "importer.",
        "a table was added or removed in vLLM: update the expected set here",
        "the scan stopped matching: check _lazy_export_table in "
        "ci_selector/codemap/graph/factories.py still recognises the shape",
    )
    assert all(counts.values()), drift_message(
        f"A lazy-export table parsed zero entries: {counts}",
        "An empty table reads exactly like a clean run while its modules lose "
        "their importer.",
        "check the table's values still look like module paths",
    )


def test_lazy_export_bare_stem_reaches_the_sibling_kernel(fg):
    """A bare stem is a relative import, so it can only reach a sibling. That
    proof is what lets these resolve; before it the inkling kernels had no
    importer at all."""
    for side, kernel in (("amd", "silu_and_mul"), ("nvidia", "silu_and_mul")):
        src = f"vllm/models/inkling/{side}/mlp.py"
        dst = f"vllm/models/inkling/{side}/ops/{kernel}.py"
        assert dst in fg.graph.imports.get(src, set()), (src, dst)
    assert "vllm/models/inkling/amd/ops/fa4_rel_attention.py" in fg.graph.imports.get(
        "tests/models/inkling/rocm/test_rel_attention.py", set()
    )


def test_lazy_export_table_of_external_values_still_counts_as_read(fg):
    """humming's every value leaves the repo, so zero edges is the complete
    answer. Reading the table is what makes its dynamic import accounted for;
    requiring an edge would push it back onto a hand list."""
    assert fg.factories.lazy_table_counts["vllm/utils/humming.py"]
    assert "vllm/utils/humming.py" in fg.graph.table_files
    assert not any(
        t.startswith("humming") for t in fg.factories.class_table_entries.values()
    )


def test_class_module_tables_parsed(fg, vllm_repo):
    """_CLASS_TO_MODULE tables are AnnAssign (annotated) assignments; the
    parser must read them, not just plain Assign."""
    import regex as _re

    total = 0
    for init_file in (
        "vllm/transformers_utils/configs/__init__.py",
        "vllm/transformers_utils/processors/__init__.py",
    ):
        src = (vllm_repo / init_file).read_text()
        assert "_CLASS_TO_MODULE" in src, f"{init_file}: table renamed"
        entries = _re.findall(r'"(\w+)":\s*\(?\s*"([\w.]+)"', src)
        assert entries, f"{init_file}: no parseable entries"
        total += sum(1 for _c, mod in entries if mod.startswith("vllm."))
    parsed = fg.factories.class_table_entries
    assert len(parsed) >= 0.9 * total, (len(parsed), total)
    assert parsed.get("MedusaConfig") == ("vllm/transformers_utils/configs/medusa.py")


def test_quant_table_oracle(fg, vllm_repo):
    src = (
        vllm_repo / "vllm/model_executor/layers/quantization/__init__.py"
    ).read_text()
    oracle = {m for m, _cls in re.findall(r'"([\w.\-]+)":\s*(\w*Config)\b', src)}
    assert len(oracle) >= 20, "quant table reshaped: update oracle regex"
    assert oracle <= set(fg.quant.methods), oracle - set(fg.quant.methods)


def test_hf_ids_oracle(fg, vllm_repo):
    n = (vllm_repo / "tests/models/registry.py").read_text().count("_HfExamplesInfo(")
    assert n > 100, "registry reshaped: update oracle"
    assert len(fg.registry.hf_ids) >= 0.8 * n, (len(fg.registry.hf_ids), n)


@pytest.mark.drift
def test_engine_entry_modules_resolve(fg):
    from ci_selector.handwritten import ENGINE_ENTRY_MODULES

    for m in ENGINE_ENTRY_MODULES:
        assert fg.index.resolve(m), drift_message(
            f"ENGINE_ENTRY_MODULES names a module that no longer resolves: {m}",
            "These anchor the worker-seam gate, which decides whether a test "
            "that only reaches a file through an engine boot really depends on "
            "it. A dead anchor turns the gate off.",
            "the module moved: update ENGINE_ENTRY_MODULES in " + HW,
        )


@pytest.mark.drift
def test_cli_entrypoint_matches_console_script(vllm_repo):
    """`vllm ...` in a job script is an edge to whatever the console script
    points at, and we hardcode that target."""
    from ci_selector.codemap.graph.spawn import CLI_ENTRYPOINT_MODULE

    cost = (
        "A job running the `vllm` CLI reaches its code through this module. "
        "Point at the wrong one and those jobs stop being selected on CLI "
        "changes."
    )
    src = (vllm_repo / "pyproject.toml").read_text()
    m = re.search(r'^vllm\s*=\s*"([\w.]+):', src, re.MULTILINE)
    assert m, drift_message(
        "pyproject.toml no longer declares a `vllm` console script in the "
        "shape this test reads.",
        cost,
        "the entry moved or changed form: update the pattern here, then check "
        "CLI_ENTRYPOINT_MODULE in ci_selector/codemap/graph/spawn.py",
    )
    assert m.group(1) == CLI_ENTRYPOINT_MODULE, drift_message(
        f"The `vllm` console script now points at {m.group(1)}, but we "
        f"hardcode {CLI_ENTRYPOINT_MODULE}.",
        cost,
        "set CLI_ENTRYPOINT_MODULE in ci_selector/codemap/graph/spawn.py to "
        f"{m.group(1)}",
    )
