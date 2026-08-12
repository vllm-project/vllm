# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wall parsers (registry flip, factories, platform, spawn, assets).

Asserted names are re-derived from the repo where cheap; a specific known
entry is first proven to exist in the source, so repo drift fails as
"update fixture" rather than as parser skew.
"""

import pytest
import regex as re
from ci_analyzer.select import select


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


def test_registry_pass_through_qualnames(fg, repo):
    src = (repo / "vllm/model_executor/models/registry.py").read_text()
    oracle = set(re.findall(r'"(vllm\.[\w.]+)"', src))
    assert oracle, "no pass-through entries left; update fixture"
    covered = {mod for mod, _cls in fg.registry.entries.values()}
    assert oracle & covered, "pass-through qualname entries were dropped"


def test_kv_connector_register_calls(fg, repo):
    factory_src = (
        repo / "vllm/distributed/kv_transfer/kv_connector/factory.py"
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


def test_lazy_parser_tables_counted_per_table(fg):
    """Every lazy table must parse entries of its own. The merged
    parser_entries dict cannot show this: parser names collide across the four
    tables, so it is last-wins and a dead tokenizers anchor leaves its size
    unchanged."""
    counts = fg.factories.parser_table_counts
    assert len(counts) >= 4, counts
    assert not [a for a, n in counts.items() if not n], (
        f"dead parser anchor(s): {sorted(a for a, n in counts.items() if not n)}"
    )
    assert sum(counts.values()) > len(fg.factories.parser_entries), (
        "parser names no longer collide across the tables; if that is real the "
        "merged emptiness row would suffice -- re-check before deleting"
    )


def test_attention_enum_backends(fg):
    assert "FLASH_ATTN" in fg.factories.enum_entries
    assert "flash_attn" in fg.factories.enum_entries["FLASH_ATTN"]


def test_module_attrs_edges(fg, repo):
    # Stored value is the module part only; an unstripped ":LLM" would make
    # every resolve fail silently (the bug this test previously masked).
    assert fg.factories.module_attrs.get("LLM") == "entrypoints.llm"
    assert fg.factories.module_attr_resolved > 0
    llm_file = "vllm/entrypoints/llm.py"

    specimen = "tests/v1/cudagraph/test_cudagraph_mode.py"
    src = (repo / specimen).read_text()
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


def test_conftest_server_fixture_suites_engine_starting(fg, repo):
    """Suites whose engine boot lives in a conftest server fixture count as
    engine-starting; the root-conftest exclude keeps the gate meaningful."""
    for conftest in (
        "tests/entrypoints/openai/responses/conftest.py",
        "tests/tool_use/conftest.py",
    ):
        src = (repo / conftest).read_text()
        assert "RemoteOpenAIServer" in src, (
            f"{conftest} no longer defines a server fixture: update specimen"
        )
    eng = fg.engine_starting_tests()
    assert any(f.startswith("tests/entrypoints/openai/responses/test_") for f in eng)
    assert any(f.startswith("tests/tool_use/") and "/test_" in f for f in eng)
    from ci_analyzer.repo import test_file_catalog

    catalog = test_file_catalog(repo)
    assert len(eng & set(catalog)) < 0.5 * len(catalog), (
        "engine-starting set covers most of the suite: a world-scale "
        "conftest is amplifying; extend CONFTESTS_NOT_ENGINE_STARTING"
    )


def test_vllm_runner_fixture_channel_populated(fg, repo):
    """engine_starting_tests has two independent channels. The test above only
    exercises the conftest one, so a rename of the vllm_runner fixture would
    empty this one with the suite green -- and an empty channel over-subtracts
    at the worker seam."""
    fixture_files = fg.graph.engine_fixture_files
    assert len(fixture_files) > 50, len(fixture_files)
    assert fixture_files <= fg.engine_starting_tests()
    specimen = next(iter(sorted(fixture_files)))
    assert "vllm_runner" in (repo / specimen).read_text(), (
        f"{specimen} no longer takes the fixture: detection drifted"
    )


def test_relative_spawner_import_is_engine_starting(fg, repo):
    """`from ...utils import RemoteOpenAIServer` binds the same name as the
    absolute form; missing it drops the test out of the seam gate."""
    import ast

    relative = []
    for f in fg.graph.engine_fixture_files | set(fg.engine_starting_tests()):
        if not f.startswith("tests/"):
            continue
        try:
            tree = ast.parse((repo / f).read_text())
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


def test_class_module_tables_parsed(fg, repo):
    """_CLASS_TO_MODULE tables are AnnAssign (annotated) assignments; the
    parser must read them, not just plain Assign."""
    import regex as _re

    total = 0
    for init_file in (
        "vllm/transformers_utils/configs/__init__.py",
        "vllm/transformers_utils/processors/__init__.py",
    ):
        src = (repo / init_file).read_text()
        assert "_CLASS_TO_MODULE" in src, f"{init_file}: table renamed"
        entries = _re.findall(r'"(\w+)":\s*\(?\s*"([\w.]+)"', src)
        assert entries, f"{init_file}: no parseable entries"
        total += sum(1 for _c, mod in entries if mod.startswith("vllm."))
    parsed = fg.factories.class_table_entries
    assert len(parsed) >= 0.9 * total, (len(parsed), total)
    assert parsed.get("MedusaConfig") == ("vllm/transformers_utils/configs/medusa.py")


def test_quant_table_oracle(fg, repo):
    src = (repo / "vllm/model_executor/layers/quantization/__init__.py").read_text()
    oracle = {m for m, _cls in re.findall(r'"([\w.\-]+)":\s*(\w*Config)\b', src)}
    assert len(oracle) >= 20, "quant table reshaped: update oracle regex"
    assert oracle <= set(fg.quant.methods), oracle - set(fg.quant.methods)


def test_hf_ids_oracle(fg, repo):
    n = (repo / "tests/models/registry.py").read_text().count("_HfExamplesInfo(")
    assert n > 100, "registry reshaped: update oracle"
    assert len(fg.registry.hf_ids) >= 0.8 * n, (len(fg.registry.hf_ids), n)


def test_engine_entry_modules_resolve(fg):
    from ci_analyzer.curated import ENGINE_ENTRY_MODULES

    for m in ENGINE_ENTRY_MODULES:
        assert fg.index.resolve(m), f"dead engine entry anchor: {m}"


def test_cli_entrypoint_matches_console_script(repo):
    from ci_analyzer.curated import CLI_ENTRYPOINT_MODULE

    src = (repo / "pyproject.toml").read_text()
    m = re.search(r'^vllm\s*=\s*"([\w.]+):', src, re.MULTILINE)
    assert m, "console script moved: update oracle"
    assert m.group(1) == CLI_ENTRYPOINT_MODULE
