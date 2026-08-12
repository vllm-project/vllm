# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import graph vs. the real checkout, each check re-derived independently.

Covers: pytest-prepend bare imports, conftest fan-in (modeled, wide by
design), examples as script nodes (not modules), and installable packages
under tests/.
"""

import time

import pytest
import regex as re
from ci_analyzer.graph.build import build_full_graph

COLD_BUILD_BUDGET_S = 60


@pytest.fixture(scope="module")
def graph_and_index(repo):
    t0 = time.monotonic()
    fg = build_full_graph(repo)
    elapsed = time.monotonic() - t0
    assert elapsed < COLD_BUILD_BUDGET_S, f"cold build took {elapsed:.1f}s"
    return fg.graph, fg.index, elapsed


def test_no_parse_errors(graph_and_index):
    graph, _, _ = graph_and_index
    assert graph.parse_errors == []


def test_function_local_bare_import_records_its_ambiguity(tmp_path):
    """An import inside a function, where the name matches two files, used to
    crash the whole build. Both spellings are here because each takes its own
    code path."""
    from ci_analyzer.graph.imports import build_graph
    from ci_analyzer.repo import build_module_index

    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm/__init__.py").write_text("")
    (tmp_path / "tests/sub").mkdir(parents=True)
    (tmp_path / "tests/utils.py").write_text("X = 1\n")
    (tmp_path / "tests/sub/utils.py").write_text("Y = 2\n")
    (tmp_path / "tests/sub/test_plain.py").write_text("def f():\n    import utils\n")
    (tmp_path / "tests/sub/test_from.py").write_text(
        "def f():\n    from utils import Y\n"
    )

    graph = build_graph(tmp_path, build_module_index(tmp_path))

    clashes = {(f, name) for f, name, _sib, _other in graph.ambiguities}
    assert ("tests/sub/test_plain.py", "utils") in clashes, graph.ambiguities
    assert ("tests/sub/test_from.py", "utils") in clashes, graph.ambiguities
    # the edge itself still defers, which is the whole point of the sink
    assert ("tests/sub/test_plain.py", "tests/sub/utils.py") in graph.lazy_imports


def test_bare_import_resolves_to_sibling_not_top_level(graph_and_index, repo):
    """tests/v1/determinism has no __init__.py; `from utils import ...`
    means the SIBLING utils.py under pytest prepend semantics."""
    graph, _, _ = graph_and_index
    det_dir = repo / "tests/v1/determinism"
    assert not (det_dir / "__init__.py").exists(), (
        "determinism dir gained __init__.py; bare-import fixture invalid"
    )
    bare_importers = [
        f.name
        for f in det_dir.glob("test_*.py")
        if re.search(r"^from utils import", f.read_text(), re.M)
    ]
    assert bare_importers, "no bare importers left; pick a new fixture case"
    for name in bare_importers:
        file = f"tests/v1/determinism/{name}"
        imports = graph.imports[file]
        assert "tests/v1/determinism/utils.py" in imports, file
        # tests/utils.py may appear only via an explicit `from tests.utils
        # import`, never from wrongly resolving the bare `utils`.
        explicitly_imports_tests_utils = re.search(
            r"^from tests\.utils import|^import tests\.utils",
            (det_dir / name).read_text(),
            re.M,
        )
        if not explicitly_imports_tests_utils:
            assert "tests/utils.py" not in imports, (
                f"{file} wrongly resolved bare `utils` to tests/utils.py"
            )


def test_reverse_closure_reaches_direct_importers(graph_and_index, repo):
    graph, _, _ = graph_and_index
    target = "vllm/lora/punica_wrapper/punica_gpu.py"
    oracle = {
        f"tests/lora/{f.name}"
        for f in (repo / "tests/lora").glob("*.py")
        if "punica_wrapper.punica_gpu" in f.read_text()
    }
    assert oracle, "oracle empty; punica import moved, update fixture"
    closure = graph.reverse_closure({target})
    assert oracle <= closure


def test_conftest_chain_edges(graph_and_index, repo):
    graph, _, _ = graph_and_index
    test_files = sorted(
        p.relative_to(repo).as_posix()
        for p in (repo / "tests/models/multimodal").rglob("test_*.py")
    )
    assert test_files
    sample = test_files[0]
    assert "tests/conftest.py" in graph.imports[sample]
    assert "tests/models/multimodal/conftest.py" in graph.imports[sample]
    # Root conftest's reverse closure must cover every collected test file.
    closure = graph.reverse_closure({"tests/conftest.py"})
    n_tests = sum(1 for f in closure if f.rsplit("/", 1)[-1].startswith("test_"))
    assert n_tests > 1000


def test_config_fan_in_at_least_grep_oracle(graph_and_index, repo):
    graph, _, _ = graph_and_index
    # Column-0 imports only: indented ones are TYPE_CHECKING-guarded or
    # function-local, which the graph deliberately treats differently.
    pat = re.compile(r"^(?:from vllm\.config[. ]|import vllm\.config)", re.M)
    oracle = {
        p.relative_to(repo).as_posix()
        for p in (repo / "vllm").rglob("*.py")
        if pat.search(p.read_text())
    }
    importers = graph.reverse.get("vllm/config/__init__.py", set())
    missing = oracle - importers - {"vllm/config/__init__.py"}  # no self-edge
    assert not missing, sorted(missing)[:10]


def test_installable_test_package_mapped(graph_and_index, repo):
    graph, index, _ = graph_and_index
    target = index.resolve("vllm_test_utils")
    assert target and target.startswith("tests/vllm_test_utils/")
    importer = next(
        (
            p.relative_to(repo).as_posix()
            for p in (repo / "tests").rglob("test_*.py")
            if re.search(r"^(?:from|import) vllm_test_utils", p.read_text(), re.M)
        ),
        None,
    )
    assert importer, "no vllm_test_utils importer found; update fixture"
    assert target in graph.imports[importer]


def test_examples_are_script_nodes_not_modules(graph_and_index, repo):
    graph, index, _ = graph_and_index
    assert not any(f.startswith("examples/") for f in index.file_to_module)
    vllm_importer = next(
        p.relative_to(repo).as_posix()
        for p in sorted((repo / "examples").rglob("*.py"))
        if re.search(r"^(?:from|import) vllm\b", p.read_text(), re.M)
    )
    assert any(dst.startswith("vllm/") for dst in graph.imports.get(vllm_importer, ()))


def test_finalize_keeps_leaf_origin_lazy_edges():
    """tests/ and benchmarks/ lazy imports are genuine leaf dependencies and
    materialize; the vllm-side edge is dropped as registry fan-out and recorded."""
    from ci_analyzer.graph.imports import ImportGraph

    g = ImportGraph()
    g.lazy_imports.add(("tests/kernels/test_x.py", "vllm/claimed.py"))
    g.lazy_imports.add(("benchmarks/bench_x.py", "vllm/claimed.py"))
    g.lazy_imports.add(("vllm/registry.py", "vllm/claimed.py"))
    g.finalize_lazy_edges({"vllm/claimed.py"})
    assert "vllm/claimed.py" in g.imports["tests/kernels/test_x.py"]
    assert "vllm/claimed.py" in g.imports["benchmarks/bench_x.py"]
    assert "vllm/registry.py" not in g.imports
    assert g.dropped_lazy == [("vllm/registry.py", "vllm/claimed.py")]
