# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoding/syntax tolerance: parsers record parse_errors and continue
instead of killing the build (preflight escalates what came back empty)."""

from ci_analyzer.curated import REGISTRY_FILE
from ci_analyzer.graph.factories import FactoryParse, add_register_call_edges
from ci_analyzer.graph.imports import ImportGraph
from ci_analyzer.graph.registry import add_registry_edges
from ci_analyzer.jobs.model import Step
from ci_analyzer.jobs.scripts import scan_script
from ci_analyzer.jobs.testmap import map_step
from ci_analyzer.repo import ModuleIndex


def test_non_utf8_register_file_recorded_not_fatal(tmp_path):
    bad = tmp_path / "vllm" / "bad.py"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"# -*- coding: latin-1 -*-\n# register_ caf\xe9\n")
    index = ModuleIndex()
    index.add("vllm.bad", "vllm/bad.py")
    graph = ImportGraph()
    add_register_call_edges(tmp_path, index, graph, FactoryParse())
    assert "vllm/bad.py" in graph.parse_errors


def test_broken_registry_yields_empty_parse_and_record(tmp_path):
    reg = tmp_path / REGISTRY_FILE
    reg.parent.mkdir(parents=True)
    reg.write_text("def broken(:\n")
    graph = ImportGraph()
    parse = add_registry_edges(tmp_path, ModuleIndex(), graph)
    assert parse.entries == {}
    assert REGISTRY_FILE in graph.parse_errors


def test_non_utf8_script_lands_in_dangling(tmp_path):
    script = tmp_path / "tests" / "bad.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"echo caf\xe9\n")
    step = Step(
        pipeline="t",
        source_file="x.yaml",
        label="s",
        key="s",
        group=None,
        commands=["bash bad.sh"],
        source_file_dependencies=None,
    )
    st = map_step(tmp_path, step, script_scanner=scan_script)
    assert "tests/bad.sh" in st.dangling


def test_pytest_and_chained_script_both_captured(tmp_path):
    """`pytest a && bash next.sh` inside a script: the pytest arg is cut at the
    `&&` and the chained script is still scanned (no early continue)."""
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "foo.py").write_text("def test_x(): pass\n")
    (tmp_path / "tests" / "next.sh").write_text("echo done\n")
    (tmp_path / "outer.sh").write_text("pytest tests/foo.py && bash tests/next.sh\n")
    step = Step(
        pipeline="t",
        source_file="x.yaml",
        label="s",
        key="s",
        group=None,
        commands=["bash outer.sh"],
        source_file_dependencies=None,
    )
    st = map_step(tmp_path, step, script_scanner=scan_script)
    assert "tests/next.sh" in st.scripts_seen
    assert any(t.path == "tests/foo.py" for t in st.targets)
