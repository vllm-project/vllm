# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoding/syntax tolerance: parsers record parse_errors and continue
instead of killing the build (preflight escalates what came back empty)."""

from ci_selector.codemap.graph.factories import FactoryParse, add_register_call_edges
from ci_selector.codemap.graph.imports import ImportGraph
from ci_selector.codemap.graph.model_registry import add_registry_edges
from ci_selector.codemap.pipeline.scripts import scan_script
from ci_selector.codemap.pipeline.step import Step
from ci_selector.codemap.pipeline.targets import map_step
from ci_selector.codemap.repo import ModuleIndex
from ci_selector.handwritten import REGISTRY_FILE


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


def _scan(tmp_path, body: str):
    (tmp_path / "outer.sh").write_text(body)
    step = Step(
        pipeline="t",
        source_file="x.yaml",
        label="s",
        key="s",
        group=None,
        commands=["bash outer.sh"],
        source_file_dependencies=None,
    )
    return map_step(tmp_path, step, script_scanner=scan_script)


def test_container_payload_paths_are_not_stale_targets(tmp_path):
    """A path named inside a container payload is relative to the image, so
    when it does not exist here it is reported rather than escalated. Without
    this the NPU steps run on every PR for a path nobody can fix from here."""
    st = _scan(tmp_path, "docker run img bash -c '\n  pytest tests/gone/\n'\n")
    assert st.container_tests == ["tests/gone/"]
    assert not st.dangling


def test_a_real_rename_inside_a_payload_is_knowingly_missed(tmp_path):
    """The accepted cost, pinned so nobody "fixes" it by accident. A renamed
    path and an image-only path are identical from this checkout: both simply
    do not resolve. Escalating both would tax every PR, so neither escalates,
    and a genuine rename inside a payload goes uncaught until someone reads
    the warning. Outside a payload the same rename still escalates."""
    st = _scan(tmp_path, "docker run img bash -c '\n  pytest tests/renamed.py\n'\n")
    assert st.container_tests == ["tests/renamed.py"] and not st.dangling

    outside = _scan(tmp_path, "pytest tests/renamed.py\n")
    assert outside.dangling == ["tests/renamed.py"] and not outside.container_tests


def test_container_payload_still_yields_targets_that_do_exist(tmp_path):
    """The vllm_repo is usually mounted into the image, so a payload path that does
    resolve is a real target. The Arm CPU steps get all their coverage this
    way, and reclassifying the whole payload would have deleted it."""
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "here.py").write_text("def test_x(): pass\n")
    st = _scan(tmp_path, 'docker exec c bash -c "\n  pytest tests/here.py\n"\n')
    assert any(t.path == "tests/here.py" for t in st.targets)
    assert not st.container_tests and not st.dangling


def test_local_pytest_after_a_closed_payload_is_unaffected(tmp_path):
    """The payload must not swallow the rest of the file. Suppressing past its
    closing quote would silently drop real targets."""
    st = _scan(tmp_path, "docker run img bash -c 'echo hi'\npytest tests/gone/\n")
    assert st.dangling == ["tests/gone/"]
    assert not st.container_tests


def test_unterminated_payload_falls_back_to_local(tmp_path):
    """No closing quote means no match, so nothing is reclassified. The failure
    direction is a spurious dangling (over-selection), never a silent drop."""
    st = _scan(tmp_path, "docker run img bash -c '\n  pytest tests/gone/\n")
    assert st.dangling == ["tests/gone/"]
    assert not st.container_tests
