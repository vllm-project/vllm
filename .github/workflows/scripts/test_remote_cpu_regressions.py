# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the stacked-PR CPU gate against silently omitting complete suites."""

from pathlib import Path

from remote_cpu_regressions import EXTERNAL_REGRESSIONS, regression_nodes


def test_every_remote_suite_is_selected_once_in_the_job_matrix():
    control = set(regression_nodes("control"))
    native = set(regression_nodes("native"))
    assert not control & native
    assert control | native == set(regression_nodes())
    assert {
        str(path) for path in Path("tests/v1/spec_decode/remote").glob("test_*.py")
    } <= control | native


def test_new_control_suite_is_automatically_included(tmp_path):
    directory = tmp_path / "tests/v1/spec_decode/remote"
    directory.mkdir(parents=True)
    (directory / "test_new_queue.py").touch()
    assert "tests/v1/spec_decode/remote/test_new_queue.py" in regression_nodes(
        "control", tmp_path
    )


def test_external_regressions_follow_actual_branch_definitions(tmp_path):
    for name, tests in EXTERNAL_REGRESSIONS.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {tests[0]}():\n    pass\n")
    nodes = regression_nodes("native", tmp_path)
    assert nodes == [
        f"{name}::{tests[0]}" for name, tests in EXTERNAL_REGRESSIONS.items()
    ]
