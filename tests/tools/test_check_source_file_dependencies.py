# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the dead ``source_file_dependencies`` pre-commit check."""

import textwrap
from pathlib import Path

import yaml


def test_dead_paths(monkeypatch):
    # The check imports its sibling checks by bare name, as pre-commit runs it.
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "tools" / "pre_commit")
    )
    from check_source_file_dependencies import dead_paths

    pipeline = yaml.safe_load(
        textwrap.dedent(
            """
            steps:
            - label: job
              source_file_dependencies:
              - vllm/  # live: a prefix of a tracked file
              - tests/b/test_x.py  # live: a tracked file
              - vllm/gone.py  # dead
              - tests/gone/  # dead
              - "!vllm/excluded/"  # dead: the "!" is ignored
              mirror:
                amd:
                  source_file_dependencies:
                  - vllm/also_gone.py  # dead, in a nested block
            """
        )
    )
    assert dead_paths(pipeline, ["vllm/a.py", "tests/b/test_x.py"]) == [
        "vllm/gone.py",
        "tests/gone/",
        "vllm/excluded/",
        "vllm/also_gone.py",
    ]
