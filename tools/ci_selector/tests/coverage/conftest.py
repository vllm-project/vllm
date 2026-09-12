# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures for the coverage half.

`tmp_repo` is a throwaway git repo, the opposite of the session-scoped
`vllm_repo` one directory up. Both were once called `repo`, and a test that
drifted between directories would have got three synthetic files instead of
vLLM and passed against nothing. Hence two names that cannot be confused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import MODULE_SOURCE, Repo


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Repo:
    root = tmp_path / "repo"
    root.mkdir()
    r = Repo(root)
    r.write("vllm/mod.py", MODULE_SOURCE)
    # A second real file, so a row keyed on it is not silently dropped as
    # absent-at-commit and mistaken for a step that recorded nothing.
    r.write("vllm/other.py", "def elsewhere():\n    return 0\n")
    r.commit("base")
    return r
