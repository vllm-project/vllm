# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the hardcoded `trust_remote_code=True` pre-commit check.

The hook passes silently on a healthy tree, so CI would not notice a rule that
stopped firing. Each case below asserts the check reports (or ignores) one
pattern.
"""

import pytest

from tools.pre_commit.check_trust_remote_code import check_source


@pytest.mark.parametrize(
    "source",
    [
        "AutoModel.from_config(cfg, trust_remote_code=True)\n",
        "get_processor(name, revision=r, trust_remote_code=True)\n",
        "x = f(g(trust_remote_code=True))\n",
        "f(\n    name,\n    trust_remote_code=True,\n)\n",
    ],
)
def test_hardcoded_true_is_reported(source):
    assert len(check_source(source)) == 1


@pytest.mark.parametrize(
    "source",
    [
        "AutoModel.from_config(cfg, trust_remote_code=cfg.trust_remote_code)\n",
        "AutoModel.from_config(cfg, trust_remote_code=False)\n",
        "f(trust_remote_code=trust_remote_code)\n",
        # Default values in signatures are not calls.
        "def f(trust_remote_code=True):\n    pass\n",
        # Docstrings, strings and comments are not code.
        '"""Example: f(trust_remote_code=True)"""\n',
        "msg = 'set trust_remote_code=True to load this model'\n",
        "# f(trust_remote_code=True)\n",
        # Explicit suppression with a reason.
        "f(trust_remote_code=True)  # trust-remote-code: ok (vetted repo)\n",
        # Unparsable files are skipped, not crashed on.
        "def broken(:\n",
    ],
)
def test_safe_code_is_not_reported(source):
    assert check_source(source) == []


def test_violation_points_at_the_keyword_line():
    source = "f(\n    name,\n    trust_remote_code=True,\n)\n"
    [violation] = check_source(source, "a.py")
    assert violation.startswith("a.py:3:")
