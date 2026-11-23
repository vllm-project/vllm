# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the agent-file pre-commit check.

The hook passes silently on a healthy tree, so CI would not notice a rule that
stopped firing. Each case breaks one convention in a minimal repo and asserts
the check reports it.
"""

import os
from pathlib import Path

import pytest

from tools.pre_commit.check_agent_files import check

SKILL_MD = ".agents/skills/demo/SKILL.md"
SKILL = "---\nname: demo\ndescription: Demo things. Use when testing.\n---\n"


def write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def tracked() -> dict[str, str]:
    """Mimic `git ls-files -s` for every file and symlink under the cwd."""
    files = {}
    for root, dirs, names in os.walk("."):
        for name in dirs + names:
            path = os.path.normpath(os.path.join(root, name))
            if os.path.islink(path):
                files[path] = "120000"
            elif os.path.isfile(path):
                files[path] = "100644"
    return files


@pytest.fixture(autouse=True)
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write("AGENTS.md", "See the [guide](docs/guide.md#some-heading).\n")
    write("docs/guide.md", "## Some Heading\n")
    write(SKILL_MD, SKILL)
    os.makedirs(".claude/skills")
    os.symlink("../../.agents/skills/demo", ".claude/skills/demo")


def test_valid_repo_passes():
    assert check(tracked()) == ([], [])


@pytest.mark.parametrize(
    ("break_repo", "expected"),
    [
        pytest.param(
            lambda: os.unlink(".claude/skills/demo"),
            "must be a symlink",
            id="skill-not-linked",
        ),
        pytest.param(
            lambda: write(".claude/skills/other/SKILL.md", SKILL),
            "only symlinks",
            id="real-file-in-claude-skills",
        ),
        pytest.param(
            lambda: os.symlink("../../.agents/skills/gone", ".claude/skills/gone"),
            "only symlinks",
            id="dangling-link",
        ),
        pytest.param(
            lambda: write(SKILL_MD, SKILL.replace("demo", "other")),
            "must be 'demo'",
            id="name-differs-from-dir",
        ),
        pytest.param(
            lambda: write(SKILL_MD, SKILL.replace("Demo things", "Demo: things")),
            "not valid YAML",
            id="unquoted-colon-in-description",
        ),
        pytest.param(
            lambda: write(SKILL_MD, "# Demo\n"),
            "missing YAML frontmatter",
            id="no-frontmatter",
        ),
        pytest.param(
            lambda: write("vllm/CLAUDE.md", "@AGENTS.md\n"),
            "remove it",
            id="claude-md",
        ),
        pytest.param(
            lambda: write("AGENTS.md", "line\n" * 201),
            "200-line budget",
            id="root-guide-over-budget",
        ),
        pytest.param(
            lambda: write("vllm/AGENTS.md", "line\n" * 301),
            "300-line budget",
            id="domain-guide-over-budget",
        ),
        pytest.param(
            lambda: write(SKILL_MD, SKILL + "line\n" * 500),
            "500-line budget",
            id="skill-over-budget",
        ),
        pytest.param(
            lambda: write("AGENTS.md", "[gone](docs/gone.md)\n"),
            "does not exist",
            id="dead-link",
        ),
        pytest.param(
            lambda: write(SKILL_MD, SKILL + "[guide](../../../docs/guide.md#gone)\n"),
            "no heading",
            id="dead-anchor",
        ),
    ],
)
def test_broken_convention_is_reported(break_repo, expected):
    break_repo()
    errors, _ = check(tracked())
    assert len(errors) == 1
    assert expected in errors[0]


def test_description_without_trigger_is_advisory():
    write(SKILL_MD, SKILL.replace(" Use when testing.", ""))
    errors, notes = check(tracked())
    assert not errors
    assert "when to use" in notes[0]
