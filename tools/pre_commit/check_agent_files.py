# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check agent instruction files and skills against vLLM's conventions.

Agents fail silently on a broken setup: a skill with no `.claude/skills` symlink
or with unparsable frontmatter is simply never offered, and a dead link reads as
missing guidance. See `docs/contributing/editing-agent-instructions.md` for the
rules enforced here.

Usage:
    python tools/pre_commit/check_agent_files.py
"""

import os
import posixpath
import subprocess
import sys
from pathlib import Path

import regex as re
import yaml

SKILLS = ".agents/skills"
CLAUDE_SKILLS = ".claude/skills"
SYMLINK = "120000"
ROOT_GUIDE_MAX_LINES = 200
DOMAIN_GUIDE_MAX_LINES = 300
SKILL_MAX_LINES = 500

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
WHEN_TO_USE = re.compile(r"(?i)\buse\b")
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)")
URL_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*:")
HEADING = re.compile(r"^#+\s+(.*)$", re.MULTILINE)


def tracked_files() -> dict[str, str]:
    """Map each tracked path to its git mode."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "-z"], capture_output=True, text=True, check=True
    ).stdout
    files = {}
    for entry in out.split("\0")[:-1]:
        meta, path = entry.split("\t", 1)
        files[path] = meta.split()[0]
    return files


def link_target(path: str) -> str:
    # Without core.symlinks, git checks a symlink out as a file holding its target.
    return os.readlink(path) if os.path.islink(path) else Path(path).read_text()


def skill_names(files: dict[str, str]) -> list[str]:
    return sorted({p.split("/")[2] for p in files if p.startswith(f"{SKILLS}/")})


def check_skill_links(files: dict[str, str]) -> list[str]:
    errors = []
    skills = skill_names(files)
    for name in skills:
        link = f"{CLAUDE_SKILLS}/{name}"
        target = f"../../{SKILLS}/{name}"
        if files.get(link) != SYMLINK or link_target(link) != target:
            errors.append(f"{link}: must be a symlink to {target}")
    for path, mode in files.items():
        if not path.startswith(f"{CLAUDE_SKILLS}/"):
            continue
        if mode != SYMLINK or path.split("/")[2] not in skills:
            errors.append(f"{path}: only symlinks to {SKILLS}/<name> belong here")
    return errors


def check_skill_frontmatter(files: dict[str, str]) -> tuple[list[str], list[str]]:
    errors, notes = [], []
    for name in skill_names(files):
        path = f"{SKILLS}/{name}/SKILL.md"
        if path not in files:
            errors.append(f"{SKILLS}/{name}: missing SKILL.md")
            continue
        match = FRONTMATTER.match(Path(path).read_text())
        try:
            meta = yaml.safe_load(match.group(1)) if match else None
        except yaml.YAMLError as exc:
            reason = getattr(exc, "problem", None) or exc
            errors.append(f"{path}: frontmatter is not valid YAML ({reason})")
            continue
        if not isinstance(meta, dict):
            errors.append(f"{path}: missing YAML frontmatter")
            continue
        if meta.get("name") != name:
            errors.append(f"{path}: name {meta.get('name')!r} must be {name!r}")
        if not SKILL_NAME.fullmatch(name) or len(name) > 64:
            errors.append(f"{path}: name must be lowercase-hyphenated, <= 64 chars")
        description = meta.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{path}: missing description")
        elif len(description) > 1024:
            errors.append(f"{path}: description exceeds 1024 chars")
        elif not WHEN_TO_USE.search(description):
            notes.append(f"{path}: description should say when to use the skill")
    return errors, notes


def check_no_claude_md(files: dict[str, str]) -> list[str]:
    return [
        f"{path}: remove it; Claude Code reads AGENTS.md, but not where a "
        "CLAUDE.md exists"
        for path in files
        if posixpath.basename(path) == "CLAUDE.md"
    ]


def check_line_budgets(files: dict[str, str]) -> list[str]:
    errors = []
    for path in files:
        if path == "AGENTS.md":
            limit = ROOT_GUIDE_MAX_LINES
        elif posixpath.basename(path) == "AGENTS.md":
            limit = DOMAIN_GUIDE_MAX_LINES
        elif path.startswith(f"{SKILLS}/") and path.endswith("/SKILL.md"):
            limit = SKILL_MAX_LINES
        else:
            continue
        lines = len(Path(path).read_text().splitlines())
        if lines > limit:
            errors.append(f"{path}: {lines} lines exceeds the {limit}-line budget")
    return errors


def anchor_key(text: str) -> str:
    # Anchor slugs differ between renderers, so compare alphanumerics only.
    return re.sub(r"[^a-z0-9]", "", text.lower())


def has_heading(path: str, anchor: str) -> bool:
    headings = HEADING.findall(CODE_BLOCK.sub("", Path(path).read_text()))
    return anchor_key(anchor) in {anchor_key(heading) for heading in headings}


def check_links(files: dict[str, str]) -> list[str]:
    errors = []
    dirs = {str(parent) for p in files for parent in Path(p).parents}
    for path in files:
        in_skill = path.startswith(f"{SKILLS}/") and path.endswith(".md")
        if not in_skill and posixpath.basename(path) != "AGENTS.md":
            continue
        text = CODE_BLOCK.sub("", Path(path).read_text())
        for link in LINK.findall(text):
            if URL_SCHEME.match(link):
                continue
            ref, _, anchor = link.partition("#")
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(path), ref))
            if dest not in files and dest not in dirs:
                errors.append(f"{path}: link {link} does not exist")
            elif anchor and dest.endswith(".md") and not has_heading(dest, anchor):
                errors.append(f"{path}: link {link} has no heading #{anchor}")
    return errors


def check(files: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (errors, advisory notes) for the tracked files under the cwd."""
    errors, notes = check_skill_frontmatter(files)
    errors += check_skill_links(files)
    errors += check_no_claude_md(files)
    errors += check_line_budgets(files)
    errors += check_links(files)
    return errors, notes


def main() -> int:
    errors, notes = check(tracked_files())
    for note in notes:
        print(f"note: {note}")
    if not errors:
        return 0
    print(f"{len(errors)} agent file problem(s):\n", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    print(
        "\nSee docs/contributing/editing-agent-instructions.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
