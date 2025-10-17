# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MkDocs hook to enable the following links to render correctly:

- Relative file links outside of the `docs/` directory, e.g.:
    - [Text](../some_file.py)
    - [Directory](../../some_directory/)
- GitHub URLs for issues, pull requests, and projects, e.g.:
    - Adds GitHub icon before links
    - Replaces raw links with descriptive text,
        e.g. <...pull/123> -> [Pull Request #123](.../pull/123)
    - Works for external repos too by including the `owner/repo` in the link title

The goal is to simplify cross-referencing common GitHub resources
in project docs.
"""

from pathlib import Path

import regex as re
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
DOC_DIR = ROOT_DIR / "docs"


gh_icon = ":octicons-mark-github-16:"

# Regex pieces
TITLE = r"(?P<title>[^\[\]<>]+?)"
REPO = r"(?P<repo>.+?/.+?)"
TYPE = r"(?P<type>issues|pull|projects)"
NUMBER = r"(?P<number>\d+)"
FRAGMENT = r"(?P<fragment>#[^\s]+)?"
URL = f"https://github.com/{REPO}/{TYPE}/{NUMBER}{FRAGMENT}"
RELATIVE = r"(?!(https?|ftp)://|#)(?P<path>[^\s]+?)"

# Common titles to use for GitHub links when none is provided in the link.
TITLES = {"issues": "Issue ", "pull": "Pull Request ", "projects": "Project "}

# Regex to match GitHub issue, PR, and project links with optional titles.
github_link = re.compile(rf"(\[{TITLE}\]\(|<){URL}(\)|>)")
# Regex to match relative file links with optional titles.
relative_link = re.compile(rf"\[{TITLE}\]\({RELATIVE}\)")


def on_page_markdown(
    markdown: str, *, page: Page, config: MkDocsConfig, files: Files
) -> str:
    def replace_relative_link(match: re.Match) -> str:
        """Replace relative file links with URLs if they point outside the docs dir."""
        title = match.group("title")
        path = match.group("path")
        path = (Path(page.file.abs_src_path).parent / path).resolve()

        # Check if the path exists and is outside the docs dir
        if not path.exists() or path.is_relative_to(DOC_DIR):
            return match.group(0)

        # Files and directories have different URL schemes on GitHub
        slug = "tree/main" if path.is_dir() else "blob/main"

        path = path.relative_to(ROOT_DIR)
        url = f"https://github.com/vllm-project/vllm/{slug}/{path}"
        return f"[{gh_icon} {title}]({url})"

    def replace_github_link(match: re.Match) -> str:
        """Replace GitHub issue, PR, and project links with enhanced Markdown links."""
        repo = match.group("repo")
        type = match.group("type")
        number = match.group("number")
        # Title and fragment could be None
        title = match.group("title") or ""
        fragment = match.group("fragment") or ""

        # Use default titles for raw links
        if not title:
            title = TITLES[type]
            if "vllm-project" not in repo:
                title += repo
            title += f"#{number}"

        url = f"https://github.com/{repo}/{type}/{number}{fragment}"
        return f"[{gh_icon} {title}]({url})"

    markdown = relative_link.sub(replace_relative_link, markdown)
    markdown = github_link.sub(replace_github_link, markdown)

    if "interface" in str(page.file.abs_src_path):
        print(markdown)

    return markdown
