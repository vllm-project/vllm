# SPDX-License-Identifier: Apache-2.0
import re

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig,
                     files: Files):
    gh_icon = ":octicons-mark-github-16:"
    gh_url = "https://github.com"
    repo_url = f"{gh_url}/vllm-project/vllm"
    org_url = f"{gh_url}/orgs/vllm-project"
    urls = {
        "issue": f"{repo_url}/issues",
        "pr": f"{repo_url}/pull",
        "project": f"{org_url}/projects",
        "dir": f"{repo_url}/tree/main",
        "file": f"{repo_url}/blob/main",
    }
    titles = {
        "issue": "Issue #",
        "pr": "Pull Request #",
        "project": "Project #",
        "dir": "",
        "file": "",
    }

    scheme = r"gh-(?P<type>.+?):(?P<path>.+?)(#(?P<fragment>.+?))?"
    inline_link = re.compile(r"\[(?P<title>[^\[]+?)\]\(" + scheme + r"\)")
    auto_link = re.compile(f"<{scheme}>")

    def replace_inline_link(match: re.Match) -> str:
        url = f'{urls[match.group("type")]}/{match.group("path")}'
        if fragment := match.group("fragment"):
            url += f"#{fragment}"

        return f'[{gh_icon} {match.group("title")}]({url})'

    def replace_auto_link(match: re.Match) -> str:
        type = match.group("type")
        path = match.group("path")
        title = f"{titles[type]}{path}"
        url = f"{urls[type]}/{path}"
        if fragment := match.group("fragment"):
            url += f"#{fragment}"

        return f"[{gh_icon} {title}]({url})"

    markdown = inline_link.sub(replace_inline_link, markdown)
    markdown = auto_link.sub(replace_auto_link, markdown)

    return markdown
