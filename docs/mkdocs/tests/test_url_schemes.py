# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

ROOT_DIR = Path(__file__).resolve().parents[3]
REPO_URL = "https://github.com/vllm-project/vllm"


def render(markdown, version_type="tag", identifier="v0.31.0", version_name="v0.31.0"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("READTHEDOCS")}
    env["READTHEDOCS_VERSION_NAME"] = version_name
    if version_type is not None:
        env["READTHEDOCS_VERSION_TYPE"] = version_type
    if identifier is not None:
        env["READTHEDOCS_GIT_IDENTIFIER"] = identifier
    return subprocess.run(
        [sys.executable, "-m", "docs.mkdocs.tests.test_url_schemes", "--render"],
        cwd=ROOT_DIR,
        env=env,
        input=markdown,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        (
            "[File](../../vllm/config/vllm.py#L10)",
            "blob/v0.31.0/vllm/config/vllm.py#L10",
        ),
        ("[Directory](../../vllm/config/)", "tree/v0.31.0/vllm/config"),
        (
            f"[File]({REPO_URL}/blob/main/vllm/config/vllm.py?plain=1#L10-L20)",
            "blob/v0.31.0/vllm/config/vllm.py?plain=1#L10-L20",
        ),
        (
            f"[Directory]({REPO_URL}/tree/main/vllm/config/)",
            "tree/v0.31.0/vllm/config/",
        ),
        (
            f"[File][source]\n\n[source]: {REPO_URL}/blob/main/README.md",
            "blob/v0.31.0/README.md",
        ),
        (f"<{REPO_URL}/blob/main/README.md>", "blob/v0.31.0/README.md"),
    ],
)
def test_release_source_links_use_tag(markdown, expected):
    link = HTMLParser(render(markdown)).css_first("a")
    assert link.attributes["href"] == f"{REPO_URL}/{expected}"


def test_source_autolink_displays_release_tag():
    document = HTMLParser(render(f"Source <{REPO_URL}/tree/main/examples>."))
    assert document.css_first("a").text() == f"{REPO_URL}/tree/v0.31.0/examples"


@pytest.mark.parametrize("tag", ["v0.31.0", "v0.32.0"])
def test_stable_documentation_uses_actual_git_tag(tag):
    html = render(
        f"[File]({REPO_URL}/blob/main/README.md)", identifier=tag, version_name="stable"
    )
    assert HTMLParser(html).css_first("a").attributes["href"] == (
        f"{REPO_URL}/blob/{tag}/README.md"
    )


@pytest.mark.parametrize(
    ("version_type", "identifier"),
    [(None, None), ("branch", "main"), ("external", "62669")],
)
def test_development_builds_keep_main_source_links(version_type, identifier):
    html = render("[File](../../vllm/config/vllm.py)", version_type, identifier)
    assert HTMLParser(html).css_first("a").attributes["href"] == (
        f"{REPO_URL}/blob/main/vllm/config/vllm.py"
    )


def test_release_preserves_other_link_targets_and_code():
    urls = [
        f"{REPO_URL}/blob/v0.30.0/README.md",
        f"{REPO_URL}/issues/62669",
        f"{REPO_URL}/blob/main-other/README.md",
        "https://github.com/another/project/blob/main/README.md",
    ]
    markdown = "\n\n".join(f"[Link]({url})" for url in urls)
    markdown += f"\n\n`{REPO_URL}/blob/main/README.md`"
    markdown += f"\n\n```text\n{REPO_URL}/blob/main/README.md\n```"
    document = HTMLParser(render(markdown))
    assert [a.attributes["href"] for a in document.css("a")] == urls
    assert [node.text() for node in document.css("code")] == [
        f"{REPO_URL}/blob/main/README.md",
        f"{REPO_URL}/blob/main/README.md\n",
    ]


def test_published_fusions_source_references_use_release_tag():
    markdown = (ROOT_DIR / "docs/design/fusions.md").read_text(encoding="utf-8")
    document = HTMLParser(render(markdown))
    hrefs = [a.attributes["href"] for a in document.css("a")]
    assert (
        f"{REPO_URL}/blob/v0.31.0/"
        "vllm/compilation/passes/fusion/allreduce_rms_fusion.py"
    ) in hrefs
    assert not any(url.startswith(f"{REPO_URL}/blob/main/") for url in hrefs)


@pytest.mark.parametrize("identifier", [None, ""])
def test_release_build_requires_tag(identifier):
    with pytest.raises(subprocess.CalledProcessError) as error:
        render("[File](../../vllm/config/vllm.py)", identifier=identifier)
    assert "READTHEDOCS_GIT_IDENTIFIER" in error.value.stderr


if __name__ == "__main__":
    from markdown import Markdown
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import File, Files
    from mkdocs.structure.pages import Page

    from docs.mkdocs.hooks import url_schemes

    config = MkDocsConfig()
    config.load_dict(
        {"site_name": "Source links", "markdown_extensions": ["fenced_code"]}
    )
    url_schemes.on_config(config)
    source = File("design/fusions.md", str(ROOT_DIR / "docs"), "site", True)
    markdown = url_schemes.on_page_markdown(
        sys.stdin.read(),
        page=Page("Fusion passes", source, config),
        config=config,
        files=Files([source]),
    )
    print(Markdown(extensions=config.markdown_extensions).convert(markdown))
