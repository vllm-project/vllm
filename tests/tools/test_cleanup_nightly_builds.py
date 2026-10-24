# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".buildkite/scripts/cleanup-nightly-builds.sh"
)
pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="requires jq")


def run_cleanup(
    tmp_path: Path, pages: list
) -> tuple[subprocess.CompletedProcess, list]:
    """Run the real CLI with an offline curl executable and real jq."""
    (tmp_path / "pages.json").write_text(json.dumps(pages))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        """
import json, os, sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

root = Path(os.environ["CLEANUP_FIXTURE_ROOT"])
url = next(arg for arg in sys.argv[1:] if arg.startswith("https://"))
assert url.startswith("https://hub.docker.com/")
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps(url) + "\\n")
if url.endswith("/users/login"):
    print('{"token":"fixture"}')
elif "DELETE" in sys.argv:
    print("{}")
else:
    page = int(parse_qs(urlparse(url).query)["page"][0])
    pages = json.loads((root / "pages.json").read_text())
    response = {"results": [], "next": None}
    if page <= len(pages):
        response = pages[page - 1]
    if response == "http-error":
        sys.exit(22)
    print(response if isinstance(response, str) else json.dumps(response))
"""
    )
    curl.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), "nightly-", "fixture/repo", "nightly-rocm"],
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CLEANUP_FIXTURE_ROOT": str(tmp_path),
            "DOCKERHUB_USERNAME": "fixture",
            "DOCKERHUB_TOKEN": "fixture",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    return result, calls


def tag(name: str, index: int = 0) -> dict:
    return {"name": name, "last_updated": f"2026-10-19T00:00:{index:02d}Z"}


@pytest.mark.parametrize("layout", [(16,), (0, 16), (8, 0, 8), (0,)])
def test_cleanup_reads_all_pages_and_preserves_protected_tags(tmp_path, layout):
    protected = [
        tag(name)
        for name in [
            "nightly-x86_64",
            "nightly-aarch64",
            "nightly-dev-build",
            "nightly-rocm-build",
        ]
    ]
    pages = []
    index = 0
    for page, count in enumerate(layout, 1):
        rows = [tag(f"nightly-{i:02d}", i) for i in range(index, index + count)]
        index += count
        rows += protected
        more = page < len(layout)
        if more:
            rows += [tag(f"release-{i}") for i in range(100 - len(rows))]
        pages.append({"results": rows, "next": "next-page" if more else None})

    result, calls = run_cleanup(tmp_path, pages)
    assert result.returncode == 0, result.stderr
    fetched = [url for url in calls if "?page=" in url]
    assert len(fetched) == len(layout)
    deleted = [url.rsplit("/", 1)[-1] for url in calls if "/tags/" in url]
    assert set(deleted) == {f"nightly-{i:02d}" for i in range(max(0, index - 14))}
    assert len(deleted) == len(set(deleted))


@pytest.mark.parametrize("bad_page", ["http-error", "not-json", {"detail": "error"}])
def test_cleanup_does_not_delete_after_a_page_fetch_fails(tmp_path, bad_page):
    first = {
        "results": [tag(f"nightly-{i:02d}", i) for i in range(16)],
        "next": "next-page",
    }
    result, calls = run_cleanup(tmp_path, [first, bad_page])
    assert result.returncode != 0
    assert not any("/tags/" in url for url in calls)
