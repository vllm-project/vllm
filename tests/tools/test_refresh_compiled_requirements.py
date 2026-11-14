# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import zipfile
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import (
    BaseHTTPRequestHandler,
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCKFILES = [
    "requirements/docs.txt",
    "requirements/test/cpu.txt",
    "requirements/test/cuda.txt",
    "requirements/test/rocm.txt",
    "requirements/test/xpu.txt",
]


@pytest.mark.parametrize("timestamps", [True, False])
def test_refresh_excludes_recent_or_undated_archives(tmp_path, timestamps):
    """Both compile passes must exclude fresh releases and fresh same-version wheels."""
    index = tmp_path / "simple" / "cooldown-probe"
    index.mkdir(parents=True)
    now = datetime.now(UTC)
    links = []
    hashes = []
    for version, build, age in [("1.0", "", 8), ("1.0", "1-", 6), ("2.0", "", 6)]:
        wheel = index / f"cooldown_probe-{version}-{build}py3-none-any.whl"
        info = f"cooldown_probe-{version}.dist-info"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("cooldown_probe.py", f"build = {build!r}\n")
            archive.writestr(
                f"{info}/METADATA",
                f"Metadata-Version: 2.1\nName: cooldown-probe\nVersion: {version}\n",
            )
            archive.writestr(
                f"{info}/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr(f"{info}/RECORD", "")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        hashes.append(digest)
        uploaded = (now - timedelta(days=age)).isoformat()
        attribute = f' data-upload-time="{uploaded}"' if timestamps else ""
        links.append(
            f'<a href="{wheel.name}#sha256={digest}"{attribute}>{wheel.name}</a>'
        )
    (index / "index.html").write_text("\n".join(links))
    (tmp_path / "requirements.in").write_text("cooldown-probe\n")
    (tmp_path / "tools").mkdir()
    helper = tmp_path / "tools" / "refresh_compiled_requirements.py"
    shutil.copyfile(ROOT / "tools/refresh_compiled_requirements.py", helper)

    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            hooks = []
            outputs = []
            for suffix in ["", "-rocm", "-xpu", "-cpu", "-docs"]:
                output = f"requirements{suffix}.txt"
                outputs.append(tmp_path / output)
                hooks.append(
                    {
                        "id": "pip-compile",
                        "alias": f"pip-compile{suffix}",
                        "args": [
                            "requirements.in",
                            "-o",
                            output,
                            "--generate-hashes",
                            "--no-config",
                            "--python",
                            sys.executable,
                            "--index-url",
                            f"http://127.0.0.1:{server.server_port}/simple",
                        ],
                    }
                )
            (tmp_path / ".pre-commit-config.yaml").write_text(
                json.dumps(
                    {
                        "repos": [
                            {
                                "repo": "https://github.com/astral-sh/uv-pre-commit",
                                "hooks": hooks,
                            }
                        ],
                    }
                )
            )
            result = subprocess.run(
                [sys.executable, str(helper)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            if timestamps and result.returncode == 0:
                before = outputs[0].read_bytes()
                subprocess.run(
                    ["uv", "pip", "compile", *hooks[0]["args"]],
                    cwd=tmp_path,
                    check=True,
                    capture_output=True,
                )
                assert outputs[0].read_bytes() == before
        finally:
            server.shutdown()
            thread.join()

    if not timestamps:
        assert result.returncode != 0, result.stdout
        assert not any(output.exists() for output in outputs)
        return

    assert result.returncode == 0, result.stderr
    for output in outputs:
        text = output.read_text()
        assert "cooldown-probe==1.0" in text
        assert f"--hash=sha256:{hashes[0]}" in text
        assert hashes[1] not in text
        assert hashes[2] not in text
        assert "--upgrade" not in text


@pytest.fixture
def review_repo(tmp_path):
    """A real Git base and refreshed locks, without installing any target packages."""
    (tmp_path / "tools").mkdir()
    shutil.copyfile(
        ROOT / "tools/review_compiled_requirements.py",
        tmp_path / "tools/review_compiled_requirements.py",
    )
    before = ["shared", "changed", "removed-cuda", "same-hash", "xpu-only"]
    for path, name in zip(LOCKFILES, before):
        lock = tmp_path / path
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"{name}==1.0 --hash=sha256:{'a' * 64}\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "requirements"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Dependency test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Base locks",
        ],
        cwd=tmp_path,
        check=True,
    )
    after = [
        "shared==1.0",
        "changed==2.0",
        "new-cuda==1.0",
        "same-hash==1.0",
        'xpu-only==1.0+vendor; sys_platform == "win32"',
    ]
    for path, requirement in zip(LOCKFILES, after):
        digest = "b" if "same-hash" in requirement else "a"
        (tmp_path / path).write_text(
            f"{requirement} \\\n    --hash=sha256:{digest * 64}\n"
        )
    # Cross the API batch boundary; the XPU-only finding is in the second batch.
    with (tmp_path / LOCKFILES[0]).open("a") as file:
        for i in range(101):
            file.write(f"batch-{i}==1.0 --hash=sha256:{'a' * 64}\n")
    return tmp_path


@pytest.mark.parametrize(
    ("response_kind", "enforce"),
    [
        ("clean", False),
        ("vulnerable", False),
        ("vulnerable", True),
        ("unavailable", False),
        ("unavailable", True),
        ("incomplete", False),
        ("paginated", False),
        ("malformed", False),
    ],
)
def test_review_is_informational_by_default(review_repo, response_kind, enforce):
    """Report all platforms/markers without blocking on findings or service errors."""
    queries = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            batch = payload["queries"]
            queries.extend(batch)
            results: list[dict] = [
                {"vulns": [{"id": "GHSA-test-advisory"}]}
                if response_kind == "vulnerable"
                and query["package"]["name"] == "xpu-only"
                and query["version"] == "1.0"
                else {}
                for query in batch
            ]
            if response_kind == "incomplete":
                results.pop()
            elif response_kind == "paginated":
                results[0] = {"next_page_token": "more"}
            elif response_kind == "malformed":
                results[0] = {"vulns": "invalid"}
            self.send_response(503 if response_kind == "unavailable" else 200)
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}).encode())

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/review_compiled_requirements.py",
                    "--base",
                    "HEAD",
                    "--output-dir",
                    "review",
                    "--osv-url",
                    f"http://127.0.0.1:{server.server_port}/v1/querybatch",
                    *(["--enforce"] if enforce else []),
                ],
                cwd=review_repo,
                capture_output=True,
                text=True,
            )
        finally:
            server.shutdown()
            thread.join()

    assert result.returncode == int(enforce and response_kind != "clean"), result.stderr
    report = (review_repo / "review/dependency-review.md").read_text()
    data = json.loads((review_repo / "review/dependency-review.json").read_text())
    assert set(data["locks"]) == set(LOCKFILES)
    assert data["enforced"] == enforce
    assert "**New dependency**" in report
    assert "Removed dependency" in report
    assert "Version changed" in report
    assert "**Same-version hash change**" in report
    change = data["locks"][LOCKFILES[3]]["changes"][0]
    assert change["before"]["hashes"] == ["a" * 64]
    assert change["after"]["hashes"] == ["b" * 64]
    if response_kind in {"clean", "vulnerable"}:
        pairs = {(q["package"]["name"], q["version"]) for q in queries}
        assert len(pairs) == 107
        assert ("shared", "1.0") in pairs
        assert ("xpu-only", "1.0+vendor") in pairs
        assert ("xpu-only", "1.0") in pairs
        assert ("removed-cuda", "1.0") not in pairs
        assert len(data["audit"]) == 107
    if response_kind == "vulnerable":
        assert "GHSA-test-advisory" in report
        assert data["complete"] is True
    elif response_kind != "clean":
        assert data["errors"]
        assert data["complete"] is False
        assert "**Advisory check incomplete:**" in report
    if not enforce:
        assert "**disabled (report only)**" in report
        assert "do not block publication" in report


@pytest.mark.parametrize(
    "requirement", ["unhashed==1.0", "unpinned>=1.0", "url @ https://example.com/a.whl"]
)
def test_review_reports_unscannable_lock_without_blocking(review_repo, requirement):
    """A parsing failure must be visible without blocking the default workflow."""
    (review_repo / LOCKFILES[-1]).write_text(requirement + "\n")
    result = subprocess.run(
        [
            sys.executable,
            "tools/review_compiled_requirements.py",
            "--base",
            "HEAD",
            "--output-dir",
            "review",
            "--osv-url",
            "http://127.0.0.1:0/v1/querybatch",
        ],
        cwd=review_repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = (review_repo / "review/dependency-review.md").read_text()
    data = json.loads((review_repo / "review/dependency-review.json").read_text())
    assert "Expected an exact hashed registry pin" in report
    assert "**Report incomplete:**" in report
    assert data["complete"] is False
    assert "error" in data["locks"][LOCKFILES[-1]]
