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
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


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
