# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upgrade locks produced by the uv pip-compile pre-commit hooks.

Reads each ``pip-compile`` hook in ``.pre-commit-config.yaml`` and reruns it
with ``--upgrade``, then once more without ``--upgrade``. The second pass
keeps the upgraded pins and hashes and rewrites the header to the pre-commit
command, so CI's compile hooks do not touch the files again.
Both passes share a seven-day upload cutoff, including archive hashes.
Handwritten requirements, build pins, and connector pins are unchanged;
Dependabot security updates cover those source manifests separately.
The hooks generate SHA-256 hashes for published package archives. uv verifies
these on download; excluded ROCm packages and separate source/build installs
are outside these locks.

``--upgrade`` stays off the pre-commit hooks. Those run on every commit and
must not float transitive pins.
"""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
UV_REPO = "https://github.com/astral-sh/uv-pre-commit"
COOLDOWN_DAYS = 7
EXPECTED_HOOKS = {
    "pip-compile",
    "pip-compile-rocm",
    "pip-compile-xpu",
    "pip-compile-cpu",
    "pip-compile-docs",
}


def compile_hooks() -> list[tuple[str, list[str]]]:
    config = yaml.safe_load(PRE_COMMIT.read_text())
    hooks: list[tuple[str, list[str]]] = []
    for repo in config["repos"]:
        if repo.get("repo") != UV_REPO:
            continue
        for hook in repo["hooks"]:
            if hook.get("id") != "pip-compile":
                continue
            name = hook.get("alias", hook["id"])
            hooks.append((name, list(hook["args"])))
    return hooks


def run(args: list[str], cutoff: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        env={**os.environ, "UV_EXCLUDE_NEWER": cutoff},
    )


def main() -> None:
    hooks = compile_hooks()
    found = {name for name, _args in hooks}
    if found != EXPECTED_HOOKS:
        missing = sorted(EXPECTED_HOOKS - found)
        extra = sorted(found - EXPECTED_HOOKS)
        sys.exit(f"Unexpected pip-compile hooks. missing={missing} extra={extra}")

    for name, args in hooks:
        if "--upgrade" in args:
            sys.exit(f"{name} already passes --upgrade; refusing to double it")
        if "--generate-hashes" not in args:
            sys.exit(f"{name} must pass --generate-hashes")

    cutoff = (datetime.now(UTC) - timedelta(days=COOLDOWN_DAYS)).isoformat()
    print(f"{COOLDOWN_DAYS}-day cooldown: excluding uploads after {cutoff}", flush=True)
    for _name, args in hooks:
        run(["uv", "pip", "compile", *args, "--upgrade"], cutoff)
        run(["uv", "pip", "compile", *args], cutoff)


if __name__ == "__main__":
    main()
