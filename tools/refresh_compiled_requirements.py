# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upgrade locks produced by the uv pip-compile pre-commit hooks.

Reads each ``pip-compile`` hook in ``.pre-commit-config.yaml`` and reruns it
with ``--upgrade``, then once more without ``--upgrade``. The second pass
keeps the upgraded pins and rewrites the header to the pre-commit command, so
CI's compile hooks do not touch the files again.

``--upgrade`` stays off the pre-commit hooks. Those run on every commit and
must not float transitive pins.
"""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
UV_REPO = "https://github.com/astral-sh/uv-pre-commit"
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


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


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
        run(["uv", "pip", "compile", *args, "--upgrade"])
        run(["uv", "pip", "compile", *args])


if __name__ == "__main__":
    main()
