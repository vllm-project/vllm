# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the stock help pages that ``tools/vllm`` serves without importing vLLM.

Each page is the canonical parser's own output at 80 columns, captured from a
separate process so nothing this script imports can change it.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PAGES = {"vllm.txt": ["--help"], "vllm-serve.txt": ["serve", "--help"]}
RENDER = (
    "import sys; sys.argv[0] = 'vllm'; "
    "from vllm.entrypoints.cli.main import main; main()"
)
# Pin everything the parser and its formatter can read from the environment.
ENV = {
    "COLUMNS": "80",
    "LINES": "24",
    "LANG": "C",
    "LC_ALL": "C",
    "NO_COLOR": "1",
    "PYTHONHASHSEED": "0",
    "TERM": "dumb",
    "VLLM_PLUGINS": "",
    "VLLM_LOGGING_LEVEL": "CRITICAL",
}


def _render(args: list[str]) -> str:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    environment |= ENV
    environment["PYTHONPATH"] = f"{ROOT}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", RENDER, *args],
        capture_output=True,
        cwd=ROOT,
        env=environment,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


def _check(directory: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        # Skip like tools/pre_commit/rust-check.sh does without cargo; CI covers it.
        print(
            "check-cli-help-pages: skipped, Torch is not installed in this interpreter",
            file=sys.stderr,
        )
        return
    stale = [
        name
        for name, args in PAGES.items()
        if not (path := directory / name).is_file()
        or path.read_text(encoding="utf-8") != _render(args)
    ]
    if stale:
        raise SystemExit(
            f"stale or missing CLI help pages: {', '.join(stale)}; "
            "run tools/generate_cli_help.py"
        )


def main() -> None:
    if sys.argv[1:] not in ([], ["--check"]):
        raise SystemExit("usage: generate_cli_help.py [--check]")
    sys.path.insert(0, str(ROOT))
    # The importable package: this source tree for a checkout or an editable
    # install, and the installed wheel in CI where the source tree is moved aside.
    origin = importlib.util.find_spec("vllm").origin
    directory = Path(origin).parent / "entrypoints" / "cli" / "_help"
    if sys.argv[1:] == ["--check"]:
        _check(directory)
        return
    # Render both pages before writing either, so a failure leaves no half cache.
    pages = {name: _render(args) for name, args in PAGES.items()}
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (directory / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
