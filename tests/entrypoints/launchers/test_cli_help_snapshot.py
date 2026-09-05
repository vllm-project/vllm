# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import runpy
import subprocess
import sys
import sysconfig
from importlib import metadata, util
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "tools/vllm"
GENERATOR = ROOT / "tools/generate_cli_help.py"
INSTALLED = Path(sysconfig.get_path("scripts")) / "vllm"
# The ROCm CI image ships tests without tools/, so use the installed command.
LAUNCHER = SOURCE if SOURCE.is_file() else INSTALLED
_SPEC = util.find_spec("vllm")
assert _SPEC is not None and _SPEC.origin is not None
# Resolved the way tools/vllm resolves it: beside the importable package.
PACKAGE = Path(_SPEC.origin).parent
HELP_DIR = PACKAGE / "entrypoints" / "cli" / "_help"
PAGES = {"top": "vllm.txt", "serve": "vllm-serve.txt"}
RUNNER = "import sys; sys.argv[0] = 'vllm'; "
RUNTIME = RUNNER + "from vllm.entrypoints.cli.main import main; main()"
RUN_LAUNCHER = RUNNER + (
    f"import runpy; runpy.run_path({str(LAUNCHER)!r}, run_name='__main__')"
)
BLOCK_IMPORTS = """\
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"vllm", "torch"}:
        raise AssertionError(f"forbidden import: {name}")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
"""
CANONICAL_ENV = dict(
    COLUMNS="80",
    LINES="24",
    LANG="C",
    LC_ALL="C",
    NO_COLOR="1",
    PYTHONHASHSEED="0",
    TERM="dumb",
    VLLM_PLUGINS="",
    VLLM_LOGGING_LEVEL="CRITICAL",
)


def _page(key):
    return (HELP_DIR / PAGES[key]).read_text(encoding="utf-8")


def _env(pythonpath=ROOT, **overrides):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    environment |= CANONICAL_ENV | overrides
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{inherited}"
    return environment


def _run(args, *, runtime=False, block_imports=False, pythonpath=ROOT, **env):
    source = RUNTIME if runtime else RUN_LAUNCHER
    if block_imports:
        source = BLOCK_IMPORTS + source
    return subprocess.run(
        [sys.executable, "-c", source, *args],
        capture_output=True,
        # cwd lands on sys.path ahead of PYTHONPATH, so it has to move too.
        cwd=pythonpath,
        env=_env(pythonpath=pythonpath, **env),
        text=True,
    )


FAST_PATHS = {
    "bare": ([], "top"),
    "top-h": (["-h"], "top"),
    "top-help": (["--help"], "top"),
    "version-v": (["-v"], None),
    "version-long": (["--version"], None),
    "serve-h": (["serve", "-h"], "serve"),
    "serve-help": (["serve", "--help"], "serve"),
}


@pytest.mark.parametrize("args, page", FAST_PATHS.values(), ids=FAST_PATHS)
def test_fast_paths_match_canonical_without_runtime_imports(args, page):
    expected = _page(page) if page else metadata.version("vllm") + "\n"
    result = _run(args, block_imports=True)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == expected
    runtime = _run(args, runtime=True)
    assert result.returncode == runtime.returncode
    assert result.stdout == runtime.stdout
    assert result.stderr == runtime.stderr


def test_installed_command_is_the_launcher():
    if not INSTALLED.is_file():
        pytest.skip("no installed vllm command")
    # Wheels and PEP 660 editable installs record a RECORD; the egg-info an
    # editable build leaves in the source tree does not, and would otherwise
    # shadow the real install on PYTHONPATH.
    dists = [
        dist
        for dist in metadata.distributions()
        if dist.metadata["Name"] == "vllm" and dist.read_text("RECORD")
    ]
    if not dists:
        pytest.skip("no installed vllm distribution")
    dist = dists[0]
    # setup.py appends a platform suffix to the distribution version, so compare
    # locations instead: a wheel whose package is the importable one, or an
    # editable install of this tree.
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    wheel = (
        Path(str(dist.locate_file("vllm/__init__.py"))).resolve()
        == (PACKAGE / "__init__.py").resolve()
    )
    editable = direct_url.get("dir_info", {}).get("editable") and (
        direct_url.get("url") == ROOT.resolve().as_uri()
    )
    if not (wheel or editable):
        pytest.skip("installed vllm is not the importable checkout")

    if not SOURCE.is_file():
        pytest.skip("launcher source is not in this tree")
    source = SOURCE.read_text(encoding="utf-8").splitlines()[:4]
    spdx = [line for line in source if line.startswith("# SPDX")]
    assert len(spdx) == 2
    # pip can replace the shebang with a multi-line sh trampoline, so match the
    # header anywhere in the installed script rather than at a fixed offset.
    installed = INSTALLED.read_text(encoding="utf-8").splitlines()
    assert all(line in installed for line in spdx)

    assert not [
        entry
        for entry in dist.entry_points
        if entry.group == "console_scripts" and entry.name == "vllm"
    ]

    result = subprocess.run(
        [sys.executable, "-X", "importtime", str(INSTALLED), "--help"],
        capture_output=True,
        cwd=ROOT,
        env=_env(),
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == _page("top")
    assert not [
        line
        for line in result.stderr.splitlines()
        if re.match(r"^import time:.*\|\s+(torch|vllm)(\.|$)", line)
    ]


def test_generator_check_pins_its_environment():
    if not GENERATOR.is_file():
        pytest.skip("generator is not in this tree")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        cwd=ROOT,
        env=os.environ
        | dict(
            VLLM_TARGET_DEVICE="cuda",
            VLLM_PLUGINS="bad",
            VLLM_USE_V1="0",
            COLUMNS="200",
            PYTHONHASHSEED="2",
        ),
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Prove the pages were compared rather than the Torch-absent skip firing.
    assert "skipped" not in result.stderr


def test_generator_check_fails_on_stale_page(tmp_path):
    if not GENERATOR.is_file():
        pytest.skip("generator is not in this tree")
    if util.find_spec("torch") is None:
        pytest.skip("the generator needs Torch to render")
    for name in PAGES.values():
        (tmp_path / name).write_bytes((HELP_DIR / name).read_bytes())
    stale = tmp_path / PAGES["top"]
    stale.write_bytes(stale.read_bytes() + b"x")
    generator = runpy.run_path(str(GENERATOR), run_name="generate_cli_help")
    with pytest.raises(SystemExit, match=PAGES["top"]):
        generator["_check"](tmp_path)


def test_pages_serve_at_generation_width_or_wider():
    top = _page("top")
    assert _run(["--help"], block_imports=True, COLUMNS="120").stdout == top
    narrow = _run(["--help"], COLUMNS="60")
    assert narrow.stdout == _run(["--help"], runtime=True, COLUMNS="60").stdout
    assert narrow.stdout != top


@pytest.mark.parametrize("content", [None, ""], ids=["missing", "empty"])
def test_missing_or_empty_page_delegates(tmp_path, content):
    # A stand-in vllm package whose CLI prints instead of importing the engine.
    package = tmp_path / "vllm"
    cli = package / "entrypoints" / "cli"
    (cli / "_help").mkdir(parents=True)
    for directory in (package, package / "entrypoints", cli):
        (directory / "__init__.py").write_text("")
    (cli / "main.py").write_text('def main():\n    print("delegated")\n')
    if content is not None:
        (cli / "_help" / PAGES["top"]).write_text(content)
    result = _run(["--help"], pythonpath=tmp_path, COLUMNS="80")
    assert result.returncode == 0
    assert result.stdout == "delegated\n"


NON_FAST_PATHS = {
    "omni": ["--omni"],
    "serve-help-all": ["serve", "--help=all"],
    "serve-help-query": ["serve", "--help=model-"],
    "after-separator": ["--", "--help"],
    "unknown-subcommand": ["bogus"],
}


@pytest.mark.parametrize("args", NON_FAST_PATHS.values(), ids=NON_FAST_PATHS)
def test_non_fast_paths_preserve_runtime_behavior(args):
    result = _run(args)
    runtime = _run(args, runtime=True)
    assert result.returncode == runtime.returncode
    assert result.stdout == runtime.stdout
    assert result.stderr == runtime.stderr
