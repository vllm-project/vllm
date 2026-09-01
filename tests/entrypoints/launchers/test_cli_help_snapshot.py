# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
ARTIFACT = json.loads((ROOT / "vllm_cli/_help.json").read_text())
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


def _run(args, *, entrypoint="vllm_cli", block_imports=False, **env):
    source = (
        "import builtins, json, sys\n"
        "args = json.loads(sys.argv[1])\n"
        "if sys.argv[2] == 'blocked':\n"
        "    original = builtins.__import__\n"
        "    def guarded(name, *args, **kwargs):\n"
        "        if name.split('.', 1)[0] in {'vllm', 'torch'}:\n"
        "            raise AssertionError(f'forbidden import: {name}')\n"
        "        return original(name, *args, **kwargs)\n"
        "    builtins.__import__ = guarded\n"
        "    import atexit\n"
        "    def assert_no_runtime_imports():\n"
        "        loaded = {name.split('.', 1)[0] for name in sys.modules}\n"
        "        assert not loaded & {'vllm', 'torch'}\n"
        "    atexit.register(assert_no_runtime_imports)\n"
        "sys.argv = ['vllm', *args]\n"
        f"from {entrypoint} import main\n"
        "main()\n"
    )
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("VLLM_")
    }
    environment |= CANONICAL_ENV | env
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), environment.get("PYTHONPATH", ""))
    )
    mode = "blocked" if block_imports else "runtime"
    return subprocess.run(
        [sys.executable, "-c", source, json.dumps(args), mode],
        capture_output=True,
        cwd=ROOT,
        env=environment,
        text=True,
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], lambda data: data["help"]["top"]),
        (["--version"], lambda data: importlib.metadata.version("vllm") + "\n"),
        (["serve", "-h"], lambda data: data["help"]["serve"]),
        (["serve", "--help=all"], lambda data: data["help"]["all"]),
        (["serve", "--help=modelconfig"], lambda data: data["queries"]["modelconfig"]),
        (
            ["serve", "--help=max-model-len"],
            lambda data: data["queries"]["max-model-len"],
        ),
    ],
)
def test_fast_paths_are_import_light(args, expected):
    result = _run(args, block_imports=True)
    assert result.returncode == 0
    assert result.stdout == expected(ARTIFACT)
    assert result.stderr == ""
    assert result.stdout == _run(args, entrypoint="vllm.entrypoints.cli.main").stdout


def test_generator_matches_checked_in_snapshot():
    environment = {
        "VLLM_TARGET_DEVICE": "cuda",
        "VLLM_PLUGINS": "bad",
        "VLLM_TEST_FLAG": "bad",
        "VLLM_USE_V1": "0",
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/generate_cli_help.py"), "--check"],
        cwd=ROOT,
        env=os.environ | environment | {"COLUMNS": "200", "PYTHONHASHSEED": "2"},
    )
    assert result.returncode == 0
    assert "/en/latest/" in json.dumps(ARTIFACT) and "/en/v" not in json.dumps(ARTIFACT)


@pytest.mark.parametrize(
    "args",
    [
        ["--omni"],
        ["bench", "--help"],
        ["serve", "--help=model-"],
        ["--", "--help"],
    ],
)
def test_non_fast_paths_preserve_runtime_behavior(args):
    result = _run(args)
    runtime = _run(args, entrypoint="vllm.entrypoints.cli.main")
    assert result.returncode == runtime.returncode
    assert result.stdout == runtime.stdout
    assert result.stderr == runtime.stderr
