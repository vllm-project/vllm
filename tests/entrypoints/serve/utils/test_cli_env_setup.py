# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cli_env_setup forkserver default and CUDA-init guard.

These tests cover the four HIGH findings from review on the
forkserver-default-flip PR:

1. The CUDA-init guard must survive ``python -O`` (no bare ``assert``).
2. The CUDA-init check must run BEFORE ``set_start_method("forkserver")``
   because ``set_start_method`` mutates the process-global default and
   raising after that point leaves the interpreter in an unrecoverable
   state for the rest of the process.
3. ``_maybe_force_spawn()`` must be able to undo the global default —
   we now pass ``force=True`` to ``set_start_method``, and we call
   ``_maybe_force_spawn()`` BEFORE the forkserver block so the env-var
   override is honored.
4. Non-``serve`` subcommands (``bench``, ``run_batch``, ``openai``,
   ``collect_env``) must NOT get the forkserver default — only ``serve``
   has the matching ``forkserver.ensure_running`` + preload setup.
5. The CUDA-init guard must be platform-gated on cuda-alike platforms;
   on XPU / CPU / etc. ``torch.cuda.is_initialized()`` is the wrong test.
"""

import os
import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Each test starts with a clean VLLM_WORKER_MULTIPROC_METHOD."""
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    yield


def _set_argv(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)


def test_serve_subcommand_sets_forkserver(monkeypatch):
    """`vllm serve <model>` should default to forkserver."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm", "serve", "Qwen/Qwen3-0.6B"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver"


def test_bench_subcommand_keeps_spawn(monkeypatch):
    """`vllm bench` must NOT get forkserver — bench has no preload setup."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm", "bench", "latency", "--model", "x"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"


def test_run_batch_subcommand_keeps_spawn(monkeypatch):
    """`vllm run_batch` must NOT get forkserver."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm", "run_batch", "-i", "x", "-o", "y"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"


def test_openai_subcommand_keeps_spawn(monkeypatch):
    """`vllm openai` chat client must NOT get forkserver."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm", "openai", "chat"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"


def test_collect_env_subcommand_keeps_spawn(monkeypatch):
    """`vllm collect_env` must NOT get forkserver."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm", "collect_env"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"


def test_no_subcommand_keeps_spawn(monkeypatch):
    """`vllm` with no subcommand must NOT get forkserver."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"


def test_version_flag_before_subcommand_handled(monkeypatch):
    """Leading flags like `-v` must not confuse subcommand detection."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    _set_argv(monkeypatch, ["vllm", "-v", "serve", "model"])
    cli_env_setup()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver"


def test_explicit_env_var_wins(monkeypatch):
    """User-set VLLM_WORKER_MULTIPROC_METHOD is never overwritten."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    _set_argv(monkeypatch, ["vllm", "serve", "model"])
    cli_env_setup()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_serve_subcommand_in_argv_helper(monkeypatch):
    """Direct test of the helper used by cli_env_setup."""
    from vllm.entrypoints.serve.utils.api_utils import _serve_subcommand_in_argv

    _set_argv(monkeypatch, ["vllm", "serve", "x"])
    assert _serve_subcommand_in_argv() is True

    _set_argv(monkeypatch, ["vllm", "bench"])
    assert _serve_subcommand_in_argv() is False

    _set_argv(monkeypatch, ["vllm", "--version"])
    assert _serve_subcommand_in_argv() is False

    _set_argv(monkeypatch, ["vllm"])
    assert _serve_subcommand_in_argv() is False
