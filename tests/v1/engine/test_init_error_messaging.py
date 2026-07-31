# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import sys
from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
from vllm.v1.engine.utils import describe_failed_procs
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_kv_cache_oom_no_memory():
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype="float16",
        )
    }

    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(config, spec, 0)


def test_kv_cache_oom_insufficient_memory(monkeypatch):
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048
    config.cache_config.block_size = 16
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,  # 100 GiB
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype="float16",
        )
    }

    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(config, spec, 1024**3)  # 1 GiB


def _exit_with_code_3():
    sys.exit(3)


def test_describe_failed_procs_reports_name_and_exit_code():
    """A proc that exited during startup must be reported with its name and
    exit code, not the bare 'Failed core proc(s): {}' of #48031 / #45626."""
    proc = multiprocessing.Process(target=_exit_with_code_3, name="EngineCore_0")
    proc.start()
    proc.join()

    msg = describe_failed_procs(SimpleNamespace(processes=[proc]), None)
    assert "EngineCore_0" in msg
    assert "3" in msg


def test_describe_failed_procs_names_proc_when_exit_code_races():
    """A proc's sentinel can fire before its exit code is collectible; the
    description must still name the proc instead of reporting an empty dict."""
    read_fd, write_fd = os.pipe()
    os.close(write_fd)  # EOF makes the read end (the fake sentinel) ready
    try:
        racing_proc = SimpleNamespace(
            sentinel=read_fd,
            name="EngineCore_0",
            exitcode=None,
            join=lambda timeout=None: None,
        )
        msg = describe_failed_procs(SimpleNamespace(processes=[racing_proc]), None)
        assert "EngineCore_0" in msg
        assert "unknown" in msg
    finally:
        os.close(read_fd)


def test_describe_failed_procs_ignores_live_procs():
    """A live proc (sentinel not ready, no exit code) must not be reported."""
    live_read_fd, live_write_fd = os.pipe()  # writer open: sentinel not ready
    coord_read_fd, coord_write_fd = os.pipe()
    try:
        live_proc = SimpleNamespace(
            sentinel=live_read_fd,
            name="EngineCore_0",
            exitcode=None,
            join=lambda timeout=None: None,
        )
        exited_coord = SimpleNamespace(
            sentinel=coord_read_fd,  # readiness unused: exitcode already set
            name="DPCoordinator",
            exitcode=-9,
            join=lambda timeout=None: None,
        )
        msg = describe_failed_procs(
            SimpleNamespace(processes=[live_proc]), exited_coord
        )
        assert "DPCoordinator" in msg
        assert "-9" in msg
        assert "EngineCore_0" not in msg
    finally:
        for fd in (live_read_fd, live_write_fd, coord_read_fd, coord_write_fd):
            os.close(fd)
