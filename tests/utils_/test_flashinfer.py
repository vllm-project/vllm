# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import re
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import vllm.utils.flashinfer as fi_utils

_FAKE_MODULE_NAME = "fake_flashinfer_module"
_FAKE_KERNEL_NAME = "slow_kernel"
_QUALNAME = f"{_FAKE_MODULE_NAME}.{_FAKE_KERNEL_NAME}"
LOGGER_NAME = "vllm.utils.flashinfer"


@pytest.fixture()
def wrapped_kernel(monkeypatch):
    """Build a _lazy_import_wrapper around a fake slow FlashInfer kernel.

    FlashInfer JIT-compiles kernels on first use (vllm-project/vllm#38246),
    so the fake kernel stands in for a slow cold first call. The slow path is
    shortened so the test stays fast.
    """
    calls = {"count": 0}

    def make(sleep_s: float = 0.0):
        module = ModuleType(_FAKE_MODULE_NAME)

        def slow_kernel(*args, **kwargs):
            calls["count"] += 1
            if sleep_s:
                time.sleep(sleep_s)
            return "done"

        setattr(module, _FAKE_KERNEL_NAME, slow_kernel)
        monkeypatch.setitem(sys.modules, _FAKE_MODULE_NAME, module)
        monkeypatch.setattr(fi_utils, "has_flashinfer", lambda: True)
        monkeypatch.setattr(fi_utils, "has_flashinfer_jit_cache", lambda: False)
        return fi_utils._lazy_import_wrapper(_FAKE_MODULE_NAME, _FAKE_KERNEL_NAME)

    return make, calls


def test_first_call_logs_jit_compile(caplog_vllm, disable_log_dedup, wrapped_kernel):
    """Cold first call logs before/after with elapsed time; warm path quiet."""
    make, calls = wrapped_kernel
    wrapper = make(sleep_s=0.05)

    with caplog_vllm.at_level(logging.INFO, logger=LOGGER_NAME):
        assert wrapper() == "done"
        assert wrapper() == "done"  # warm path

    assert calls["count"] == 2
    messages = [record.getMessage() for record in caplog_vllm.records]
    before = [msg for msg in messages if "may JIT-compile kernels on first use" in msg]
    after = [msg for msg in messages if "finished first use in" in msg]

    assert len(before) == 1
    assert _QUALNAME in before[0]
    assert "flashinfer-jit-cache" in before[0]

    assert len(after) == 1
    assert _QUALNAME in after[0]
    assert re.search(r"in \d+\.\d\d seconds\.$", after[0]), after[0]


def test_first_call_quiet_with_jit_cache(caplog_vllm, monkeypatch, wrapped_kernel):
    """Pre-compiled kernels (flashinfer-jit-cache) -> cold path stays quiet."""
    make, calls = wrapped_kernel
    wrapper = make()
    monkeypatch.setattr(fi_utils, "has_flashinfer_jit_cache", lambda: True)

    with caplog_vllm.at_level(logging.INFO, logger=LOGGER_NAME):
        assert wrapper() == "done"

    assert calls["count"] == 1
    assert not any(
        "JIT-compile" in record.getMessage() for record in caplog_vllm.records
    )


def test_missing_flashinfer_uses_fallback_without_logs(caplog_vllm, monkeypatch):
    """Without FlashInfer the fallback runs and nothing is logged."""
    monkeypatch.setattr(fi_utils, "has_flashinfer", lambda: False)
    fallback = MagicMock(return_value="fallback")
    wrapper = fi_utils._lazy_import_wrapper(
        _FAKE_MODULE_NAME, _FAKE_KERNEL_NAME, fallback_fn=fallback
    )

    with caplog_vllm.at_level(logging.INFO, logger=LOGGER_NAME):
        assert wrapper() == "fallback"

    fallback.assert_called_once()
    assert not any(
        "JIT-compile" in record.getMessage() for record in caplog_vllm.records
    )


def test_failed_first_call_keeps_logging_enabled(
    caplog_vllm, disable_log_dedup, monkeypatch, wrapped_kernel
):
    """A failed cold call must not silence logging of the retry."""
    _make, calls = wrapped_kernel
    monkeypatch.setattr(fi_utils, "has_flashinfer", lambda: True)
    monkeypatch.setattr(fi_utils, "has_flashinfer_jit_cache", lambda: False)

    module = ModuleType(_FAKE_MODULE_NAME)

    def flaky_kernel():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("compile failed")
        return "done"

    setattr(module, _FAKE_KERNEL_NAME, flaky_kernel)
    monkeypatch.setitem(sys.modules, _FAKE_MODULE_NAME, module)
    wrapper = fi_utils._lazy_import_wrapper(_FAKE_MODULE_NAME, _FAKE_KERNEL_NAME)

    with caplog_vllm.at_level(logging.INFO, logger=LOGGER_NAME):
        with pytest.raises(RuntimeError, match="compile failed"):
            wrapper()
        assert wrapper() == "done"
        assert wrapper() == "done"  # warm path

    messages = [record.getMessage() for record in caplog_vllm.records]
    before = [msg for msg in messages if "may JIT-compile kernels on first use" in msg]
    after = [msg for msg in messages if "finished first use in" in msg]
    assert len(before) == 2
    assert len(after) == 1
