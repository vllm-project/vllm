# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.utils.flashinfer as flashinfer_utils


@pytest.fixture(autouse=True)
def _reset_flashinfer_jit_cache():
    flashinfer_utils._may_need_flashinfer_jit_compile.cache_clear()
    yield
    flashinfer_utils._may_need_flashinfer_jit_compile.cache_clear()


def test_lazy_wrapper_logs_when_flashinfer_may_jit_compile(monkeypatch):
    op = Mock(return_value="result")
    info_once = Mock()

    monkeypatch.setattr(flashinfer_utils, "has_flashinfer", lambda: True)
    monkeypatch.setattr(flashinfer_utils, "has_flashinfer_cubin", lambda: False)
    monkeypatch.setattr(flashinfer_utils.shutil, "which", lambda cmd: "/usr/bin/nvcc")
    monkeypatch.setattr(
        flashinfer_utils,
        "_get_submodule",
        lambda module_name: SimpleNamespace(op=op),
    )
    monkeypatch.setattr(flashinfer_utils.logger, "info_once", info_once)

    wrapper = flashinfer_utils._lazy_import_wrapper("flashinfer.test", "op")

    assert wrapper() == "result"
    op.assert_called_once_with()
    info_once.assert_called_once_with(flashinfer_utils._FLASHINFER_JIT_COMPILE_MESSAGE)


@pytest.mark.parametrize(
    ("has_cubin", "nvcc_path"),
    [
        (True, "/usr/bin/nvcc"),
        (False, None),
    ],
)
def test_flashinfer_jit_log_skipped_when_compile_is_unlikely(
    has_cubin, nvcc_path, monkeypatch
):
    info_once = Mock()

    monkeypatch.setattr(flashinfer_utils, "has_flashinfer_cubin", lambda: has_cubin)
    monkeypatch.setattr(flashinfer_utils.shutil, "which", lambda cmd: nvcc_path)
    monkeypatch.setattr(flashinfer_utils.logger, "info_once", info_once)

    flashinfer_utils._log_flashinfer_jit_compile_once()

    info_once.assert_not_called()
