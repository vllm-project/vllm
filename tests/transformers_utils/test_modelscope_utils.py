# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

from vllm.transformers_utils import modelscope_utils
from vllm.transformers_utils.modelscope_utils import configure_modelscope_runtime


def test_configure_modelscope_runtime_preserves_existing_env(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        m.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
        m.setenv("MODELSCOPE_CACHE", "/custom/cache")
        m.setenv("MODELSCOPE_CREDENTIALS_PATH", "/custom/creds")

        configure_modelscope_runtime()

        assert os.environ["NO_PROXY"] == "localhost,127.0.0.1,::1"
        assert os.environ["MODELSCOPE_CACHE"] == "/custom/cache"
        assert os.environ["MODELSCOPE_CREDENTIALS_PATH"] == "/custom/creds"


def test_modelscope_is_available_rejects_unsupported_version(
    monkeypatch, tmp_path: Path
):
    package_dir = tmp_path / "modelscope"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("__version__ = '1.18.0'\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("vllm.envs.VLLM_USE_MODELSCOPE", True)

    modelscope_utils.modelscope_is_available.cache_clear()
    try:
        assert modelscope_utils.modelscope_is_available() is False
        assert modelscope_utils.should_use_modelscope() is False
    finally:
        modelscope_utils.modelscope_is_available.cache_clear()


def test_modelscope_is_available_rejects_broken_install(
    monkeypatch, tmp_path: Path
):
    package_dir = tmp_path / "modelscope"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "raise RuntimeError('broken modelscope')\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("vllm.envs.VLLM_USE_MODELSCOPE", True)

    modelscope_utils.modelscope_is_available.cache_clear()
    try:
        assert modelscope_utils.modelscope_is_available() is False
        assert modelscope_utils.should_use_modelscope() is False
    finally:
        modelscope_utils.modelscope_is_available.cache_clear()
