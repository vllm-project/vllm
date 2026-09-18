# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the pre-downloaded cubin detection and the artifactory probe.

These run without a GPU: everything network- or package-related is patched.
"""

from unittest.mock import patch

import pytest
import requests

from vllm.utils import flashinfer as fi


@pytest.fixture(autouse=True)
def _clear_caches():
    fi.has_flashinfer_cubin.cache_clear()
    fi.has_nvidia_artifactory.cache_clear()
    yield
    fi.has_flashinfer_cubin.cache_clear()
    fi.has_nvidia_artifactory.cache_clear()


@pytest.fixture
def _no_cubin_package():
    with patch("vllm.utils.flashinfer.importlib.util.find_spec", return_value=None):
        yield


@pytest.mark.usefixtures("_no_cubin_package")
def test_has_flashinfer_cubin_env_flag(monkeypatch):
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    with patch("vllm.envs.VLLM_HAS_FLASHINFER_CUBIN", True):
        assert fi.has_flashinfer_cubin() is True


@pytest.mark.usefixtures("_no_cubin_package")
def test_has_flashinfer_cubin_local_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_CUBIN_DIR", str(tmp_path))
    with patch("vllm.envs.VLLM_HAS_FLASHINFER_CUBIN", False):
        assert fi.has_flashinfer_cubin() is True


@pytest.mark.usefixtures("_no_cubin_package")
def test_has_flashinfer_cubin_missing_dir_is_not_enough(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_CUBIN_DIR", str(tmp_path / "does-not-exist"))
    with patch("vllm.envs.VLLM_HAS_FLASHINFER_CUBIN", False):
        assert fi.has_flashinfer_cubin() is False


@pytest.mark.usefixtures("_no_cubin_package")
def test_has_flashinfer_cubin_none(monkeypatch):
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    with patch("vllm.envs.VLLM_HAS_FLASHINFER_CUBIN", False):
        assert fi.has_flashinfer_cubin() is False


def test_artifactory_skipped_with_local_cubins():
    with (
        patch("vllm.utils.flashinfer.has_flashinfer_cubin", return_value=True),
        patch("vllm.utils.flashinfer.requests.get") as mock_get,
    ):
        assert fi.has_nvidia_artifactory() is True
        mock_get.assert_not_called()


def test_artifactory_unreachable_warns_with_consequence(caplog_vllm):
    with (
        patch("vllm.utils.flashinfer.has_flashinfer_cubin", return_value=False),
        patch(
            "vllm.utils.flashinfer.requests.get",
            side_effect=requests.ConnectionError("no route to host"),
        ),
    ):
        assert fi.has_nvidia_artifactory() is False
    assert "TRTLLM attention kernels" in caplog_vllm.text
    assert "FLASHINFER_CUBIN_DIR" in caplog_vllm.text


def test_artifactory_bad_status_warns_with_consequence(caplog_vllm):
    class _Resp:
        status_code = 503

    with (
        patch("vllm.utils.flashinfer.has_flashinfer_cubin", return_value=False),
        patch("vllm.utils.flashinfer.requests.get", return_value=_Resp()),
    ):
        assert fi.has_nvidia_artifactory() is False
    assert "503" in caplog_vllm.text
    assert "TRTLLM attention kernels" in caplog_vllm.text


def test_artifactory_reachable():
    class _Resp:
        status_code = 200

    with (
        patch("vllm.utils.flashinfer.has_flashinfer_cubin", return_value=False),
        patch("vllm.utils.flashinfer.requests.get", return_value=_Resp()),
    ):
        assert fi.has_nvidia_artifactory() is True
