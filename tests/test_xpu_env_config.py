# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
from unittest.mock import MagicMock, patch

from vllm.platforms.xpu_env import configure_ccl_env, detect_xpu_interconnect


def test_detect_xpu_interconnect_user_override_pcie():
    with patch.dict(os.environ, {"VLLM_XPU_INTERCONNECT": "pcie"}):
        assert detect_xpu_interconnect() == "pcie"


def test_detect_xpu_interconnect_user_override_xelink():
    with patch.dict(os.environ, {"VLLM_XPU_INTERCONNECT": "xelink"}):
        assert detect_xpu_interconnect() == "xelink"


def test_detect_xpu_interconnect_sysfs_xelink():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "glob.glob",
            side_effect=lambda p: (
                ["/sys/class/drm/card0/gt/gt0/fabric_ports"]
                if "fabric_ports" in p
                else []
            ),
        ),
    ):
        assert detect_xpu_interconnect() == "xelink"


def test_detect_xpu_interconnect_sysfs_pcie():
    mock_torch = MagicMock()
    mock_props = MagicMock()
    mock_props.name = "Intel Data Center GPU Flex 170"
    mock_torch.xpu.get_device_properties.return_value = mock_props
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("glob.glob", return_value=[]),
        patch.dict(sys.modules, {"torch": mock_torch}),
    ):
        assert detect_xpu_interconnect() == "pcie"


def test_detect_xpu_interconnect_fallback_pcie():
    mock_torch = MagicMock()
    mock_torch.xpu.get_device_properties.side_effect = RuntimeError("no device")
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("glob.glob", side_effect=OSError("no sysfs")),
        patch.dict(sys.modules, {"torch": mock_torch}),
    ):
        assert detect_xpu_interconnect() == "pcie"


def test_configure_ccl_env_pcie():
    with patch.dict(os.environ, {}, clear=True):
        result = configure_ccl_env("pcie", 2)
    assert result["CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK"] == "0"
    assert result["CCL_ZE_IPC_EXCHANGE"] == "sockets"
    assert result["FI_PROVIDER"] == "shm"
    assert result["CCL_ATL_TRANSPORT"] == "ofi"


def test_configure_ccl_env_xelink():
    with patch.dict(os.environ, {}, clear=True):
        result = configure_ccl_env("xelink", 2)
    assert "CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK" not in result


def test_configure_ccl_env_respects_user_override():
    with patch.dict(
        os.environ,
        {"CCL_ATL_TRANSPORT": "mpi"},
        clear=True,
    ):
        result = configure_ccl_env("pcie", 2)
    assert "CCL_ATL_TRANSPORT" not in result


def test_configure_ccl_env_large_world_size():
    with patch.dict(os.environ, {}, clear=True):
        result = configure_ccl_env("pcie", 8)
    assert result["CCL_WORKER_COUNT"] == "1"
