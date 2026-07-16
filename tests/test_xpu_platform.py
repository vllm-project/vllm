# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU platform only"
)


def test_mem_info_falls_back_to_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.platforms import xpu

    kernel_query = Mock(return_value=(0, 24 * 1024**3))
    pytorch_query = Mock(return_value=(16 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.ops._C_cache_ops, "getMemoryInfo", kernel_query)
    monkeypatch.setattr(torch.xpu, "mem_get_info", pytorch_query)

    assert xpu.get_mem_info_wrapper(0) == pytorch_query.return_value
    kernel_query.assert_called_once_with(0)
    pytorch_query.assert_called_once_with(0)
