# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import VllmConfig
from vllm.plugins.io_processors import get_io_processor


def test_loading_missing_plugin(monkeypatch):
    monkeypatch.setenv("VLLM_USE_IO_PROCESSOR_PLUGIN", "plugin")

    vllm_config = VllmConfig()
    with pytest.raises(ValueError):
        get_io_processor(vllm_config)
