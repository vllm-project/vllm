# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import VllmConfig
from vllm.plugins.multimodal_data_processors import (
    get_multimodal_data_processor)


def test_loading_missing_plugin(monkeypatch):
    monkeypatch.setenv("VLLM_USE_MULTIMODAL_DATA_PROCESSOR_PLUGIN", "plugin")

    vllm_config = VllmConfig()
    with pytest.raises(ValueError):
        get_multimodal_data_processor(vllm_config)
