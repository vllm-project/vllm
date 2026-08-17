# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.vllm import VllmConfig

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_quantization_config_forwards_model_revision(monkeypatch: pytest.MonkeyPatch):
    quant_config = Mock()
    quant_config.get_min_capability.return_value = 0
    quant_config.get_supported_act_dtypes.return_value = [torch.float32]
    model_config = SimpleNamespace(
        quantization="auto_gptq",
        model="repo/model",
        hf_config=object(),
        dtype=torch.float32,
        revision="pinned-revision",
    )

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.get_quant_config",
        lambda model_config, load_config: quant_config,
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_device_capability", lambda: None
    )

    VllmConfig._get_quantization_config(model_config, SimpleNamespace())

    quant_config.maybe_update_config.assert_called_once_with(
        "repo/model",
        hf_config=model_config.hf_config,
        revision="pinned-revision",
    )
