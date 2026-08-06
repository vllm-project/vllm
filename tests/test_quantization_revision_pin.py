# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.vllm import VllmConfig

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _QuantConfig:
    def __init__(self):
        self.calls: list[tuple[str, object, str | None]] = []

    def get_min_capability(self):
        return 0

    def get_supported_act_dtypes(self):
        return [torch.float32]

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        self.calls.append((model_name, hf_config, revision))


def test_quantization_config_forwards_model_revision(monkeypatch: pytest.MonkeyPatch):
    quant_config = _QuantConfig()
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

    result = VllmConfig._get_quantization_config(model_config, SimpleNamespace())

    assert result is quant_config
    assert quant_config.calls == [
        ("repo/model", model_config.hf_config, "pinned-revision")
    ]
