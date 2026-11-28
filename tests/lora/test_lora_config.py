# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm.config.lora as lora_config_module
from vllm.config.lora import LoRAConfig

pytestmark = pytest.mark.skip_global_cleanup

GPU_PUNICA_WRAPPER = "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
CPU_PUNICA_WRAPPER = "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"


def _model_config(dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(dtype=dtype, architectures=[])


def _use_punica_wrapper(monkeypatch: pytest.MonkeyPatch, wrapper: str) -> None:
    monkeypatch.setattr(
        lora_config_module.current_platform, "get_punica_wrapper", lambda: wrapper
    )


@pytest.mark.parametrize(
    "model_dtype, lora_dtype",
    [
        (torch.float32, "auto"),
        (torch.float16, torch.float32),
        (torch.float32, "float16"),
    ],
)
def test_triton_lora_rejects_unsupported_dtype(monkeypatch, model_dtype, lora_dtype):
    _use_punica_wrapper(monkeypatch, GPU_PUNICA_WRAPPER)
    lora_config = LoRAConfig(lora_dtype=lora_dtype)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        lora_config.verify_with_model_config(_model_config(model_dtype))


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_triton_lora_accepts_half_precision(monkeypatch, model_dtype):
    _use_punica_wrapper(monkeypatch, GPU_PUNICA_WRAPPER)
    lora_config = LoRAConfig()
    lora_config.verify_with_model_config(_model_config(model_dtype))
    assert lora_config.lora_dtype == model_dtype


def test_cpu_lora_allows_float32(monkeypatch):
    _use_punica_wrapper(monkeypatch, CPU_PUNICA_WRAPPER)
    lora_config = LoRAConfig()
    lora_config.verify_with_model_config(_model_config(torch.float32))
    assert lora_config.lora_dtype == torch.float32
