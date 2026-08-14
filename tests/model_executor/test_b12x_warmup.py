# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.b12x_warmup import b12x_warmup


@pytest.mark.parametrize(
    ("module_name", "warmup_name", "import_name"),
    [
        (
            "vllm.model_executor.kernels.linear.scaled_mm.b12x_block",
            "warmup_b12x_block_fp8_linear",
            "_import_b12x_blockscaled",
        ),
        (
            "vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor",
            "warmup_b12x_tensor_fp8_linear",
            "_import_b12x_tensor_fp8",
        ),
        (
            "vllm.model_executor.kernels.linear.mxfp8.b12x",
            "warmup_b12x_mxfp8_linear",
            "_import_b12x_mxfp8",
        ),
        (
            "vllm.model_executor.kernels.linear.mxfp4.b12x",
            "warmup_b12x_mxfp4_linear",
            "_import_b12x_blockscaled",
        ),
        (
            "vllm.model_executor.kernels.linear.nvfp4.b12x",
            "warmup_b12x_nvfp4_linear",
            "_import_b12x_blockscaled",
        ),
    ],
)
def test_b12x_linear_warmup_skips_unused_provider(
    monkeypatch,
    module_name: str,
    warmup_name: str,
    import_name: str,
) -> None:
    module = importlib.import_module(module_name)
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    monkeypatch.setattr(module, "current_platform", platform)

    def fail_import():
        pytest.fail("unused B12X provider was imported")

    monkeypatch.setattr(module, import_name, fail_import)

    warmup = getattr(module, warmup_name)
    assert warmup(torch.nn.Module(), max_tokens=128) == 0


def test_b12x_warmup_covers_linear_serving_shapes(monkeypatch) -> None:
    import vllm.model_executor.warmup.b12x_warmup as warmup_mod

    model = torch.nn.Module()
    linear_calls: list[tuple[torch.nn.Module, dict]] = []

    def linear_warmup(model, **kwargs):
        linear_calls.append((model, kwargs))
        return 0

    monkeypatch.setattr(warmup_mod, "warmup_b12x_block_fp8_linear", linear_warmup)
    monkeypatch.setattr(warmup_mod, "warmup_b12x_mxfp4_linear", linear_warmup)
    monkeypatch.setattr(warmup_mod, "warmup_b12x_mxfp8_linear", linear_warmup)
    monkeypatch.setattr(warmup_mod, "warmup_b12x_nvfp4_linear", linear_warmup)
    monkeypatch.setattr(warmup_mod, "warmup_b12x_tensor_fp8_linear", linear_warmup)

    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=256,
            max_num_scheduled_tokens=320,
        ),
        model_config=SimpleNamespace(dtype=torch.float16),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(compile_sizes=[32, "dynamic", 64])
        ),
    )

    b12x_warmup(worker, [8, 16])

    expected_linear_kwargs = {
        "max_tokens": 256,
        "cudagraph_capture_sizes": [8, 16],
        "output_dtype": torch.float16,
    }
    assert linear_calls == [(model, expected_linear_kwargs)] * 5
