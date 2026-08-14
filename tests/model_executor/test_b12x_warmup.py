# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    B12xFp8BlockScaledMMKernel,
    B12xMxFp4LinearKernel,
    B12xMxfp8LinearKernel,
    B12xNvFp4LinearKernel,
    B12xTensorFP8ScaledMMLinearKernel,
)
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts


def test_b12x_warmup_token_counts_cover_serving_regimes() -> None:
    assert b12x_warmup_token_counts(
        max_tokens=2048,
        cudagraph_capture_sizes=[1, 2, 8, 128],
    ) == (1, 2, 8, 128, 2048)


@pytest.mark.parametrize(
    ("kernel_cls", "module_name", "call_name", "layer", "name"),
    [
        (
            B12xMxFp4LinearKernel,
            "vllm.model_executor.kernels.linear.mxfp4.b12x",
            "_apply_b12x_mxfp4_linear",
            SimpleNamespace(
                weight=torch.empty((48, 64), dtype=torch.uint8),
                weight_scale=torch.empty((128, 4), dtype=torch.uint8),
            ),
            "MXFP4",
        ),
        (
            B12xNvFp4LinearKernel,
            "vllm.model_executor.kernels.linear.nvfp4.b12x",
            "_apply_b12x_nvfp4_linear",
            SimpleNamespace(
                weight=torch.empty((48, 64), dtype=torch.uint8),
                weight_scale=torch.empty((128, 8), dtype=torch.float8_e4m3fn),
                input_global_scale_inv=torch.tensor(2.0),
                alpha=torch.tensor(0.25),
            ),
            "NVFP4",
        ),
        (
            B12xFp8BlockScaledMMKernel,
            "vllm.model_executor.kernels.linear.scaled_mm.b12x",
            "_run_b12x_fp8_block_scaled_mm",
            SimpleNamespace(
                weight=torch.empty((256, 128), dtype=torch.float8_e4m3fn),
                weight_scale_inv=torch.empty((2, 1), dtype=torch.float32),
            ),
            "block-FP8",
        ),
    ],
)
def test_b12x_warmup_units_cover_token_counts(
    monkeypatch,
    kernel_cls,
    module_name: str,
    call_name: str,
    layer,
    name: str,
) -> None:
    calls = []
    monkeypatch.setattr(
        importlib.import_module(module_name),
        call_name,
        lambda *args: calls.append(args),
    )
    kernel = object.__new__(kernel_cls)

    unit = kernel.get_b12x_warmup_unit(layer, (1, 8), torch.bfloat16)
    unit.compile()

    assert unit.name == name
    assert [args[0].shape[0] for args in calls] == [1, 8]
    assert unit.key[-1] == torch.bfloat16


def test_b12x_mxfp8_warmup_unit(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: SimpleNamespace(
            mm=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=object()),
    )
    packed_weight = SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=256,
        weight=SimpleNamespace(values=torch.empty(1)),
    )
    layer = SimpleNamespace(b12x_mxfp8_packed_weight=packed_weight)
    kernel = object.__new__(B12xMxfp8LinearKernel)

    unit = kernel.get_b12x_warmup_unit(layer, (1, 8), torch.float16)
    unit.compile()

    assert [args[0].shape for args, _ in calls] == [(1, 128), (8, 128)]
    assert [kwargs["expected_m"] for _, kwargs in calls] == [1, 8]


def test_b12x_tensor_fp8_warmup_unit(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

    calls = []
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: SimpleNamespace(
            prewarm=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=object()),
    )
    packed_weight = SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=256,
        values=torch.empty(1),
    )
    layer = SimpleNamespace(b12x_tensor_fp8_packed_weight=packed_weight)
    kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)

    unit = kernel.get_b12x_warmup_unit(layer, (1, 8), torch.bfloat16)
    unit.compile()

    assert calls[0][0] == (packed_weight, (1, 8))
    assert calls[0][1]["out_dtype"] == torch.bfloat16


def test_b12x_warmup_deduplicates_registered_signatures(monkeypatch) -> None:
    import vllm.model_executor.warmup.b12x_warmup as warmup_mod

    calls: list[tuple[str, tuple[int, ...], torch.dtype]] = []

    class Provider:
        def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
            return B12xWarmupUnit(
                name="fake",
                key=(type(self), layer.shape, output_dtype),
                compile=lambda: calls.append((layer.name, token_counts, output_dtype)),
            )

    provider = Provider()
    layers = [
        SimpleNamespace(name="first", shape=(128, 256), b12x_warmup_provider=provider),
        SimpleNamespace(
            name="duplicate", shape=(128, 256), b12x_warmup_provider=provider
        ),
        SimpleNamespace(name="second", shape=(256, 256), b12x_warmup_provider=provider),
        SimpleNamespace(),
    ]
    scans = 0

    def modules():
        nonlocal scans
        scans += 1
        return iter(layers)

    worker = SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=modules),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    synchronized = []
    monkeypatch.setattr(warmup_mod, "current_platform", platform)
    monkeypatch.setattr(
        warmup_mod.torch.accelerator,
        "synchronize",
        lambda: synchronized.append(True),
    )

    b12x_warmup(worker, [1, 2])

    assert scans == 1
    assert calls == [
        ("first", (1, 2, 8), torch.bfloat16),
        ("second", (1, 2, 8), torch.bfloat16),
    ]
    assert synchronized == [True]
