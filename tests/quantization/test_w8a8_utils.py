# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, cast

import torch


def _load_w8a8_utils(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_ops = types.ModuleType("vllm._custom_ops")
    fake_platforms = types.ModuleType("vllm.platforms")
    fake_vllm_any = cast(Any, fake_vllm)
    fake_ops_any = cast(Any, fake_ops)
    fake_platforms_any = cast(Any, fake_platforms)

    fake_ops_any.cutlass_scaled_mm_supports_fp8 = lambda capability: False
    fake_ops_any.cutlass_scaled_mm_supports_block_fp8 = lambda capability: False
    fake_ops_any.cutlass_group_gemm_supported = lambda capability: False
    fake_platforms_any.current_platform = types.SimpleNamespace(
        is_cuda=lambda: False,
        get_device_capability=lambda: None,
    )
    fake_vllm_any._custom_ops = fake_ops

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)

    module_path = (
        Path(__file__).parents[2]
        / "vllm"
        / "model_executor"
        / "layers"
        / "quantization"
        / "utils"
        / "w8a8_utils.py"
    )
    spec = importlib.util.spec_from_file_location("w8a8_utils_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_e4m3fn_to_e4m3fnuz_handles_e8m0fnu_scales(monkeypatch):
    w8a8_utils = _load_w8a8_utils(monkeypatch)
    weight = torch.tensor([1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([1.0, 2.0], dtype=torch.float32).to(
        torch.float8_e8m0fnu
    )
    input_scale = torch.tensor([4.0], dtype=torch.float32).to(torch.float8_e8m0fnu)

    _, normalized_weight_scale, normalized_input_scale = (
        w8a8_utils.normalize_e4m3fn_to_e4m3fnuz(
            weight,
            weight_scale,
            input_scale,
        )
    )

    assert normalized_weight_scale.dtype == torch.float8_e8m0fnu
    assert normalized_input_scale is not None
    assert normalized_input_scale.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(
        normalized_weight_scale.float(),
        torch.tensor([2.0, 4.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        normalized_input_scale.float(),
        torch.tensor([8.0], dtype=torch.float32),
    )
