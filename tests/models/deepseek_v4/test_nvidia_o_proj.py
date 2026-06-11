# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_o_proj_module() -> Any:
    root = Path(__file__).parents[3]
    module_path = (
        root / "vllm" / "models" / "deepseek_v4" / "nvidia" / "ops" / "o_proj.py"
    )

    stub_names = (
        "vllm.models.deepseek_v4.common.ops",
        "vllm.platforms",
        "vllm.utils.deep_gemm",
    )
    original_modules: dict[str, types.ModuleType] = {
        name: sys.modules[name] for name in stub_names if name in sys.modules
    }
    missing_modules = {name for name in stub_names if name not in sys.modules}
    try:
        common_ops = types.ModuleType("vllm.models.deepseek_v4.common.ops")
        common_ops.fused_inv_rope_fp8_quant = None  # type: ignore[attr-defined]
        sys.modules["vllm.models.deepseek_v4.common.ops"] = common_ops

        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = None  # type: ignore[attr-defined]
        sys.modules["vllm.platforms"] = platforms

        deep_gemm = types.ModuleType("vllm.utils.deep_gemm")
        deep_gemm.fp8_einsum = None  # type: ignore[attr-defined]
        sys.modules["vllm.utils.deep_gemm"] = deep_gemm

        spec = importlib.util.spec_from_file_location("test_o_proj", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in original_modules.items():
            sys.modules[name] = module
        for name in missing_modules:
            sys.modules.pop(name, None)


o_proj = load_o_proj_module()
get_fp8_weight_scale = o_proj.get_fp8_weight_scale
inv_rope_bf16_o_proj = o_proj.inv_rope_bf16_o_proj
deep_gemm_fp8_o_proj = o_proj.deep_gemm_fp8_o_proj


def test_get_fp8_weight_scale_prefers_weight_scale_inv():
    layer = nn.Module()
    layer.weight_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(torch.tensor([2.0]), requires_grad=False)

    assert get_fp8_weight_scale(layer) is layer.weight_scale_inv


def test_get_fp8_weight_scale_accepts_weight_scale():
    layer = nn.Module()
    layer.weight_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    assert get_fp8_weight_scale(layer) is layer.weight_scale


def test_get_fp8_weight_scale_returns_none_without_scale():
    assert get_fp8_weight_scale(nn.Module()) is None


class FakeWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return x[..., :1]


def test_inv_rope_bf16_o_proj_uses_unquantized_linear_path():
    wo_a = FakeWoA()
    o = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
    cos_sin_cache = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    out = inv_rope_bf16_o_proj(
        o,
        torch.tensor([0], dtype=torch.long),
        cos_sin_cache,
        wo_a,
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=1,
    )

    assert out.shape == (1, 1, 1)
    assert wo_a.input is not None
    expected = torch.tensor([[[1.0, 2.0, 4.0, -3.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(wo_a.input, expected)


class FakeGroupedWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                ],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )


class FakeSingleGroupWoA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )


def test_inv_rope_bf16_o_proj_reshapes_flat_grouped_weight():
    o = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=torch.bfloat16
    )
    out = inv_rope_bf16_o_proj(
        o,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeGroupedWoA(),
        n_groups=2,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
    )

    expected = torch.tensor([[[1.0, 2.0], [10.0, 12.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


class FakeWoB(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return x + 1


def test_deep_gemm_fp8_o_proj_uses_bf16_fallback_without_scale():
    wo_b = FakeWoB()
    out = deep_gemm_fp8_o_proj(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        FakeSingleGroupWoA(),
        wo_b,
        n_groups=1,
        heads_per_group=1,
        nope_dim=2,
        rope_dim=2,
        o_lora_rank=2,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )

    assert wo_b.input is not None
    torch.testing.assert_close(
        wo_b.input, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))
