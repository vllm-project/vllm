# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch contract for DSv4 o_proj on SM12x (#43743).

These tests do not run the Triton kernel. They lock the capability
predicate so SM12x cannot select DeepGEMM's SM100 scale recipe
(layout.hpp:97 / Unknown SF transformation).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.parametrize(
    ("major", "recipe", "tma"),
    [
        (9, (1, 128, 128), False),
        (10, (1, 1, 128), True),
        (12, (1, 128, 128), False),
    ],
)
def test_fp8_einsum_recipe_is_per_capability(monkeypatch, major, recipe, tma):
    from vllm.models.deepseek_v4.nvidia.ops import o_proj

    monkeypatch.setattr(
        o_proj.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=major, minor=1),
    )
    got_recipe, got_tma = o_proj.compute_fp8_einsum_recipe()
    assert got_recipe == recipe
    assert got_tma is tma


def test_sm12x_triton_predicate_rejects_sm100_recipe(monkeypatch):
    from vllm.models.deepseek_v4.nvidia.ops import fp8_einsum as fe

    monkeypatch.setattr(
        fe.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=12, minor=1),
    )
    scale = torch.zeros(1, dtype=torch.float32)
    assert fe._use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", [1, 128, 128], scale
    )
    assert not fe._use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", [1, 1, 128], scale
    )


def test_sm100_does_not_take_triton_fallback(monkeypatch):
    from vllm.models.deepseek_v4.nvidia.ops import fp8_einsum as fe

    monkeypatch.setattr(
        fe.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=10, minor=0),
    )
    scale = torch.zeros(1, dtype=torch.float32)
    assert not fe._use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", [1, 128, 128], scale
    )


def test_o_proj_calls_triton_not_deepgemm_on_sm12x(monkeypatch):
    from vllm.models.deepseek_v4.nvidia.ops import o_proj

    monkeypatch.setattr(
        o_proj.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=12, minor=1),
    )
    monkeypatch.setattr(
        o_proj,
        "fused_inv_rope_fp8_quant",
        lambda *a, **k: (MagicMock(), MagicMock()),
    )
    triton_calls = []
    deepgemm_calls = []
    monkeypatch.setattr(
        o_proj,
        "_use_deepseek_v4_sm12x_triton_fp8_einsum",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        o_proj, "deepseek_v4_fp8_einsum", lambda *a, **k: triton_calls.append(1)
    )
    monkeypatch.setattr(o_proj, "fp8_einsum", lambda *a, **k: deepgemm_calls.append(1))

    o = torch.empty(2, 8, 192)
    wo_a = SimpleNamespace(
        weight=torch.empty(4, 8),
        weight_scale=torch.empty(1, dtype=torch.float32),
    )
    wo_b = lambda x: x  # noqa: E731
    o_proj.deep_gemm_fp8_o_proj(
        o,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(1),
        wo_a,
        wo_b,
        n_groups=1,
        heads_per_group=8,
        nope_dim=128,
        rope_dim=64,
        o_lora_rank=4,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )
    assert triton_calls == [1]
    assert deepgemm_calls == []
