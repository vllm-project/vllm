# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch

import vllm.model_executor.kernels.linear.nvfp4.b12x as linear_b12x_module
import vllm.model_executor.layers.fused_moe.experts.b12x_nvfp4_moe as moe_b12x_module
from vllm.model_executor.kernels.linear import _NVFP4_BACKEND_TO_KERNEL
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.b12x import B12xNvFp4LinearKernel
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.b12x_nvfp4_moe import (
    B12xExperts,
    _get_b12x_workspace_pool,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    backend_to_kernel_cls,
    map_nvfp4_backend,
)


def test_b12x_nvfp4_linear_backend_registered() -> None:
    assert _NVFP4_BACKEND_TO_KERNEL["b12x"] is B12xNvFp4LinearKernel


def test_b12x_moe_backend_maps_to_b12x() -> None:
    assert map_nvfp4_backend("b12x") is NvFp4MoeBackend.B12X
    assert backend_to_kernel_cls(NvFp4MoeBackend.B12X) == [B12xExperts]


def test_b12x_experts_workspace_contract() -> None:
    experts = object.__new__(B12xExperts)
    assert experts.workspace_shapes(
        M=64,
        N=256,
        K=512,
        topk=8,
        global_num_experts=16,
        local_num_experts=16,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    ) == ((0,), (0,), (64, 512))


def test_b12x_linear_backend_imports_required_submodules(monkeypatch) -> None:
    imported_modules: list[str] = []
    grouped_scale_view = object()
    swizzle_block_scale = object()
    dense_gemm = object()

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        if module_name == "b12x.cute.fp4":
            return SimpleNamespace(
                as_grouped_scale_view=grouped_scale_view,
                swizzle_block_scale=swizzle_block_scale,
            )
        if module_name == "b12x.gemm.dense":
            return SimpleNamespace(dense_gemm=dense_gemm)
        raise ImportError(module_name)

    monkeypatch.setattr(linear_b12x_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(linear_b12x_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        linear_b12x_module.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )

    kernel = B12xNvFp4LinearKernel(NvFp4LinearLayerConfig())

    assert kernel._as_grouped_scale_view is grouped_scale_view
    assert kernel._swizzle_block_scale is swizzle_block_scale
    assert kernel._dense_gemm is dense_gemm
    assert "b12x.cute.fp4" in imported_modules
    assert "b12x.gemm.dense" in imported_modules


def test_b12x_workspace_pool_imports_tp_moe_submodule(monkeypatch) -> None:
    imported_modules: list[str] = []
    workspace_pool = object()

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        if module_name == "b12x.integration.tp_moe":
            return SimpleNamespace(
                allocate_tp_moe_workspace_pool=lambda: workspace_pool
            )
        raise ImportError(module_name)

    monkeypatch.setattr(moe_b12x_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(moe_b12x_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(moe_b12x_module, "_B12X_MOE_WORKSPACE_POOLS", {})

    assert _get_b12x_workspace_pool(torch.device("cuda")) is workspace_pool
    assert _get_b12x_workspace_pool(torch.device("cuda")) is workspace_pool
    assert imported_modules == ["b12x.integration.tp_moe"]


def test_b12x_experts_pass_reciprocal_input_scales(monkeypatch) -> None:
    captured: dict[str, object] = {}
    workspace_pool = object()

    def fake_b12x_moe_fp4(
        *args,
        workspace,
        output=None,
        input_scales_are_reciprocal=False,
        input_scales_static=False,
        **kwargs,
    ):
        del args, kwargs
        captured["workspace"] = workspace
        captured["output"] = output
        captured["input_scales_are_reciprocal"] = input_scales_are_reciprocal
        captured["input_scales_static"] = input_scales_static

    experts = object.__new__(B12xExperts)
    experts.quant_config = SimpleNamespace(
        a1_gscale=torch.ones(1, dtype=torch.float32),
        a2_gscale=torch.ones(1, dtype=torch.float32),
        g1_alphas=torch.ones(1, dtype=torch.float32),
        g2_alphas=torch.ones(1, dtype=torch.float32),
        w1_scale=torch.ones(1, dtype=torch.float32),
        w2_scale=torch.ones(1, dtype=torch.float32),
    )
    experts._b12x_moe_fp4 = fake_b12x_moe_fp4
    monkeypatch.setattr(
        moe_b12x_module,
        "_get_b12x_workspace_pool",
        lambda _device: workspace_pool,
    )

    hidden_states = torch.randn(2, 8)
    output = torch.empty_like(hidden_states)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(1, dtype=torch.uint8),
        w2=torch.empty(1, dtype=torch.uint8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=1,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured == {
        "workspace": workspace_pool,
        "output": output,
        "input_scales_are_reciprocal": True,
        "input_scales_static": True,
    }


def test_b12x_experts_fuse_input_scales_into_alphas_once() -> None:
    experts = object.__new__(B12xExperts)
    experts.quant_config = SimpleNamespace()

    layer = SimpleNamespace(
        w13_weight_scale_2=torch.nn.Parameter(torch.tensor([2.0, 3.0])),
        w2_weight_scale_2=torch.nn.Parameter(torch.tensor([5.0, 7.0])),
        w13_input_scale=torch.nn.Parameter(torch.tensor([11.0, 13.0])),
        w2_input_scale=torch.nn.Parameter(torch.tensor([17.0, 19.0])),
    )

    experts.process_weights_after_loading(layer)
    experts.process_weights_after_loading(layer)

    torch.testing.assert_close(
        layer.w13_weight_scale_2,
        torch.tensor([22.0, 39.0]),
    )
    torch.testing.assert_close(
        layer.w2_weight_scale_2,
        torch.tensor([85.0, 133.0]),
    )
