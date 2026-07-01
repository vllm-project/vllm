# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only tests for the after_eplb_rearrangement hook.

The GPU/distributed EPLB tests drive ``rearrange_expert_weights_inplace``
directly and never go through ``EplbState``, so they never exercise the
post-rearrangement hooks. These tests cover that gap: the dispatch wiring
and the NVFP4 derived-scale refresh, both runnable without a GPU.
"""

from types import SimpleNamespace

import torch

from vllm.distributed.eplb.eplb_state import _run_after_eplb_rearrangement_hooks


class _RecordingQuantMethod:
    def __init__(self):
        self.calls: list[object] = []

    def after_eplb_rearrangement(self, layer):
        self.calls.append(layer)


def _make_model(num_layers: int):
    """Fake MixtureOfExperts whose moe_layers mimic MoERunner.

    Like a real ``MoERunner``, each layer exposes ``routed_experts`` but has
    no public ``quant_method`` attribute -- that lives on routed_experts.
    """
    moe_layers = []
    for _ in range(num_layers):
        routed_experts = SimpleNamespace(quant_method=_RecordingQuantMethod())
        moe_layers.append(SimpleNamespace(routed_experts=routed_experts))
    return SimpleNamespace(moe_layers=moe_layers)


def test_hook_dispatches_via_routed_experts():
    # Regression: the hook must reach quant_method through routed_experts and
    # pass the routed_experts (a RoutedExperts) as the layer arg. A MoERunner
    # has no public quant_method, so ``moe_layer.quant_method`` would raise.
    model = _make_model(num_layers=3)

    for moe_layer in model.moe_layers:
        assert not hasattr(moe_layer, "quant_method")

    _run_after_eplb_rearrangement_hooks(model)

    for moe_layer in model.moe_layers:
        routed_experts = moe_layer.routed_experts
        assert routed_experts.quant_method.calls == [routed_experts]


def test_nvfp4_after_eplb_rearrangement_refreshes_scale_2():
    # after_eplb_rearrangement ignores self, so bypass the CUDA-dependent
    # __init__ and drive the real method body against a fake RoutedExperts.
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )

    method = object.__new__(CompressedTensorsW4A4Nvfp4MoEMethod)

    num_experts = 4
    torch.manual_seed(0)
    layer = SimpleNamespace(
        w13_weight_global_scale=torch.rand(num_experts, 2) + 0.5,
        w2_weight_global_scale=torch.rand(num_experts) + 0.5,
        w13_input_scale=torch.rand(num_experts) + 0.5,
        w2_input_scale=torch.rand(num_experts) + 0.5,
        w13_weight_scale_2=torch.zeros(num_experts),
        w2_weight_scale_2=torch.zeros(num_experts),
    )

    method.after_eplb_rearrangement(layer)
    torch.testing.assert_close(
        layer.w13_weight_scale_2,
        (1.0 / layer.w13_weight_global_scale[:, 0]) * layer.w13_input_scale,
    )
    torch.testing.assert_close(
        layer.w2_weight_scale_2,
        (1.0 / layer.w2_weight_global_scale) * layer.w2_input_scale,
    )

    # EPLB permutes the per-expert globals/inputs in place; the derived
    # scale_2 (separate storage) must be re-derived to match.
    scale_2_before = layer.w13_weight_scale_2.clone()
    perm = torch.tensor([2, 0, 3, 1])
    layer.w13_weight_global_scale = layer.w13_weight_global_scale[perm].contiguous()
    layer.w2_weight_global_scale = layer.w2_weight_global_scale[perm].contiguous()
    layer.w13_input_scale = layer.w13_input_scale[perm].contiguous()
    layer.w2_input_scale = layer.w2_input_scale[perm].contiguous()

    method.after_eplb_rearrangement(layer)
    torch.testing.assert_close(layer.w13_weight_scale_2, scale_2_before[perm])
