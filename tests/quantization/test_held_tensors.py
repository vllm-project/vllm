# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensors held by quant methods must be registered on their layer."""

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    make_fp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import register_held_tensors


class _MoEMethod(QuantizeMethodBase):
    def __init__(self, moe_quant_config: FusedMoEQuantConfig) -> None:
        self.moe_quant_config = moe_quant_config

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError


def test_computed_moe_quant_config_scales_are_registered():
    """FlashInfer CUTLASS per-tensor FP8 computes ``a1_gscale = 1 / a1_scale``
    in the quant config, so only the quant method holds it. It must become a
    buffer of the layer, or sleep mode and weight reload leave it stale."""
    layer = torch.nn.Module()
    for name in ("w13_weight_scale", "w2_weight_scale"):
        layer.register_parameter(name, torch.nn.Parameter(torch.rand(4)))
    for name in ("w13_input_scale", "w2_input_scale"):
        layer.register_parameter(name, torch.nn.Parameter(torch.rand(1)))
    config = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.FLASHINFER_CUTLASS,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a1_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )
    layer.quant_method = _MoEMethod(config)

    register_held_tensors(layer)

    registered = {t.untyped_storage().data_ptr() for t in layer.buffers()}
    for tensor in (
        config.a1_gscale,
        config.a2_gscale,
        config.g1_alphas,
        config.g2_alphas,
    ):
        assert tensor is not None
        assert tensor.untyped_storage().data_ptr() in registered
    # Scales the layer already registers as parameters are not duplicated.
    assert len(list(layer.buffers())) == 4
