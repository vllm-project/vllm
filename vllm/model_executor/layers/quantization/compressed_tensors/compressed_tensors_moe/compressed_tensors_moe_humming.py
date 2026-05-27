# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CompressedTensors FusedMoE method that dispatches pack-quantized INT experts to
the humming MoE kernel. Built for bit widths that Marlin / WNA16 does not
support (e.g. 2-bit GSQ checkpoints).
"""

import torch
from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_humming import (  # noqa: E501
    build_humming_layer_quant_config,
)
from vllm.model_executor.layers.quantization.humming import HummingMoEMethod

__all__ = ["CompressedTensorsHummingMoEMethod"]


class CompressedTensorsHummingMoEMethod(HummingMoEMethod):
    """
    Pack-quantized INT FusedMoE backed by the humming MoE kernel, built from a
    compressed-tensors ``QuantizationArgs``. The model's HF config stays
    ``quant_method: compressed-tensors`` — no user override required.

    The parent ``HummingMoEMethod.process_weights_after_loading`` ends with an
    ``assert self.moe_quant_config is not None`` and uses the value immediately
    after, but the parent never populates it. Other vLLM MoE methods (e.g.
    ``UnquantizedFusedMoEMethod``) populate ``moe_quant_config`` inside their
    pwal; we bridge to that pattern here.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        format_: str,
        moe: FusedMoEConfig,
    ):
        quant_config = build_humming_layer_quant_config(weight_quant, format_)
        super().__init__(quant_config, moe)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(self, "processed", False):
            return

        try:
            super().process_weights_after_loading(layer)
        except AssertionError:
            # Recover only the documented failure case: the final-block assert
            # in the parent. Re-raise if the state doesn't match what we expect
            # (loop completed, experts not yet built, moe_quant_config still
            # None), so we don't swallow unrelated assertion failures.
            if (
                self.moe_quant_config is not None
                or getattr(self, "experts", None) is not None
                or not getattr(self, "processed", False)
                or not getattr(layer, "weight_schemas", None)
            ):
                raise

            from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (  # noqa: E501
                HummingGroupedExperts,
                HummingIndexedExperts,
                get_humming_moe_gemm_type,
            )

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            if get_humming_moe_gemm_type() == "indexed":
                self.experts = HummingIndexedExperts(
                    layer, self.moe, self.moe_quant_config
                )
            else:
                self.experts = HummingGroupedExperts(
                    layer, self.moe, self.moe_quant_config
                )
