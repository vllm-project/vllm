# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

_PACKED_WEIGHTS = ("w13_weight", "w2_weight")


class CompressedTensorsW4A8Mxfp4MoEMethod(Mxfp4MoEMethod):
    """MXFP4 weights with dynamic FP8 activations in compressed-tensors format.

    Only the checkpoint naming differs from the native MXFP4 path, so backend
    selection, weight conversion and execution are inherited.
    """

    # The packed -> unpacked rename means a processed state dict no longer
    # matches the parameters registered by create_weights.
    supports_pre_processed_weights = False

    def create_weights(self, layer: RoutedExperts, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        for name in _PACKED_WEIGHTS:
            param = getattr(layer, name)
            delattr(layer, name)
            layer.register_parameter(f"{name}_packed", param)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for name in _PACKED_WEIGHTS:
            param = getattr(layer, f"{name}_packed")
            delattr(layer, f"{name}_packed")
            layer.register_parameter(name, param)
        super().process_weights_after_loading(layer)
