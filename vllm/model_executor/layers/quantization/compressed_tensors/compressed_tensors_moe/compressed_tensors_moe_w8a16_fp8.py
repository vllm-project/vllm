# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


class CompressedTensorsW8A16Fp8MoEMethod(Fp8MoEMethod):
    """Load FP8 weights with a scale per output channel per 128 inputs."""

    def __init__(self, layer: RoutedExperts):
        super().__init__(Fp8Config(weight_block_size=[1, 128]), layer)
        # Compressed-tensors uses weight_scale for grouped weights.
        self.weight_scale_name = "weight_scale"
