# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

__all__ = ["CompressedTensorsW4A8Mxfp4"]


class CompressedTensorsW4A8Mxfp4(CompressedTensorsW4A4Mxfp4):
    """MXFP4 weights with dynamic FP8 activations in compressed-tensors format.

    The weight layout matches W4A4; only the activation quantization differs,
    and it is taken from the checkpoint rather than chosen per platform.
    """

    def __init__(self, activation_quant_key: QuantKey):
        super().__init__(activation_quant_key=activation_quant_key)
