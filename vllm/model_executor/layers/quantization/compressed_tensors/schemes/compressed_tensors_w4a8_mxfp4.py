# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A8Mxfp4"]


class CompressedTensorsW4A8Mxfp4(CompressedTensorsW4A4Mxfp4):
    """MXFP4 weights with dynamic FP8 activations in compressed-tensors format.

    The weight layout matches W4A4; only the activation quantization differs.
    The checkpoint's FP8 activations are used when a kernel supports them,
    otherwise this falls back to the W4A4 activation format.
    """

    def __init__(self, activation_quant_key: QuantKey):
        try:
            super().__init__(activation_quant_key=activation_quant_key)
        except ValueError as e:
            logger.warning_once(
                "No MXFP4 linear kernel supports %s activations on this "
                "platform; falling back to MXFP4 activations. %s",
                activation_quant_key,
                e,
            )
            super().__init__()
