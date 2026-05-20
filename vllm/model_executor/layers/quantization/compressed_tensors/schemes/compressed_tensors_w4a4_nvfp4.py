# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.quantization import QuantSpec
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.quant_spec_scheme import (  # noqa: E501
    QuantSpecScheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(QuantSpecScheme):
    def __init__(self):
        super().__init__(
            QuantSpec(
                weight=kNvfp4Static,
                activation=kNvfp4Dynamic,
            )
        )
