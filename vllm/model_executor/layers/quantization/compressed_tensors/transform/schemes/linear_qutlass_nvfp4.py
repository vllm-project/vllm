# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod)


# Because qutlass fuses hadamard with quantization, it cannot automatically be
# composed with kernels in the way CompressedTensorsLinearTransformMethod does.
# Therefore, a separate scheme must be created for each quantized dtype
class QutlassLinearMethodNvFP4(CompressedTensorsLinearTransformMethod):

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # fused hadamard quant linear method
        raise NotImplementedError()
