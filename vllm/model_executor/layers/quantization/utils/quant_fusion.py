from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


@dataclass
class QuantizedActivation:
    """Pre-quantized activation payload consumed directly by linear kernels."""

    q: torch.Tensor
    scale: torch.Tensor | None = None
    block_scale: torch.Tensor | None = None
    quant_key: QuantKey | None = None
