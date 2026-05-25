# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)


def get_static_fp8_attn_output_scale(
    attn: Any, output_proj: Any
) -> torch.Tensor | None:
    """Return the output scale for manual attention + static FP8 quant fusion."""
    if not getattr(attn, "use_fused_attn_quant", False):
        return None

    fused_supported = getattr(attn.impl, "fused_output_quant_supported", None)
    if fused_supported is None or not fused_supported(kFp8StaticTensorSym):
        return None

    if getattr(output_proj, "input_quant_key", None) != kFp8StaticTensorSym:
        return None

    return getattr(output_proj, "input_scale", None)
