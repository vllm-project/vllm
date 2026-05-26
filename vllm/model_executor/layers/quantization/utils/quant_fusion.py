# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
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


def get_mla_attn_quant_params(
    attn: Any, output_proj: Any
) -> tuple[torch.Tensor | None, torch.Tensor | None, int | None]:
    """Return the output scale, block scale, and group size for manual MLA attention + quant fusion.

    Returns:
        (output_scale, output_block_scale, quant_group_size)
    """
    if not getattr(attn, "use_fused_attn_quant", False):
        return None, None, None

    fused_supported = getattr(attn.impl, "fused_output_quant_supported", None)
    if fused_supported is None:
        return None, None, None

    # Check which quantization scheme is supported
    output_quant_key = getattr(output_proj, "input_quant_key", None)

    if output_quant_key == kFp8StaticTensorSym:
        if not fused_supported(kFp8StaticTensorSym):
            return None, None, None
        return getattr(output_proj, "input_scale", None), None, None

    elif output_quant_key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym):
        if not fused_supported(output_quant_key):
            return None, None, None
        output_scale = getattr(output_proj, "input_scale", None)
        output_block_scale = getattr(output_proj, "input_block_scale", None)
        quant_group_size = getattr(output_proj, "quant_group_size", None)
        return output_scale, output_block_scale, quant_group_size

    return None, None, None
