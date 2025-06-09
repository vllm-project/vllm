# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

OCP_MX_BLOCK_SIZE = 32


def per_token_group_quant_mxfp4(x: torch.Tensor,
                                block_k: int,
                                scale_calculation_mode: str = "even"
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            fake_quantize_fp4_fp6_per_group_with_scale)
        from quark.torch.quantization.utils import (even_round,
                                                    reshape_to_blocks)
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`.") from err

    axis = -1
    block_x = reshape_to_blocks(x, block_k, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    amax = amax.squeeze(-1)

    # TODO: there are other rounding strategies supported in quark and in the
    # config.json that we do not check for here!
    if scale_calculation_mode != "even":
        raise NotImplementedError(
            f"Scale calculation mode {scale_calculation_mode} is not yet "
            "supported in MX-FP4 quantization")
    scale = even_round(amax, "fp4")

    # Apply dequantize(quantize(x)).
    x = fake_quantize_fp4_fp6_per_group_with_scale(
        x,
        scale.to(x.device),
        axis=axis,
        group_size=block_k,
        quant_dtype="fp4",
    )

    return x, scale
