# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

import torch

import vllm.envs as envs

OCP_MX_BLOCK_SIZE = 32


def per_token_group_dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor,
                                  block_k: int,
                                  float_dtype: torch.dtype) -> torch.Tensor:
    if envs.VLLM_QUARK_USE_KERNELS:
        try:
            from quark.torch.kernel.hw_emulation.extensions import kernel_ext
        except ImportError as err:
            raise ImportError("The package `amd-quark` is required to use "
                            "MX-FP4 models. Please install it with `pip install "
                            "amd-quark`.") from err

        dequant_weight_shape = (*x.shape[:-1], x.shape[-1] * 2)

        dq_w = torch.empty(dequant_weight_shape, device=x.device, dtype=float_dtype)
        kernel_ext.dq_uint8_mxfp4_to_half(x, scale, dq_w, OCP_MX_BLOCK_SIZE)

        return dq_w
    else:
        try:
            from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
                dequantize_fp4_fp6_per_group)
            from quark.torch.utils import pack
        except ImportError as e:
            raise ImportError("The package `amd-quark` is required to use "
                            "MX-FP4 models. Please install it with `pip install "
                            "amd-quark`.") from e

        # TODO: Both arguments are unused.
        pack_method = pack.Pack_fp4(None, dtype="fp4")
        # TODO: Both 'reorder' and 'origin_packed_axis_size' are unused.
        unpacked_x = pack_method.unpack(x, reorder=False)

        scale = 2**(scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

        # TODO: `dequantize_fp4_fp6_per_group` and `prepare_inputs_per_group` always return fp32.
        return dequantize_fp4_fp6_per_group(unpacked_x,
                                            scale,
                                            axis=-1,
                                            group_size=block_k,
                                            quant_dtype="fp4").to(float_dtype)


def per_token_group_quant_mxfp4(x: torch.Tensor,
                                block_k: int,
                                scale_calculation_mode: str = "even"
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    if envs.VLLM_QUARK_USE_KERNELS:
        try:
            from quark.torch.kernel.hw_emulation.extensions import kernel_ext
        except ImportError as err:
            raise ImportError("The package `amd-quark` is required to use "
                            "MX-FP4 models. Please install it with `pip install "
                            "amd-quark`.") from err

        x = kernel_ext.qdq_mxfp4(x, OCP_MX_BLOCK_SIZE)

        return x
    else:
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

        return x
