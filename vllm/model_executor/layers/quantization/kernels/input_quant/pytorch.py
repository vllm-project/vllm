# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    group_broadcast,
    prep_scale_for_group_broadcast,
)

from .InputQuantKernel import (
    _FP8_DTYPE,
    _FP8_MAX,
    _FP8_MIN,
    InputQuantConfig,
    InputQuantKernel,
)

_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


class PytorchInputQuantKernel(InputQuantKernel[InputQuantConfig]):
    @classmethod
    def is_supported(cls):
        return True, ""

    @classmethod
    def can_implement(cls, config: InputQuantConfig):
        if config.group_shape.is_per_group() and config.static:
            return (
                False,
                "Native pytorch group quant does not support static quantization.",
            )
        return True, ""

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type[InputQuantKernel[InputQuantConfig]]]:
        return [cls]

    def apply_group_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert scale is None, "Dynamic group quantization does not use scale"

        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales_raw = absmax / _FP8_MAX
        if self.config.use_ue8m0:
            scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
        scales = (scales_raw).clamp(min=_FP8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.is_column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales

    def apply_per_token_per_tensor_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (scale is not None) == self.is_static_quant
        assert scale_ub is None or (
            not self.is_static_quant
            and self.group_shape.is_per_token()
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape.is_per_token():
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
        else:
            scale = prep_scale_for_group_broadcast(scale, x, self.group_shape)

        out = (
            x.to(torch.float32)
            * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
        )
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.config.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale
