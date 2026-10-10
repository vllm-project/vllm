# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_einsum,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class DeepGemmMxfp8BmmLinearKernel(Mxfp8LinearKernel):
    """Grouped MXFP8 einsum with per-row, 32-element weight scales."""

    def __init__(self, config: Mxfp8LinearLayerConfig):
        super().__init__(config)
        self.recipe = (1, 1, 32)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda() or not is_deep_gemm_supported():
            return False, "DeepGEMM BMM requires a supported CUDA device."
        if not current_platform.is_device_capability_family(100):
            return False, "DeepGEMM MXFP8 BMM requires Blackwell."
        return True, None

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if config.bmm_batch_size is None or config.bmm_batch_size <= 0:
            return False, "DeepGEMM BMM requires a positive batch size."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.ndim == 3:
            return
        scale = layer.weight_scale.data
        assert weight.dtype == torch.float8_e4m3fn and weight.ndim == 2
        assert scale.shape == (weight.shape[0], weight.shape[1] // 32)
        assert self.config.bmm_batch_size is not None
        weight, scale = deepgemm_post_process_fp8_weight_block(
            wq=weight,
            ws=scale,
            quant_block_shape=(1, 32),
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=self.config.bmm_batch_size,
        )
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", scale)
        layer.weight_block_size = [1, 32]

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project [tokens, groups, K] input to [tokens, groups, N]."""
        if isinstance(x, tuple):
            q_input, input_scale = x
        else:
            tokens, groups, k = x.shape
            q_input, input_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                x.reshape(tokens, groups * k),
                group_size=32,
                use_ue8m0=True,
            )
            q_input = q_input.view(tokens, groups, k)
            input_scale = input_scale.view(tokens, groups, k // 128)
        output = torch.ops.vllm.deepgemm_mxfp8_bmm(
            q_input, input_scale, layer.weight, layer.weight_scale, list(self.recipe)
        )
        if bias is not None:
            output = output + bias.view(output.shape[1:])
        return output


def _deepgemm_mxfp8_bmm_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    recipe: list[int],
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0], weight.shape[1]),
        device=x.device,
        dtype=torch.bfloat16,
    )


def _deepgemm_mxfp8_bmm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    recipe: list[int],
) -> torch.Tensor:
    output = torch.empty(
        (x.shape[0], weight.shape[0], weight.shape[1]),
        device=x.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (x, x_scale),
        (weight, weight_scale),
        output,
        recipe=tuple(recipe),
    )
    return output


direct_register_custom_op(
    "deepgemm_mxfp8_bmm", _deepgemm_mxfp8_bmm, fake_impl=_deepgemm_mxfp8_bmm_fake
)
