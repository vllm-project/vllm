# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8DynamicPacked,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_einsum,
    fp8_gemm_nt,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# DeepGEMM's MXFP8 recipe: per-row activations and weights in groups of 32.
_RECIPE = (1, 1, 32)


class DeepGemmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 GEMM on DeepGEMM, with per-row, 32-element UE8M0 scales.

    A 2D layer runs ``fp8_gemm_nt``; a grouped layer (``bmm_batch_size``)
    projects ``[tokens, groups, K]`` with ``fp8_einsum``. Activations arrive
    either in BF16 and are quantized here, or already quantized by their
    producer in DeepGEMM's packed layout (``kMxfp8DynamicPacked``).
    """

    def __init__(self, config: Mxfp8LinearLayerConfig):
        super().__init__(config)
        self.bmm_batch_size = config.bmm_batch_size

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda() or not is_deep_gemm_supported():
            return False, "DeepGEMM MXFP8 requires a supported CUDA device."
        if not current_platform.is_device_capability_family(100):
            return False, "DeepGEMM MXFP8 requires Blackwell."
        return True, None

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if config.bmm_batch_size is not None and config.bmm_batch_size <= 0:
            return False, "DeepGEMM BMM requires a positive batch size."
        return True, None

    def input_quant_key(self) -> QuantKey:
        return kMxfp8DynamicPacked

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.ndim == 3:
            return
        scale = layer.weight_scale.data
        n, k = weight.shape
        assert weight.dtype == torch.float8_e4m3fn and k % 32 == 0
        assert scale.shape == (n, k // 32), (tuple(scale.shape), (n, k // 32))
        weight, scale = deepgemm_post_process_fp8_weight_block(
            wq=weight,
            ws=scale,
            quant_block_shape=(1, 32),
            use_e8m0=False,
            is_bmm=self.bmm_batch_size is not None,
            bmm_batch_size=self.bmm_batch_size or 0,
        )
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", scale)
        layer.weight_block_size = [1, 32]
        if self.bmm_batch_size is None:
            layer.deep_gemm_warmup_provider = self

    def get_deep_gemm_warmup_weights(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer.weight, layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project ``[..., K]`` to ``[..., N]``."""
        if self.bmm_batch_size is not None:
            return self._apply_bmm(layer, x, bias)
        qa = as_quantized_activation(x, kMxfp8DynamicPacked)
        if qa is not None:
            input_shape = qa.orig_shape
            q_input, input_scale = qa.data.view(-1, input_shape[-1]), qa.scale
        else:
            assert isinstance(x, torch.Tensor)
            input_shape = x.shape
            q_input, input_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                x.reshape(-1, input_shape[-1]), group_size=32, use_ue8m0=True
            )
        output = torch.ops.vllm.deepgemm_mxfp8_gemm(
            q_input, input_scale, layer.weight, layer.weight_scale
        )
        if bias is not None:
            output = output + bias
        return output.view(*input_shape[:-1], output.shape[-1])

    def _apply_bmm(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project ``[tokens, groups, K]`` to ``[tokens, groups, N]``."""
        qa = as_quantized_activation(x, kMxfp8DynamicPacked)
        if qa is not None:
            q_input, input_scale = qa.data, qa.scale
        else:
            assert isinstance(x, torch.Tensor)
            tokens, groups, k = x.shape
            q_input, input_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                x.reshape(tokens, groups * k), group_size=32, use_ue8m0=True
            )
            q_input = q_input.view(tokens, groups, k)
            input_scale = input_scale.view(tokens, groups, k // 128)
        output = torch.ops.vllm.deepgemm_mxfp8_bmm(
            q_input, input_scale, layer.weight, layer.weight_scale
        )
        if bias is not None:
            output = output + bias.view(output.shape[1:])
        return output


def _deepgemm_mxfp8_gemm_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def _deepgemm_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    output = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
    fp8_gemm_nt(
        (x, x_scale),
        (weight, weight_scale),
        output,
        recipe=_RECIPE,
        is_deep_gemm_e8m0_used=True,
    )
    return output


def _deepgemm_mxfp8_bmm_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0], weight.shape[0], weight.shape[1]), dtype=torch.bfloat16
    )


def _deepgemm_mxfp8_bmm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    output = x.new_empty(
        (x.shape[0], weight.shape[0], weight.shape[1]), dtype=torch.bfloat16
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (x, x_scale),
        (weight, weight_scale),
        output,
        recipe=_RECIPE,
    )
    return output


direct_register_custom_op(
    "deepgemm_mxfp8_gemm", _deepgemm_mxfp8_gemm, fake_impl=_deepgemm_mxfp8_gemm_fake
)
direct_register_custom_op(
    "deepgemm_mxfp8_bmm", _deepgemm_mxfp8_bmm, fake_impl=_deepgemm_mxfp8_bmm_fake
)
