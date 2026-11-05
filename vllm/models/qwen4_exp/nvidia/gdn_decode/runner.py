# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-specific GDN tail dispatch; the shared attention class stays unchanged."""

from typing import cast

import torch

from vllm import envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization.modelopt import (
    FormatScheme,
    ModelOptLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)

from .dispatch import is_gdn_decode_context
from .kernel import GdnNormQuant

logger = init_logger(__name__)


class Qwen38GdnNormQuantAttention(QwenGatedDeltaNetAttention):
    def _output_projection(
        self, core_attn_out: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        batch = core_attn_out.shape[0]
        producer = getattr(self, "_gdn_decode_kernels", {}).get(batch)
        if producer is not None and is_forward_context_available():
            context = get_forward_context()
            if (
                not envs.VLLM_BATCH_INVARIANT
                and is_gdn_decode_context(
                    context.attn_metadata, batch, context.cudagraph_runtime_mode
                )
                and core_attn_out.shape == z.shape == (batch, 48, 128)
                and core_attn_out.dtype == z.dtype == torch.bfloat16
                and core_attn_out.stride() == (6144, 128, 1)
                and z.stride() == (6144 if batch == 1 else 16384, 128, 1)
                and core_attn_out.device
                == z.device
                == self.norm.weight.device
                == producer.quantized.device
                and self.norm.weight.shape == (128,)
                and self.norm.weight.dtype == torch.bfloat16
                and self.norm.weight.is_contiguous()
                and get_input_quant_key(self.out_proj) == kMxfp8Dynamic
                and getattr(self.out_proj.quant_method, "kernel", None)
                is self._gdn_decode_consumer
            ):
                data, scale = producer.run(core_attn_out, z, self.norm.weight)
                activation = QuantizedActivation(
                    data=data,
                    scale=scale,
                    orig_dtype=torch.bfloat16,
                    orig_shape=torch.Size((batch, 6144)),
                    quant_key=kMxfp8Dynamic,
                )
                output, _ = self.out_proj(activation)
                return output
        return super()._output_projection(core_attn_out, z)


def validate_gdn_layer(layer: Qwen38GdnNormQuantAttention) -> None:
    norm, linear = layer.norm, layer.out_proj
    method = linear.quant_method
    if not (
        type(norm) is RMSNormGated
        and getattr(norm._forward_method, "__func__", None) is RMSNormGated.forward_cuda
        and norm.eps == 1e-6
        and norm.activation == "sigmoid"
        and norm.norm_before_gate
        and norm.group_size is None
        and norm.bias is None
        and norm.weight.shape == (128,)
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and type(linear) is RowParallelLinear
        and linear.input_is_parallel
        and linear.params_dtype == torch.bfloat16
        and linear.tp_size == layer.tp_size == 1
        and linear.bias is None
        and linear.return_bias
        and type(method) is ModelOptLinearMethod
        and type(method.fmt) is FormatScheme
        and method.spec.weight == kMxfp8Static
        and method.spec.activation == kMxfp8Dynamic
        and type(method.kernel) is FlashInferCutlassMxfp8LinearKernel
        and get_input_quant_key(linear) == kMxfp8Dynamic
        and linear.weight.shape == (2560, 6144)
        and linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight.is_contiguous()
        and linear.weight_scale.shape == (491520,)
        and linear.weight_scale.dtype == torch.uint8
        and linear.weight_scale.is_contiguous()
        and norm.weight.device == linear.weight.device == linear.weight_scale.device
        and norm.weight.device.type == "cuda"
        and layer.num_v_heads == 48
        and layer.head_v_dim == 128
        and not layer.gqa_interleaved_layout
        and layer.speculative_config is None
        and not layer.enable_fused_gdn_spec_decode
        and layer._fused_gdn_gated_rms_norm is None
    ):
        raise ValueError(
            f"{layer.prefix}: GDN decode requires native BF16 RMS128/sigmoid and "
            "the processed MXFP8 CUTLASS RowParallelLinear(6144,2560) consumer"
        )


@torch.inference_mode()
def initialize_gdn_decode(model: torch.nn.Module) -> None:
    layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, Qwen38GdnNormQuantAttention)
    ]
    if len(layers) != 36:
        raise ValueError("gdn_norm_quant_decode requires exactly 36 GDN layers")
    for layer in layers:
        validate_gdn_layer(layer)
        if hasattr(layer, "_gdn_decode_kernels"):
            raise ValueError(
                "GDN decode does not support weight reload or reinitialization"
            )
    for layer in layers:
        kernels = {}
        for batch in (1, 2):
            device = layer.norm.weight.device
            producer = GdnNormQuant(batch, device=device)
            x = torch.zeros((batch, 48, 128), dtype=torch.bfloat16, device=device)
            backing = torch.zeros((batch, 16384), dtype=torch.bfloat16, device=device)
            z = backing[:, 10240:].view(batch, 48, 128)
            data, scale = producer.run(x, z, layer.norm.weight)
            layer.out_proj(
                QuantizedActivation(
                    data,
                    scale,
                    torch.bfloat16,
                    torch.Size((batch, 6144)),
                    kMxfp8Dynamic,
                )
            )
            kernels[batch] = producer
        layer._gdn_decode_kernels = kernels
        layer._gdn_decode_consumer = cast(
            ModelOptLinearMethod, layer.out_proj.quant_method
        ).kernel
    torch.accelerator.synchronize()
    logger.info("Initialized GDN norm+MXFP8 decode for 36 layers, physical B1/B2")
