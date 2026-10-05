# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only quantized copy of the target lm_head for drafting."""

import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)


class DraftLMHeadMethod:
    """Projects through the quantized copy for LogitsProcessor."""

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if layer.fp8_linear is not None:
            return layer.fp8_linear.apply_weights(layer, x, bias)
        return apply_fp4_marlin_linear(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.workspace,
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            bias,
        )


class QuantizedDraftLMHead(nn.Module):
    """Private quantized copy of this rank's lm_head shard for the drafter.

    Exposes what LogitsProcessor reads from a ParallelLMHead, so the drafter
    projects through it unchanged while the target keeps its own head. FP8 uses
    per-row weight scales and per-token activation scales on the platform's FP8
    linear kernel. NVFP4 stores E2M1 weights with E4M3 scales per 16 weights and
    runs on the Marlin kernel with activations in the model dtype.
    """

    def __init__(self, lm_head: nn.Module, quantization: str):
        super().__init__()
        if quantization == "nvfp4" and not is_fp4_marlin_supported():
            raise ValueError(
                "draft_lm_head_quantization='nvfp4' needs a CUDA GPU with "
                "compute capability 7.5 or newer."
            )
        weight = lm_head.weight.detach()
        if (
            not isinstance(
                lm_head.quant_method,
                (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
            )
            or getattr(lm_head, "bias", None) is not None
            or weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(
                "draft_lm_head_quantization needs an unquantized fp16/bf16 "
                "lm_head without bias."
            )
        self.tp_size = lm_head.tp_size
        self.shard_indices = lm_head.shard_indices
        self.org_vocab_size = lm_head.org_vocab_size
        self.output_size_per_partition, self.input_size_per_partition = weight.shape
        self.params_dtype = weight.dtype
        self.quant_method = DraftLMHeadMethod()
        self.fp8_linear = None
        if quantization == "nvfp4":
            global_scale = (weight.abs().amax().float() / (6.0 * 448.0)).clamp(
                min=1e-12
            )
            packed, scales = ops.scaled_fp4_quant(
                weight, 1.0 / global_scale, is_sf_swizzled_layout=False
            )
            self.weight = nn.Parameter(packed, requires_grad=False)
            self.weight_scale = nn.Parameter(scales, requires_grad=False)
            self.weight_global_scale = nn.Parameter(global_scale, requires_grad=False)
            prepare_fp4_layer_for_marlin(self)
            self.register_buffer(
                "workspace", marlin_make_workspace_new(weight.device), persistent=False
            )
        else:
            self.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTokenSym,
                weight_quant_key=kFp8StaticChannelSym,
                input_dtype=weight.dtype,
                out_dtype=weight.dtype,
                weight_shape=weight.shape,
                module_name="draft lm_head",
            )
            qweight, scale = ops.scaled_fp8_quant(weight, use_per_token_if_dynamic=True)
            self.logical_widths = [weight.shape[0]]
            self.orig_dtype = weight.dtype
            self.input_scale = None
            self.input_scale_ub = None
            self.weight = nn.Parameter(qweight.t(), requires_grad=False)
            self.weight_scale = nn.Parameter(
                scale.float().view(-1, 1), requires_grad=False
            )
            self.fp8_linear.process_weights_after_loading(self)
        # Buffers keep the copy out of the drafter's parameters (weight loading
        # and reloading) and out of its state dict.
        for name, param in list(self.named_parameters(recurse=False)):
            delattr(self, name)
            self.register_buffer(name, param.data, persistent=False)
        logger.info(
            "Drafting with a %s copy of the lm_head (%d x %d per rank).",
            quantization,
            *weight.shape,
        )
