# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only quantized copy of the target lm_head for drafting."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_GROUP_SIZE = 16


def quantize_nvfp4(
    weight: torch.Tensor, chunk_rows: int = 16384
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize rows to NVFP4 in the ModelOpt checkpoint layout.

    Returns packed E2M1 codes (low nibble first), E4M3 scales per 16 weights and
    the fp32 global scale.
    """
    n, k = weight.shape
    group_amax = weight.view(n, -1, NVFP4_GROUP_SIZE).abs().amax(-1).float()
    global_scale = (group_amax.max() / (6.0 * 448.0)).clamp(min=1e-12)
    scales = (group_amax / (6.0 * global_scale)).to(torch.float8_e4m3fn)
    values = torch.tensor(E2M1_VALUES, device=weight.device)
    midpoints = (values[1:] + values[:-1]) / 2
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        step = scales[rows].float().repeat_interleave(NVFP4_GROUP_SIZE, 1)
        x = weight[rows].float() / (step * global_scale).clamp(min=1e-30)
        codes = torch.bucketize(x.abs(), midpoints).to(torch.uint8)
        codes |= (x < 0).to(torch.uint8) << 3
        packed[rows] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales, global_scale


class QuantizedDraftLMHead(nn.Module):
    """Private weight-only copy of this rank's lm_head shard for the drafter.

    Exposes what LogitsProcessor reads from a ParallelLMHead, so the drafter
    projects through it unchanged while the target keeps its own head. FP8
    stores E4M3 rows with one scale per row; NVFP4 stores E2M1 weights with
    E4M3 scales per 16 weights. Both run on the Marlin kernel with activations
    in the model dtype.
    """

    def __init__(self, lm_head: nn.Module, quantization: str):
        super().__init__()
        supported = {"fp8": is_fp8_marlin_supported, "nvfp4": is_fp4_marlin_supported}
        if not supported[quantization]():
            raise ValueError(
                f"draft_lm_head_quantization={quantization!r} needs a CUDA GPU with "
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
        self.quantization = quantization
        # LogitsProcessor projects through lm_head.quant_method.apply().
        self.quant_method = SimpleNamespace(apply=self._project)
        if quantization == "nvfp4":
            packed, scales, global_scale = quantize_nvfp4(weight)
            self.weight = nn.Parameter(packed, requires_grad=False)
            self.weight_scale = nn.Parameter(scales, requires_grad=False)
            self.weight_global_scale = nn.Parameter(global_scale, requires_grad=False)
            prepare_fp4_layer_for_marlin(self)
        else:
            qweight, scale = ops.scaled_fp8_quant(weight, use_per_token_if_dynamic=True)
            self.weight = nn.Parameter(qweight, requires_grad=False)
            self.weight_scale = nn.Parameter(scale, requires_grad=False)
            self.orig_dtype = weight.dtype
            prepare_fp8_layer_for_marlin(self, size_k_first=False)
        # Buffers keep the copy out of the drafter's parameters (weight loading
        # and reloading) and out of its state dict.
        for name, param in list(self.named_parameters(recurse=False)):
            delattr(self, name)
            self.register_buffer(name, param.data, persistent=False)
        self.register_buffer(
            "workspace", marlin_make_workspace_new(weight.device), persistent=False
        )
        logger.info(
            "Drafting with a %s copy of the lm_head (%d x %d per rank).",
            quantization,
            *weight.shape,
        )

    def _project(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.quantization == "nvfp4":
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
        return apply_fp8_marlin_linear(
            x,
            layer.weight,
            layer.weight_scale,
            layer.workspace,
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            bias,
        )
