# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional NVFP4 (W4A4) projection of the Muse Spark language-model heads.

Selected for the target and draft heads by
`additional_config={"nvfp4_lm_head": true}` with the specialized decode. The
head keeps its BF16 weight; a packed NVFP4 copy with FP8 block scales and one
global scale is made when the weights finish loading. A projection of at most
`MAX_ROWS` rows (decode and verification) quantizes the hidden states to NVFP4
with a static global scale (`ACT_AMAX`, a few times the largest hidden state
magnitude seen on real head inputs) and runs the FlashInfer CuTe DSL FP4 GEMM,
which reads a quarter of the weight bytes; any other projection runs the BF16
head. The logits are rounded to BF16 like the unquantized head's.
"""

from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .quantize_experts import MIN_AMAX, NVFP4_RANGE

logger = init_logger(__name__)

MAX_ROWS = 128
ROW_ALIGN = 128
FP4_BLOCK = 16
ACT_AMAX = 512.0
QUANT_BACKEND = "flashinfer-cutedsl"
GEMM_BACKEND = "cute-dsl"


def use_nvfp4_lm_head(lm_head: torch.nn.Module) -> None:
    """Make `lm_head` an NVFP4 head (`nvfp4_lm_head`), before its weights load."""
    lm_head.quant_method = Nvfp4LMHeadMethod()


def processor_is_identity(processor: "LogitsProcessor") -> bool:
    """Whether `processor` returns the head's logits unchanged (no scale, soft
    cap or logits passed in as input)."""
    return (
        not processor.logits_as_input
        and processor.soft_cap is None
        and processor.scale == 1.0
    )


def padded_head_logits(lm_head: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`lm_head`'s BF16 logits of `x` (cast to the weight dtype), without the
    output copy of an NVFP4 head (`Nvfp4LMHeadMethod.apply_padded`): unit
    stride in the vocab, possibly the padded row stride of its GEMM."""
    x = x.to(lm_head.weight.dtype)
    method = lm_head.quant_method
    if isinstance(method, Nvfp4LMHeadMethod):
        return method.apply_padded(lm_head, x)
    return method.apply(lm_head, x)


class Nvfp4LMHeadMethod(UnquantizedEmbeddingMethod):
    """The unquantized head with an NVFP4 W4A4 path for small projections."""

    def __init__(self) -> None:
        # The GEMM's B operands: the packed weight and its swizzled block
        # scales, transposed.
        self.weight_fp4_t: torch.Tensor | None = None
        self.weight_scale_t: torch.Tensor | None = None
        self.input_global_scale: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None
        self.rows = 0
        self.mm_fp4 = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight = layer.weight
        if (
            not current_platform.is_cuda()
            or not current_platform.is_device_capability_family(100)
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.shape[1] % FP4_BLOCK
        ):
            return
        from flashinfer import mm_fp4

        rows, hidden = weight.shape
        # The RTN recipe of `quantize_nvfp4_weight`, but the global scale is
        # the range over the amax (not one over amax over the range), which
        # rounds differently.
        weight_global = NVFP4_RANGE / weight_amax(weight).float().clamp(min=MIN_AMAX)
        weight_global = weight_global.reshape(1).to(torch.float32)
        if rows % ROW_ALIGN:
            padded = round_up(rows, ROW_ALIGN)
            weight = torch.cat((weight, weight.new_zeros(padded - rows, hidden)))
        packed, scale = ops.scaled_fp4_quant(
            weight, weight_global, is_sf_swizzled_layout=False
        )
        del weight
        self.mm_fp4 = mm_fp4
        self.weight_fp4_t = packed.t()
        self.weight_scale_t = swizzle_blockscale(scale).t()
        self.input_global_scale = torch.full(
            (1,),
            NVFP4_RANGE / ACT_AMAX,
            dtype=torch.float32,
            device=packed.device,
        )
        self.alpha = (1.0 / (self.input_global_scale * weight_global)).float()
        self.rows = rows
        logger.info(
            "Muse Spark NVFP4 lm_head: %d x %d shard on %s.",
            rows,
            hidden,
            packed.device,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.apply_padded(layer, x, bias).contiguous()

    def apply_padded(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`apply`, but a `[rows, num_rows]` view of the GEMM's padded output:
        row stride above `num_rows`, unit stride in the last dimension, no copy."""
        if (
            self.weight_fp4_t is None
            or bias is not None
            or x.dim() != 2
            or x.dtype != torch.bfloat16
            or not 1 <= x.shape[0] <= MAX_ROWS
        ):
            return super().apply(layer, x, bias)
        x_fp4, x_scale = ops.scaled_fp4_quant(
            x,
            self.input_global_scale,
            is_sf_swizzled_layout=True,
            backend=QUANT_BACKEND,
        )
        assert self.mm_fp4 is not None and self.weight_scale_t is not None
        logits = self.mm_fp4(
            x_fp4,
            self.weight_fp4_t,
            x_scale,
            self.weight_scale_t,
            self.alpha,
            torch.bfloat16,
            backend=GEMM_BACKEND,
            enable_pdl=True,
        )
        return logits[:, : self.rows]
