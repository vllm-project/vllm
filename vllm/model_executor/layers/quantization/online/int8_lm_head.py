# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time int8 quantization of an unquantized lm_head on RDNA (gfx11/gfx12).

Decode reads the whole lm_head every step, so on bandwidth-bound RDNA GPUs an
fp16/bf16 head is one of the largest per-token costs even when the checkpoint
quantizes every other linear layer. ``Int8LMHeadMethod`` quantizes the loaded
weight to symmetric int8 with one scale per output channel and runs it on the
RDNA W8A16 op (HIP skinny GEMM for decode, Triton for prefill). It halves the
head's traffic, versus ~3.4x for int4, but flips far fewer top-1 tokens.
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from vllm.logger import init_logger

# Importing rdna_w8a16 registers torch.ops.vllm.rdna_w8a16_apply.
from vllm.model_executor.kernels.linear.mixed_precision.rdna_w8a16 import (
    empty_int8_weight,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# -128 is left unused so the code range is symmetric around 0.
_INT8_MAX = 127
# wvSplitK_int8 loads K in 16-byte chunks and, at large vocab, tiles rows by 4.
_K_ALIGN = 16
_N_ALIGN = 4


@torch.no_grad()
def quantize_lm_head_int8(
    weight: torch.Tensor, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization of ``weight`` [N, K].

    Returns the codes in the RDNA W8A16 layout (int8 [N, K], see
    ``empty_int8_weight``) and the scales [N] in the weight dtype, such that
    ``codes * scale[:, None]`` approximates ``weight``. Rows are processed in
    chunks to bound the fp32 working set.
    """
    n, k = weight.shape
    codes = empty_int8_weight(n, k, weight.device)
    scales = torch.empty(n, dtype=weight.dtype, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        w = weight[rows].float()
        # Quantize against the stored scale so the kernel dequantizes exactly
        # the values the codes were rounded for.
        scale = (
            (w.abs().amax(dim=-1) / _INT8_MAX)
            .clamp(min=torch.finfo(weight.dtype).tiny)
            .to(weight.dtype)
        )
        scales[rows] = scale
        codes[rows] = (
            (w / scale.float().unsqueeze(-1)).round().clamp(-_INT8_MAX, _INT8_MAX)
        )
    return codes, scales


def _on_gfx1x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


class Int8LMHeadMethod(QuantizeMethodBase):
    """Loads the fp16/bf16 lm_head as usual, then replaces it with an int8
    per-channel copy run by the RDNA W8A16 op.

    A tied head only rebinds its own ``weight``, so the input embeddings keep
    their full-precision weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self._unquantized = UnquantizedEmbeddingMethod()

    @classmethod
    def from_config(
        cls, quant_method: QuantizeMethodBase, vllm_config: "VllmConfig", prefix: str
    ) -> "Int8LMHeadMethod":
        """Check that ``lm_head_quantization`` can replace ``quant_method``."""
        if not _on_gfx1x():
            raise ValueError(
                "lm_head_quantization='int8_per_channel' needs a ROCm gfx11 or "
                "gfx12 GPU."
            )
        if not isinstance(
            quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            raise ValueError(
                f"lm_head_quantization needs an unquantized lm_head, but {prefix} "
                f"uses {type(quant_method).__name__}."
            )
        model_config = vllm_config.model_config
        if (
            model_config is not None
            and model_config.head_dtype is not None
            and model_config.head_dtype != model_config.dtype
        ):
            raise ValueError(
                "lm_head_quantization is not supported with a head_dtype that "
                "differs from the model dtype."
            )
        if vllm_config.lora_config is not None:
            raise ValueError("lm_head_quantization is not supported with LoRA.")
        return cls()

    def create_weights(self, layer: Module, *weight_args, **extra_weight_attrs):
        self._unquantized.create_weights(layer, *weight_args, **extra_weight_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        n, k = weight.shape
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"lm_head_quantization needs an fp16/bf16 lm_head, got {weight.dtype}."
            )
        if n % _N_ALIGN or k % _K_ALIGN:
            raise ValueError(
                f"lm_head_quantization needs a vocab shard divisible by {_N_ALIGN} "
                f"and a hidden size divisible by {_K_ALIGN}, got {n} x {k}."
            )
        w_q, w_s = quantize_lm_head_int8(weight.data)
        replace_parameter(layer, "weight", w_q)
        layer.weight_scale = Parameter(w_s, requires_grad=False)
        logger.info_once(
            "Quantized the lm_head to int8 per-channel (%d x %d per rank).", n, k
        )

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.ops.vllm.rdna_w8a16_apply(
            x_2d, layer.weight, layer.weight_scale, bias, num_compute_units()
        )
        return output.reshape(*x.shape[:-1], output.shape[-1])
