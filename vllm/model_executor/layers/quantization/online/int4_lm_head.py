# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time int4 quantization of an unquantized lm_head on RDNA (gfx11/gfx12).

Decode reads the whole lm_head every step, so on bandwidth-bound RDNA GPUs an
fp16/bf16 head is one of the largest per-token costs even when the checkpoint
quantizes every other linear layer. ``Int4LMHeadMethod`` quantizes the loaded
weight to asymmetric uint4 with one scale and zero point per 32 input channels
and runs it on the RDNA hybrid W4A16 op (HIP skinny GEMM for decode, Triton
for prefill).
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    pack_int4_exllama_shuffle,
    skinny_int4_layout,
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

GROUP_SIZE = 32
_UINT4_MAX = 15


def quantize_int4_rows(
    weight: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uint4 quantization of the rows of ``weight`` [N, K].

    Returns the codes [N, K] uint8, the scales [N, K // group_size] in the
    weight dtype and the zero points [N, K // group_size] uint8, such that
    ``(codes - zero_point) * scale`` approximates ``weight``. Each group's range
    is widened to include 0, so the zero point always fits in uint4.
    """
    n, k = weight.shape
    groups = weight.float().view(n, k // group_size, group_size)
    w_min = groups.amin(dim=-1).clamp(max=0)
    w_max = groups.amax(dim=-1).clamp(min=0)
    # Quantize against the stored scale so the kernel dequantizes exactly the
    # values the codes were rounded for.
    scale = (
        ((w_max - w_min) / _UINT4_MAX)
        .clamp(min=torch.finfo(weight.dtype).tiny)
        .to(weight.dtype)
    )
    scale_f = scale.float()
    zero_point = (-w_min / scale_f).round().clamp(0, _UINT4_MAX)
    codes = ((groups / scale_f.unsqueeze(-1)).round() + zero_point.unsqueeze(-1)).clamp(
        0, _UINT4_MAX
    )
    return (
        codes.to(torch.uint8).view(n, k),
        scale,
        zero_point.to(torch.uint8),
    )


def pack_zero_points(zero_point: torch.Tensor) -> torch.Tensor:
    """Pack uint4 zero points along rows: [N, G] -> [N // 8, G] int32.

    Row n lands in word n // 8 at bits 4 * (n % 8).
    """
    n, g = zero_point.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=zero_point.device)
    words = zero_point.to(torch.int32).view(n // 8, 8, g) << shifts.view(1, 8, 1)
    # The nibbles are disjoint, so the (wrapping) sum is their bitwise OR.
    return words.sum(dim=1, dtype=torch.int32)


@torch.no_grad()
def quantize_lm_head_int4(
    weight: torch.Tensor, group_size: int = GROUP_SIZE, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` [N, K] into the RDNA hybrid W4A16 layout.

    Returns the packed weight (int8 [N, K // 2], ExLlama shuffled), the scales
    [N, K // group_size] and the packed zero points [N // 8, K // group_size].
    Rows are processed in chunks to bound the fp32 working set.
    """
    n, k = weight.shape
    shuffled = torch.empty(n, k // 8, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    zero_points = torch.empty(
        n, k // group_size, dtype=torch.uint8, device=weight.device
    )
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        codes, scales[rows], zero_points[rows] = quantize_int4_rows(
            weight[rows], group_size
        )
        shuffled[rows] = pack_int4_exllama_shuffle(codes)
    return skinny_int4_layout(shuffled), scales, pack_zero_points(zero_points)


def _on_gfx1x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


class Int4LMHeadMethod(QuantizeMethodBase):
    """Loads the fp16/bf16 lm_head as usual, then replaces it with an int4
    per-group-32 copy run by the RDNA hybrid W4A16 op.

    A tied head only rebinds its own ``weight``, so the input embeddings keep
    their full-precision weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self._unquantized = UnquantizedEmbeddingMethod()

    @classmethod
    def from_config(
        cls, quant_method: QuantizeMethodBase, vllm_config: "VllmConfig", prefix: str
    ) -> "Int4LMHeadMethod":
        """Check that ``lm_head_quantization`` can replace ``quant_method``."""
        if not _on_gfx1x():
            raise ValueError(
                "lm_head_quantization='int4_per_group_32' needs a ROCm gfx11 or "
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
        if n % 8 or k % GROUP_SIZE:
            raise ValueError(
                f"lm_head_quantization needs a vocab shard divisible by 8 and a "
                f"hidden size divisible by {GROUP_SIZE}, got {n} x {k}."
            )
        w_q, w_s, w_zp = quantize_lm_head_int4(weight.data)
        replace_parameter(layer, "weight", w_q)
        layer.weight_scale = Parameter(w_s, requires_grad=False)
        layer.weight_zero_point = Parameter(w_zp, requires_grad=False)
        logger.info_once(
            "Quantized the lm_head to int4 per-group-%d (%d x %d per rank).",
            GROUP_SIZE,
            n,
            k,
        )

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            x_2d,
            layer.weight,
            layer.weight_scale,
            layer.weight_zero_point,
            bias,
            num_compute_units(),
            GROUP_SIZE,
        )
        return output.reshape(*x.shape[:-1], output.shape[-1])
