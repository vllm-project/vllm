# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time quantization of an unquantized lm_head on RDNA (gfx11/gfx12).

Decode reads the whole lm_head every step, so on bandwidth-bound RDNA GPUs an
fp16/bf16 head is one of the largest per-token costs even when the checkpoint
quantizes every other linear layer. ``RdnaLMHeadMethod`` holds what every
``KernelConfig.lm_head_quantization`` scheme shares; subclasses quantize the
loaded weight and run it.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

import torch
from torch.nn import Module, Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _on_gfx1x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


class RdnaLMHeadMethod(QuantizeMethodBase):
    """Loads the fp16/bf16 lm_head as usual, then replaces its weight with a
    quantized copy run by an RDNA op.

    A tied head only rebinds its own ``weight``, so the input embeddings keep
    their full-precision weight.
    """

    scheme: ClassVar[str]
    """The ``KernelConfig.lm_head_quantization`` value this method implements."""
    n_align: ClassVar[int]
    """Required divisor of the vocab shard (output rows)."""
    k_align: ClassVar[int]
    """Required divisor of the hidden size (input columns)."""

    def __init__(self) -> None:
        super().__init__()
        self._unquantized = UnquantizedEmbeddingMethod()

    @classmethod
    def from_config(
        cls, quant_method: QuantizeMethodBase, vllm_config: "VllmConfig", prefix: str
    ) -> Self:
        """Check that ``lm_head_quantization`` can replace ``quant_method``."""
        if not _on_gfx1x():
            raise ValueError(
                f"lm_head_quantization={cls.scheme!r} needs a ROCm gfx11 or gfx12 GPU."
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
        if n % self.n_align or k % self.k_align:
            raise ValueError(
                f"lm_head_quantization={self.scheme!r} needs a vocab shard "
                f"divisible by {self.n_align} and a hidden size divisible by "
                f"{self.k_align}, got {n} x {k}."
            )
        self.quantize_weight(layer, weight.data)
        logger.info_once(
            "Quantized the lm_head to %s (%d x %d per rank).", self.scheme, n, k
        )

    @staticmethod
    def set_weight(layer: Module, weight: torch.Tensor) -> None:
        """Register ``weight`` as a new ``layer.weight`` parameter.

        Not `replace_parameter`, which keeps the loaded parameter and may copy
        into its storage: a tied head shares both with the input embeddings, and
        an untied head would keep its fp16 storage allocated.
        """
        layer.weight = Parameter(weight, requires_grad=False)

    @abstractmethod
    def quantize_weight(self, layer: Module, weight: torch.Tensor) -> None:
        """Replace ``layer.weight`` with the quantized copy of ``weight`` (via
        `set_weight`) and register any scale tensors the op needs."""
        raise NotImplementedError

    @abstractmethod
    def matmul(
        self, layer: Module, x_2d: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Return ``x_2d @ W.T + bias`` for the quantized weight, [M, N]."""
        raise NotImplementedError

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output = self.matmul(layer, x_2d, bias)
        return output.reshape(*x.shape[:-1], output.shape[-1])
