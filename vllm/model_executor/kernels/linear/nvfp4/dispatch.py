# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 W4A16 linear that picks Marlin or FlashInfer CuTe-DSL by token count.

Marlin is faster for the few tokens of a small decode batch and CuTe-DSL for large
ones. Both weight formats stay resident, so this costs extra memory. The branch lives
inside a custom op, so torch.compile does not specialize on the number of tokens and
CUDA graphs bake in the kernel for each captured size.

Enabled by setting VLLM_NVFP4_W4A16_DISPATCH_M to the token count from which CuTe-DSL
is used.
"""

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
from .flashinfer import FlashInferCuteDslNvFp4W4A16LinearKernel
from .marlin import MarlinNvFp4LinearKernel

logger = init_logger(__name__)

# One entry per layer:
# (marlin_kernel, marlin_layer, cute_kernel, cute_layer, threshold_tokens)
_ENTRIES: list = []


def _dispatch_impl(x: torch.Tensor, idx: int, n: int) -> torch.Tensor:
    marlin_k, marlin_l, cute_k, cute_l, threshold = _ENTRIES[idx]
    if x.shape[0] >= threshold:
        return cute_k.apply_weights(cute_l, x)
    return marlin_k.apply_weights(marlin_l, x)


def _dispatch_fake(x: torch.Tensor, idx: int, n: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="nvfp4_w4a16_dispatch",
    op_func=_dispatch_impl,
    mutates_args=[],
    fake_impl=_dispatch_fake,
)


class MarlinCuteDslNvFp4W4A16DispatchKernel(NvFp4LinearKernel):
    """Marlin below VLLM_NVFP4_W4A16_DISPATCH_M tokens, CuTe-DSL at or above it."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        ok_m, why_m = MarlinNvFp4LinearKernel.is_supported()
        ok_c, why_c = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported(
            compute_capability
        )
        if ok_m and ok_c:
            return True, None
        return False, why_m or why_c

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.threshold = envs.VLLM_NVFP4_W4A16_DISPATCH_M
        self.marlin = MarlinNvFp4LinearKernel(config)
        self.cute = FlashInferCuteDslNvFp4W4A16LinearKernel(config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Each backend repacks its weights in place, so CuTe-DSL gets a copy to repack.
        shadow = nn.Module()
        for name in ("weight", "weight_scale", "weight_global_scale"):
            param = nn.Parameter(getattr(layer, name).data.clone(), requires_grad=False)
            setattr(shadow, name, param)
        shadow.output_size_per_partition = layer.output_size_per_partition
        shadow.input_size_per_partition = layer.input_size_per_partition
        shadow.params_dtype = layer.params_dtype
        self.cute.process_weights_after_loading(shadow)
        self.marlin.process_weights_after_loading(layer)
        self._n = int(layer.output_size_per_partition)
        layer._nvfp4_dispatch_idx = len(_ENTRIES)
        _ENTRIES.append((self.marlin, layer, self.cute, shadow, self.threshold))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.nvfp4_w4a16_dispatch(
            x_2d, layer._nvfp4_dispatch_idx, self._n
        )
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], self._n)
