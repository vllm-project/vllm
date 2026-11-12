# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp GEMM selection on AMD CDNA3.

Linear layers whose local `(N, K)` shape has rows in AITER's tuned A16W16
table for this device use AITER `tgemm` at token counts with a tuned row.
Other token counts use the standard ROCm linear implementation.
"""

import functools

import torch
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.utils import (
    lmhead_skinny_guard_permits,
    rocm_unquantized_gemm_impl,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# The standard path serves up to this many tokens with the wvSplitK and LLMM1
# skinny kernels, except weights the LM-head guard sends to the generic GEMM.
_SKINNY_MAX_TOKENS = 5
_PROBE_TOKENS = (*range(8, 513, 8), *(1 << i for i in range(10, 17)))


@functools.lru_cache(maxsize=4096)
def _has_tuned_row(
    num_tokens: int, out_features: int, in_features: int, dtype: torch.dtype
) -> bool:
    from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config

    config = get_GEMM_A16W16_tuned_config(
        num_tokens, out_features, in_features, False, str(dtype), str(dtype)
    )
    # A torch row means tuning found nothing faster than F.linear.
    return config is not None and config["libtype"] != "torch"


class _Qwen4ExpTgemmApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_rocm_tgemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpTgemmLinearMethod(_Qwen4ExpTgemmApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpTgemmEmbeddingMethod(_Qwen4ExpTgemmApply, UnquantizedEmbeddingMethod):
    pass


def _qwen4_exp_rocm_tgemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    num_tokens = x.numel() // x.shape[-1]
    skinny = num_tokens <= _SKINNY_MAX_TOKENS and lmhead_skinny_guard_permits(
        weight, num_tokens
    )
    if not skinny and _has_tuned_row(
        num_tokens, weight.shape[0], weight.shape[1], x.dtype
    ):
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, weight)
    return rocm_unquantized_gemm_impl(x, weight)


def _qwen4_exp_rocm_tgemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_rocm_tgemm",
    op_func=_qwen4_exp_rocm_tgemm,
    fake_impl=_qwen4_exp_rocm_tgemm_fake,
)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """On CDNA3, route linears with tuned AITER rows to `tgemm`."""
    if (
        dtype != torch.bfloat16
        or not current_platform.is_rocm()
        or not rocm_aiter_ops.is_linear_enabled()
    ):
        return
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() != 3:
        return

    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        if not any(
            _has_tuned_row(t, weight.shape[0], weight.shape[1], dtype)
            for t in _PROBE_TOKENS
        ):
            continue
        if is_linear:
            child.quant_method = Qwen4ExpTgemmLinearMethod()
        else:
            child.quant_method = Qwen4ExpTgemmEmbeddingMethod()
