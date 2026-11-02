# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch LM head: FlashInfer's CuTe DSL ``mm_bf16`` (split-K) for up to
``LM_HEAD_MAX_ROWS`` rows.

``SpecializedDecodeLogitsProcessor`` projects through it, so a model that shares the
head module but not the processor keeps the default GEMM. The runner computes
logits outside the CUDA graph, and the GEMM has no cross-rank state, so the
gate is the row count alone.
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.utils import _can_use_flashinfer_cutedsl_bf16
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_bf16_mm,
    is_flashinfer_cutedsl_bf16_gemm_supported,
)

logger = init_logger(__name__)

LM_HEAD_MAX_ROWS = 16


def low_batch_lm_head_applies(lm_head: nn.Module, x: torch.Tensor) -> bool:
    """Whether ``low_batch_lm_head`` takes ``x``: an unquantized BF16 head
    and 1 to ``LM_HEAD_MAX_ROWS`` rows."""
    return (
        isinstance(lm_head, ParallelLMHead)
        and type(lm_head.quant_method) is UnquantizedEmbeddingMethod
        and x.dim() == 2
        and 0 < x.shape[0] <= LM_HEAD_MAX_ROWS
        and is_flashinfer_cutedsl_bf16_gemm_supported()
        and _can_use_flashinfer_cutedsl_bf16(x, lm_head.weight, None)
    )


def low_batch_lm_head(lm_head: nn.Module, x: torch.Tensor, pdl: bool) -> torch.Tensor:
    return flashinfer_bf16_mm(x, lm_head.weight.t(), None, pdl, "cute-dsl")


def _zeros(head: ParallelLMHead, rows: int) -> torch.Tensor:
    weight = head.weight
    return torch.zeros(
        (rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
    )


def _heads(model: nn.Module) -> list[ParallelLMHead]:
    """The LM heads of ``model`` that the GEMM takes."""
    heads = []
    for head in model.modules():
        if not isinstance(head, ParallelLMHead):
            continue
        if low_batch_lm_head_applies(head, _zeros(head, 1)):
            heads.append(head)
        else:
            logger.warning_once(
                "The LM head keeps its default GEMM: it is not an unquantized "
                "BF16 head, or FlashInfer CuTe DSL mm_bf16 is unavailable."
            )
    return heads


def autotune_low_batch_lm_head(model: nn.Module) -> None:
    """Tune each power-of-two row bucket inside FlashInfer's autotune context."""
    pdl = current_platform.is_arch_support_pdl()
    for head in _heads(model):
        for rows in (1 << i for i in range(LM_HEAD_MAX_ROWS.bit_length())):
            low_batch_lm_head(head, _zeros(head, rows), pdl)


def warmup_low_batch_lm_head(model: nn.Module) -> None:
    """Compile the selected kernel for every row count (CuTe DSL compiles per
    M)."""
    pdl = current_platform.is_arch_support_pdl()
    for head in _heads(model):
        for rows in range(1, LM_HEAD_MAX_ROWS + 1):
            low_batch_lm_head(head, _zeros(head, rows), pdl)
