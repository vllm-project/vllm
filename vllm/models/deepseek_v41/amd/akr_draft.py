# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 lm_head and DSpark draft projection on amd_kernels.

lm_head: the bf16 head GEMM behind `lm_head.quant_method.apply` (the draft
    shares the target's head), at the row counts where it beat vLLM's own
    dispatch. The row-major weight is used as it is.
main_proj: the draft's MXFP8 input projection of the concatenated target
    hidden states, from 640 rows (on par below).
"""

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41.amd import akr_config, akr_hooks, akr_kernels

logger = init_logger(__name__)


def _install_lm_head(head) -> bool:
    qm = head.quant_method
    if getattr(qm, "akr_stock_apply", None) is not None:
        return True
    if (
        head.weight.dtype != torch.bfloat16
        or not head.weight.is_cuda
        or not akr_kernels.bf16_linear_ok(head.weight)
    ):
        return False
    stock = qm.apply

    def apply(layer, x, bias=None):
        if (
            bias is None
            and x.dim() == 2
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and akr_config.serves("lm_head", x.shape[0])
        ):
            return akr_kernels.bf16_linear(x, layer.weight)
        return stock(layer, x, bias)

    qm.akr_stock_apply = stock
    qm.apply = apply
    return True


def main_proj(model, x: torch.Tensor) -> torch.Tensor | None:
    """`model.main_proj(x)` on amd_kernels, or None for the stock linear."""
    w = getattr(model, "akr_main_proj", None)
    if (
        w is None
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not akr_config.serves("main_proj", x.shape[0])
    ):
        return None
    return akr_kernels.mxfp8_dense(x, w)


def post_load(model) -> None:
    if akr_hooks.is_draft(model):
        if akr_config.on("main_proj"):
            inner = model.model
            inner.akr_main_proj = akr_kernels.mxfp8_dense_weight(inner.main_proj)
            if inner.akr_main_proj is None:
                akr_config.fail("main_proj", "main_proj is not an MXFP8 linear")
        return
    if not akr_config.on("lm_head"):
        return
    head = getattr(model, "lm_head", None)
    if head is None or not _install_lm_head(head):
        akr_config.fail("lm_head", "lm_head is not one rank's bf16 weight")
        return
    akr_kernels.bf16_linear_warmup(head.weight)
