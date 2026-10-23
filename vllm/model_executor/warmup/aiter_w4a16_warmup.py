# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile AITER gemm_a16w4 for every M bucket before serving.

AITER picks a tuned config per power-of-two M bucket (M_LEQ_1 ... M_LEQ_8192,
then "any"), and Triton further specializes M on M == 1 and M % 16 == 0.
CUDA graph capture only covers the decode sizes, so prefill chunks would
otherwise compile new variants during serving.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _warmup_m_values(max_tokens: int) -> list[int]:
    m_values = {1, max_tokens, max_tokens - 1}
    b = 2
    while b <= max_tokens:
        m_values.update((b, b - 1))
        b *= 2
    return sorted(m for m in m_values if 1 <= m <= max_tokens)


@torch.inference_mode()
def aiter_w4a16_warmup(model: torch.nn.Module, max_tokens: int) -> None:
    from vllm.model_executor.kernels.linear.mixed_precision.aiter_w4a16 import (
        AiterW4A16LinearKernel,
    )

    m_values = _warmup_m_values(max_tokens)
    seen = set()
    x_buf = None
    for module in model.modules():
        # GPTQ/AWQ keep the kernel on quant_method, compressed-tensors on scheme.
        kernels = [
            getattr(getattr(module, owner, None), "kernel", None)
            for owner in ("quant_method", "scheme")
        ]
        kernel = next(
            (k for k in kernels if isinstance(k, AiterW4A16LinearKernel)), None
        )
        if kernel is None:
            continue
        w_q, w_s, w_zp = kernel._get_weight_params(module)
        bias = (
            None
            if getattr(module, "skip_bias_add", False)
            else getattr(module, "bias", None)
        )
        act_type = kernel.config.act_type
        key = (
            tuple(w_q.shape),
            tuple(w_s.shape),
            w_s.dtype,
            None if w_zp is None or not kernel.config.zero_points else w_zp.dtype,
            None if bias is None else bias.dtype,
            act_type,
        )
        if key in seen:
            continue
        seen.add(key)
        K = kernel.config.partition_weight_shape[0]
        if x_buf is None or x_buf.shape[1] != K or x_buf.dtype != act_type:
            x_buf = torch.zeros((max_tokens, K), dtype=act_type, device=w_q.device)
        for m in m_values:
            kernel.apply_weights(module, x_buf[:m], bias)
    if seen:
        torch.accelerator.synchronize()
        logger.info(
            "Warmed up AITER gemm_a16w4 for %d weight shapes x %d M values.",
            len(seen),
            len(m_values),
        )
