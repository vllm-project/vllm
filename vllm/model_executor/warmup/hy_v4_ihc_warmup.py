# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the HY V4 Triton iHC kernels before serving requests.

``triton_ihc_pre`` / ``post`` / ``head`` run in every decoder layer, and the
large-batch (tensor-core) path's compile keys depend on the token count, so
``warmup_token_sizes`` enumerates one representative batch size per key and
each is launched once here instead of on the first request that hits it.
No-op for other models and for the HPC / eager paths.
"""

import time

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_WARMUP_MAX_TOKENS = 16_384


@instrument(span_name="HY V4 iHC warmup")
def hy_v4_ihc_warmup(
    model: torch.nn.Module,
    *,
    dtype: torch.dtype,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "hy_v4":
        return

    from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer
    from vllm.models.hy_v4.nvidia.triton_ihc import (
        triton_ihc_supported,
        warmup_token_sizes,
    )

    hc_layer = next(
        (m for m in model.modules() if isinstance(m, HYV4HCLayer) and m.enable_ihc),
        None,
    )
    if hc_layer is None or hc_layer.hc_pre.hpc_op is not None:
        return
    device = hc_layer.hc_pre.hc_fn.weight.device
    hidden_size = int(config.hidden_size)
    hc_mult = int(config.hc_mult)
    max_tokens = min(max_tokens, _WARMUP_MAX_TOKENS)
    sizes = set(warmup_token_sizes(hidden_size, hc_mult, max_tokens, device.index or 0))
    sizes.update(s for s in (cudagraph_capture_sizes or []) if 1 <= s <= max_tokens)
    if not sizes:
        return
    x = torch.zeros(max(sizes), hc_mult, hidden_size, dtype=dtype, device=device)
    if not triton_ihc_supported(x):
        return
    head = next((m for m in model.modules() if isinstance(m, HYV4HCHeadLayer)), None)

    started = time.perf_counter()
    with torch.inference_mode():
        for size in sorted(sizes):
            reduced, post_gates, residual = hc_layer.pre(x[:size])
            hc_layer.post(reduced, residual, post_gates)
            if head is not None and head.hpc_op is None:
                head(x[:size])
        torch.accelerator.synchronize()
    logger.info(
        "HY V4 iHC Triton warmup: %d token sizes in %.2f s",
        len(sizes),
        time.perf_counter() - started,
    )
