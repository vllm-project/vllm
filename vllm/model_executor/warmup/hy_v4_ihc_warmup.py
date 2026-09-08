# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the HY V4 Triton iHC kernels before serving requests.

``triton_ihc_pre`` / ``post`` / ``head`` run in every decoder layer. Their
compile keys depend only on the model shape (hidden size, ``hc_mult``, eps and
magnitude constants), so one launch per op compiles everything the model will
ever use. No-op for other models and for the HPC / eager paths.
"""

import time

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_WARMUP_TOKENS = (1, 16)


@instrument(span_name="HY V4 iHC warmup")
def hy_v4_ihc_warmup(model: torch.nn.Module, *, dtype: torch.dtype) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "hy_v4":
        return

    from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer
    from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_supported

    hc_layer = next(
        (m for m in model.modules() if isinstance(m, HYV4HCLayer) and m.enable_ihc),
        None,
    )
    if hc_layer is None or hc_layer.hc_pre.hpc_op is not None:
        return
    device = hc_layer.hc_pre.hc_fn.weight.device
    hidden_size = int(config.hidden_size)
    hc_mult = int(config.hc_mult)
    x = torch.zeros(
        max(_WARMUP_TOKENS), hc_mult, hidden_size, dtype=dtype, device=device
    )
    if not triton_ihc_supported(x):
        return
    head = next((m for m in model.modules() if isinstance(m, HYV4HCHeadLayer)), None)

    started = time.perf_counter()
    with torch.inference_mode():
        for size in _WARMUP_TOKENS:
            reduced, post_gates, residual = hc_layer.pre(x[:size])
            hc_layer.post(reduced, residual, post_gates)
            if head is not None and head.hpc_op is None:
                head(x[:size])
        torch.accelerator.synchronize()
    logger.info(
        "HY V4 iHC Triton warmup finished in %.2f s", time.perf_counter() - started
    )
