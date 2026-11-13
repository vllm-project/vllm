# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels ``top_p`` route: the sampler's split top-p chain (<= 64 rows, top-p
without top-k: seven Triton launches) as one launch of
``amd_kernels.sampling.top_p_mask_f32_v200064``, bit for bit (same masked logits)."""

import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd import akr_config

logger = init_logger(__name__)

_STATE: dict[str, str | None] = {}


def configure(vllm_config) -> None:
    """From the target's ``__init__``: check, warm up and set the sampler's hook."""
    if "blocker" in _STATE or not akr_config.on("top_p"):
        return
    from amd_kernels.sampling.top_p_mask_f32_v200064 import (
        V,
        top_p_blocker,
        top_p_mask,
        warmup_top_p_mask,
    )

    from vllm.v1.sample.ops import topk_topp_triton

    why = _STATE["blocker"] = top_p_blocker(
        vllm_config.model_config.get_vocab_size(), float("-inf")
    )
    if why is not None:
        logger.warning("amd_kernels top_p stays off: %s", why)
        return
    warmup_top_p_mask(torch.device("cuda", torch.accelerator.current_device_index()))

    def split_top_p(logits: torch.Tensor, p: torch.Tensor, mask_value: float) -> bool:
        if mask_value != float("-inf") or logits.shape[1] != V:
            return False
        if not p.is_contiguous():
            return False
        top_p_mask(logits, p)
        return True

    topk_topp_triton.split_top_p_override = split_top_p
    logger.info("amd_kernels top_p: the split top-p chain runs top_p_mask")
