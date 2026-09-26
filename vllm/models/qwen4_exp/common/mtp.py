# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared MTP drafter state for Qwen4Exp."""

import torch

from vllm.config import VllmConfig
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig


def make_mtp_hidden_buffer(
    vllm_config: VllmConfig,
    config: Qwen4ExpTextConfig,
    *,
    is_last_rank: bool,
) -> torch.Tensor | None:
    """Buffer holding the pre-final-mixer multi-stream hidden state, or None.

    When speculative ``method == "mtp"`` the drafter feeds a real multi-stream
    backbone hidden on its first step, so the last stage retains
    ``[T, hc_count * H]``. Whether it is needed is derived from the config
    alone, not from node identity, so prefill and decode nodes agree.
    """
    spec_config = vllm_config.speculative_config
    needs_mtp_hidden = (
        spec_config is not None
        and getattr(spec_config, "method", None) == "mtp"
        and is_last_rank
    )
    if not needs_mtp_hidden:
        return None
    return torch.empty(
        vllm_config.scheduler_config.max_num_batched_tokens,
        config.hc_count * config.hidden_size,
        dtype=vllm_config.model_config.dtype,
        device=vllm_config.device_config.device,
    )
