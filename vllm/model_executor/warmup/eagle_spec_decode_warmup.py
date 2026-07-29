# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup Eagle spec-decode Triton kernels via VllmJitKernel wrappers.

No-op when Eagle spec decoding is not configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _warmup_eagle_prepare_next_token_kernel(vllm_config: VllmConfig) -> None:
    from vllm.v1.spec_decode.utils import _EAGLE_PREPARE_NEXT_TOKEN_KERNEL

    _EAGLE_PREPARE_NEXT_TOKEN_KERNEL.warmup(vllm_config)


def _warmup_eagle_prepare_inputs_kernel(vllm_config: VllmConfig) -> None:
    from vllm.v1.spec_decode.utils import _EAGLE_PREPARE_INPUTS_KERNEL

    _EAGLE_PREPARE_INPUTS_KERNEL.warmup(vllm_config)


def _warmup_mtp_shared_head_rmsnorm_kernel(vllm_config: VllmConfig) -> None:
    from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
        _MTP_SHARED_HEAD_RMSNORM_KERNEL,
    )

    _MTP_SHARED_HEAD_RMSNORM_KERNEL.warmup(vllm_config)


def _warmup_eagle_step_slot_mapping_metadata_kernel(
    vllm_config: VllmConfig,
    block_size: int,
    max_model_len: int,
) -> None:
    from vllm.v1.spec_decode.utils import _EAGLE_STEP_SLOT_MAPPING_KERNEL

    _EAGLE_STEP_SLOT_MAPPING_KERNEL.warmup(
        vllm_config,
        block_size=block_size,
        max_model_len=max_model_len,
    )


def eagle_eagle_kernel_warmup(
    device: torch.device,
    num_speculative_tokens: int | None,
    vllm_config: VllmConfig,
    block_size: int | None = None,
    max_model_len: int | None = None,
) -> None:
    if num_speculative_tokens is None or num_speculative_tokens <= 0:
        return
    if device.type != "cuda":
        return

    logger.info(
        "Warming up Eagle spec-decode kernels on %s (num_speculative_tokens=%d)",
        device,
        num_speculative_tokens,
    )

    _warmup_eagle_prepare_next_token_kernel(vllm_config)
    _warmup_eagle_prepare_inputs_kernel(vllm_config)
    _warmup_mtp_shared_head_rmsnorm_kernel(vllm_config)
    if block_size is not None and max_model_len is not None:
        _warmup_eagle_step_slot_mapping_metadata_kernel(
            vllm_config, block_size, max_model_len
        )

    logger.info("Eagle spec-decode kernel warmup finished.")
