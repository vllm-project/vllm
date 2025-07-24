# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

logger = init_logger(__name__)


def is_valid_flashinfer_cutlass_fused_moe(hidden_states: torch.Tensor,
                                          w1: torch.Tensor,
                                          w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the FlashInfer CUTLASS MoE 
    kernel.
    """
    if not has_flashinfer_cutlass_fused_moe():
        logger.debug_once("FlashInferExperts disabled: "
                          "flashinfer_cutlass_fused_moe not available.")
        return False
    # Data type checks
    if (w1.dtype != torch.uint8 or w2.dtype != torch.uint8
            or hidden_states.dtype
            not in [torch.float32, torch.float16, torch.bfloat16]):
        logger.debug_once(
            "FlashInferExperts disabled: w1/w2 must be torch.uint8 "
            f"(got w1={w1.dtype}, w2={w2.dtype}), hidden_states must be "
            f"float32, float16, or bfloat16 (got {hidden_states.dtype}).")
        return False
    return True
