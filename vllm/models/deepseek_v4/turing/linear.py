from __future__ import annotations


def is_marlin_fp8_block_supported() -> bool:
    """Marlin runs on 7.5+; block-FP8 weights cover the dense matmuls."""
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    supported, _ = MarlinFP8ScaledMMLinearKernel.is_supported()
    return supported
