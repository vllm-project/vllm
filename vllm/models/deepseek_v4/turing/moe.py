from __future__ import annotations


def is_marlin_mxfp4_available() -> bool:
    """True when Marlin MXFP4 MoE is selectable on this device."""
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.has_device_capability((7, 5))


def expert_quant_activation() -> None:
    """Marlin MXFP4 consumes unquantized (FP16/BF16) activations."""
    return None
