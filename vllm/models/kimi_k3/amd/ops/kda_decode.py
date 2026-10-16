# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry points for the fused Kimi-K3 KDA decode kernel.

The kernel in ``csrc/kimi_k3/fused_kda_decode_kernel_rocm.cu``
replaces, for a pure non-speculative decode batch, the three Triton launches
and two copies the AMD KDA layer otherwise runs per layer: the packed causal
conv1d update, the recurrent delta-rule step, and the gated output RMSNorm.

The kernel wants a width-major conv weight and an fp32 norm weight, so the
layer stages both once in ``process_weights_after_loading``.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

logger = init_logger(__name__)

# Head counts the kernel is instantiated for (Kimi-K3 has 96 KDA heads, so this
# covers TP1/2/4/8).
SUPPORTED_NUM_HEADS = (12, 24, 48, 96)


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    """Whether the fused decode kernel can serve this layer on this device."""
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    if (
        num_heads not in SUPPORTED_NUM_HEADS
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
        or is_conv_state_dim_first()
        or not hasattr(torch.ops._C, "fused_kda_decode")
    ):
        return False
    # gfx950 (MI355X) and gfx942 (MI325X): both CDNA, sharing the wave64 / DPP /
    # bf16 primitives the kernel relies on.
    return on_gfx950() or on_gfx942()
