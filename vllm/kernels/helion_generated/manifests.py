# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated-kernel manifests keyed by kernel and canonical GPU platform."""

from vllm.kernels.helion_generated.kernels.fused_qk_norm_rope.nvidia_h100.manifest import (  # noqa: E501
    KERNELS as H100_FUSED_QK_NORM_ROPE_KERNELS,
)
from vllm.kernels.helion_generated.kernels.per_token_group_fp8_quant.nvidia_b200.manifest import (  # noqa: E501
    KERNELS as B200_PER_TOKEN_GROUP_FP8_QUANT_KERNELS,
)
from vllm.kernels.helion_generated.kernels.per_token_group_fp8_quant.nvidia_h100.manifest import (  # noqa: E501
    KERNELS as H100_PER_TOKEN_GROUP_FP8_QUANT_KERNELS,
)
from vllm.kernels.helion_generated.kernels.rms_norm_per_block_quant.nvidia_h100.manifest import (  # noqa: E501
    KERNELS as H100_RMS_NORM_PER_BLOCK_QUANT_KERNELS,
)
from vllm.kernels.helion_generated.kernels.silu_and_mul_per_block_quant.nvidia_h100.manifest import (  # noqa: E501
    KERNELS as H100_SILU_AND_MUL_PER_BLOCK_QUANT_KERNELS,
)

GENERATED_KERNEL_MANIFESTS = {
    "fused_qk_norm_rope": {
        "nvidia_h100": H100_FUSED_QK_NORM_ROPE_KERNELS,
    },
    "per_token_group_fp8_quant": {
        "nvidia_b200": B200_PER_TOKEN_GROUP_FP8_QUANT_KERNELS,
        "nvidia_h100": H100_PER_TOKEN_GROUP_FP8_QUANT_KERNELS,
    },
    "rms_norm_per_block_quant": {
        "nvidia_h100": H100_RMS_NORM_PER_BLOCK_QUANT_KERNELS,
    },
    "silu_and_mul_per_block_quant": {
        "nvidia_h100": H100_SILU_AND_MUL_PER_BLOCK_QUANT_KERNELS,
    },
}

MANIFESTS = GENERATED_KERNEL_MANIFESTS["per_token_group_fp8_quant"]
