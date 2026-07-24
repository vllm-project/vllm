# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated-kernel manifests keyed by canonical GPU platform."""

from vllm.kernels.helion_generated.kernels.nvidia_b200.per_token_group_fp8_quant.manifest import (  # noqa: E501
    KERNELS as B200_KERNELS,
)
from vllm.kernels.helion_generated.kernels.nvidia_h100.per_token_group_fp8_quant.manifest import (  # noqa: E501
    KERNELS as H100_KERNELS,
)

MANIFESTS = {
    "nvidia_b200": B200_KERNELS,
    "nvidia_h100": H100_KERNELS,
}
